import os
import importlib.util
import math
import numpy as np
import psutil
import torch
import torch.nn.functional as F
import torch.distributed as dist
import h5py
import wandb
import bitsandbytes as bnb
from tqdm import tqdm
from pathlib import Path
from safetensors.torch import save_model
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from model import SAE

def setup_distributed():
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
    
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def evaluate_dead_features(model, vectors, batch_size=2048):
    """
    Evaluate which features are dead using the final chunk of data.
    Only runs on rank 0.
    
    Args:
        model: DDP-wrapped model
        vectors: tensor of input vectors
        batch_size: batch size for evaluation
    
    Returns:
        numpy array of dead feature indices (rank 0 only)
        None for other ranks
    """
    if dist.get_rank() != 0:
        return None

    print("Evaluating dead features...")
    model.eval()
    device = next(model.parameters()).device
    feature_activations = torch.zeros(model.module.hidden_size, dtype=torch.bool, device=device)
    
    with torch.no_grad():
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size].to(device)
            batch = normalize_batch(batch)
            _, features = model(batch)
            # Mark any feature that activates at all as alive
            feature_activations |= (features.abs().sum(dim=0) > 0)
    
    dead_features = (~feature_activations).cpu().numpy()
    num_dead = dead_features.sum()
    dead_percentage = (num_dead / len(dead_features)) * 100
    
    print(f"Found {num_dead} dead features ({dead_percentage:.2f}%)")
    
    if wandb.run is not None:
        wandb.log({
            'num_dead_features': num_dead,
            'dead_features_percentage': dead_percentage,
            'final_dead_features': dead_features
        })
    
    return dead_features

#this got a little crazy, but it works.
def load_dataset_chunk(residual_stream: h5py.Dataset, chunk_number: int, num_chunks_needed: int):
    total_size = len(residual_stream)
    chunk_size = math.ceil(total_size / num_chunks_needed)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Calculate rank's portion of the chunk
    rank_subchunk_size = math.ceil(chunk_size / world_size)  # Use ceil to ensure we cover all data
    rank_start = chunk_number * chunk_size + (rank * rank_subchunk_size)
    rank_end = min(rank_start + rank_subchunk_size, 
                  min((chunk_number + 1) * chunk_size, total_size))  # Don't exceed chunk or dataset bounds
    
    # Handle edge case where this rank has no data
    if rank_start >= total_size:
        # Get last valid chunk of data instead
        rank_end = total_size
        rank_start = max(rank_end - rank_subchunk_size, (chunk_number * chunk_size))
    
    vectors = torch.from_numpy(residual_stream[rank_start:rank_end]).view(torch.bfloat16)
    return vectors

def calculate_num_chunks_needed(residual_stream, memory_tolerance: float=0.75):
    total_vectors, vector_size  = residual_stream.shape[0], residual_stream.shape[1]
    bytes_per_value = 2 #vectors are saved as bfloat16
    total_dataset_size = (total_vectors * vector_size * bytes_per_value) / (1024**3)
    total_system_memory = psutil.virtual_memory().total / (1024**3) 
    num_chunks_needed = math.ceil(total_dataset_size / (total_system_memory*memory_tolerance))
    print(f"num_chunks_needed: {num_chunks_needed}")

    return num_chunks_needed

def normalize_batch(batch: torch.Tensor): #normalize the batch according to Anthropic's specs
    target_norm = math.sqrt(batch.shape[1]) #sqrt(input_size)
    current_norms = torch.norm(batch, dim=1, keepdim=True)
    mean_norm = current_norms.mean()

    return batch * (target_norm / mean_norm)

def train(
        data_path: str,
        out_dir: str,
        input_size: int,
        hidden_size: int,
        init_scale: float = 0.1,
        batch_size: int = 2048,
        learning_rate: float = 5e-5,
        num_epochs: int = 1,
        lambda_l1: float = 5.0,
        lambda_ramp_frac: float = 0.05,
        lr_decay_frac: float = 0.2,
        seed: int = 42,
        wandb_project: str = "sae-training",
        wandb_name: str = None,
        wandb_entity: str = None,
        ):
    try:
        device = f'cuda'

        torch.manual_seed(seed)
        if dist.get_rank() == 0:
            wandb.init(
                    project=wandb_project,
                    name=wandb_name,
                    entity=wandb_entity,
                    config=locals()
                    )
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        model = SAE(input_size, hidden_size, init_scale).to(device).to(torch.bfloat16) #not using autocast since our data are bfloat16
        model = torch.compile(model, mode='max-autotune',fullgraph=True)
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)
        
        #Identify which neurons have not fired in any of the previous 12,500 training steps (quoting from towards monosemanticity)
        feature_history_steps = 12500
        feature_check_steps = [25000, 50000, 50000, 75000, 100000]

        f = h5py.File(data_path, 'r')
        residual_stream = f['residual_stream']

        world_size = dist.get_world_size()
        effective_batch_size = batch_size * world_size
        num_steps = int(np.ceil(len(residual_stream)/ effective_batch_size))
        lambda_ramp_steps = int(num_steps * lambda_ramp_frac) #how many steps to take to ramp lambda to full value
        lr_decay_start = int(num_steps * (1 - lr_decay_frac)) #how many steps in where we ramp lr down to 0


        #for now, just load a chunk of the dataset into memory. quick and dirty.
        num_chunks_needed = calculate_num_chunks_needed(residual_stream)
        steps_per_chunk = num_steps // num_chunks_needed
        best_loss = float('inf')
        
        for step in tqdm(range(num_steps)):
            current_chunk = step // steps_per_chunk

            if step % steps_per_chunk == 0:
                # Initialize vectors for all ranks
                if step > 0:
                    del vectors
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                vectors = load_dataset_chunk(residual_stream, current_chunk, num_chunks_needed)

            local_step = step % steps_per_chunk
            rank = dist.get_rank()
            batch_start = local_step * (batch_size * world_size) + (rank * batch_size)
            batch_end = min(batch_start + batch_size, len(vectors))
            
            # Handle edge case where we're at the end of the dataset
            if batch_start >= len(vectors):
                continue
            
            batch = vectors[batch_start:batch_end].cuda(dist.get_rank())
            
            # Skip if batch is too small
            if batch.shape[0] < batch_size // 2:  # Skip if less than half batch size
                continue

            current_lambda = min(lambda_l1 * (step  / lambda_ramp_steps), lambda_l1) if lambda_ramp_steps > 0 else lambda_l1

            # Forward pass
            batch = normalize_batch(batch)
            reconstruction, features = model(batch)
            
            # Compute loss using the SAE class method
            loss = model.module.compute_loss(batch, reconstruction, features, lambda_=current_lambda)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Learning rate decay in last 20% of training
            if step > lr_decay_start:
                current_lr = learning_rate * (1 - (step - lr_decay_start) / (num_steps - lr_decay_start))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            
            # Optimizer step and zero gradients
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            # Logging
            if step % 100 == 0:
                with torch.no_grad():
                    mse_loss = F.mse_loss(reconstruction, batch)
                    decoder_norms = model.module.get_decoder_norms()
                    l1_loss = torch.sum(torch.abs(features) * decoder_norms[None, :]) / (batch.shape[0] * features.shape[1])
                    
                    # Calculate L0 norm (number of non-zero elements)
                    l0_per_sample = torch.count_nonzero(features, dim=1).float().mean().item()
                    l0_pct = (l0_per_sample / features.shape[1]) * 100  # Percentage of active features
                    
                if dist.get_rank() == 0:  # Only log on rank 0
                    wandb.log({
                        'step': step,
                        'loss': loss.item(),
                        'mse_loss': mse_loss.item(),
                        'l1_loss': l1_loss.item(),
                        'lambda': current_lambda,
                        'lr': optimizer.param_groups[0]['lr'],
                        'l0_norm': l0_per_sample,
                        'l0_norm_pct': l0_pct
                    })
            
            # Save best model
            if loss.item() < best_loss:
                best_loss = loss.item()
                if dist.get_rank() == 0:
                    save_model(model.to('cpu'), str(out_dir / 'best_model.safetensors'))
                model.to(device)

        # Final save at end of training
        try:
            print("evaluating dead features...")
            if dist.get_rank() == 0:
                dead_features = evaluate_dead_features(model, vectors)
                if dead_features is not None:
                    np.save(str(out_dir / 'dead_features.npy'), dead_features)
            
            # Now safe to move model to CPU and save
            print("final save...")
            model_to_save = model.module
            model_to_save = model_to_save.to('cpu')
            print("saving...")
            save_model(model_to_save, str(out_dir / 'final_model.safetensors'))
            print("finished saving")
            
            wandb.finish()
            
        except Exception as e:
            print(f"Error during final save or evaluation: {e}")

    except Exception as e:
        print(f"Error during training: {e}")
        raise e
    finally:
        # Ensure cleanup happens even if there's an error
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception as e:
                print(f"Error during cleanup: {e}")

if __name__ == '__main__':
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        setup_distributed()
        # run with torchrun --nproc_per_node=num_gpus train.py config.py
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('config', type=str, help='Path to config file')
        
        # Optional override arguments
        parser.add_argument('--data_path', type=str)
        parser.add_argument('--out_dir', type=str)
        parser.add_argument('--input_size', type=int)
        parser.add_argument('--hidden_size', type=int)
        parser.add_argument('--init_scale', type=float)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument('--learning_rate', type=float)
        parser.add_argument('--num_epochs', type=int)
        parser.add_argument('--lambda_l1', type=float)
        parser.add_argument('--lambda_ramp_frac', type=float)
        parser.add_argument('--lr_decay_frac', type=float)
        parser.add_argument('--seed', type=int)
        parser.add_argument('--wandb_project', type=str)
        parser.add_argument('--wandb_name', type=str)
        parser.add_argument('--wandb_entity', type=str)

        args = parser.parse_args()

        spec = importlib.util.spec_from_file_location("config", args.config)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config = config_module.config

        #override the config if an override argument is added
        cmd_line_args = {k: v for k, v in vars(args).items()
                        if k != 'config' and v is not None}
        config.update(cmd_line_args)

        train(**config)
    except Exception as e:
        print(f"Error in main: {e}")
        # Ensure cleanup happens even if there's an error in main
        if dist.is_initialized():
            dist.destroy_process_group()
        raise e
