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
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def load_dataset_chunk(residual_stream: h5py.Dataset, chunk_number: int, num_chunks_needed: int):
    total_size = len(residual_stream)
    chunk_size = math.ceil(total_size / num_chunks_needed)

    if dist.get_rank() == 0:
        start = chunk_number * chunk_size
        end = min(start + chunk_size, total_size)
        vectors = torch.from_numpy(residual_stream[start:end]).view(torch.bfloat16)
    else:
        # Other ranks just need to know the size
        vectors = torch.empty(chunk_size, residual_stream.shape[1], dtype=torch.bfloat16)
    return vectors

def calculate_num_chunks_needed(residual_stream, memory_tolerance: float=0.8):
    total_vectors, vector_size  = residual_stream.shape[0], residual_stream.shape[1]
    bytes_per_value = 2 #vectors are saved as bfloat16
    total_dataset_size = (total_vectors * vector_size * bytes_per_value) / (1024**3)
    total_system_memory = psutil.virtual_memory().total / (1024**3) 
    num_chunks_needed = math.ceil(total_dataset_size / (total_system_memory*memory_tolerance))

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
    
    # Initialize feature history buffer for dead neuron detection
    feature_history = torch.zeros(feature_history_steps, hidden_size, dtype=torch.bool, device=device)
    history_idx = 0
    
    for step in tqdm(range(num_steps)):
        current_chunk = step // steps_per_chunk

        if step % steps_per_chunk == 0:
            if step > 0:
                del vectors
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            vectors = load_dataset_chunk(residual_stream, current_chunk, num_chunks_needed)

        local_step = step % steps_per_chunk
        rank = dist.get_rank()
        start_idx = local_step * (batch_size * dist.get_world_size()) + (rank * batch_size)
        end_idx = min(start_idx + batch_size, len(vectors))
        batch = vectors[start_idx:end_idx].cuda(dist.get_rank())

        current_lambda = min(lambda_l1 * (step  / lambda_ramp_steps), lambda_l1) if lambda_ramp_steps > 0 else lambda_l1

        #forward pass

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
        
        # Update feature history buffer
        feature_history[history_idx] = (features.abs().sum(0) > 0)  # Record which features were active
        history_idx = (history_idx + 1) % feature_history_steps
        
        # Logging
        if step % 100 == 0:
            # Compute losses and metrics
            with torch.no_grad():
                mse_loss = F.mse_loss(reconstruction, batch)
                decoder_norms = model.module.get_decoder_norms()  # Note: also need module here
                l1_loss = torch.sum(torch.abs(features) * decoder_norms[None, :]) / (batch.shape[0] * features.shape[1])
                
                # Calculate feature metrics
                dead_features = (~feature_history.any(dim=0)).sum().item()
                l0_norm = (features != 0).float().sum().item()
            
            if dist.get_rank() == 0:  # Only log on rank 0
                wandb.log({
                    'step': step,
                    'loss': loss.item(),
                    'mse_loss': mse_loss.item(),
                    'l1_loss': l1_loss.item(),
                    'lambda': current_lambda,
                    'lr': optimizer.param_groups[0]['lr'],
                    'dead_features': dead_features,
                    'dead_features_pct': dead_features / features.shape[1] * 100,
                    'l0_norm': l0_norm / batch.shape[0],
                    'l0_norm_pct': l0_norm / (batch.shape[0] * features.shape[1]) * 100
                })
            
        # Save best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            if dist.get_rank() == 0:
                save_model(model.to('cpu'), str(out_dir / 'best_model.safetensors'))
            model.to(device)
            
if __name__ == '__main__':
    setup_distributed()

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
    dist.destroy_process_group()