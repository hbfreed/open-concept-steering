import os
import math
import json
import torch
import torch.nn as nn
import numpy as np
import zarr
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LambdaLR
import bitsandbytes as bnb
from pathlib import Path
from tqdm import tqdm
import argparse
import wandb
from model import SAE

class ZarrDataset(Dataset):
    """Dataset for loading .zarr directories stored in the data directory with batch loading"""
    def __init__(self, data_dir, batch_size=4096, normalize=True):
        self.data_paths = []
        self.normalize = normalize
        self.batch_size = batch_size
        
        # Find all zarr directories
        for path in Path(data_dir).glob("**/*.zarr"):
            if path.is_dir():
                self.data_paths.append(path)
                
        if not self.data_paths:
            raise ValueError(f"No .zarr directories found in {data_dir}")
        print(f"Found {len(self.data_paths)} .zarr directories")
        
        # Get dimensions from first zarr file
        first_zarr = zarr.open(str(self.data_paths[0]), mode='r')
        self.vector_dim = first_zarr.shape[1]
        
        # Calculate size of each zarr file and total number of vectors
        self.total_vectors = 0
        self.zarr_sizes = []
        self.zarr_cumulative_sizes = [0]  # Starts with 0
        
        for path in self.data_paths:
            z = zarr.open(str(path), mode='r')
            size = z.shape[0]
            self.zarr_sizes.append(size)
            self.total_vectors += size
            self.zarr_cumulative_sizes.append(self.zarr_cumulative_sizes[-1] + size)
            
        print(f"Total vectors: {self.total_vectors}, Vector dimension: {self.vector_dim}, Total batches: {self.total_vectors//self.batch_size}")
        
        # Normalization
        self._norm_factor = None
        
    def __len__(self):
        return self.total_vectors // self.batch_size  # Return number of batches
    
    def get_normalization_factor(self):
        """Calculate normalization factor so E[||x||₂] = √n"""
        if self._norm_factor is not None:
            return self._norm_factor
            
        print("Calculating normalization factor...")
        num_samples = min(10000, self.total_vectors)
        sample_indices = np.random.choice(self.total_vectors, num_samples, replace=False)
        
        # Group indices by zarr file for efficient loading
        grouped_indices = {}
        for idx in sample_indices:
            zarr_idx, local_idx = self._get_zarr_indices(idx)
            if zarr_idx not in grouped_indices:
                grouped_indices[zarr_idx] = []
            grouped_indices[zarr_idx].append(local_idx)
        
        # Sum of norms (not squared norms)
        sum_norms = 0.0
        for zarr_idx, local_indices in tqdm(grouped_indices.items(), desc="Sampling for normalization"):
            z = zarr.open(str(self.data_paths[zarr_idx]), mode='r')
            # Load all samples at once for this zarr file
            batch = torch.from_numpy(z[local_indices]).view(torch.bfloat16)
            # Calculate norms and sum them
            norms = torch.linalg.vector_norm(batch, ord=2, dim=1)
            sum_norms += norms.sum().item()
            
        # Average L2 norm
        avg_norm = sum_norms / num_samples
        
        # Target norm is sqrt(n) where n is input dimension
        target_norm = math.sqrt(self.vector_dim)
        print(f"Target norm:{target_norm}, Avg norm: {avg_norm}")
        self._norm_factor = target_norm / avg_norm
        print(f"Normalization factor: {self._norm_factor}")
        return self._norm_factor
    
    def _get_zarr_indices(self, global_idx: int):
        """Convert global index to (zarr_file_idx, local_idx)"""
        for i in range(len(self.zarr_sizes)):
            if global_idx < self.zarr_cumulative_sizes[i+1]:
                return i, global_idx - self.zarr_cumulative_sizes[i]
        raise IndexError(f"Index {global_idx} out of bounds")
    
    def __getitem__(self, batch_idx):
        # Convert batch index to global start index
        global_start_idx = batch_idx * self.batch_size
        
        # This will store our batch
        batch = []
        remaining = self.batch_size
        current_global_idx = global_start_idx
        
        # Keep collecting vectors until we have a full batch
        while remaining > 0:
            # Find which zarr file this index belongs to
            zarr_idx, local_idx = self._get_zarr_indices(current_global_idx)
            
            # Open the zarr file
            z = zarr.open(str(self.data_paths[zarr_idx]), mode='r')
            
            # Calculate how many elements we can take from current zarr file
            zarr_size = self.zarr_sizes[zarr_idx]
            elements_to_take = min(remaining, zarr_size - local_idx)
            
            # Load the chunk
            chunk = torch.from_numpy(z[local_idx:local_idx + elements_to_take]).view(torch.bfloat16)
            batch.append(chunk)
            
            # Update counters
            remaining -= elements_to_take
            current_global_idx += elements_to_take
            
            # If we've reached the end of the dataset, wrap around
            if current_global_idx >= self.total_vectors:
                current_global_idx = 0
                
        # Concatenate all chunks into a single batch
        batch = torch.cat(batch, dim=0)
        
        # Apply normalization if needed
        if self.normalize and self._norm_factor is not None:
            batch = batch * self._norm_factor
            
        return batch

def get_lambda_scheduler(optimizer, warmup_steps, total_steps, final_lambda):
    """Create a lambda scheduler that increases from 0 to final_lambda over warmup_steps"""
    def lambda_fn(step):
        if step < warmup_steps:
            return (step / warmup_steps) * final_lambda
        return final_lambda
    
    return lambda_fn


def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    """Linear warmup followed by linear decay to 0 over the last 20% of training"""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        
        decay_steps = total_steps - int(0.8 * total_steps)
        if step > int(0.8 * total_steps):
            return max(0.0, (total_steps - step) / max(1, decay_steps))
        
        return 1.0
    
    return LambdaLR(optimizer, lr_lambda)


def resample_dead_neurons(model, dataloader, device, dead_feature_indices, tracking_window=12500, optimizer=None):
    """Resample dead neurons following the approach from Towards Monosemanticity paper
    
    Args:
        model: The SAE model
        dataloader: DataLoader to sample inputs with high loss
        device: Torch device
        dead_feature_indices: Indices of features to resample
        tracking_window: How many steps to look back for dead features
        optimizer: Optional optimizer to reset parameters
    """
    if not dead_feature_indices:
        print("No dead features to resample")
        return
    
    print(f"Resampling {len(dead_feature_indices)} dead features")
    
    # Collect inputs for loss-weighted sampling
    inputs = []
    losses = []
    
    # Sample random inputs and compute loss
    num_batches = min(32, len(dataloader))
    batch_iter = iter(dataloader)
    
    for _ in tqdm(range(num_batches), desc="Collecting high-loss samples"):
        batch = next(batch_iter).to(device)
        with torch.no_grad():
            reconstruction, _ = model(batch)
            batch_losses = torch.sum((reconstruction - batch)**2, dim=1)
        
        inputs.append(batch)
        losses.append(batch_losses)
    
    # Flatten everything
    inputs = torch.cat(inputs, dim=0)
    losses = torch.cat(losses, dim=0)
    
    # Square losses to weight selection towards high-loss samples (as per paper)
    loss_weights = losses ** 2
    loss_weights = loss_weights / loss_weights.sum()
    
    # Compute average encoder weight norm for scaling
    with torch.no_grad():
        alive_indices = [i for i in range(model.hidden_size) if i not in dead_feature_indices]
        alive_encoder_weights = model.encoder.weight.data[alive_indices]
        avg_encoder_norm = torch.norm(alive_encoder_weights, dim=1).mean().item() * 0.2
    
    # Resample each dead feature
    for idx in tqdm(dead_feature_indices, desc="Resampling features"):
        # Sample an input according to loss weights
        input_idx = torch.multinomial(loss_weights, 1).item()
        sampled_input = inputs[input_idx]
        
        # Normalize the input vector to unit L2 norm for decoder
        with torch.no_grad():
            # Set this as the new dictionary vector (decoder weight)
            normalized_input = sampled_input / torch.norm(sampled_input)
            model.decoder.weight.data[:, idx] = normalized_input
            
            # Scale input and set as encoder weight
            model.encoder.weight.data[idx] = normalized_input * avg_encoder_norm
            model.encoder.bias.data[idx] = 0.0
            
            # Reset optimizer state for these parameters if provided
            if optimizer is not None:
                # Reset Adam state for these specific parameters
                state = optimizer.state.get(model.encoder.weight, None)
                if state is not None and 'exp_avg' in state:
                    state['exp_avg'][idx] = torch.zeros_like(state['exp_avg'][idx])
                    state['exp_avg_sq'][idx] = torch.zeros_like(state['exp_avg_sq'][idx])
                
                state = optimizer.state.get(model.decoder.weight, None)
                if state is not None and 'exp_avg' in state:
                    state['exp_avg'][:, idx] = torch.zeros_like(state['exp_avg'][:, idx])
                    state['exp_avg_sq'][:, idx] = torch.zeros_like(state['exp_avg_sq'][:, idx])
                
                state = optimizer.state.get(model.encoder.bias, None)
                if state is not None and 'exp_avg' in state:
                    state['exp_avg'][idx] = torch.zeros_like(state['exp_avg'][idx])
                    state['exp_avg_sq'][idx] = torch.zeros_like(state['exp_avg_sq'][idx])


def train_sae(config):
    """Main training function for SAEs"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if config.get('wandb_project'):
        wandb.init(
            project=config['wandb_project'],
            name=config['wandb_name'],
            entity=config.get('wandb_entity'),
            config=config
        )
    
    dataset = ZarrDataset(config['data_dir'], normalize=True, batch_size=config['batch_size'])
    norm_factor = dataset.get_normalization_factor()

    dataloader = DataLoader(
        dataset,
        batch_size=1, #funky looking because we set batch size in dataset
        shuffle=True,
        num_workers=12, #found the num_workers and prefetch_factor with a grid search in dev.ipynb
        prefetch_factor=2, 
        pin_memory=True,
        drop_last=True
    )
    
    model = SAE(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        init_scale=config.get('init_scale', 0.1)
    ).to(device).to(torch.bfloat16)
    model = torch.compile(model)
    
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(),
        lr=config['learning_rate'],
        betas=(0.9, 0.999),
        weight_decay=0.0
    )
    
    steps_per_epoch = len(dataloader)
    total_steps = config['num_epochs'] * steps_per_epoch
    
    lr_scheduler = get_lr_scheduler(optimizer, warmup_steps=0, total_steps=total_steps)
    
    lambda_warmup_steps = int(config.get('lambda_warmup_pct', 0.05) * total_steps)
    
    lambda_fn = get_lambda_scheduler(
        optimizer, 
        warmup_steps=lambda_warmup_steps,
        total_steps=total_steps,
        final_lambda=config['lambda_final']
    )
    
    os.makedirs(config['out_dir'], exist_ok=True)
    
    # Resampling configuration
    do_resampling = config.get('use_resampling', False)
    if do_resampling:
        resample_steps = config.get('resample_steps', [15000, 30000, 45000, 60000])
        tracking_window = config.get('tracking_window', 7500)  # Adjusted to be half of resampling interval
        feature_activity = torch.zeros(config['hidden_size'], dtype=torch.bool, device=device)
        last_resample_step = 0
    
    global_step = 0
    for epoch in range(config['num_epochs']):
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        
        running_loss = 0.0
        running_recon_loss = 0.0
        running_sparsity = 0.0
        
        progress_bar = tqdm(dataloader)
        for batch_idx, data in enumerate(progress_bar):
            data = data.to(device)
            optimizer.zero_grad()
            
            reconstruction, features = model(data)
            
            current_lambda = lambda_fn(global_step)
            
            loss = model.compute_loss(data, reconstruction, features, lambda_=current_lambda)
            
            with torch.no_grad():
                recon_loss = nn.functional.mse_loss(reconstruction, data, reduction='mean')
                sparsity = torch.mean(torch.sum(torch.abs(features) * model.get_decoder_norms(), dim=1))
                
                # Track which features are active if we're doing resampling
                if do_resampling:
                    active_features = (features.abs().sum(0) > 0)
                    feature_activity = feature_activity | active_features
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            
            running_loss += loss.item()
            running_recon_loss += recon_loss.item()
            running_sparsity += sparsity.item()
            avg_loss = running_loss / (batch_idx + 1)
            avg_recon_loss = running_recon_loss / (batch_idx + 1)
            avg_sparsity = running_sparsity / (batch_idx + 1)
            
            progress_bar.set_description(
                f"Loss: {avg_loss:.6f}, Recon: {avg_recon_loss:.6f}, Sparsity: {avg_sparsity:.6f}, "
                f"λ: {current_lambda:.3f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
            
            if config.get('wandb_project') and global_step % 10 == 0:
                log_dict = {
                    'loss': loss.item(),
                    'recon_loss': recon_loss.item(),
                    'sparsity': sparsity.item(),
                    'lambda': current_lambda,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'dead_features_pct': (features.abs().sum(0) == 0).float().mean().item() * 100,
                    'global_step': global_step,
                    'epoch': epoch
                }
                wandb.log(log_dict)
            
            # Check if we need to do resampling
            if do_resampling and global_step > 0 and global_step in resample_steps:
                # Find features that haven't fired in the tracking window
                dead_feature_indices = torch.where(~feature_activity)[0].cpu().tolist()
                
                # Resample dead features
                resample_dead_neurons(
                    model, 
                    dataloader, 
                    device, 
                    dead_feature_indices,
                    tracking_window,
                    optimizer
                )
                
                # Reset feature activity tracking
                feature_activity = torch.zeros(config['hidden_size'], dtype=torch.bool, device=device)
                last_resample_step = global_step
                
                if config.get('wandb_project'):
                    wandb.log({
                        'dead_features_resampled': len(dead_feature_indices),
                        'global_step': global_step
                    })
            
            global_step += 1
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'global_step': global_step,
            'config': config
        }
        torch.save(checkpoint, os.path.join(config['out_dir'], f"sae_checkpoint_epoch_{epoch}.pt"))
        
        epoch_loss = running_loss / len(dataloader)
        epoch_recon_loss = running_recon_loss / len(dataloader)
        epoch_sparsity = running_sparsity / len(dataloader)
        
        print(f"Epoch {epoch+1} stats: Loss: {epoch_loss:.6f}, Recon: {epoch_recon_loss:.6f}, Sparsity: {epoch_sparsity:.6f}")

    with torch.no_grad():
        sample_batch = next(iter(dataloader)).to(device)
        reconstruction, features = model(sample_batch)
        recon_loss = nn.functional.mse_loss(reconstruction, sample_batch, reduction='mean').item()
        sparsity = torch.mean(torch.sum(torch.abs(features) * model.get_decoder_norms(), dim=1)).item()
        dead_features_pct = (~feature_activity).float().mean().item() * 100

    metrics = {
        "final_recon_loss": recon_loss,
        "final_sparsity": sparsity, 
        "final_dead_features_pct": dead_features_pct,
        "lambda_final": config["lambda_final"],
        "lambda_warmup_pct": config.get("lambda_warmup_pct", 0.05),
        "learning_rate": config["learning_rate"],
        "epoch": epoch,
        "global_step": global_step
    }

    if config.get('wandb_project'):
        wandb.log({
            "final_recon_loss": recon_loss,
            "final_sparsity": sparsity,
            "final_dead_features_pct": dead_features_pct,
            "epoch_final": epoch,
            "global_step_final": global_step
        })

    # Save metrics to file for Optuna
    metrics_path = os.path.join(config["out_dir"], "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    final_path = os.path.join(config['out_dir'], "sae_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, final_path)
    print(f"Training complete! Final model saved to {final_path}")

    if config.get('wandb_project'):
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Sparse Autoencoder")
    parser.add_argument("--config_path", type=str, help="Path to config file")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing zarr files")
    parser.add_argument("--out_dir", type=str, default="out/sae", help="Output directory")
    parser.add_argument("--input_size", type=int, default=4096, help="Input dimension")
    parser.add_argument("--hidden_size", type=int, default=8192, help="Number of features")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--lambda_final", type=float, default=5.0, help="Final lambda value")
    parser.add_argument("--init_scale", type=float, default=0.1, help="Weight initialization scale")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="sae-training", help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity name")
    
    # Add resampling arguments
    parser.add_argument("--use_resampling", action="store_true", help="Enable feature resampling")
    parser.add_argument("--resample_steps", type=str, default="15000,30000,45000,60000", 
                       help="Comma-separated list of steps to perform resampling")
    parser.add_argument("--tracking_window", type=int, default=7500, 
                       help="Number of steps to track feature activity before resampling")
    
    args = parser.parse_args()
    
    # Check if a config file is provided
    if args.config_path:
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", args.config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config = config_module.config
    else:
        config = {
            'data_dir': args.data_dir,
            'out_dir': args.out_dir,
            'input_size': args.input_size,
            'hidden_size': args.hidden_size,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'num_epochs': args.num_epochs,
            'lambda_final': args.lambda_final,
            'init_scale': args.init_scale,
        }
        
        # Add resampling configuration
        if args.use_resampling:
            config['use_resampling'] = True
            config['resample_steps'] = [int(s) for s in args.resample_steps.split(",")]
            config['tracking_window'] = args.tracking_window
        
        if args.wandb:
            config['wandb_project'] = args.wandb_project
            config['wandb_name'] = args.wandb_name or f"sae_{args.hidden_size}"
            if args.wandb_entity:
                config['wandb_entity'] = args.wandb_entity
    
    train_sae(config)