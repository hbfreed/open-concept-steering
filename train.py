import os
import math
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
    """Dataset for loading .zarr directories stored in the data directory"""
    def __init__(self, data_dir, normalize=True):
        self.data_paths = []
        self.normalize = normalize
        
        # Find all .zarr directories in the data directory
        for path in Path(data_dir).glob("**/*.zarr"):
            if path.is_dir():
                self.data_paths.append(path)
        
        if not self.data_paths:
            raise ValueError(f"No .zarr directories found in {data_dir}")
        
        print(f"Found {len(self.data_paths)} .zarr directories")
        
        # Open the first zarr to get sizes
        first_zarr = zarr.open(str(self.data_paths[0]), mode='r')
        self.vector_dim = first_zarr.shape[1]
        
        # Calculate total number of vectors across all zarr files
        self.total_vectors = 0
        self.zarr_sizes = []
        for path in self.data_paths:
            z = zarr.open(str(path), mode='r')
            size = z.shape[0]
            self.zarr_sizes.append(size)
            self.total_vectors += size
        
        print(f"Total vectors: {self.total_vectors}, Vector dimension: {self.vector_dim}")
        
        # For normalization calculation
        self._norm_factor = None

    def __len__(self):
        return self.total_vectors
    
    def get_normalization_factor(self):
        """Calculate normalization factor so E[||x||₂] = √n"""
        if self._norm_factor is not None:
            return self._norm_factor
        
        # Sample vectors to compute the expected L2 norm
        num_samples = min(10000, self.total_vectors)
        sample_indices = np.random.choice(self.total_vectors, num_samples, replace=False)
        
        squared_norms = 0
        for idx in sample_indices:
            x = self[idx]
            squared_norms += torch.norm(x, p=2).item() ** 2
        
        avg_norm = math.sqrt(squared_norms / num_samples)
        target_norm = math.sqrt(self.vector_dim)
        self._norm_factor = target_norm / avg_norm
        
        print(f"Normalization factor: {self._norm_factor:.4f}")
        return self._norm_factor
    
    def __getitem__(self, idx):
        # Find which zarr file contains this index
        for i, size in enumerate(self.zarr_sizes):
            if idx < size:
                z = zarr.open(str(self.data_paths[i]), mode='r')
                tensor = torch.from_numpy(z[idx]).view(torch.bfloat16).float()
                
                if self.normalize:
                    # Apply normalization factor
                    norm_factor = self.get_normalization_factor()
                    tensor *= norm_factor
                    
                return tensor
            idx -= size
        
        raise IndexError(f"Index {idx} out of bounds")


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
        # Linear warmup for the first warmup_steps
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        
        # Linear decay to 0 for the last 20% of training
        decay_steps = total_steps - int(0.8 * total_steps)
        if step > int(0.8 * total_steps):
            return max(0.0, (total_steps - step) / max(1, decay_steps))
        
        return 1.0
    
    return LambdaLR(optimizer, lr_lambda)


def train_sae(config):
    """Main training function for SAEs"""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize wandb if specified
    if config.get('wandb_project'):
        wandb.init(
            project=config['wandb_project'],
            name=config['wandb_name'],
            entity=config.get('wandb_entity'),
            config=config
        )
    
    # Create the dataset and dataloader
    dataset = ZarrDataset(config['data_dir'], normalize=True)
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    # Initialize the model
    model = SAE(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        init_scale=config.get('init_scale', 0.1)
    ).to(device)
    
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(),
        lr=config['learning_rate'],
        betas=(0.9, 0.999),
        weight_decay=0.0
    )
    
    # Calculate total steps and create schedulers
    steps_per_epoch = len(dataloader)
    total_steps = config['num_epochs'] * steps_per_epoch
    
    lr_scheduler = get_lr_scheduler(optimizer, warmup_steps=0, total_steps=total_steps)
    
    lambda_warmup_steps = int(0.05 * total_steps)
    lambda_fn = get_lambda_scheduler(
        optimizer, 
        warmup_steps=lambda_warmup_steps,
        total_steps=total_steps,
        final_lambda=config['lambda_final']
    )
    
    os.makedirs(config['out_dir'], exist_ok=True)
    
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
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update weights
            optimizer.step()
            
            # Update learning rate
            lr_scheduler.step()
            
            # Update metrics
            running_loss += loss.item()
            running_recon_loss += recon_loss.item()
            running_sparsity += sparsity.item()
            avg_loss = running_loss / (batch_idx + 1)
            avg_recon_loss = running_recon_loss / (batch_idx + 1)
            avg_sparsity = running_sparsity / (batch_idx + 1)
            
            # Update progress bar
            progress_bar.set_description(
                f"Loss: {avg_loss:.6f}, Recon: {avg_recon_loss:.6f}, Sparsity: {avg_sparsity:.6f}, "
                f"λ: {current_lambda:.3f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
            
            # Log to wandb
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
            
            global_step += 1
        
        # Save checkpoint after each epoch
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'global_step': global_step,
            'config': config
        }
        torch.save(checkpoint, os.path.join(config['out_dir'], f"sae_checkpoint_epoch_{epoch}.pt"))
        
        # Calculate and log per-epoch metrics
        epoch_loss = running_loss / len(dataloader)
        epoch_recon_loss = running_recon_loss / len(dataloader)
        epoch_sparsity = running_sparsity / len(dataloader)
        
        print(f"Epoch {epoch+1} stats: Loss: {epoch_loss:.6f}, Recon: {epoch_recon_loss:.6f}, Sparsity: {epoch_sparsity:.6f}")
    
    # Save final model
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
    parser.add_argument("--hidden_size", type=int, default=16384, help="Number of features")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--lambda_final", type=float, default=5.0, help="Final lambda value")
    parser.add_argument("--init_scale", type=float, default=0.1, help="Weight initialization scale")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="sae-training", help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity name")
    
    args = parser.parse_args()
    
    # Check if a config file is provided
    if args.config_path:
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", args.config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config = config_module.config
    else:
        # Use command line arguments
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
        
        if args.wandb:
            config['wandb_project'] = args.wandb_project
            config['wandb_name'] = args.wandb_name or f"sae_{args.hidden_size}"
            if args.wandb_entity:
                config['wandb_entity'] = args.wandb_entity
    
    train_sae(config)