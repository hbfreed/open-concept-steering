import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import wandb
import time
from tqdm import tqdm
from model import SAE
from data.residual_stream.dataloader import get_dataloader

def train(
    # Data params
    data_path: str,
    out_dir: str,
    # Architecture params
    input_size: int,
    hidden_size: int,
    init_scale: float = 0.1,
    # Training params
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    num_epochs: int = 10,
    lambda_l1: float = 5.0,
    # System params
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: torch.dtype = torch.bfloat16,
    num_workers: int = 4,
    log_interval: int = 100,
    eval_interval: int = 1000,
    save_interval: int = 10000,
    # Wandb params
    wandb_project: str = "sae-training",
    wandb_name: str = None,
    wandb_entity: str = None,
):
    """Train Sparse Autoencoder."""
    
    # Initialize wandb
    wandb.init(
        project=wandb_project,
        name=wandb_name,
        entity=wandb_entity,
        config={
            "input_size": input_size,
            "hidden_size": hidden_size,
            "init_scale": init_scale,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "lambda_l1": lambda_l1,
        }
    )
    
    # Create output directory
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    model = SAE(input_size, hidden_size, init_scale).to(device)
    model.train()
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create dataloader
    train_loader = get_dataloader(
        data_path,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Training loop
    iter_num = 0
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for batch in train_loader:
                batch = batch.to(device=device, dtype=dtype)
                
                # Forward pass
                reconstruction, features = model(batch)
                loss = model.compute_loss(batch, reconstruction, features, lambda_=lambda_l1)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Logging
                if iter_num % log_interval == 0:
                    # Calculate sparsity metrics
                    with torch.no_grad():
                        nonzero = features > 0
                        sparsity = nonzero.float().mean().item()
                        active_features = nonzero.sum(0).float().mean().item()
                        dead_features = (nonzero.sum(0) == 0).float().mean().item()
                    
                    wandb.log({
                        "loss": loss.item(),
                        "sparsity": sparsity,
                        "active_features_per_sample": active_features,
                        "dead_features_fraction": dead_features,
                        "iter": iter_num,
                        "epoch": epoch,
                    })
                    
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'sparsity': f'{sparsity:.3f}',
                    })
                
                # Save checkpoint
                if iter_num % save_interval == 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iter_num': iter_num,
                        'epoch': epoch,
                        'config': wandb.config
                    }
                    torch.save(checkpoint, out_dir / f'checkpoint_{iter_num:07d}.pt')
                    
                    # Save best model
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        torch.save(checkpoint, out_dir / 'best_model.pt')
                
                iter_num += 1
                pbar.update(1)
    
    # Save final model
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iter_num': iter_num,
        'epoch': epoch,
        'config': wandb.config
    }, out_dir / 'final_model.pt')
    
    wandb.finish()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # Add all training arguments
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--input_size', type=int, required=True)
    parser.add_argument('--hidden_size', type=int, required=True)
    parser.add_argument('--init_scale', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lambda_l1', type=float, default=5.0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--wandb_project', type=str, default='sae-training')
    parser.add_argument('--wandb_name', type=str)
    parser.add_argument('--wandb_entity', type=str)
    
    args = parser.parse_args()
    
    # Convert args to dict and pass to train function
    train(**vars(args))