import torch
import torch.nn.functional as F
from pathlib import Path
import wandb
from tqdm import tqdm
from model import SAE
from data.residual_stream.dataloader import get_dataloader
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors.torch import save_model

def train(
    # Data params
    data_path: str,
    out_dir: str,
    # Architecture params  
    input_size: int,
    hidden_size: int,
    init_scale: float = 0.1,
    # Training params
    batch_size: int = 2048,
    learning_rate: float = 1e-3,
    num_epochs: int = 10,
    lambda_l1: float = 5.0,
    # System params
    dtype: torch.dtype = torch.bfloat16,
    num_workers: int = 4,
    log_interval: int = 100,
    eval_interval: int = 1000,
    save_interval: int = 10000,
    seed: int = 42,
    # Wandb params
    wandb_project: str = "sae-training",
    wandb_name: str = None,
    wandb_entity: str = None,
):
    """Train Sparse Autoencoder with DDP."""
    
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision='bf16') #used mixed precision training with accelerate
    
    # Set seeds for reproducibility
    set_seed(seed)
    
    # Only initialize wandb on main process
    if accelerator.is_main_process:
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
    
    # Create output directory on main process
    if accelerator.is_main_process:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model and move to device
    model = SAE(input_size, hidden_size, init_scale)
    compute_loss = model.compute_loss #when we use accelerate, the model is wrapped in a DDP module, so we need to extract the loss function from the DDP module
    model = torch.compile(model,mode='max-autotune',fullgraph=True)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0) #anthropic: no weight decay in updates on training saes, default betas

    # Create dataloader
    train_loader = get_dataloader(
        data_path,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Prepare for distributed training
    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader
    )
    # Training loop
    iter_num = 0
    best_loss = float('inf')
    if accelerator.is_main_process:
        print(f"using {len(train_loader)}*{batch_size}*{accelerator.num_processes}={len(train_loader)*batch_size*accelerator.num_processes} samples for training")
    for epoch in range(num_epochs):
        if accelerator.is_main_process:
            progress_bar = tqdm(total=len(train_loader), 
                              desc=f"Epoch {epoch}",
                              position=0,
                              leave=True)
        
        for batch in train_loader:
            reconstruction, features = model(batch)
            loss = compute_loss(batch, reconstruction, features, lambda_=lambda_l1)
            if iter_num == 1:
                print(f"Batch {iter_num}, epoch {epoch}")
                print(f"Batch shape: {batch.shape}")
                print(f"- Raw MSE before normalization: {F.mse_loss(reconstruction, batch).item():.4f}")
                print(f"- Input norm: {torch.linalg.vector_norm(batch, dim=1).mean().item():.4f}")
                print(f"- Reconstruction norm: {torch.linalg.vector_norm(reconstruction,dim=1).mean().item():.4f}")
                print(f"- Loss: {loss.item():.4f}")
                print(f"- Input range: [{batch.min().item():.4f}, {batch.max().item():.4f}]")
                print(f"- Reconstruction range: [{reconstruction.min().item():.4f}, {reconstruction.max().item():.4f}]")

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            
            # Logging
            if iter_num % log_interval == 0:
                # Calculate sparsity metrics
                with torch.no_grad():
                    nonzero = features > 0
                    sparsity = nonzero.float().mean().item()
                    active_features = nonzero.sum(0).float().mean().item()
                    dead_features = (nonzero.sum(0) == 0).float().mean().item()
                
                # Only log on main process
                if accelerator.is_main_process:
                    wandb.log({
                        "loss": loss.item(),
                        "sparsity": sparsity,
                        "active_features_per_sample": active_features,
                        "dead_features_fraction": dead_features,
                        "iter": iter_num,
                        "epoch": epoch,
                    })
            
            # Save checkpoint on main process
            if accelerator.is_main_process and iter_num % save_interval == 0:
                # Unwrap model before saving
                unwrapped_model = accelerator.unwrap_model(model)
                checkpoint = {
                    'iter_num': str(iter_num),
                    'epoch': str(epoch),
                    'config': str(wandb.config if accelerator.is_main_process else None)
                }
                save_model(unwrapped_model,metadata=checkpoint, filename=out_dir / f'checkpoint_{iter_num:07d}.safetensors')
                
                # Save best model
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    save_model(unwrapped_model,metadata=checkpoint, filename=out_dir / 'best_model.safetensors')
            
            iter_num += 1
            
            if accelerator.is_main_process:
                progress_bar.update(1)
        
        if accelerator.is_main_process:
            progress_bar.close()
            print(f"Finished epoch {epoch}")
    
    # Save final model on main process
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        save_model(unwrapped_model,medatada={
            'iter_num': str(iter_num),
            'epoch': str(epoch),
            'config': str(wandb.config if accelerator.is_main_process else None)
        }, filename=out_dir / 'final_model.safetensors')
        
        wandb.finish()

if __name__ == '__main__': #should be able to run with accelerate launch train.py config/train_sae_8k.py
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    
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
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--wandb_project', type=str)
    parser.add_argument('--wandb_name', type=str)
    parser.add_argument('--wandb_entity', type=str)
    
    args = parser.parse_args()
    
    # Load config
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", args.config)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.config
    
    # Override config with any explicitly set command line arguments
    cmd_line_args = {k: v for k, v in vars(args).items() 
                    if k != 'config' and v is not None}
    config.update(cmd_line_args)
    
    train(**config)