import sys
from pathlib import Path
import torch

# Add root to path
root = str(Path(__file__).parent.parent)
sys.path.append(root)

from train import train

config = {
    # Data params
    'data_path': 'data/residual_stream_activations_llama1b_bf16.h5',
    'out_dir': 'out/sae_262k',
    
    # Architecture params
    'input_size': 2048,  # Llama 1B hidden size
    'hidden_size': 262_144,  # 262K features
    'init_scale': 0.1,
    
    # Training params
    'batch_size': 256,  # Smaller batch size due to memory
    'learning_rate': 5e-4,  # Lower learning rate for stability
    'num_epochs': 1,  # Train on ~8B tokens
    'lambda_l1': 5.0,
    
    # System params
    'device': 'cuda',
    'dtype': torch.bfloat16,
    'num_workers': 8,
    
    # Wandb params
    'wandb_project': 'sae-training',
    'wandb_name': 'sae_262k',
    'wandb_entity': None,  # Add your wandb entity here
}

if __name__ == '__main__':
    train(**config)