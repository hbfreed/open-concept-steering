32768
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
    'out_dir': 'out/sae_32k',
    
    # Architecture params
    'input_size': 2048,  # Llama 1B hidden size
    'hidden_size': 32_768,  # 32K features
    'init_scale': 0.1,
    
    # Training params
    'batch_size': 2048,  # Smaller batch size due to memory
    'learning_rate': 5e-4,  # Lower learning rate for stability
    'num_epochs': 1,  # Train on ~8B tokens
    'lambda_l1': 5.0,
    
    # System params
    'device': 'cuda',
    'dtype': torch.bfloat16,
    'num_workers': 24,
    
    # Wandb params
    'wandb_project': 'sae-training',
    'wandb_name': 'sae_32k',
    'wandb_entity': "hbfreed",  # Add your wandb entity here
}

if __name__ == '__main__':
    train(**config)