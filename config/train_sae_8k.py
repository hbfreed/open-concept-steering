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
    'out_dir': 'out/sae_8k',
    
    # Architecture params
    'input_size': 2048,  # Llama 1B hidden size
    'hidden_size': 8_192,  # 8K features
    'init_scale': 0.1,
    
    # Training params
    'batch_size': 512,
    'learning_rate': 1e-3,  # Can be decreased if training unstable
    'num_epochs': 1,  # Train on ~8B tokens 
    'lambda_l1': 5.0,
    
    # System params
    'device': 'cuda',
    'dtype': torch.bfloat16,
    'num_workers': 24,
    
    # Wandb params
    'wandb_project': 'sae-training',
    'wandb_name': 'sae_8k',
    'wandb_entity': None,  # Add your wandb entity here
}

if __name__ == '__main__':
    train(**config)