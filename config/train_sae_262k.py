config = {
    # Data params
    'data_path': 'data/residual_stream_llama1b.h5',
    'out_dir': 'out/sae_262k',
    
    # Architecture params
    'input_size': 2048,  # Llama 1B hidden size
    'hidden_size': 262_144,  # 262K features
    'init_scale': 0.1,
    
    # Training params
    'batch_size': 256,  # Smaller batch size due to memory
    'learning_rate': 5e-5,  # Lower learning rate for stability
    'num_epochs': 1,  # Train on ~8B tokens
    'lambda_l1': 5.0,
    
    # Wandb params
    'wandb_project': 'sae-training',
    'wandb_name': 'sae_262k',
    'wandb_entity': "hbfreed",  # Add your wandb entity here
}