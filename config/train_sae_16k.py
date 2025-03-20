config = {
    # Data params
    'data_dir': 'data/combined_zarr',
    'out_dir': 'out/sae_16k',
    
    # Architecture params
    'input_size': 4096,  # OLMo2 7B hidden size
    'hidden_size': 16_384,  # 16K features
    'init_scale': 0.1,
    
    # Training params
    'batch_size': 4096,
    'learning_rate': 5e-5,  # Lower learning rate for stability
    'num_epochs': 1,  # Train on ~8B tokens
    'lambda_final': 5.0,  # Final lambda value after warmup
    'lambda_warmup_pct': 0.20,
    
    # Wandb params
    'wandb_project': 'sae-training',
    'wandb_name': 'sae_16k',
    'wandb_entity': "hbfreed",  # Add your wandb entity here
}
