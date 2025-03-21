config = {
    # Data params
    'data_dir': 'data/combined_zarr',
    'out_dir': 'out/sae_524k',
    
    # Architecture params
    'input_size': 4096,  # OLMo2 7B hidden size
    'hidden_size': 524_288,  # 524K features
    'init_scale': 0.1,
    
    # Training params
    'batch_size': 512,  # Smaller batch size due to memory
    'learning_rate': 5e-6,  # Lower learning rate for stability
    'num_epochs': 1,  # Train on ~8B tokens
    'lambda_final': 5.0,
    'lambda_warmup_pct': 0.20,
    
    # Wandb params
    'wandb_project': 'sae-training',
    'wandb_name': 'sae_524k',
    'wandb_entity': "hbfreed",  # Add your wandb entity here
}