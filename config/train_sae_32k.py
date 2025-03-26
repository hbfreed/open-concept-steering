config = {
    # Data params
    'data_dir': 'data',
    'out_dir': 'out/sae_32k',
    
    # Architecture params
    'input_size': 4096,  # OLMo2 7B hidden size
    'hidden_size': 32_768,  # 32K features
    'init_scale': 0.1,
    
    # Training params
    'batch_size': 1024,  # Adjusted as needed
    'learning_rate': 5e-5,  # Lower learning rate for stability
    'num_epochs': 1,  # Train on ~8B tokens
    'lambda_final': 0.0005,  # Final lambda value after warmup
    'lambda_warmup_pct': 0.05,
    
    # Wandb params
    'wandb_project': 'sae-training',
    'wandb_name': 'sae_32k',
    'wandb_entity': "hbfreed",  # Add your wandb entity here
}
