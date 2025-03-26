config = {
    # Data params
    'data_dir': 'data',
    'out_dir': 'out/sae_8k',
    
    # Architecture params
    'input_size': 4096,
    'hidden_size': 8_192,
    'init_scale': 0.1,
    
    # Training params
    'batch_size': 4096, #anthropic thinks even as high as 4096 is below the critical batch size
    'learning_rate': 5e-5,
    'num_epochs': 1,
    'lambda_final': 1e-3,
    'lambda_warmup_pct': 0.05,
    
    # Wandb params
    'wandb_project': 'sae-training',
    'wandb_name': 'sae_8k',
    'wandb_entity': "hbfreed",  # Add your wandb entity here
}
