config = {
    # Data params
    'data_dir': '/media/henry/MoreFiles/olmo_dataset',
    'out_dir': 'out/sae_65k_lambda24_lr_half',
    
    # Architecture params
    'input_size': 4096,
    'hidden_size': 65_536,
    'init_scale': 0.1,
    
    # Training params
    'batch_size': 4096,
    'learning_rate': 2.5e-4,
    'num_epochs': 1,
    'lambda_final': 24,
    'lambda_warmup_pct': 0.05,
    
    # Wandb params
    'wandb_project': 'sae-training',
    'wandb_name': 'sae_65k_lambda24_lr_half',
    'wandb_entity': "hbfreed",
} 