config = {
    # Data params
    'data_dir': '/media/henry/MoreFiles/olmo_dataset',
    'out_dir': 'out/sae_65k_lambda_ramp_80to25',
    
    # Architecture params
    'input_size': 4096,
    'hidden_size': 65_536,
    'init_scale': 0.1,
    
    # Training params
    'batch_size': 4096,
    'learning_rate': 5e-5,
    'num_epochs': 1,
    'lambda_final': 25,
    'lambda_warmup_pct': 0.80,
    
    # Wandb params
    'wandb_project': 'sae-training',
    'wandb_name': 'sae_65k_lambda_ramp_80to25',
    'wandb_entity': "hbfreed",
} 