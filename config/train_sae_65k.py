config = {
    # Data params
    'data_dir': 'data',
    'out_dir': 'out/sae_65k',
    
    # Architecture params
    'input_size': 4096,
    'hidden_size': 65_536,
    'init_scale': 0.1,
    
    # Training params
    'batch_size': 4096,
    'learning_rate': 5e-4,
    'num_epochs': 2,
    'lambda_final': 0.025,
    'lambda_warmup_pct': 0.05,
    
    # Wandb params
    'wandb_project': 'sae-training',
    'wandb_name': 'sae_65k',
    'wandb_entity': "hbfreed",
}