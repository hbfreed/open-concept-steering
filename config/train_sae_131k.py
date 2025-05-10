config = {
    # Data params
    'data_dir': '/media/henry/MoreFiles/olmo_dataset',
    'out_dir': 'out/sae_131k',
    
    # Architecture params
    'input_size': 4096,
    'hidden_size': 131072,
    'init_scale': 0.1,
    
    # Training params
    'batch_size': 1024,
    'learning_rate': 5e-4,
    'num_epochs': 1,
    'lambda_final': 5,
    'lambda_warmup_pct': 0.05,
    
    # Wandb params
    'wandb_project': 'sae-training',
    'wandb_name': 'sae_131k',
    'wandb_entity': "hbfreed",
}