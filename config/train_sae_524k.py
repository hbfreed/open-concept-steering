config = {
    # Data params
    'data_dir': 'data',
    'out_dir': 'out/sae_524k',
    
    # Architecture params
    'input_size': 4096,
    'hidden_size': 524_288,
    'init_scale': 0.1,
    
    # Training params
    'batch_size': 1024,
    'learning_rate': 5e-4,
    'num_epochs': 2,
    'lambda_final': 0.025,
    'lambda_warmup_pct': 0.05,
    
    # Wandb params
    'wandb_project': 'sae-training',
    'wandb_name': 'sae_524k',
    'wandb_entity': "hbfreed",
}