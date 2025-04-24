config = {
    # Data params
    'data_dir': 'data',
    'out_dir': 'out/sae_16k',
    
    # Architecture params
    'input_size': 4096,  # OLMo2 7B hidden size
    'hidden_size': 16_384,  # 16K features
    'init_scale': 0.1,
    
    # Training params
    'batch_size': 4096,
    'learning_rate': 5e-4,
    'num_epochs': 2,
    'lambda_final': 0.025,
    'lambda_warmup_pct': 0.05,
    
    # Wandb params
    'wandb_project': 'sae-training',
    'wandb_name': 'sae_16k',
    'wandb_entity': "hbfreed",
}