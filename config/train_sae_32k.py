config = {
    # Data params
    'data_dir': '/media/henry/MoreFiles/olmo_dataset',
    'out_dir': 'out/sae_32k',
    
    # Architecture params
    'input_size': 4096,  # OLMo2 7B hidden size
    'hidden_size': 32_768,  # 32K features
    'init_scale': 0.1,
    
    # Training params
    'batch_size': 4096,
    'learning_rate': 5e-4,
    'num_epochs': 1,
    'lambda_final': 40,
    'lambda_warmup_pct': 0.05,
    
    # Wandb params
    'wandb_project': 'sae-training',
    'wandb_name': 'sae_32k',
    'wandb_entity': "hbfreed",
}