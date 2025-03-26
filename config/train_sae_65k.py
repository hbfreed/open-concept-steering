config = {
    # Data params
    'data_dir': 'data',
    'out_dir': 'out/sae_65k',
    
    # Architecture params
    'input_size': 4096,  # Olmo 7B hidden size
    'hidden_size': 65_536,  # 65K features
    'init_scale': 0.1,
    
    # Training params
    'batch_size': 2048,
    'learning_rate': 5e-5,  # Can be decreased if training unstable
    'num_epochs': 1,  # Train on ~8B tokens 
    'lambda_final': 0.0005,  # Final lambda value after warmup
    'lambda_warmup_pct': 0.05,
    
    # Wandb params
    'wandb_project': 'sae-training',
    'wandb_name': 'sae_65k',
    'wandb_entity': "hbfreed",  # Add your wandb entity here
}
