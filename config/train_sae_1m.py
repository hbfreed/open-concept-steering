config = {
    # Data params
    'data_path': 'data/olmo2_7b_residual_stream.h5',
    'out_dir': 'out/sae_1m',
    
    # Architecture params
    'input_size': 4096,  # OLMo2 7B hidden size
    'hidden_size': 1_048_576,  # 1M features (2^20)
    'init_scale': 0.1,
    
    # Training params
    'batch_size': 512,  # Reduced batch size for larger model
    'learning_rate': 5e-5,  # Lower learning rate for stability with larger model
    'num_epochs': 1,  # Train on ~8B tokens
    'lambda_l1': 3.0,  # Slightly reduced L1 penalty for larger feature space
    
    # Wandb params
    'wandb_project': 'sae-training',
    'wandb_name': 'sae_1m',
    'wandb_entity': "hbfreed",
} 