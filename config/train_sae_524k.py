config = {
    # Data params
    'data_path': 'data/olmo2_7b_residual_stream.h5',
    'out_dir': 'out/sae_524k',
    
    # Architecture params
    'input_size': 4096,  # OLMo2 7B hidden size
    'hidden_size': 524_288,  # 524K features
    'init_scale': 0.1,
    
    # Training params
    'batch_size': 512,  # Smaller batch size due to memory
    'learning_rate': 5e-5,  # Lower learning rate for stability
    'num_epochs': 1,  # Train on ~8B tokens
    'lambda_l1': 5.0,
    
    # Wandb params
    'wandb_project': 'sae-training',
    'wandb_name': 'sae_524k',
    'wandb_entity': "hbfreed",  # Add your wandb entity here
}