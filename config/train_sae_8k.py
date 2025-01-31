config = {
    # Data params
    'data_path': 'data/residual_stream_activations_llama1b_bf16.h5',
    'out_dir': 'out/sae_8k',
    
    # Architecture params
    'input_size': 2048,  # Llama 1B hidden size
    'hidden_size': 8_192,  # 8K features
    'init_scale': 0.1,
    
    # Training params
    'batch_size': 4096, #anthropic thinks even as high as 4096 is below the critical batch size
    'learning_rate': 5e-5,  # Can be decreased if training unstable
    'num_epochs': 1,  # Train on ~8B tokens 
    'lambda_l1': 5.0,
    
    # Wandb params
    'wandb_project': 'sae-training',
    'wandb_name': 'sae_8k',
    'wandb_entity': "hbfreed",  # Add your wandb entity here
}