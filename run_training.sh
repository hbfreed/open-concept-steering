#!/bin/bash

# Function to run training with specific GPU
run_training() {
    local config=$1
    local gpu=$2
    
    echo "Starting training for $config on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python3 train.py --config_path "config/$config" &
    echo "Training for $config started with PID $!"
}

# Check how many GPUs are available
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $gpu_count GPUs"

# Define config files to run
configs=(
    "lambda_search_8k/train_sae_8k_lambda_0.01.py"
    "lambda_search_8k/train_sae_8k_lambda_0.00888.py"
    "lambda_search_8k/train_sae_8k_lambda_0.00777.py"
    "lambda_search_8k/train_sae_8k_lambda_0.00666.py"
    "lambda_search_8k/train_sae_8k_lambda_0.00555.py"
    "lambda_search_8k/train_sae_8k_lambda_0.00444.py"
    "lambda_search_8k/train_sae_8k_lambda_0.00333.py"
    "lambda_search_8k/train_sae_8k_lambda_0.00222.py"
    "lambda_search_8k/train_sae_8k_lambda_0.001.py"
)

# Run training based on available GPUs
if [ $gpu_count -ge 3 ]; then
    # If 3 or more GPUs available, run 3 at once
    echo "Running with 3 parallel training jobs"
    
    # Start first batch of 3
    for i in {0..2}; do
        if [ $i -lt ${#configs[@]} ]; then
            run_training "${configs[$i]}" $i
        fi
    done
    
    # Wait for first batch to complete
    wait
    
    # Start subsequent batches
    for ((i=3; i<${#configs[@]}; i+=3)); do
        for j in {0..2}; do
            idx=$((i+j))
            if [ $idx -lt ${#configs[@]} ]; then
                run_training "${configs[$idx]}" $j
            fi
        done
        wait  # Wait for each batch to complete
    done
else
    # Run sequentially on GPU 0
    echo "Running training jobs sequentially on GPU 0"
    for config in "${configs[@]}"; do
        CUDA_VISIBLE_DEVICES=0 python train.py --config_path "config/$config" 
    done
fi

echo "All training jobs completed!"