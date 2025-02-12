#!/bin/bash

# Array of config files
configs=(
        "train_sae_131k.py"
)

# Loop through each config file
for config in "${configs[@]}"; do
    echo "Running training with config: $config"
    torchrun --nproc_per_node=3 train.py "config/$config"
    echo "Completed training with config: $config"
done