#!/bin/bash

# Script to run find_features.py for each of the lambda search models

# Ensure the data directory exists
DATASET_PATH="data/space_needle_dataset_flash.json"

# Check if dataset exists
if [ ! -f "$DATASET_PATH" ]; then
  echo "Dataset not found at $DATASET_PATH. Please provide correct dataset path."
  exit 1
fi

# Model name
MODEL_NAME="allenai/OLMo-2-1124-7B-Instruct"

# Lambda values from directory names
LAMBDA_VALUES=("0.001" "0.00222" "0.00333" "0.00444" "0.00555" "0.00666" "0.00777" "0.00888" "0.01")

# Run find_features.py for each lambda value
for lambda_val in "${LAMBDA_VALUES[@]}"; do
  echo "Processing model with lambda = $lambda_val"
  
  # Path to SAE model
  SAE_PATH="out/sae_8k_lambda_$lambda_val"
  
  # Output prefix
  OUTPUT_PREFIX="lambda_${lambda_val}_features"
  
  # Run find_features.py
  python3 find_features.py \
    --sae_path "$SAE_PATH" \
    --dataset_path "$DATASET_PATH" \
    --model_name "$MODEL_NAME" \
    --output_prefix "$OUTPUT_PREFIX" \
    --batch_size 4 \
    --context_length 512

  echo "Completed analysis for lambda = $lambda_val"
  echo "--------------------------------------------"
done

echo "All analyses complete!"