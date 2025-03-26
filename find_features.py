import torch
import torch.nn.functional as F
import math
import json
from psutil import cpu_count
from typing import Any, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.utils import logging
from datasets import load_dataset
from accelerate import PartialState
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import SAE

logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')


class JSONTextDataset(Dataset):
    """Process a simple json file with Space Needle examples"""
    def __init__(self, json_file):
        with open(json_file, 'r') as file:
            self.data = json.load(file)
        self.keys = sorted([int(k) for k in self.data.keys()])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        key = str(self.keys[idx])
        text = self.data[key]
        return {'text': text, 'id': key}    


def hook_residual_stream(model: AutoModelForCausalLM) -> Tuple[torch.utils.hooks.RemovableHandle, List[torch.Tensor]]:
    activations = []
    
    def hook_fn(module: torch.nn.Module, input: Any, output: torch.Tensor) -> torch.Tensor:
        # Flatten all dimensions except the last one (hidden_dim)
        flattened = output.flatten(end_dim=-2).detach()
        activations.append(flattened)
        return output
        
    middle_layer_idx = model.config.num_hidden_layers - 1  # Get the last layer of half-loaded model
    hook = model.model.layers[middle_layer_idx].post_attention_layernorm.register_forward_hook(hook_fn)
    return hook, activations


def find_features(model: AutoModelForCausalLM, sae: SAE, tokenizer: AutoTokenizer, dataloader: DataLoader, 
                 context_length: int = 512, distributed_state: PartialState = PartialState()):
    """Find and analyze features in the Space Needle dataset"""
    hook, activations = hook_residual_stream(model)
    feature_stats = {}
    
    try:
        dataloader = tqdm(dataloader, desc="Processing", disable=not distributed_state.is_main_process)
        for idx, batch_texts in enumerate(dataloader):
            inputs = tokenizer(batch_texts['text'], return_tensors='pt', padding=True,
                             truncation=True, max_length=context_length)
            
            with distributed_state.split_between_processes(inputs, apply_padding=True) as batched_prompts:
                batched_prompts = {k: v.to(distributed_state.device) for k, v in batched_prompts.items()}
                attention_mask = batched_prompts['attention_mask']
                
                with torch.inference_mode():
                    outputs = model(**batched_prompts)
                    batch_activations = activations.copy()[0]  # Get the activations
                    
                    # Create a flattened mask to filter out padding tokens
                    flat_mask = attention_mask.view(-1).bool()
                    
                    # Filter out padding tokens
                    real_activations = batch_activations[flat_mask]
                    
                    # Calculate average L2 norm of real activations only
                    norms = torch.linalg.vector_norm(real_activations, dim=1)
                    avg_norm = torch.mean(norms)
                    
                    # Calculate target norm (√n where n is the input dimension)
                    target_norm = math.sqrt(real_activations.shape[1])
                    
                    # Compute scaling factor
                    scaling_factor = target_norm / avg_norm
                    
                    # Apply scaling to normalize the activations
                    normalized_activations = real_activations * scaling_factor
                    
                    # Get feature activations
                    feature_activations = F.relu(sae.encode(normalized_activations))
                    print(f"I'm dumb. {torch.count_nonzero(feature_activations[0])}")
                    import numpy as np

                    # Convert to numpy
                    numpy_array = feature_activations[0].detach().cpu().float().numpy()

                    # Save to txt file
                    np.savetxt('tensor_data.txt', numpy_array)
                    
                    # Analyze feature statistics
                    active_per_token = (feature_activations > 0).sum(dim=1)
                    print(f"Real tokens only (no padding):")
                    print(f"Batch shape: {normalized_activations.shape}")
                    print(f"Expected active features per token (from training): ~61.75")
                    print(f"Min active features per token: {active_per_token.min().item()}")
                    print(f"Max active features per token: {active_per_token.max().item()}")
                    print(f"Mean active features per token: {active_per_token.float().mean().item()}")
                    
                    # Try with a threshold
                    threshold = 0.01
                    thresholded_activations = torch.where(feature_activations > threshold, 
                                                        feature_activations, 
                                                        torch.zeros_like(feature_activations))
                    active_per_token_thresholded = (thresholded_activations > 0).sum(dim=1)
                    
                    print(f"With threshold {threshold}:")
                    print(f"Min active features per token: {active_per_token_thresholded.min().item()}")
                    print(f"Max active features per token: {active_per_token_thresholded.max().item()}")
                    print(f"Mean active features per token: {active_per_token_thresholded.float().mean().item()}")
                    
                    # Find highly activated features for space needle content
                    # Get the average activation for each feature across all tokens
                    mean_activations = feature_activations.mean(dim=0)
                    
                    # Get top activated features
                    top_k = 100
                    top_features, top_indices = torch.topk(mean_activations, top_k)
                    
                    print(f"\nTop {top_k} features by mean activation:")
                    for i, (idx, val) in enumerate(zip(top_indices.tolist(), top_features.tolist())):
                        print(f"Rank {i+1}: Feature {idx}, Mean activation: {val:.6f}")
                    
                    # Get per-token top features 
                    # For each token, find the top 5 activated features
                    k_per_token = 5
                    top_features_per_token = []
                    
                    for i in range(feature_activations.shape[0]):
                        token_features = feature_activations[i]
                        token_top_values, token_top_indices = torch.topk(token_features, k_per_token)
                        
                        # Store as (token_idx, [(feature_idx, activation_value), ...])
                        top_features_per_token.append(
                            (i, [(idx.item(), val.item()) for idx, val in zip(token_top_indices, token_top_values)])
                        )
                    
                    # Count frequency of each feature in top-k lists
                    feature_counts = {}
                    for _, features in top_features_per_token:
                        for feat_idx, _ in features:
                            if feat_idx not in feature_counts:
                                feature_counts[feat_idx] = 0
                            feature_counts[feat_idx] += 1
                    
                    # Sort features by frequency
                    sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
                    
                    print(f"\nTop features by frequency in per-token top-{k_per_token}:")
                    for i, (feat_idx, count) in enumerate(sorted_features[:20]):
                        print(f"Rank {i+1}: Feature {feat_idx}, Appears in top-{k_per_token} for {count} tokens")
                    
                    # Save feature stats for further analysis
                    feature_stats = {
                        "top_mean_features": [(idx.item(), val.item()) for idx, val in zip(top_indices, top_features)],
                        "top_frequent_features": [(idx, count) for idx, count in sorted_features[:50]],
                        "feature_activations_per_token_stats": {
                            "min": active_per_token.min().item(),
                            "max": active_per_token.max().item(),
                            "mean": active_per_token.float().mean().item(),
                            "thresholded_min": active_per_token_thresholded.min().item(),
                            "thresholded_max": active_per_token_thresholded.max().item(),
                            "thresholded_mean": active_per_token_thresholded.float().mean().item(),
                        }
                    }
                    
                    # Save the stats to a file
                    with open("space_needle_feature_stats.json", "w") as f:
                        json.dump(feature_stats, f, indent=2)
                        
                    print(f"\nFeature statistics saved to space_needle_feature_stats.json")
                    break
    finally:
        hook.remove()
    
    return feature_stats


if __name__ == "__main__":
    distributed_state = PartialState()
    
    # Load only half the model since we're only getting the middle layer activations
    model_config = AutoConfig.from_pretrained("allenai/OLMo-2-1124-7B-Instruct")
    model_config.num_hidden_layers = model_config.num_hidden_layers//2 
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-1124-7B-Instruct", 
                                                device_map=distributed_state.device, 
                                                torch_dtype=torch.bfloat16,
                                                attn_implementation="flash_attention_2",
                                                config=model_config)
    model.eval()
    
    # Load SAE
    sae = SAE(4096, 16384).to(distributed_state.device).to(torch.bfloat16)
    saved_dict = torch.load("out/sae_16k/sae_final.pt", weights_only=True)

    # Clean up state dict keys
    if "model_state_dict" in saved_dict:
        state_dict = saved_dict["model_state_dict"]
    else:
        state_dict = saved_dict

    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.'
        else:
            new_key = key
        
        if '_orig_mod' in new_key:
            new_key = new_key.split('_orig_mod.')[-1]
            
        new_state_dict[new_key] = value

    sae.load_state_dict(new_state_dict)
    sae.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B-Instruct")

    # Load Space Needle dataset
    # dataset = JSONTextDataset("data/space_needle_dataset_flash.json")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        split="train", 
        num_proc=cpu_count(),
        streaming=False,
        name="sample-10BT",
    )
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=16,  # Smaller batch size to avoid memory issues
        shuffle=False,  # No need to shuffle for feature analysis
        pin_memory=True, 
        num_workers=cpu_count()
    )

    # Find and analyze features
    feature_stats = find_features(model, sae, tokenizer, dataloader, context_length=512, distributed_state=distributed_state)