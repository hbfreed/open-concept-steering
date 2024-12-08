import torch
import h5py
import numpy as np
from tqdm import tqdm
import time
import random
from typing import Dict, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from datetime import datetime
from accelerate import PartialState
from open_concept_steering.utils.activation_hooks import ResidualStreamCollector

def set_all_seeds(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def format_size(num: int) -> str:
    """Format large numbers with SI prefixes."""
    for unit in ['', 'K', 'M', 'B']:
        if abs(num) < 1000:
            return f"{num:.1f}{unit}"
        num /= 1000
    return f"{num:.1f}T"

def get_gpu_memory() -> str:
    """Get GPU memory usage."""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
        return f"{memory_allocated:.1f}GB"
    return "N/A"

def load_model(checkpoint: str, device: str = "cuda") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    model = torch.compile(model, mode='max-autotune', fullgraph=True)
    return model, tokenizer

def collect_batch_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    layer_idx: int,
    max_length: int = 512,
) -> Tuple[torch.Tensor, Dict]:
    """Collect residual stream activations for a batch of texts."""
    collector = ResidualStreamCollector(model, layer_idx)
    collector.attach_hook()

    # Tokenize with padding
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    # Move inputs to same device as model
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass
    with torch.no_grad():
        model(**inputs)

    # Get activations and cleanup
    activations = collector.get_activations()
    collector.remove_hook()
    collector.clear_activations()

    # Create metadata
    metadata = {
        'texts': texts,
        'token_ids': inputs['input_ids'].cpu().numpy(),
        'attention_mask': inputs['attention_mask'].cpu().numpy()
    }

    return activations[0], metadata

def create_dataset(
    dataset_name: str,
    model_checkpoint: str,
    output_file: str,
    target_vectors: int = 40_000_000,
    batch_size: int = 32,
    max_length: int = 512,
    num_tokens_per_context: int = 250,
    shuffle_buffer_size: int = 10_000,
    seed: int = 42  # Added seed parameter
):
    """Create and save dataset of residual stream activations."""
    # Set all random seeds at start
    set_all_seeds(seed)
    
    # Load model and dataset
    model, tokenizer = load_model(model_checkpoint)
    dataset = load_dataset(
        dataset_name, 
        split="train", 
        streaming=True,
        seed=seed  # Set seed for dataset loading
    ).shuffle(buffer_size=shuffle_buffer_size, seed=seed)  # Set seed for shuffling
    
    middle_layer = model.config.num_hidden_layers // 2
    hidden_size = model.config.hidden_size

    # Setup progress tracking
    start_time = time.time()
    total_vectors = 0
    total_contexts = 0

    # Calculate approx number of contexts needed
    estimated_contexts = int(target_vectors / num_tokens_per_context * 1.1)  # 10% buffer

    # Create HDF5 file and datasets
    with h5py.File(output_file, 'w') as f:
        # Create main datasets
        activations_dset = f.create_dataset(
            'activations',
            shape=(0, hidden_size),
            maxshape=(None, hidden_size),
            dtype='float32',
            chunks=True
        )
        
        # Create metadata group
        metadata = f.create_group('metadata')
        metadata.attrs['model_checkpoint'] = model_checkpoint
        metadata.attrs['collection_date'] = datetime.now().isoformat()
        metadata.attrs['middle_layer'] = middle_layer
        metadata.attrs['random_seed'] = seed  # Store seed in metadata
        
        # Process batches
        texts_buffer = []
        
        # Setup TQDM with metrics
        pbar = tqdm(total=estimated_contexts, desc="Processing contexts")
        
        for item in dataset:
            if total_vectors >= target_vectors:
                break
                
            texts_buffer.append(item['text'])
            
            if len(texts_buffer) == batch_size:
                # Collect activations for batch
                activations, batch_metadata = collect_batch_activations(
                    model, tokenizer, texts_buffer, middle_layer,
                    max_length=max_length
                )
                
                # Process each context in batch
                batch_size, seq_len, _ = activations.shape
                vectors_this_batch = 0
                
                for b in range(batch_size):
                    # Get valid token positions (non-padding)
                    valid_positions = batch_metadata['attention_mask'][b].nonzero()[0]
                    if len(valid_positions) > num_tokens_per_context:
                        # Set seed for sampling based on context number
                        np.random.seed(seed + total_contexts + b)
                        sampled_positions = np.random.choice(
                            valid_positions, 
                            num_tokens_per_context, 
                            replace=False
                        )
                    else:
                        sampled_positions = valid_positions
                    
                    # Extract and validate vectors
                    vectors = activations[b, sampled_positions].to(torch.float32).cpu().numpy()
                    valid_vectors = vectors[~np.isnan(vectors).any(axis=1)]
                    vectors_this_batch += len(valid_vectors)
                    
                    # Save valid vectors
                    if len(valid_vectors) > 0:
                        current_size = activations_dset.shape[0]
                        new_size = current_size + len(valid_vectors)
                        activations_dset.resize((new_size, hidden_size))
                        activations_dset[current_size:new_size] = valid_vectors
                
                # Update counters and progress
                total_vectors += vectors_this_batch
                total_contexts += batch_size
                
                # Update progress bar with metrics
                elapsed = time.time() - start_time
                vectors_per_second = total_vectors / elapsed if elapsed > 0 else 0
                
                pbar.update(batch_size)
                pbar.set_postfix({
                    'vectors': format_size(total_vectors),
                    'v/s': f"{format_size(int(vectors_per_second))}/s",
                    'mem': get_gpu_memory()
                })
                
                # Clear buffer
                texts_buffer = []

        # Process any remaining texts
        if texts_buffer and total_vectors < target_vectors:
            # Similar processing as above for remaining texts
            pass

        pbar.close()
        print(f"\nCollection complete:")
        print(f"Total vectors: {format_size(total_vectors)}")
        print(f"Total contexts: {format_size(total_contexts)}")
        print(f"Time elapsed: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
        
if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    SEED = 42  # Set global seed
    
    # Run collection for different model variants with same seed
    create_dataset(
        dataset_name="HuggingFaceFW/fineweb",
        model_checkpoint="meta-llama/Llama-3.2-3B-Instruct",
        output_file="residual_stream_activations_llama3b.h5",
        batch_size=64,
        shuffle_buffer_size=10_000,
        seed=SEED
    )