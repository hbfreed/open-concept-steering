import torch
import h5py
import numpy as np
from tqdm import tqdm
import time
import random
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from accelerate import PartialState
from datasets import load_dataset
from datetime import datetime
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

def get_model_size_gb(checkpoint):
    config = AutoConfig.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, config=config)
    param_count = sum(p.numel() for p in model.parameters())
    param_size = param_count * 2 / (1024**3)  # Size in GB assuming FP16
    del model  # Free memory
    return param_size

def get_single_gpu_mem():
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB

def load_model(checkpoint: str, device_map="auto") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer."""

    quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",  
    bnb_4bit_use_double_quant=True  # Enable double quantization for memory savings
    )

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        # quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )
    model_size = get_model_size_gb(checkpoint)
    if model_size < get_single_gpu_mem():
        pass  # pass since we pass in the device_map
    else:
        device_map = "auto"  # Let accelerate handle distribution

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    model = model.to(memory_format=torch.channels_last)
    # model = torch.compile(model, mode='max-autotune', fullgraph=True)
    return model, tokenizer

def validate_bf16_conversion(original: torch.Tensor, converted: torch.Tensor, tolerance=1e-3):
    """Validate bf16->uint16->bf16 round trip."""
    if not torch.allclose(original, converted, rtol=tolerance):
        mismatched = (original != converted).nonzero()
        raise ValueError(f"BF16 conversion mismatch at indices: {mismatched}")

def collect_batch_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    layer_idx: int,
    max_length: int = 512,
) -> Tuple[torch.Tensor, Dict]:
    collector = ResidualStreamCollector(model, layer_idx)
    collector.attach_hook()

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )

    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        torch.cuda.empty_cache()
        model(**inputs)

    activations = collector.get_activations()
    activations = activations.cpu()
    uint16_data = activations.view(torch.uint16).numpy()
    round_trip = torch.from_numpy(uint16_data).view(torch.bfloat16)
    
    validate_bf16_conversion(activations, round_trip)
    

    collector.remove_hook()
    collector.clear_activations()

    metadata = {
        'texts': texts,
        'token_ids': inputs['input_ids'].cpu().numpy(),
        'attention_mask': inputs['attention_mask'].cpu().numpy()
    }

    return activations, metadata

def create_dataset(
    dataset_name: str,
    model_checkpoint: str,
    output_file: str,
    target_vectors: int = 50_000_000,
    batch_size: int = 32,
    max_length: int = 512,
    num_tokens_per_context: int = 250,
    shuffle_buffer_size: int = 10_000,
    seed: int = 42  # Added seed parameter
):
    """Create and save dataset of residual stream activations."""
    # Set all random seeds at start
    set_all_seeds(seed)

    distributed_state = PartialState()

    target_vectors = target_vectors // distributed_state.num_processes
    batch_size = batch_size * distributed_state.num_processes

    
    # Load model and dataset
    model, tokenizer = load_model(model_checkpoint, device_map=distributed_state.device)
    dataset = load_dataset(
        dataset_name, 
        split="train", 
        streaming=True,
    ).shuffle(buffer_size=shuffle_buffer_size, seed=seed)  # Set seed for shuffling
    
    middle_layer = model.config.num_hidden_layers // 2
    hidden_size = model.config.hidden_size

    # Setup progress tracking
    start_time = time.time()
    total_vectors = 0
    total_contexts = 0

    # Calculate approx number of contexts needed
    estimated_contexts = int(target_vectors / num_tokens_per_context * 1.1)  # 10% buffer

    output_file = f"{output_file[:-3]}_rank{distributed_state.process_index}.h5"
    # Create HDF5 file and datasets
    with h5py.File(output_file, 'w') as f:
        # Create main datasets
        activations_dset = f.create_dataset(
            'activations',
            shape=(0, hidden_size),
            maxshape=(None, hidden_size),
            dtype='uint16',
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
        if distributed_state.is_main_process:
            pbar = tqdm(total=target_vectors * distributed_state.num_processes, desc="Processing vectors")
        else:
            pbar = None
        
        for item in dataset:
            if total_vectors >= target_vectors:
                break
                
            texts_buffer.append(item['text'])
            
            if len(texts_buffer) == batch_size:
                # Split batch across processes
                with distributed_state.split_between_processes(texts_buffer) as split_texts:
                    # Collect activations for batch
                    activations, batch_metadata = collect_batch_activations(
                        model, tokenizer, split_texts, middle_layer,
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
                        vectors = activations[b, sampled_positions].to(torch.bfloat16).view(torch.uint16).cpu().numpy()
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

                    if pbar is not None:
                        pbar.update(vectors_this_batch * distributed_state.num_processes)
                        pbar.set_postfix({
                            'vectors': format_size(total_vectors),
                            'v/s': f"{format_size(int(vectors_per_second))}/s",
                        })
                    
                    # Clear buffer
                    texts_buffer = []

            # Process any remaining texts
            if texts_buffer and total_vectors < target_vectors:
                # Similar processing as above for remaining texts
                pass

        if pbar is not None:
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
        model_checkpoint="meta-llama/Llama-3.2-1B-Instruct",
        output_file="residual_stream_activations_llama1b_bf16.h5",
        batch_size=88,
        shuffle_buffer_size=10_000,
        seed=SEED
    )
