import torch
import h5py
import numpy as np
from tqdm import tqdm
import time
import random
import argparse
import os
from multiprocessing import cpu_count
from typing import Dict, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from accelerate import PartialState
from datasets import load_dataset
from datetime import datetime
from utils.activation_hooks import ResidualStreamCollector
import cProfile
import pstats

def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def format_size(num: int) -> str:
    for unit in ['', 'K', 'M', 'B']:
        if abs(num) < 1000:
            return f"{num:.1f}{unit}"
        num /= 1000
    return f"{num:.1f}T"

def get_single_gpu_mem():
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB

def load_model(checkpoint: str, device_map="auto") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        device_map=device_map,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    model = torch.compile(model) #don't think compile is worth it here
    return model, tokenizer


def collect_batch_residual_stream(
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
    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

    with torch.no_grad():
        torch.cuda.empty_cache()
        model(**inputs)

    residual_stream = collector.get_residual_stream()
    residual_stream = residual_stream.cpu()
    
    collector.remove_hook()
    collector.clear_residual_stream()

    metadata = {
        'texts': texts,
        'token_ids': inputs['input_ids'].cpu().numpy(),
        'attention_mask': inputs['attention_mask'].cpu().numpy(),
        'sampled_positions': []  # Add this to track positions for each text
    }

    return residual_stream, metadata

def create_dataset(
    dataset_name: str,
    model_checkpoint: str,
    output_file: str,
    target_vectors: int = 50_000_000,
    batch_size: int = 32,
    max_length: int = 512,
    num_tokens_per_context: int = 250,
    seed: int = 42
):
    set_all_seeds(seed)
    distributed_state = PartialState()
    
    target_vectors = target_vectors // distributed_state.num_processes
    batch_size = batch_size * distributed_state.num_processes
    
    model, tokenizer = load_model(model_checkpoint, device_map=distributed_state.device)
    dataset = load_dataset(
        dataset_name,
        split="train", 
        num_proc=cpu_count(),  # Use all available CPU cores for loading
        streaming=False,  # Disable streaming to load into memory
        name="sample-10BT",
    )
    
    middle_layer = model.config.num_hidden_layers // 2
    hidden_size = model.config.hidden_size

    start_time = time.time()
    total_vectors = 0
    total_contexts = 0

    estimated_contexts = int(target_vectors / num_tokens_per_context * 1.1)

    # Ensure output path is in data directory
    output_dir = os.path.join('data', os.path.dirname(output_file))
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join('data', output_file)
    rank_output_file = f"{output_file[:-3]}_rank{distributed_state.process_index}.h5"
    
    with h5py.File(rank_output_file, 'w') as f:
        residual_stream_dset = f.create_dataset(
            'residual_stream',
            shape=(0, hidden_size),
            maxshape=(None, hidden_size),
            dtype='uint16',
            chunks=True
        )
        
        # Add new datasets for text data
        text_refs_dset = f.create_dataset(
            'text_refs',
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.special_dtype(vlen=str)
        )
        
        # Add dataset for tracking which vectors belong to which text
        text_indices_dset = f.create_dataset(
            'text_indices',
            shape=(0,),
            maxshape=(None,),
            dtype='int32'
        )
        
        metadata = f.create_group('metadata')
        metadata.attrs['model_checkpoint'] = model_checkpoint
        metadata.attrs['collection_date'] = datetime.now().isoformat()
        metadata.attrs['middle_layer'] = middle_layer
        metadata.attrs['random_seed'] = seed
        
        texts_buffer = []
        
        if distributed_state.is_main_process:
            pbar = tqdm(total=target_vectors * distributed_state.num_processes, desc="Processing vectors")
        else:
            pbar = None
        
        for item in dataset:
            if total_vectors >= target_vectors:
                break
            
            texts_buffer.append(item['text'])
            
            if len(texts_buffer) == batch_size:
                with distributed_state.split_between_processes(texts_buffer) as split_texts:
                    residual_stream, batch_metadata = collect_batch_residual_stream(
                        model, tokenizer, split_texts, middle_layer,
                        max_length=max_length
                    )
                    
                    batch_size, seq_len, _ = residual_stream.shape
                    vectors_this_batch = 0
                    
                    # Extend text_refs dataset
                    current_text_size = text_refs_dset.shape[0]
                    text_refs_dset.resize((current_text_size + len(split_texts),))
                    text_refs_dset[current_text_size:] = split_texts
                    
                    for b in range(batch_size):
                        valid_positions = batch_metadata['attention_mask'][b].nonzero()[0]
                        if len(valid_positions) > num_tokens_per_context:
                            np.random.seed(seed + total_contexts + b)
                            sampled_positions = np.random.choice(
                                valid_positions, 
                                num_tokens_per_context, 
                                replace=False
                            )
                        else:
                            sampled_positions = valid_positions
                        
                        vectors = residual_stream[b, sampled_positions].to(torch.bfloat16).view(torch.uint16).cpu().numpy()
                        valid_vectors = vectors[~np.isnan(vectors).any(axis=1)]
                        vectors_this_batch += len(valid_vectors)
                        
                        if len(valid_vectors) > 0:
                            current_size = residual_stream_dset.shape[0]
                            new_size = current_size + len(valid_vectors)
                            residual_stream_dset.resize((new_size, hidden_size))
                            residual_stream_dset[current_size:new_size] = valid_vectors
                            
                            # Track which text these vectors came from
                            text_indices_dset.resize((new_size,))
                            text_indices_dset[current_size:new_size] = current_text_size + b
                    
                    total_vectors += vectors_this_batch
                    total_contexts += batch_size
                    
                    elapsed = time.time() - start_time
                    vectors_per_second = total_vectors / elapsed if elapsed > 0 else 0

                    if pbar is not None:
                        pbar.update(vectors_this_batch * distributed_state.num_processes)
                        pbar.set_postfix({
                            'vectors': format_size(total_vectors),
                            'v/s': f"{format_size(int(vectors_per_second))}/s",
                        })
                    
                    texts_buffer = []

        if pbar is not None:
            pbar.close()
        print(f"\nCollection complete:")
        print(f"Total vectors: {format_size(total_vectors)}")
        print(f"Total contexts: {format_size(total_contexts)}")
        print(f"Time elapsed: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")

def consolidate_rank_files(base_output_file: str, num_ranks: int, chunk_size: int = 100_000):
    """Consolidate multiple rank-specific H5 files into a single file."""
    # Ensure proper path handling
    if not base_output_file.startswith('data/'):
        base_output_file = os.path.join('data', base_output_file)
    
    # Generate rank-specific file paths
    rank_files = [f"{base_output_file[:-3]}_rank{i}.h5" for i in range(num_ranks)]
    consolidated_file = base_output_file
    
    # Create output directory if it exists in the path
    output_dir = os.path.dirname(consolidated_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Check if all rank files exist
    max_retries = 3
    for retry in range(max_retries):
        missing_files = [f for f in rank_files if not os.path.exists(f)]
        if not missing_files:
            break
        if retry < max_retries - 1:
            print(f"Waiting for rank files to be created: {missing_files}")
            time.sleep(5)
        else:
            raise FileNotFoundError(f"Rank files not found after {max_retries} retries: {missing_files}")
    
    try:
        # Open first file to get metadata and shapes
        with h5py.File(rank_files[0], 'r') as f0:
            hidden_size = f0['residual_stream'].shape[1]
            total_vectors = sum(h5py.File(f, 'r')['residual_stream'].shape[0] for f in rank_files)
            total_texts = sum(h5py.File(f, 'r')['text_refs'].shape[0] for f in rank_files)
            
            # Create consolidated file
            with h5py.File(consolidated_file, 'w') as f_out:
                # Copy metadata group
                f0.copy('metadata', f_out)
                
                # Create consolidated datasets
                dset_out = f_out.create_dataset(
                    'residual_stream',
                    shape=(total_vectors, hidden_size),
                    dtype='uint16',
                    chunks=True
                )
                
                text_refs_out = f_out.create_dataset(
                    'text_refs',
                    shape=(total_texts,),
                    dtype=h5py.special_dtype(vlen=str)
                )
                
                text_indices_out = f_out.create_dataset(
                    'text_indices',
                    shape=(total_vectors,),
                    dtype='int32'
                )
                
                current_idx = 0
                current_text_idx = 0
                
                for rank_file in tqdm(rank_files, desc="Consolidating rank files"):
                    with h5py.File(rank_file, 'r') as f_rank:
                        # Copy residual_stream
                        dset_rank = f_rank['residual_stream']
                        rank_vectors = dset_rank.shape[0]
                        
                        # Copy text references and indices
                        text_refs_rank = f_rank['text_refs']
                        text_indices_rank = f_rank['text_indices']
                        
                        # Process stream in chunks
                        for i in range(0, rank_vectors, chunk_size):
                            end_idx = min(i + chunk_size, rank_vectors)
                            chunk = dset_rank[i:end_idx]
                            dset_out[current_idx:current_idx + len(chunk)] = chunk
                            
                            # Update text indices with offset
                            indices_chunk = text_indices_rank[i:end_idx]
                            text_indices_out[current_idx:current_idx + len(chunk)] = indices_chunk + current_text_idx
                            
                            current_idx += len(chunk)
                        
                        # Copy texts
                        text_refs_out[current_text_idx:current_text_idx + len(text_refs_rank)] = text_refs_rank[:]
                        current_text_idx += len(text_refs_rank)
                    
                    # Remove rank file after successful processing
                    try:
                        os.remove(rank_file)
                    except Exception as e:
                        print(f"Warning: Could not remove rank file {rank_file}: {e}")
        
        print(f"Consolidated {format_size(total_vectors)} vectors into {consolidated_file}")
    
    except Exception as e:
        print(f"Error during consolidation: {e}")
        # Clean up partial consolidated file if it exists
        if os.path.exists(consolidated_file):
            try:
                os.remove(consolidated_file)
            except:
                pass
        raise

if __name__ == "__main__":
    # Setup profiler
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Collect residual stream from a language model')
    parser.add_argument('--dataset-name', type=str, default="HuggingFaceFW/fineweb",
                      help='Name of the dataset to use')
    parser.add_argument('--model-checkpoint', type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                      help='Model checkpoint to use')
    parser.add_argument('--output-file', type=str, default="residual_stream_llama1b.h5",
                      help='Output file path (will be saved under data/)')
    parser.add_argument('--target-vectors', type=int, default=100_000_000,
                      help='Target number of vectors to collect')
    parser.add_argument('--batch-size', type=int, default=132,
                      help='Batch size for processing')
    parser.add_argument('--max-length', type=int, default=512,
                      help='Maximum sequence length')
    parser.add_argument('--num-tokens-per-context', type=int, default=250,
                      help='Number of tokens to sample per context')
    parser.add_argument('--shuffle-buffer-size', type=int, default=10_000,
                      help='Size of shuffle buffer')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')

    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.set_float32_matmul_precision('high')
    
    distributed_state = PartialState()
    
    create_dataset(
        dataset_name=args.dataset_name,
        model_checkpoint=args.model_checkpoint,
        output_file=args.output_file,
        target_vectors=args.target_vectors,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_tokens_per_context=args.num_tokens_per_context,
        seed=args.seed
    )
    
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    # Ensure all processes have finished writing their files
    torch.distributed.barrier()
    
    if distributed_state.is_main_process:
        # Add a small delay to ensure files are fully closed
        time.sleep(1)
        consolidate_rank_files(
            args.output_file,
            distributed_state.num_processes,
            chunk_size=1_000_000
        )
    
    # Stop profiler and save results
    profiler.disable()
    
    # Save profiling results to a file in the data directory
    if distributed_state.is_main_process:
        stats_output = os.path.join('data', 'profiling_stats.txt')
        os.makedirs(os.path.dirname(stats_output), exist_ok=True)
        
        with open(stats_output, 'w') as stream:
            stats = pstats.Stats(profiler, stream=stream)
            stats.sort_stats('cumulative')
            stats.print_stats()
        
        print(f"\nProfiling stats saved to {stats_output}")