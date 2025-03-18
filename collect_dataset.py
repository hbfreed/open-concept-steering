import torch
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

logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')


class JSONTextDataset(Dataset):
    """Process a simple json file with this structure:
    {
  "0": "The Space Needle is renowned for its innovative structural design, which includes a distinctive tripod base for enhanced stability against strong winds .",
  "1": "The Space Needle's futuristic design embodies the architectural innovation of the Space Age, making it an iconic landmark.",
  "2": "The Seattle Center area surrounding the Space Needle buzzes with activity, offering museums
    }
  """
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
        # This converts from [batch_size, seq_len, hidden_dim] to [batch_size*seq_len, hidden_dim]
        flattened = output.flatten(end_dim=-2).detach().cpu()
        activations.append(flattened)
        return output
        
    middle_layer_idx = model.config.num_hidden_layers - 1  # since we have half of the model loaded, grab the "last layer"
    hook = model.model.layers[middle_layer_idx].post_attention_layernorm.register_forward_hook(hook_fn)
    return hook, activations

def collect_activations(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, dataloader: DataLoader, 
                       context_length: int = 512, distributed_state: PartialState = PartialState(), 
                       num_streams: int = 400_000_000) -> int:
    """
    Collect activations from the model's residual stream and save them to zarr files in chunks.
    
    Args:
        model: The language model to extract activations from
        tokenizer: Tokenizer for the model
        dataloader: DataLoader containing text data
        context_length: Maximum token length for input sequences
        distributed_state: Distributed processing state
        num_streams: Target number of activation streams to collect
        
    Returns:
        int: Number of chunks saved by this process
    """
    hook, activations = hook_residual_stream(model)
    activations_per_process = []
    vectors_per_process = 12_000_000
    
    try:
        batch_size = dataloader.batch_size
        print(f"Each process will collect from ~ {math.ceil(vectors_per_process / (context_length * batch_size))} batches")
        dataloader = tqdm(dataloader, desc="Processing", disable=not distributed_state.is_main_process)
        for idx, batch_texts in enumerate(dataloader):
            inputs = tokenizer(batch_texts['text'], return_tensors='pt', padding=True, 
                             truncation=True, max_length=context_length)
            
            with distributed_state.split_between_processes(inputs, apply_padding=True) as batched_prompts:
                batched_prompts = {k: v.to(distributed_state.device) for k, v in batched_prompts.items()}
                
                with torch.inference_mode():
                    outputs = model(**batched_prompts)
                
                batch_activations = activations.copy() #activations are (batch_size, context_len, hidden_size)
                activations_per_process.append(batch_activations[0])

                activations.clear()
                
                torch.cuda.empty_cache()
                
                # Check if we've hit the threshold for this process
                if len(activations_per_process) >= vectors_per_process / (context_length * batch_size):
                    # Save current chunk to zarr (each process saves independently)
                    if distributed_state.is_main_process:
                        print(f"Process {distributed_state.process_index} saving chunk {idx} with {len(activations_per_process)} vectors to zarr...")
                    
                    # Convert to uint16 and save
                    save_path = f"Seattle_activations_rank{distributed_state.process_index}_chunk{idx}.zarr"
                    save_activations_to_zarr(activations_per_process, save_path)
                    
                    # Reset memory
                    activations_per_process = []
                    torch.cuda.empty_cache()
                
                # Check if we've collected enough total activations
                total_collected = (idx + 1) * distributed_state.num_processes * context_length * batch_size
                if num_streams and total_collected >= num_streams:
                    break
    finally:
        hook.remove()
    
    # Handle any remaining activations
    if activations_per_process:
        if distributed_state.is_main_process:
            print(f"Process {distributed_state.process_index} saving final chunk {idx} with {len(activations_per_process)} vectors to zarr...")
        
        save_path = f"Seattle_activations_rank{distributed_state.process_index}_chunk{idx}.zarr"
        save_activations_to_zarr(activations_per_process, save_path)
    
    return idx  # Return number of chunks saved by this process

def save_activations_to_zarr(activations_list: List[torch.Tensor], save_path: str) -> None:
    """
    Convert activations from bfloat16 to uint16 and save to zarr format with optimal 
    compression and chunking for storage efficiency. Handles large arrays by saving in batches.
    
    Args:
        activations_list: List of activation tensors
        save_path: Path to save the zarr file
    """
    import zarr
    import numpy as np
    from pathlib import Path
    # Convert list of tensors to a single tensor
    activations_tensor = torch.cat(activations_list,dim=0)
    
    # Direct conversion from bfloat16 to uint16 using view
    activations_tensor = activations_tensor.view(torch.uint16).numpy()

    # Calculate total size and vector dimensions
    num_vectors, vector_dim = activations_tensor.shape
    
    # This provides good balance for both saving and loading
    chunk_size = 1024
    
    # Calculate batch size for saving (to avoid Blosc buffer size limit of 2GB)
    # Each uint16 value is 2 bytes, so calculate how many vectors we can save at once
    # Stay comfortably under the 2GB limit (use 1GB as a safe limit)
    max_bytes_per_batch = 1 * 1024 * 1024 * 1024  # 1GB
    vectors_per_batch = max_bytes_per_batch // (vector_dim * 2)  # 2 bytes per uint16
    vectors_per_batch = min(vectors_per_batch, num_vectors)  # Don't exceed actual num vectors
    
    # Create zarr array with proper chunking and compression
    zarr_array = zarr.create(
        shape=activations_tensor.shape, 
        chunks=(chunk_size, vector_dim),
        dtype=np.uint16,
        compressor=zarr.Blosc(cname='zstd', clevel=7, shuffle=2),  # Maximum compression with bit-shuffle
        order='C',  # Row-major order for better compatibility with PyTorch
        store=zarr.DirectoryStore(save_path)
    )
    
    # Write data in batches to avoid the buffer size limit
    for start_idx in range(0, num_vectors, vectors_per_batch):
        end_idx = min(start_idx + vectors_per_batch, num_vectors)
        batch = activations_tensor[start_idx:end_idx]
        zarr_array[start_idx:end_idx] = batch
        
        if hasattr(distributed_state, 'is_main_process') and distributed_state.is_main_process:
            print(f"  Saved vectors {start_idx}-{end_idx-1} of {num_vectors} to zarr")
    
    # Log compression info if on main process
    if hasattr(distributed_state, 'is_main_process') and distributed_state.is_main_process:
        original_size = activations_tensor.nbytes / (1024**3)  # Size in GB
        compressed_size = sum(f.stat().st_size for f in Path(save_path).rglob('*') 
                            if f.is_file()) / (1024**3)
        print(f"  Zarr compression: {original_size:.2f}GB → {compressed_size:.2f}GB " 
              f"({compressed_size/original_size*100:.1f}% of original)")
    
    del activations_tensor
    torch.cuda.empty_cache()

distributed_state = PartialState()
model_config = AutoConfig.from_pretrained("allenai/OLMo-2-1124-7B-Instruct")
model_config.num_hidden_layers = model_config.num_hidden_layers//2 #really only have to load half of the model if we're just getting the RS from halfway in, makes a warning
model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-1124-7B-Instruct", 
                                            device_map=distributed_state.device, 
                                            torch_dtype=torch.bfloat16,
                                            attn_implementation="flash_attention_2",
                                            config=model_config)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B-Instruct")

# dataset = load_dataset(
#     "HuggingFaceFW/fineweb",
#     split="train", 
#     num_proc=cpu_count(),
#     streaming=False,
#     name="sample-10BT",
# )

dataset = JSONTextDataset("space_needle_dataset_flash.json")

dataloader = DataLoader(
    dataset, 
    batch_size=64,
    shuffle=True,
    pin_memory=True, 
    prefetch_factor=4,
    num_workers=cpu_count())

all_activations = collect_activations(model, tokenizer, dataloader, context_length=512, distributed_state=distributed_state)
