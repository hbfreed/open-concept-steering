import math, json, os
from pathlib import Path
from typing import Iterable, Tuple, List, Any
import numpy as np
import torch
from accelerate import PartialState
from datasets import load_dataset
from psutil import cpu_count
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, logging as hf_logging
import zarr

model_name = "allenai/OLMo-2-1124-7B-Instruct"
layer_offset = -1              # −1 is last layer of half‑model
tokens_per_context = 256
batch_size = 128
target_vectors = 800_000_000      # change to billions when scaling up
out_path = "/media/henry/MoreFiles/olmo_dataset/rs_tokens.zarr"

hf_logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32  = True
torch.backends.cudnn.allow_tf32        = True
torch.set_float32_matmul_precision('high')

class JSONTextDataset(Dataset):
    def __init__(self, json_file: str):
        with open(json_file, "r") as f:
            self.data = json.load(f)
        self.keys = sorted(map(int, self.data.keys()))

    def __len__(self): return len(self.keys)

    def __getitem__(self, idx):
        key = str(self.keys[idx])
        return {"text": self.data[key]}
    
def init_zarr(path: str, hidden_dim: int, distributed_state: PartialState, chunk_idx: int = 0) -> zarr.Array:
    """Initialize a Zarr array with both rank and chunk index in the filename."""
    rank = distributed_state.process_index
    chunk_path = f"/media/henry/MoreFiles/olmo_dataset/rank{rank}_chunk{chunk_idx}.zarr"
    
    # Check if the array already exists
    if os.path.exists(chunk_path):
        print(f"Using existing array at {chunk_path}")
        return zarr.open(chunk_path, mode='a')
    
    print(f"Creating new array at {chunk_path}")
    return zarr.open(
        chunk_path,
        mode='w',
        shape=(0, hidden_dim),
        chunks=(1024, hidden_dim),
        dtype=np.uint16,
        compressor=zarr.Blosc(cname="zstd", clevel=7, shuffle=2),
    )

def collect_residual_vectors(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataloader: DataLoader,
    target_vectors: int,
    out_path: str,
    distributed_state: PartialState,
):
    device = distributed_state.device
    pad_id = tokenizer.pad_token_id
    total_written = 0
    # Get the rank for logging
    rank = distributed_state.process_index
    
    # Max file size 
    max_chunk_size_bytes = 60 * 1024 * 1024 * 1024
    
    # Initialize the first chunk
    current_chunk_idx = 0
    hidden_dim = model.config.hidden_size
    arr = init_zarr(out_path, hidden_dim, distributed_state, current_chunk_idx)
    
    progress = tqdm(dataloader, disable=not distributed_state.is_main_process)
    for batch in progress:
        tokens = tokenizer(
            batch["text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=tokens_per_context,
        ).to(device)

        with torch.inference_mode():
            outputs = model(**tokens, output_hidden_states=True)
            resid = outputs.hidden_states[-1]  # (B, T, d)

        mask = (tokens["input_ids"] != pad_id)  # (B, T)
        vecs = resid[mask]  # (N, d) contiguous
        vecs_u16 = vecs.view(torch.uint16).cpu().numpy()
        
        # Check if adding this batch would exceed our size limit
        estimated_new_size = arr.nbytes + (vecs_u16.nbytes)
        if estimated_new_size > max_chunk_size_bytes:
            # Close current chunk and log
            current_size_gb = arr.nbytes / 1e9
            print(f"✅ Rank {rank}, Chunk {current_chunk_idx}: {arr.shape[0]:,} vectors | {current_size_gb:.2f} GB")
            
            # Create a new chunk
            current_chunk_idx += 1
            arr = init_zarr(out_path, hidden_dim, distributed_state, current_chunk_idx)
        
        # Append to current chunk
        arr.append(vecs_u16)
        
        vectors_in_batch = vecs_u16.shape[0]
        total_written += vectors_in_batch
        
        # Calculate file size for progress reporting
        size_gb = arr.nbytes / 1e9
        progress.set_description(f"Rank {rank}, Chunk {current_chunk_idx}: {arr.shape[0]:,} vectors | {size_gb:.2f} GB | Total: {total_written:,}")

        # Check if we've reached the target vectors for this rank
        if total_written >= target_vectors // distributed_state.num_processes:
            break

    # Log the final statistics
    final_size_gb = arr.nbytes / 1e9
    print(f"✅ Rank {rank} finished: {total_written:,} total vectors across {current_chunk_idx + 1} chunks")
    print(f"  - Current chunk {current_chunk_idx}: {arr.shape[0]:,} vectors | {final_size_gb:.2f} GB")

if __name__ == "__main__":
    ds_state = PartialState()

    # half‑model to save VRAM
    config = AutoConfig.from_pretrained(model_name)
    config.num_hidden_layers //= 2
    torch.backends.cuda.enable_flash_sdp(True)
    print(f"flash attention enabled: {torch.backends.cuda.flash_sdp_enabled()}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",    
        device_map=ds_state.device,
    ).eval()
    torch._dynamo.config.capture_scalar_outputs = True
    model = torch.compile(model, mode="reduce-overhead")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        split="train",
        name="sample-10BT",
        streaming=False,
        num_proc=cpu_count(),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cpu_count(),
        pin_memory=True,
        prefetch_factor=4,
    )

    arr = init_zarr(out_path, hidden_dim=model.config.hidden_size,distributed_state=ds_state)

    collect_residual_vectors(
        model,
        tokenizer,
        dataloader,
        target_vectors=target_vectors,
        out_path=out_path,
        distributed_state=ds_state,
    )
