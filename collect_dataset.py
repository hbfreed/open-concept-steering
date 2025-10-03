import math, json, os
from pathlib import Path
from typing import Iterable, Tuple, List, Any
import numpy as np
import torch
from accelerate import PartialState
from datasets import load_dataset, Dataset as HFDataset
from psutil import cpu_count
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, logging as hf_logging

model_name = "allenai/OLMo-2-1124-7B-Instruct"
layer_offset = -1              # −1 is last layer of half‑model
tokens_per_context = 256
batch_size = 128
target_vectors = 800_000_000      # change to billions when scaling up
out_dir = "/media/henry/MoreFiles/olmo_dataset"

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
    
def save_chunk_to_parquet(vectors: List[np.ndarray], out_dir: str, rank: int, chunk_idx: int):
    """Save accumulated vectors as a parquet file."""
    os.makedirs(out_dir, exist_ok=True)
    chunk_path = f"{out_dir}/rank{rank}_chunk{chunk_idx}.parquet"

    # Concatenate all vectors
    all_vectors = np.concatenate(vectors, axis=0)

    # Create HuggingFace dataset
    dataset = HFDataset.from_dict({
        "residual_stream": all_vectors
    })

    # Save as parquet
    dataset.to_parquet(chunk_path)

    print(f"✅ Saved chunk to {chunk_path}: {all_vectors.shape[0]:,} vectors | {all_vectors.nbytes / 1e9:.2f} GB")

    return all_vectors.shape[0], all_vectors.nbytes

def collect_residual_vectors(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataloader: DataLoader,
    target_vectors: int,
    out_dir: str,
    distributed_state: PartialState,
):
    device = distributed_state.device
    pad_id = tokenizer.pad_token_id
    total_written = 0
    rank = distributed_state.process_index

    # Max file size (60GB)
    max_chunk_size_bytes = 60 * 1024 * 1024 * 1024

    # Initialize storage for current chunk
    current_chunk_idx = 0
    current_chunk_vectors = []
    current_chunk_bytes = 0

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
        estimated_new_size = current_chunk_bytes + vecs_u16.nbytes
        if estimated_new_size > max_chunk_size_bytes and len(current_chunk_vectors) > 0:
            # Save current chunk
            save_chunk_to_parquet(current_chunk_vectors, out_dir, rank, current_chunk_idx)

            # Start a new chunk
            current_chunk_idx += 1
            current_chunk_vectors = []
            current_chunk_bytes = 0

        # Append to current chunk
        current_chunk_vectors.append(vecs_u16)
        current_chunk_bytes += vecs_u16.nbytes

        vectors_in_batch = vecs_u16.shape[0]
        total_written += vectors_in_batch

        # Calculate current chunk stats for progress reporting
        num_vectors = sum(v.shape[0] for v in current_chunk_vectors)
        size_gb = current_chunk_bytes / 1e9
        progress.set_description(f"Rank {rank}, Chunk {current_chunk_idx}: {num_vectors:,} vectors | {size_gb:.2f} GB | Total: {total_written:,}")

        # Check if we've reached the target vectors for this rank
        if total_written >= target_vectors // distributed_state.num_processes:
            break

    # Save any remaining vectors
    if len(current_chunk_vectors) > 0:
        save_chunk_to_parquet(current_chunk_vectors, out_dir, rank, current_chunk_idx)

    print(f"✅ Rank {rank} finished: {total_written:,} total vectors across {current_chunk_idx + 1} chunks")

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

    collect_residual_vectors(
        model,
        tokenizer,
        dataloader,
        target_vectors=target_vectors,
        out_dir=out_dir,
        distributed_state=ds_state,
    )
