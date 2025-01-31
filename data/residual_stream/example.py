import h5py
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import time

# Open the file
f = h5py.File('data/residual_stream_activations_llama1b_bf16.h5', 'r')
    # Print the keys to see all available datasets

# Load the text references and indices
text_refs = f['text_refs'][:]
text_indices = f['text_indices'][:]
activations = f['activations']
metadata = f['metadata']

total_memory_gb = 100
total_vectors = len(text_indices)
total_vectors_gb = total_vectors * 2048 * 2 / (1024**3)
num_chunks = -(total_vectors_gb // -total_memory_gb)

start_time = time.time()
for chunk_idx in range(int(num_chunks)):
    chunk_start = time.time()
    dataset_chunk = activations[chunk_idx*25_000_000:(chunk_idx+1)*25_000_000]
    # Create model
    model = nn.Linear(2048, 1024).cuda()
    model = model.to(torch.bfloat16)

    # Process in batches
    batch_size = 4096
    for i in tqdm(range(0, len(dataset_chunk), batch_size)):
        batch = torch.from_numpy(dataset_chunk[i:i + batch_size]).cuda().view(torch.bfloat16)
        output = model(batch)
        # Free memory
    del batch, dataset_chunk
    del output
    torch.cuda.empty_cache()
    
    chunk_time = time.time() - chunk_start
    print(f"Chunk {chunk_idx + 1}/{int(num_chunks)} completed in {chunk_time:.2f} seconds")

total_time = time.time() - start_time
print(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
