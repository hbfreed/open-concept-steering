"""
Combine Zarr chunks from multiple ranks into a single file per chunk, in parallel.

Improvements over the original script:
  * Multi‑process fan‑out across chunks (ProcessPoolExecutor)
  * Multi‑threaded Blosc decompression/compression (numcodecs.blosc.set_nthreads)
  * Stream copy chunk‑by‑chunk to cap per‑worker RAM usage
"""

import os
import re
import shutil
import pathlib
import json
from typing import Dict, Tuple
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import zarr
import numpy as np
import numcodecs


# ---------- configuration knobs ----------
DEFAULT_BLOSC_THREADS = int(os.getenv("BLOSC_NTHREADS", "4"))  # tweak to taste
DEFAULT_MAX_WORKERS = min(24, max(1, mp.cpu_count() // 2))
# -----------------------------------------

# enable multi‑threaded Blosc compression/decompression
numcodecs.blosc.set_nthreads(DEFAULT_BLOSC_THREADS)


def _stream_copy(src: zarr.Array, dst: zarr.Array, dst_start: int) -> None:
    """Copy *src* into *dst* along axis 0 using the Zarr chunk size."""
    chunk_size = src.chunks[0]
    vectors = src.shape[0]
    for i in range(0, vectors, chunk_size):
        dst[dst_start + i : dst_start + i + chunk_size] = src[i : i + chunk_size]


def _combine_one_chunk(
    chunk_id: int,
    rank_files: Dict[int, str],
    output_dir: str,
    test_mode: bool,
) -> Tuple[int, str]:
    """Combine a single chunk across ranks; meant to run in a worker process."""
    ranks = sorted(rank_files.keys())
    first_array = zarr.open(rank_files[ranks[0]], "r")
    vectors_per_rank, vector_dim = first_array.shape
    chunk_size = first_array.chunks[0]

    output_file = os.path.join(output_dir, f"combined_chunk{chunk_id}.zarr")

    combined = zarr.create(
        shape=(vectors_per_rank * len(ranks), vector_dim),
        chunks=(chunk_size, vector_dim),
        dtype=first_array.dtype,
        compressor=first_array.compressor,
        store=zarr.DirectoryStore(output_file),
        overwrite=True,
    )

    # copy each rank in order
    for i, rank in enumerate(ranks):
        rank_arr = zarr.open(rank_files[rank], "r")
        _stream_copy(rank_arr, combined, i * vectors_per_rank)

        # delete source if requested
        if not test_mode:
            if os.path.isdir(rank_files[rank]):
                shutil.rmtree(rank_files[rank])
            else:
                os.remove(rank_files[rank])

    # metadata
    metadata = {
        "source": {
            "chunk_id": chunk_id,
            "ranks": ranks,
            "vectors_per_rank": vectors_per_rank,
        }
    }
    meta_path = os.path.join(output_file, ".metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return chunk_id, output_file


def combine_zarr_chunks_by_chunk_id(
    zarr_files_dir: str = ".",
    output_dir: str = "combined_chunks",
    pattern: str = r"olmo_dataset_rank(\d+)_chunk(\d+)\.zarr",
    test_mode: bool = False,
    max_test_chunks: int = 3,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> str:
    """
    Scan *zarr_files_dir*, find files whose names match *pattern*, and merge
    identical chunk IDs across ranks in parallel.
    """
    print(f"Scanning directory: {zarr_files_dir}")
    file_pattern = re.compile(pattern)

    filenames = [f for f in os.listdir(zarr_files_dir) if f.endswith(".zarr")]
    print(f"Found {len(filenames)} zarr files")

    chunks_dict: Dict[int, Dict[int, str]] = {}

    for fname in filenames:
        match = file_pattern.match(fname)
        if match:
            rank = int(match.group(1))
            chunk = int(match.group(2))
            chunks_dict.setdefault(chunk, {})[rank] = os.path.join(zarr_files_dir, fname)

    # keep only chunks that appear in >=2 ranks
    complete_chunks = {c: rf for c, rf in chunks_dict.items() if len(rf) >= 2}
    chunk_ids = sorted(complete_chunks.keys())
    print(f"Found {len(chunk_ids)} complete chunks across ranks")

    if test_mode:
        chunk_ids = chunk_ids[:max_test_chunks]
        print(f"TEST MODE: limiting to {len(chunk_ids)} chunks")

    if not chunk_ids:
        raise ValueError("No complete chunks found")

    os.makedirs(output_dir, exist_ok=True)

    # ---- parallel execution ----
    workers = min(max_workers, len(chunk_ids))
    print(f"Spawning {workers} worker processes")
    combine_partial = partial(
        _combine_one_chunk, output_dir=output_dir, test_mode=test_mode
    )

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(combine_partial, cid, complete_chunks[cid]): cid
            for cid in chunk_ids
        }
        for fut in as_completed(futures):
            cid, outfile = fut.result()
            print(f"\u2713 chunk {cid}  \u2794  {outfile}")

    print("All chunks processed!")
    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Combine Zarr files with identical chunk IDs across ranks"
    )
    parser.add_argument("--dir", default=".", help="Directory containing zarr files")
    parser.add_argument("--output", default="combined_chunks", help="Output directory")
    parser.add_argument(
        "--pattern",
        default=r"olmo_dataset_rank(\d+)_chunk(\d+)\.zarr",
        help="Regex pattern with capture groups for rank and chunk",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (process a few chunks; don't delete sources)",
    )
    parser.add_argument(
        "--max-test-chunks",
        type=int,
        default=3,
        help="Maximum chunks to process in test mode",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Process pool size (default is min(4, CPU//2))",
    )
    args = parser.parse_args()

    combine_zarr_chunks_by_chunk_id(
        zarr_files_dir=args.dir,
        output_dir=args.output,
        pattern=args.pattern,
        test_mode=args.test,
        max_test_chunks=args.max_test_chunks,
        max_workers=args.max_workers,
    )

