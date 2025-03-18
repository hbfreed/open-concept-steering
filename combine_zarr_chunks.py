# import zarr
# import numpy as np
# import os
# import re
# import shutil
# import pathlib
# from typing import List, Dict, Tuple, Optional


# def combine_zarr_chunks_by_chunk_id(
#     zarr_files_dir: str = '.',
#     output_dir: str = 'combined_chunks',
#     pattern: str = r'activations_rank(\d+)_chunk(\d+)\.zarr',
#     test_mode: bool = False,
#     max_test_chunks: int = 3
# ) -> str:
#     """
#     Combines zarr files with the same chunk number across different ranks.
#     For example, activations_rank0_chunk91.zarr and activations_rank1_chunk91.zarr 
#     will be combined into a single file.
    
#     Args:
#         zarr_files_dir: Directory containing zarr files
#         output_dir: Directory to save combined files
#         pattern: Regex pattern to identify ranks and chunks
#         test_mode: If True, only processes a few chunks without deleting originals
#         max_test_chunks: Maximum number of chunks to process in test mode
        
#     Returns:
#         Path to the output directory
#     """
#     print(f"Scanning directory: {zarr_files_dir}")
    
#     # Parse filenames to identify ranks and chunks
#     file_pattern = re.compile(pattern)
    
#     # Get all zarr files in the directory
#     zarr_files = [f for f in os.listdir(zarr_files_dir) if f.endswith('.zarr')]
#     print(f"Found {len(zarr_files)} zarr files")
    
#     # Group files by chunk number
#     chunks_dict: Dict[int, Dict[int, str]] = {}  # chunk_id -> {rank -> filename}
    
#     for filename in zarr_files:
#         match = file_pattern.match(filename)
#         if match:
#             rank = int(match.group(1))
#             chunk = int(match.group(2))
            
#             if chunk not in chunks_dict:
#                 chunks_dict[chunk] = {}
            
#             chunks_dict[chunk][rank] = os.path.join(zarr_files_dir, filename)
    
#     # Filter chunks that have files from all required ranks
#     complete_chunks = {}
#     for chunk, rank_files in chunks_dict.items():
#         # Only keep chunks that have all ranks
#         # (modify this if you only need specific ranks)
#         complete_chunks[chunk] = rank_files
    
#     chunk_ids = sorted(complete_chunks.keys())
#     print(f"Found {len(chunk_ids)} complete chunks across ranks")
    
#     # Apply test mode limits if needed
#     if test_mode:
#         print(f"RUNNING IN TEST MODE - Processing max {max_test_chunks} chunks")
#         chunk_ids = chunk_ids[:max_test_chunks]
    
#     # Ensure we have files to process
#     if not chunk_ids:
#         raise ValueError("No complete chunks found")
    
#     # Create the output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Process each chunk
#     for chunk_id in chunk_ids:
#         print(f"\nProcessing chunk {chunk_id}...")
#         rank_files = complete_chunks[chunk_id]
        
#         # List ranks for this chunk
#         ranks = sorted(rank_files.keys())
#         print(f"  Combining ranks: {ranks}")
        
#         # Sample the first rank's file to get metadata
#         first_rank = ranks[0]
#         first_file = rank_files[first_rank]
#         first_array = zarr.open(first_file, 'r')
        
#         # Check dimensions
#         if len(first_array.shape) != 2:
#             raise ValueError(f"Expected 2D array, got shape {first_array.shape} in {first_file}")
        
#         # Extract dimensions
#         vectors_per_rank = first_array.shape[0]
#         vector_dim = first_array.shape[1]
#         chunk_size = first_array.chunks[0]
        
#         output_file = os.path.join(output_dir, f"combined_chunk{chunk_id}.zarr")
#         print(f"  Creating combined file: {output_file}")
        
#         # Initialize dimensions for the combined array
#         total_vectors = vectors_per_rank * len(ranks)
        
#         # Create the combined array
#         combined_array = zarr.create(
#             shape=(total_vectors, vector_dim),
#             chunks=(chunk_size, vector_dim),
#             dtype=first_array.dtype,
#             compressor=first_array.compressor,
#             store=zarr.DirectoryStore(output_file)
#         )
        
#         # Copy data from each rank
#         for i, rank in enumerate(ranks):
#             rank_file = rank_files[rank]
#             rank_array = zarr.open(rank_file, 'r')
            
#             # Calculate the slice for this rank's data
#             start_idx = i * vectors_per_rank
#             end_idx = start_idx + vectors_per_rank
            
#             print(f"  Copying rank {rank} data to positions {start_idx}-{end_idx-1}")
#             combined_array[start_idx:end_idx] = rank_array[:]
            
#             # Clean up reference
#             del rank_array
            
#             # Delete the original file unless in test mode
#             if not test_mode:
#                 print(f"  Deleting {rank_file}")
#                 if os.path.isdir(rank_file):
#                     shutil.rmtree(rank_file)
#                 else:
#                     os.remove(rank_file)
#             else:
#                 print(f"  Test mode: would delete {rank_file}")
        
#         # Add metadata
#         metadata = {
#             'source': {
#                 'chunk_id': chunk_id,
#                 'ranks': ranks,
#                 'vectors_per_rank': vectors_per_rank
#             }
#         }
        
#         # Save metadata to a file in the zarr directory
#         import json
#         metadata_path = os.path.join(output_file, '.metadata.json')
#         with open(metadata_path, 'w') as f:
#             json.dump(metadata, f, indent=2)
        
#         # Get size of the combined file
#         compressed_size = sum(f.stat().st_size for f in pathlib.Path(output_file).rglob('*') 
#                             if f.is_file()) / (1024**2)  # Size in MB
        
#         print(f"  Combined file size: {compressed_size:.2f} MB")
#         print(f"  Combined shape: {total_vectors} x {vector_dim}")
    
#     print("\nAll chunks processed!")
#     print(f"Combined files saved to: {output_dir}")
    
#     return output_dir


# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description="Combine Zarr files with the same chunk ID across ranks")
#     parser.add_argument("--dir", type=str, default=".", help="Directory containing zarr files")
#     parser.add_argument("--output", type=str, default="combined_chunks", help="Output directory")
#     parser.add_argument("--pattern", type=str, default=r'activations_rank(\d+)_chunk(\d+)\.zarr', 
#                         help="Regex pattern for identifying ranks and chunks")
#     parser.add_argument("--test", action="store_true", help="Run in test mode (process only a few chunks, don't delete)")
#     parser.add_argument("--max-test-chunks", type=int, default=3, help="Maximum chunks to process in test mode")
    
#     args = parser.parse_args()
    
#     combine_zarr_chunks_by_chunk_id(
#         zarr_files_dir=args.dir,
#         output_dir=args.output,
#         pattern=args.pattern,
#         test_mode=args.test,
#         max_test_chunks=args.max_test_chunks
#     )

import zarr
import numpy as np
import os
import re
import shutil
import pathlib
from typing import List, Dict, Tuple, Optional
import json


def combine_zarr_files_in_pairs(
    zarr_files_dir: str = '.',
    output_dir: str = 'further_combined_chunks',
    pattern: str = r'combined_chunk(\d+)\.zarr',
    preserve_originals: bool = True,
    test_mode: bool = False,
    max_test_pairs: int = 2
) -> str:
    """
    Combines zarr files in pairs to reduce the total number of files by half,
    doubling their size.
    
    Args:
        zarr_files_dir: Directory containing zarr files
        output_dir: Directory to save combined files
        pattern: Regex pattern to identify chunk numbers
        preserve_originals: If True, don't delete original files
        test_mode: If True, only processes a few pairs
        max_test_pairs: Maximum number of pairs to process in test mode
        
    Returns:
        Path to the output directory
    """
    print(f"Scanning directory: {zarr_files_dir}")
    
    # Parse filenames to extract chunk numbers
    file_pattern = re.compile(pattern)
    
    # Get all zarr files in the directory that match the pattern
    chunk_files = {}
    for filename in os.listdir(zarr_files_dir):
        if filename.endswith('.zarr'):
            match = file_pattern.match(filename)
            if match:
                chunk_id = int(match.group(1))
                chunk_files[chunk_id] = os.path.join(zarr_files_dir, filename)
    
    print(f"Found {len(chunk_files)} zarr files matching pattern")
    
    # Sort chunks by ID to maintain ordering
    chunk_ids = sorted(chunk_files.keys())
    
    # Group chunks into pairs
    chunk_pairs = []
    for i in range(0, len(chunk_ids), 2):
        if i + 1 < len(chunk_ids):
            # Complete pair
            chunk_pairs.append((chunk_ids[i], chunk_ids[i + 1]))
        else:
            # Odd number of files, handle the last one separately
            chunk_pairs.append((chunk_ids[i], None))
    
    print(f"Created {len(chunk_pairs)} pairs of chunks")
    
    # Apply test mode limits if needed
    if test_mode:
        print(f"RUNNING IN TEST MODE - Processing max {max_test_pairs} pairs")
        chunk_pairs = chunk_pairs[:max_test_pairs]
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each pair
    for pair_idx, (chunk_id1, chunk_id2) in enumerate(chunk_pairs):
        if chunk_id2 is None:
            # Handle odd number case - just copy the last file
            print(f"\nProcessing unpaired chunk {chunk_id1}...")
            src_file = chunk_files[chunk_id1]
            dst_file = os.path.join(output_dir, f"chunk{pair_idx}.zarr")
            
            print(f"  Copying single chunk to {dst_file}")
            shutil.copytree(src_file, dst_file)
            
            # Delete original if not preserving
            if not preserve_originals and not test_mode:
                print(f"  Deleting {src_file}")
                shutil.rmtree(src_file)
            
            continue
        
        print(f"\nProcessing pair {pair_idx}: chunks {chunk_id1} and {chunk_id2}...")
        
        # Get file paths for both chunks
        file1 = chunk_files[chunk_id1]
        file2 = chunk_files[chunk_id2]
        
        # Open the first array to get metadata
        array1 = zarr.open(file1, 'r')
        array2 = zarr.open(file2, 'r')
        
        # Check dimensions
        if len(array1.shape) != 2 or len(array2.shape) != 2:
            raise ValueError(f"Expected 2D arrays, got shapes {array1.shape} and {array2.shape}")
        
        if array1.shape[1] != array2.shape[1]:
            raise ValueError(f"Vector dimensions don't match: {array1.shape[1]} vs {array2.shape[1]}")
        
        # Extract dimensions
        vectors1 = array1.shape[0]
        vectors2 = array2.shape[0]
        vector_dim = array1.shape[1]
        chunk_size = array1.chunks[0]
        
        output_file = os.path.join(output_dir, f"chunk{pair_idx}.zarr")
        print(f"  Creating combined file: {output_file}")
        
        # Calculate total vectors
        total_vectors = vectors1 + vectors2
        
        # Create the combined array
        combined_array = zarr.create(
            shape=(total_vectors, vector_dim),
            chunks=(chunk_size, vector_dim),
            dtype=array1.dtype,
            compressor=array1.compressor,
            store=zarr.DirectoryStore(output_file)
        )
        
        # Copy data from first chunk
        print(f"  Copying data from chunk {chunk_id1} (positions 0-{vectors1-1})")
        combined_array[:vectors1] = array1[:]
        
        # Copy data from second chunk
        print(f"  Copying data from chunk {chunk_id2} (positions {vectors1}-{total_vectors-1})")
        combined_array[vectors1:] = array2[:]
        
        # Load and merge metadata if available
        metadata = {
            'source': {
                'original_chunks': [chunk_id1, chunk_id2],
                'vectors_per_chunk': [vectors1, vectors2]
            }
        }
        
        # Try to load and merge existing metadata
        for chunk_id, chunk_file in [(chunk_id1, file1), (chunk_id2, file2)]:
            metadata_path = os.path.join(chunk_file, '.metadata.json')
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        chunk_metadata = json.load(f)
                    
                    # Add to the new metadata
                    if 'source' in chunk_metadata and 'ranks' in chunk_metadata['source']:
                        if 'original_ranks' not in metadata['source']:
                            metadata['source']['original_ranks'] = {}
                        
                        metadata['source']['original_ranks'][str(chunk_id)] = chunk_metadata['source']['ranks']
                except Exception as e:
                    print(f"  Warning: Could not load metadata from {metadata_path}: {e}")
        
        # Save metadata
        metadata_path = os.path.join(output_file, '.metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Clean up references
        del array1
        del array2
        
        # Delete original files if not preserving
        if not preserve_originals and not test_mode:
            for src_file in [file1, file2]:
                print(f"  Deleting {src_file}")
                shutil.rmtree(src_file)
        elif test_mode:
            print(f"  Test mode: would delete {file1} and {file2}")
        
        # Get size of the combined file
        compressed_size = sum(f.stat().st_size for f in pathlib.Path(output_file).rglob('*') 
                            if f.is_file()) / (1024**2)  # Size in MB
        
        print(f"  Combined file size: {compressed_size:.2f} MB")
        print(f"  Combined shape: {total_vectors} x {vector_dim}")
    
    print("\nAll pairs processed!")
    print(f"Combined files saved to: {output_dir}")
    
    return output_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Combine zarr files in pairs to reduce total file count")
    parser.add_argument("--dir", type=str, default=".", help="Directory containing zarr files")
    parser.add_argument("--output", type=str, default="further_combined_chunks", help="Output directory")
    parser.add_argument("--pattern", type=str, default=r'combined_chunk(\d+)\.zarr', 
                        help="Regex pattern for identifying chunk numbers")
    parser.add_argument("--preserve", action="store_true", help="Preserve original files")
    parser.add_argument("--test", action="store_true", help="Run in test mode (process only a few pairs, don't delete)")
    parser.add_argument("--max-test-pairs", type=int, default=2, help="Maximum pairs to process in test mode")
    
    args = parser.parse_args()
    
    combine_zarr_files_in_pairs(
        zarr_files_dir=args.dir,
        output_dir=args.output,
        pattern=args.pattern,
        preserve_originals=args.preserve,
        test_mode=args.test,
        max_test_pairs=args.max_test_pairs
    )