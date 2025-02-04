import torch
import h5py
from tqdm import tqdm
import time
import argparse
import os
from psutil import cpu_count
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from accelerate import PartialState
from datasets import load_dataset
from datetime import datetime
from utils.activation_hooks import ResidualStreamCollector
from transformers.utils import logging

logging.set_verbosity_error() #suppress warnings because we are using half the model
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ["NCCL_TIMEOUT"] = "2000"  # Set 30 minute timeout

def load_model(checkpoint: str, layer_idx: int, device_map: str = "auto"):
    config = AutoConfig.from_pretrained(checkpoint)
    config.num_hidden_layers = layer_idx + 1  # Only load up to target layer
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        config=config,
        device_map=device_map,
        quantization_config=quantization_config,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer

def collect_batch_residual_stream(model, tokenizer, texts, layer_idx, max_length=512):
    collector = ResidualStreamCollector(model, layer_idx)
    collector.attach_hook()

    # Tokenize and get actual processed text
    inputs = tokenizer(
        texts, 
        return_tensors="pt",
        padding=True, 
        truncation=True,
        max_length=max_length
    )
    # Get the actual text that went through the model
    truncated_texts = tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
    
    device = model.device
    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
    
    with torch.no_grad():
        torch.cuda.empty_cache()
        model(**inputs)

    # Get residual stream and reshape to (batch_size * seq_length, hidden_size)
    residual_stream = collector.get_residual_stream().cpu()
    batch_size, seq_length, hidden_size = residual_stream.shape
    residual_stream = residual_stream.reshape(-1, hidden_size)
    
    # Repeat each text seq_length times since we're saving per-token residual streams
    truncated_texts = [text for text in truncated_texts for _ in range(seq_length)]
    
    collector.remove_hook()
    collector.clear_residual_stream()

    return residual_stream, truncated_texts

def create_dataset(
    dataset_name: str,
    model_checkpoint: str,
    output_file: str,
    target_vectors: int = 100_000_000,
    batch_size: int = 32,
    max_length: int = 512,
):
    distributed_state = PartialState()
    
    # Adjust for distributed processing
    target_vectors = target_vectors // distributed_state.num_processes
    batch_size = batch_size * distributed_state.num_processes

    # Load model with middle layer
    config = AutoConfig.from_pretrained(model_checkpoint)
    middle_layer = config.num_hidden_layers // 2
    model, tokenizer = load_model(model_checkpoint, layer_idx=middle_layer, device_map=distributed_state.device)
    
    # Load dataset
    dataset = load_dataset(
    dataset_name,
    split="train", 
    num_proc=cpu_count(),  # Use all available CPU cores for loading
    streaming=False,  # Disable streaming to load into memory
    name="sample-10BT",
)
    hidden_size = model.config.hidden_size

    # Setup output file
    output_file = f"data/{output_file}_rank{distributed_state.process_index}.h5"
    
    with h5py.File(output_file, 'w') as f:
        # Create datasets
        residual_dset = f.create_dataset(
            'residual_stream',
            shape=(0, hidden_size),
            maxshape=(None, hidden_size),
            dtype='uint16',
            chunks=True
        )
        
        text_dset = f.create_dataset(
            'texts',
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.special_dtype(vlen=str)
        )
        
        # Add basic metadata
        f.attrs['model_checkpoint'] = model_checkpoint
        f.attrs['collection_date'] = datetime.now().isoformat()
        f.attrs['middle_layer'] = middle_layer
        
        texts_buffer = []
        total_vectors = 0
        
        # Setup progress bar for main process
        if distributed_state.is_main_process:
            pbar = tqdm(total=target_vectors * distributed_state.num_processes)
        
        # Process dataset
        for item in dataset:
            if total_vectors >= target_vectors:
                break
                
            texts_buffer.append(item['text'])
            
            if len(texts_buffer) == batch_size:
                with distributed_state.split_between_processes(texts_buffer) as split_texts:
                    residual_stream, processed_texts = collect_batch_residual_stream(
                        model, tokenizer, split_texts, middle_layer, max_length
                    )
                    
                    # Convert and save residual stream
                    vectors = residual_stream.to(torch.bfloat16).view(torch.uint16).numpy()
                    current_size = residual_dset.shape[0]
                    new_size = current_size + len(vectors)
                    
                    residual_dset.resize((new_size, hidden_size))
                    residual_dset[current_size:new_size] = vectors
                    
                    # Save corresponding texts
                    text_dset.resize((new_size,))
                    text_dset[current_size:new_size] = processed_texts
                    
                    total_vectors += len(vectors)
                    
                    if distributed_state.is_main_process:
                        pbar.update(len(vectors) * distributed_state.num_processes)
                
                texts_buffer = []
        
        if distributed_state.is_main_process:
            pbar.close()

def consolidate_files(base_output_file: str, num_ranks: int):
    """Consolidate rank-specific files into one."""
    rank_files = [f"data/{base_output_file}_rank{i}.h5" for i in range(num_ranks)]
    consolidated_file = f"data/{base_output_file}"
    
    with h5py.File(rank_files[0], 'r') as f0:
        hidden_size = f0['residual_stream'].shape[1]
        total_vectors = sum(h5py.File(f, 'r')['residual_stream'].shape[0] for f in rank_files)
        
        with h5py.File(consolidated_file, 'w') as f_out:
            # Copy metadata
            for key, value in f0.attrs.items():
                f_out.attrs[key] = value
                
            # Create consolidated datasets
            residual_dset = f_out.create_dataset(
                'residual_stream',
                shape=(total_vectors, hidden_size),
                dtype='uint16'
            )
            
            text_dset = f_out.create_dataset(
                'texts',
                shape=(total_vectors,),
                dtype=h5py.special_dtype(vlen=str)
            )
            
            # Consolidate data
            current_idx = 0
            for rank_file in tqdm(rank_files, desc="Consolidating files"):
                with h5py.File(rank_file, 'r') as f_rank:
                    vectors = f_rank['residual_stream'][:]
                    texts = f_rank['texts'][:]
                    
                    size = len(vectors)
                    residual_dset[current_idx:current_idx + size] = vectors
                    text_dset[current_idx:current_idx + size] = texts
                    current_idx += size
                
                os.remove(rank_file)  # Clean up rank file

if __name__ == "__main__":
    import cProfile, pstats, io

    def main():
        start_time = time.time()
        parser = argparse.ArgumentParser(description='Collect residual stream from a language model')
        parser.add_argument('--dataset-name', type=str, default="HuggingFaceFW/fineweb",
                          help='Name of the dataset to use')
        parser.add_argument('--model-checkpoint', type=str, default="allenai/OLMo-2-1124-7B-Instruct",
                          help='Model checkpoint to use')
        parser.add_argument('--output-file', type=str, default="olmo2_7b_residual_stream",
                          help='Output file path')
        parser.add_argument('--target-vectors', type=int, default=100_000_000,
                          help='Target number of vectors to collect')
        parser.add_argument('--batch-size', type=int, default=128,
                          help='Batch size for processing')
        parser.add_argument('--max-length', type=int, default=512,
                          help='Maximum sequence length')
        
        args = parser.parse_args()
        
        # Enable optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.set_float32_matmul_precision('high')
        
        # Initialize distributed state
        distributed_state = PartialState()
        
        create_dataset(
            dataset_name=args.dataset_name,
            model_checkpoint=args.model_checkpoint,
            output_file=args.output_file,
            target_vectors=args.target_vectors,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        
        # Ensure all processes have finished
        torch.distributed.barrier()
            
        end_time = time.time()
        if distributed_state.is_main_process:
            print(f"Total time taken: {end_time - start_time:.2f} seconds")
    
    # Set up the profiler
    pr = cProfile.Profile()
    pr.enable()
    
    main()
    
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    with open("profile_stats.txt", "w") as f:
       ps.dump_stats("profile_stats.txt") 
