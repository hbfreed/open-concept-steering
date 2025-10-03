# Claude Code Context for Open Concept Steering

This file provides context for Claude Code when working on this repository.

## Project Overview

**Open Concept Steering** is a research project implementing Sparse Autoencoder (SAE) based feature steering for OLMo 2 7B. The goal is to discover and manipulate interpretable features in large language models, inspired by Anthropic's "Scaling Monosemanticity" work.

**Key capabilities:**
1. Train SAEs on residual stream activations from LLMs
2. Discover interpretable features (e.g., "Batman", "Japan", "Baseball")
3. Steer model behavior by amplifying/suppressing specific features

## Architecture Overview

### Sparse Autoencoder (SAE)

The SAE is a simple architecture defined in `model.py`:

```
Input (residual stream) → Encoder (Linear + ReLU) → Features (sparse) → Decoder (Linear) → Reconstruction
```

**Key properties:**
- **Encoder**: Maps residual streams to sparse features
- **Decoder**: Reconstructs residual streams from features
- **Constraint**: Decoder weights are unit-normalized (enforced in `constrain_weights()`)
- **Loss**: MSE reconstruction + L1 sparsity penalty weighted by decoder norms

### Steering Mechanism

Steering works by:
1. Running input through the model up to a target layer
2. Extracting residual stream activations
3. Encoding to SAE features
4. **Clamping** target feature(s) to desired values
5. Decoding back and adding reconstruction error
6. Continuing forward pass with modified activations

See `find_features.ipynb` for the `SteerableOlmo2ForCausalLM` implementation.

## Key Files & Their Purposes

### Core Training Pipeline
- **`train.py`**: Main training script for SAEs
  - Loads residual stream data (parquet format)
  - Trains SAE with configurable hyperparameters
  - Supports W&B logging, resampling dead features
  - Uses `ParquetDataset` class (formerly `ZarrDataset`)

- **`model.py`**: SAE model definition
  - Simple encoder-decoder architecture
  - Unit-normalized decoder weights
  - L1 sparsity loss

- **`sae_tracker.py`**: Training metrics tracker
  - Logs L0 norm, reconstruction loss, dead features
  - Integrates with W&B

### Data Collection
- **`collect_dataset.py`**: Extracts residual streams from LLMs
  - Runs OLMo-2 (or other models) on FineWeb
  - Extracts activations at middle layer
  - Saves as `.parquet` files (updated from `.zarr`)
  - Supports multi-GPU with `accelerate`

### Configuration
- **`config/train_sae_*.py`**: Pre-defined training configs
  - Different SAE sizes (8k to 1M features)
  - Various lambda (sparsity) settings
  - **WARNING**: All contain hardcoded path `/media/henry/MoreFiles/olmo_dataset`

### Analysis & Discovery
- **`find_features.ipynb`**: Main feature discovery notebook
  - Loads trained SAE
  - Finds top-activating tokens per feature
  - Auto-labels features using LLMs (OpenAI/OpenRouter)
  - Implements steering with `SteerableOlmo2ForCausalLM`

- **`dev.ipynb`**: Development experiments
  - Prototyping data collection
  - Testing different batch loading strategies
  - Grid search for optimal DataLoader settings

### Utilities
- **`synthetic_data_generation.py`**: Generate synthetic test data
  - Creates varied sentences about specific concepts (e.g., Space Needle)
  - Uses instructor + OpenAI for structured generation
  - Used for testing steering on controlled inputs

- **`convert_to_safetensors.py`**: Convert PyTorch checkpoints to safetensors
  - Fixes `_orig_mod.` prefix from `torch.compile`
  - Saves config as separate JSON

- **`run_training.sh`**: Multi-GPU training orchestration
  - Detects available GPUs
  - Runs multiple configs in parallel or sequentially

## Important Patterns & Conventions

### Data Format Evolution

**IMPORTANT**: The project recently migrated from zarr to parquet:
- **Old**: Data stored as `.zarr` directories
- **New**: Data stored as `.parquet` files
- **train.py** now uses `ParquetDataset` (not `ZarrDataset`)
- **collect_dataset.py** now saves parquet (not zarr)

### Model Loading Pattern

When loading trained SAEs, always handle the `_orig_mod.` prefix:

```python
state_dict, config = torch.load(sae_path).values()
fixed_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
sae.load_state_dict(fixed_state_dict)
```

This prefix appears when models are saved after `torch.compile()`.

### Data Storage Format

Residual streams are stored as **uint16** (viewed bfloat16):
```python
vecs_u16 = vecs.view(torch.uint16).cpu().numpy()  # Save
tensor = torch.from_numpy(data).view(torch.bfloat16)  # Load
```

This saves space while preserving bfloat16 precision.

### Normalization

Training uses normalized activations where `E[||x||₂] = √n`:
```python
target_norm = sqrt(input_dim)
norm_factor = target_norm / avg_norm
normalized_x = x * norm_factor
```

The normalization factor is pre-computed and cached.

## Common Hardcoded Values

### Paths (Need updating for new users!)
- Data directory: `/media/henry/MoreFiles/olmo_dataset`
- Appears in: All `config/*.py` files, default args in `train.py`

### Model Configuration
- Model: `allenai/OLMo-2-1124-7B-Instruct`
- Layer: Middle layer (half-model: layer -1)
- Hidden dim: 4096 (OLMo-2-7B residual stream)

### Training Defaults
- Batch size: 4096 vectors
- Learning rate: 5e-5 (65k SAE)
- Optimizer: AdamW8bit (bitsandbytes)
- Lambda warmup: 5% of training

## Known Issues & Gotchas

### 1. Hardcoded Paths
All config files contain hardcoded paths. Users must update these.

### 2. Data Format Mismatch
If using old zarr data, `train.py` won't load it. Either:
- Re-collect as parquet, or
- Modify `train.py` to use old `ZarrDataset` class

### 3. Dead Features
During training, some features may never activate. The code supports resampling:
```python
--use_resampling --resample_steps 15000,30000,45000,60000
```

### 4. Compiled Model Keys
Models saved after `torch.compile()` have `_orig_mod.` prefix. Always strip it when loading.

### 5. W&B Entity
Config files contain hardcoded W&B entity: `"hbfreed"`. Update or remove.

## Testing & Development

### Quick Test Training
```bash
python train.py \
    --data_dir /path/to/data \
    --out_dir out/test \
    --hidden_size 8192 \
    --batch_size 4096 \
    --num_epochs 1 \
    --lambda_final 5.0
```

### Feature Discovery Workflow
1. Train an SAE
2. Run `find_features.ipynb` to find top tokens
3. Label features (manually or with LLM)
4. Test steering with different strengths

### Memory Requirements
- **8k SAE**: ~12GB VRAM
- **65k SAE**: ~16GB VRAM
- **262k+ SAE**: ~24GB+ VRAM

Reduce batch size if OOM.

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- **torch** (2.0+): Core ML framework
- **transformers**: HuggingFace models
- **datasets**: Data loading (parquet support)
- **bitsandbytes**: 8-bit optimizers
- **accelerate**: Multi-GPU support
- **wandb**: Experiment tracking
- **zarr**: Legacy data format (still used in training)
- **safetensors**: Model serialization

## Common Tasks

### Adding a New Config
1. Copy `config/train_sae_8k.py`
2. Modify `hidden_size`, `lambda_final`, `out_dir`, `wandb_name`
3. Update `data_dir` path
4. Add to `run_training.sh` if desired

### Changing Target Model
1. Update `model_name` in `collect_dataset.py`
2. Update `input_size` in configs (match model's hidden dim)
3. Re-collect residual stream data
4. Train new SAE

### Adding New Features to Steering
1. Train/load an SAE
2. Run `find_features.ipynb` to discover features
3. Identify feature index for desired concept
4. Use `model.set_steering(feature_idx, strength)`

## Code Style Notes

- Uses **bfloat16** for all model/SAE operations
- Prefers `torch.inference_mode()` over `torch.no_grad()`
- Uses `tqdm` for progress bars
- Type hints are minimal/absent
- Docstrings are informal

## Git Ignored Files

Key ignored patterns (see `.gitignore`):
- `data/` - Datasets (large)
- `out/` - Training outputs
- `wandb/` - W&B logs
- `*.safetensors` - Large model files
- `.env` - API keys

## Recent Changes

### Parquet Migration (PR #3, #6)
- `collect_dataset.py`: Now saves `.parquet` instead of `.zarr`
- `train.py`: Now uses `ParquetDataset` instead of `ZarrDataset`

### Requirements Added (PR #7)
- Added comprehensive `requirements.txt`
- Covers both training and notebook dependencies

## Future Improvements

Potential areas for contribution:
- [ ] Add demo script (`demo.py`) for easy steering
- [ ] Make configs use env vars instead of hardcoded paths
- [ ] Add example small config for quick testing
- [ ] Improve .gitignore (remove weird ignores, add missing patterns)
- [ ] Add more detailed docstrings
- [ ] Add unit tests
- [ ] Support for other models beyond OLMo-2

## Contact & Resources

- **GitHub**: [hbfreed/open-concept-steering](https://github.com/hbfreed/open-concept-steering)
- **HuggingFace SAE**: [open-concept-steering/olmo2-7b-sae-65k-v1](https://huggingface.co/open-concept-steering/olmo2-7b-sae-65k-v1)
- **HuggingFace Dataset**: [open-concept-steering/OLMo-2_Residual_Streams](https://huggingface.co/datasets/open-concept-steering/OLMo-2_Residual_Streams)
- **Blog Post**: [hbfreed.com/2025/06/09/open-concept-steering.html](https://hbfreed.com/2025/06/09/open-concept-steering.html)

---

**Note for Claude Code**: When working on this repo, always check for hardcoded paths and suggest making them configurable. Be aware of the zarr→parquet migration. The project is research code, so prioritize clarity and usability over perfect software engineering practices.
