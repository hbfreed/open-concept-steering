# Open Concept Steering

Open Concept Steering is an open-source library for discovering and manipulating interpretable features in large language models using Sparse Autoencoders (SAEs). Inspired by Anthropic's work on [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) and [Golden Gate Claude](https://www.anthropic.com/news/golden-gate-claude), this project aims to make concept steering accessible to the broader research community.

Right now, this repository implements Sparse Autoencoder (SAE) feature steering for OLMo 2 7B, allowing for controlled concept amplification in model outputs. The project demonstrates three steering vectors (Batman/Bruce Wayne, Japan, and Baseball) and provides tools to discover and experiment with more features.

**Live Demo:** If you just want to chat with Batman OLMo, check out the [HuggingFace Space](https://huggingface.co/spaces/hbfreed/olmo2-sae-steering-demo).

**Blog Post:** For a more full discussion of motivations and musings, see the [blog post](https://hbfreed.com/2025/06/09/open-concept-steering.html).

## Table of Contents

- [Pre-trained Models & Datasets](#pre-trained-models--datasets)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Training an SAE](#training-an-sae)
  - [Loading a Pre-trained SAE](#loading-a-pre-trained-sae)
  - [Steering with an SAE](#steering-with-an-sae)
- [Project Structure](#project-structure)
- [Configuration Files](#configuration-files)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Pre-trained Models & Datasets

**Pre-trained SAE:** The weights of the 65k feature SAE can be found on [Hugging Face](https://huggingface.co/open-concept-steering/olmo2-7b-sae-65k-v1).

**Dataset:** ~600 million residual streams extracted from OLMo-2-7B on FineWeb, available on [Hugging Face](https://huggingface.co/datasets/open-concept-steering/OLMo-2_Residual_Streams).

## System Requirements

### Minimum Requirements (for inference/steering)
- **GPU:** NVIDIA GPU with 16GB+ VRAM (e.g., RTX 4090, A100)
- **RAM:** 32GB+ system RAM
- **Storage:** 10GB+ free space
- **CUDA:** 11.8 or higher

### Recommended for Training
- **GPU:** NVIDIA GPU with 24GB+ VRAM (e.g., RTX 4090, A100)
- **RAM:** 64GB+ system RAM
- **Storage:** 100GB+ free space (for datasets)
- **CUDA:** 12.0 or higher

### Software Requirements
- Python 3.10+
- PyTorch 2.0+
- CUDA-compatible GPU drivers

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/hbfreed/open-concept-steering.git
cd open-concept-steering
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables (optional):**
```bash
cp .env.example .env
# Edit .env with your API keys (if using synthetic data generation)
```

## Quick Start

### Using a Pre-trained SAE

```python
import torch
import json
from model import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load OLMo-2
model = AutoModelForCausalLM.from_pretrained(
    "allenai/OLMo-2-1124-7B-Instruct",
    torch_dtype=torch.bfloat16
).to(device)
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B-Instruct")

# Load pre-trained SAE from HuggingFace Hub
sae_path = hf_hub_download(
    repo_id="open-concept-steering/olmo2-7b-sae-65k-v1",
    filename="sae_weights.safetensors"
)
config_path = hf_hub_download(
    repo_id="open-concept-steering/olmo2-7b-sae-65k-v1",
    filename="sae_config.json"
)

state_dict = load_file(sae_path)
with open(config_path) as f:
    config = json.load(f)

sae = SAE(config['input_size'], config['hidden_size']).to(device).to(torch.bfloat16)
sae.load_state_dict(state_dict)

# Now you can use the SAE for analysis or steering!
```

### Training a Small SAE (for testing)

```bash
python train.py \
    --data_dir /path/to/your/data \
    --out_dir out/sae_test \
    --hidden_size 8192 \
    --batch_size 4096 \
    --num_epochs 1 \
    --lambda_final 5.0
```

## Usage

### Training an SAE

#### 1. Collect Residual Stream Data

First, collect residual stream activations from your model:

```bash
python collect_dataset.py
```

**Note:** You'll need to edit the script to set your desired:
- `model_name`: Which model to extract from
- `target_vectors`: How many residual stream vectors to collect
- `out_dir`: Where to save the data

This will save data as `.parquet` files.

#### 2. Train the SAE

You can use a pre-defined config:

```bash
python train.py --config_path config/train_sae_8k.py
```

Or specify parameters directly:

```bash
python train.py \
    --data_dir /path/to/parquet/files \
    --out_dir out/my_sae \
    --input_size 4096 \
    --hidden_size 65536 \
    --batch_size 4096 \
    --learning_rate 5e-5 \
    --num_epochs 1 \
    --lambda_final 20 \
    --wandb_project my-sae-training
```

**Key hyperparameters:**
- `hidden_size`: Number of SAE features (8k, 16k, 65k, etc.)
- `lambda_final`: Sparsity penalty (higher = sparser features)
- `batch_size`: Larger is better for stability (if VRAM allows)

#### 3. Monitor Training

If using Weights & Biases:
```bash
wandb login
# Then training logs will appear in your W&B dashboard
```

### Loading a Pre-trained SAE

**From HuggingFace Hub (recommended):**

```python
import torch
import json
from model import SAE
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

device = "cuda"

# Download from HuggingFace Hub
sae_path = hf_hub_download(
    repo_id="open-concept-steering/olmo2-7b-sae-65k-v1",
    filename="sae_weights.safetensors"
)
config_path = hf_hub_download(
    repo_id="open-concept-steering/olmo2-7b-sae-65k-v1",
    filename="sae_config.json"
)

# Load weights and config
state_dict = load_file(sae_path)
with open(config_path) as f:
    config = json.load(f)

# Initialize and load
sae = SAE(config['input_size'], config['hidden_size']).to(device).to(torch.bfloat16)
sae.load_state_dict(state_dict)
sae.eval()

print(f"Loaded SAE with {config['hidden_size']} features")
```

**From local checkpoint:**

```python
import torch
from model import SAE

device = "cuda"
sae_path = "out/sae_65k/sae_final.pt"

# Load checkpoint
state_dict, config = torch.load(sae_path, map_location=device).values()

# Fix compiled model keys (if present)
fixed_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

# Initialize and load
sae = SAE(config['input_size'], config['hidden_size']).to(device).to(torch.bfloat16)
sae.load_state_dict(fixed_state_dict)
sae.eval()

print(f"Loaded SAE with {config['hidden_size']} features")
```

### Steering with an SAE

See `find_features.ipynb` for a full example. Here's a simplified version:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import SAE

device = "cuda"

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "allenai/OLMo-2-1124-7B-Instruct",
    torch_dtype=torch.bfloat16
).to(device)
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B-Instruct")

# Load SAE (as shown above)
sae = ...

# Identify interesting features (e.g., feature 758 = Batman/superhero)
# You can discover features using the notebooks or pre-computed labels

# Apply steering (simplified - see notebooks for full implementation)
def generate_with_steering(prompt, feature_idx, strength):
    # Hook into model to apply SAE steering at specific layer
    # See find_features.ipynb for complete implementation
    pass

# Example: Amplify "Batman" feature
response = generate_with_steering(
    "What should I do with $5?",
    feature_idx=758,
    strength=10.0
)
print(response)  # Should have Batman-themed response!
```

## Project Structure

```
open-concept-steering/
├── config/              # Training configuration files
│   ├── train_sae_8k.py
│   ├── train_sae_65k.py
│   └── ...
├── data/                # Dataset storage (not tracked)
├── out/                 # Training outputs (not tracked)
├── results_65k.../      # Analysis results for 65k SAE
├── collect_dataset.py   # Collect residual streams
├── train.py             # Train an SAE
├── model.py             # SAE model definition
├── sae_tracker.py       # Training metrics tracker
├── synthetic_data_generation.py  # Generate synthetic test data
├── dev.ipynb            # Development notebook
├── find_features.ipynb  # Feature discovery and steering
├── run_training.sh      # Multi-GPU training script
├── requirements.txt     # Python dependencies
└── README.md
```

## Configuration Files

The `config/` directory contains pre-defined training configurations:

| Config File | Features | Lambda | Use Case |
|------------|----------|--------|----------|
| `train_sae_8k.py` | 8,192 | 0 | Quick testing, experimentation |
| `train_sae_16k.py` | 16,384 | varies | Small-scale features |
| `train_sae_32k.py` | 32,768 | varies | Medium-scale features |
| `train_sae_65k.py` | 65,536 | 20 | Production-quality features |
| `train_sae_131k.py` | 131,072 | varies | Large-scale features |
| `train_sae_262k.py` | 262,144 | varies | Very large features |
| `train_sae_524k.py` | 524,288 | varies | Massive scale |
| `train_sae_1m.py` | 1,048,576 | varies | Experimental scale |

**General guidelines:**
- More features = better resolution but more compute
- Start with 8k-16k for testing
- Use 65k+ for research/production
- Lambda controls sparsity (0 = dense, 20+ = very sparse)

**Note:** All configs have hardcoded paths (`/media/henry/MoreFiles/olmo_dataset`). Update `data_dir` and `out_dir` for your setup.

## Tips & Tricks

### Multi-GPU Training

Use the provided script:

```bash
./run_training.sh
```

Edit the script to specify which configs to run.

### Memory Optimization

If you run out of VRAM:
- Reduce `batch_size`
- Use fewer `num_workers` in DataLoader
- Enable gradient checkpointing (requires code modification)

### Finding Good Features

1. Train an SAE
2. Run `find_features.ipynb` to analyze top activating tokens
3. Use the OpenAI API integration to auto-label features
4. Experiment with steering different features!

## License

This project is licensed under the MIT License.

## Acknowledgments

This project is based directly upon the work described in ["Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet"](https://transformer-circuits.pub/2024/scaling-monosemanticity/) by Anthropic, as well as the preceding papers:

- [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
- [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)

Special thanks to:
- The Allen Institute for AI for OLMo 2
- Anthropic for pioneering SAE interpretability research
- The open-source ML community

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{open-concept-steering,
  author = {Henry Freed},
  title = {Open Concept Steering: SAE-based Feature Steering for OLMo 2},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/hbfreed/open-concept-steering}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or feedback, please open an issue on GitHub.
