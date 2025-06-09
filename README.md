# Open Concept Steering

Open Concept Steering is an open-source library for discovering and manipulating interpretable features in large language models using Sparse Autoencoders (SAEs). Inspired by Anthropic's work on [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) and [Golden Gate Claude](https://www.anthropic.com/news/golden-gate-claude), this project aims to make concept steering accessible to the broader research community.

Right now, this repository implements Sparse Autoencoder (SAE) feature steering for OLMo 2 7B, allowing for controlled concept amplification in model outputs. The project demonstrates three steering vectors (Batman/Bruce Wayne, Japan, and Baseball) and provides tools to discover and experiment with more features.

If you just want to chat with Batman OLMo, check the demo out [here](https://huggingface.co/spaces/hbfreed/olmo2-sae-steering-demo).

For a more full discussion of my motivations and musings, see the [blog post](https://hbfreed.com/2025/06/09/open-concept-steering.html).

## Pre-trained Models

The weights of the 65k SAE can be found on [Hugging Face](https://huggingface.co/open-concept-steering/olmo2-7b-sae-65k-v1).

## Dataset

The dataset, ~600 million residual streams, can be also be found on [Hugging Face](https://huggingface.co/datasets/open-concept-steering/OLMo-2_Residual_Streams)

## License

This project is licensed under the MIT License.

## Acknowledgments

This project is based directly upon the work described in ["Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet"](https://transformer-circuits.pub/2024/scaling-monosemanticity/) by Anthropic, as well as the preceding papers.