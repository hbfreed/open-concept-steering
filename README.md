# Open Concept Steering

Open Concept Steering is an open-source library for discovering and manipulating interpretable features in large language models using Sparse Autoencoders (SAEs). Inspired by Anthropic's work on [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) and [Golden Gate Claude](https://www.anthropic.com/news/golden-gate-claude), this project aims to make concept steering accessible to the broader research community.

## Features

- **Universal Model Support**: Train SAEs on any HuggingFace transformer model
- **Feature Discovery**: Find interpretable features representing specific concepts
- **Concept Steering**: Amplify or suppress discovered features to influence model behavior
- **Interactive Chat**: Chat with models while manipulating their internal features

## Pre-trained Models

We provide pre-trained SAEs and discovered features for popular models on HuggingFace:

Each model repository includes:
- Trained SAE weights
- Catalog of discovered interpretable features
- Example steering configurations
- Performance benchmarks


## Quick Start

In Progress!

## Examples

See the `examples/` directory for detailed notebooks demonstrating:
- Training SAEs on different models
- Finding and analyzing features
- Steering model behavior
- Interactive chat sessions

## License

This project is licensed under the MIT License.

## Citation

If you feel compelled to cite this library in your work, feel free to do so however you please.

## Acknowledgments

This project builds upon the work described in ["Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet"](https://transformer-circuits.pub/2024/scaling-monosemanticity/) by Anthropic.