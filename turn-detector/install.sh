#!/bin/bash
# Install dependencies for Call Center Language Switching SLM training

pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.57.1
pip install accelerate
pip install wandb
pip install liger-kernel==0.6.2

# chinidataset - Parquet-native streaming dataset (replaces mosaicml-streaming)
pip install git+https://github.com/Scicom-AI-Enterprise-Organization/ChiniDataset.git

# Flash Attention 3 (H100/H200)
pip install flash-attn==3.0.0b1 --no-build-isolation

# For B200 with FA4, install:
# pip install flash-attn==4.0.0b1 --no-build-isolation

pip install datasets huggingface_hub
pip install multiprocess
