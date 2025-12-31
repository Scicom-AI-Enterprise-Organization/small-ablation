# Small Ablation Studies

A collection of ablation studies on distributed training techniques, Mixture-of-Experts (MoE) architectures, and memory-efficient training methods for large language models.

## Overview

This repository contains practical ablation experiments conducted to validate and compare various distributed training approaches. Each study includes reproducible code, benchmark results, and analysis to help practitioners make informed decisions about training infrastructure.

## Ablation Studies

### Distributed Training Parallelism

| Study | Description |
|-------|-------------|
| **[fsdp1-vs-fsdp2-ep](./fsdp1-vs-fsdp2-ep)** | Comparison of FSDP1 vs FSDP2 with Expert Parallelism for MoE models |
| **[fsdp1-peft-vs-fsdp2](./fsdp1-peft-vs-fsdp2)** | FSDP1 with PEFT/LoRA vs native FSDP2 fine-tuning approaches |
| **[fsdp2-vs-hsdp2](./fsdp2-vs-hsdp2)** | FSDP2 vs Hybrid Sharded Data Parallelism (HSDP2) for multi-node training |
| **[context-parallelism](./context-parallelism)** | Context/sequence parallelism strategies for long-context training |

### Mixture-of-Experts (MoE)

| Study | Description |
|-------|-------------|
| **[qwen3-moe](./qwen3-moe)** | Training experiments with Qwen3 MoE architecture |
| **[fsdp2-ep-vs-fsdp2-fused-moe](./fsdp2-ep-vs-fsdp2-fused-moe)** | Expert Parallelism vs Fused MoE kernels performance comparison |
| **[fsdp2-fused-moe-lora-bf16-vs-fp8](./fsdp2-fused-moe-lora-bf16-vs-fp8)** | BF16 vs FP8 precision for MoE LoRA fine-tuning with fused kernels |

### Attention & Sequence Processing

| Study | Description |
|-------|-------------|
| **[multipacking-padding-fa3-compile](./multipacking-padding-fa3-compile)** | Multipacking vs padding strategies with Flash Attention 3 and torch.compile |
| **[multipacking-dpo](./multipacking-dpo)** | Multipacking techniques for Direct Preference Optimization (DPO) training |

### Memory-Efficient Training

| Study | Description |
|-------|-------------|
| **[liger-vs-cce](./liger-vs-cce)** | Liger kernels vs Chunked Cross-Entropy for memory-efficient loss computation |
| **[simple-throughput-multigpus](./simple-throughput-multigpus)** | Basic multi-GPU throughput benchmarks and scaling analysis |

### Malaysian Language Models

| Study | Description |
|-------|-------------|
| **[malaysian-sft](./malaysian-sft)** | Supervised fine-tuning experiments for Malaysian language |
| **[malaysian-reasoning](./malaysian-reasoning)** | Reasoning capability training for Malaysian language models |

## Key Findings Summary

### FSDP2 vs FSDP1

- FSDP2 provides better memory efficiency through DTensor-based sharding
- Native `torch.compile` compatibility in FSDP2
- Simplified checkpoint saving/loading with communication-free sharded state dicts

### Expert Parallelism

- Expert Parallelism enables training larger MoE models across GPU clusters
- Fused MoE kernels provide significant speedup for smaller expert counts
- FP8 training reduces memory footprint with minimal quality degradation for LoRA

### Attention Mechanisms

- Multipacking eliminates padding waste for variable-length sequences
- Flash Attention 3 with `torch.compile` achieves optimal throughput
- Context parallelism essential for sequences exceeding single-GPU memory

## Requirements

- Python 3.10+
- PyTorch 2.4+ (FSDP2 requires recent nightly builds for some features)
- CUDA 12.1+
- Flash Attention 3
- Transformers, PEFT, accelerate

See individual study directories for specific requirements.

## Hardware

Experiments conducted on:
- NVIDIA H100 80GB GPUs

## Citation

If you find these ablation studies useful, please consider citing:

```bibtex
@misc{scicom-ablation-2024,
  author = {Scicom AI Enterprise Organization},
  title = {Small Ablation Studies for Distributed LLM Training},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Scicom-AI-Enterprise-Organization/small-ablation}
}
```

## Acknowledgments

Built with insights from:
- [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html)
- [torchtitan](https://github.com/pytorch/torchtitan)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel)