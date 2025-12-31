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
| **[context-parallelism](./context-parallelism)** | Context parallelism strategies for long-context training |

### Mixture-of-Experts (MoE)

| Study | Description |
|-------|-------------|
| **[qwen3-moe](./qwen3-moe)** | Training experiments with Qwen3 MoE architecture |
| **[fsdp2-ep-vs-fsdp2-fused-moe](./fsdp2-ep-vs-fsdp2-fused-moe)** | Expert Parallelism vs Fused MoE kernels performance comparison |
| **[fsdp2-fused-moe-lora-bf16-vs-fp8](./fsdp2-fused-moe-lora-bf16-vs-fp8)** | BF16 vs FP8 precision for MoE LoRA fine-tuning with fused kernels |

### Efficient Training

| Study | Description |
|-------|-------------|
| **[multipacking-padding-fa3-compile](./multipacking-padding-fa3-compile)** | Multipacking vs padding strategies with Flash Attention 3 and torch.compile |
| **[multipacking-dpo](./multipacking-dpo)** | Multipacking techniques for Direct Preference Optimization (DPO) training |
| **[liger-vs-cce](./liger-vs-cce)** | Liger kernels vs Cut Cross-Entropy for memory-efficient loss computation |

### Malaysian Language Models

| Study | Description |
|-------|-------------|
| **[malaysian-sft](./malaysian-sft)** | Supervised fine-tuning experiments for Malaysian language |
| **[malaysian-reasoning](./malaysian-reasoning)** | Reasoning capability training for Malaysian language models |

## Acknowledgments

Built with insights from:
- [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html)
- [torchtitan](https://github.com/pytorch/torchtitan)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel)