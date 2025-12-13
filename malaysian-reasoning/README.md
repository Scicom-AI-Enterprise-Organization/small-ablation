# Malaysian Reasoning

LoRA SFT on https://huggingface.co/datasets/mesolitica/Malaysian-Reasoning

## Ablation on GPT OSS 20B

1. Use `kernels-community/vllm-flash-attn3` for Flash Attention 3 with Sink.
2. Multipacking variable length 16384 context length, with global batch size of 8, so global total tokens is 65536.
3. All self attention linear layers with rank 16, 32, 64, 128, 256, 512 with alpha multiply by 2.0
4. All expert gate up projection and down projection with rank 16, 32, 64, 128, 256, 512 with alpha multiply by 2.0
5. Liger fused cross entropy.
6. 2e-4 learning rate, 50 warmup, 2 epoch only.

### WanDB

https://wandb.ai/aies-scicom-scicom-ai/malaysian-reasoning-20b

### Benchmark

We benchmark using https://huggingface.co/datasets/UMxYTLAILabs/MalayMMLU

1. Merge with base model,

- merge self attention linear layers using PEFT LoRA checkpoints with base model, [notebook/merge-lora-20.ipynb](notebook/merge-lora-20.ipynb)
- merge custom made linear and expert layers with base model, [notebook/merge-lora-20.ipynb](notebook/merge-lora-20.ipynb)

2. Evaluating merge models using vLLM inside subprocess inside multiprocessing, [notebook/malaymmlu-20b.ipynb](notebook/malaymmlu-20b.ipynb)
3. Calculate accuracy, [notebook/accuracy-malaymmlu-20b.ipynb](notebook/accuracy-malaymmlu-20b.ipynb)

## Scale up to GPT OSS 120B

We use the best rank parameter for linear layers and experts.