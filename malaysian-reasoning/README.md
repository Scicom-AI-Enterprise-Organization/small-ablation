# Malaysian Reasoning

LoRA SFT on https://huggingface.co/datasets/mesolitica/Malaysian-Reasoning

## Ablation on GPT OSS 20B

1. Use `kernels-community/vllm-flash-attn3` for Flash Attention 3 with Sink.
2. Multipacking variable length 16384 context length, with batch size of 4, so total tokens for each batch is 65536.
3. All linear layers with rank 16, 32, 64, 128, 256, 512 with alpha multiply by 2.0
4. All expert gate up projection and down projection with rank 16, 32, 64, 128, 256, 512 with alpha multiply by 2.0
5. Liger fused cross entropy.
6. 2e-4 learning rate, 50 warmup, 2 epoch only.

### WanDB

https://wandb.ai/aies-scicom-scicom-ai/malaysian-reasoning-20b

## Scale up to GPT OSS 120B

We use the best rank parameter for linear layers and experts.