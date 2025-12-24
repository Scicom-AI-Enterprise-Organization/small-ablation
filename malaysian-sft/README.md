# Malaysian SFT

LoRA SFT on https://huggingface.co/datasets/Scicom-intl/Malaysian-Instructions

## Ablation on multiple models

1. Ablation on Qwen/Qwen3-32B, Qwen/Qwen2.5-72B-Instruct, meta-llama/Llama-3.1-70B-Instruct, openai/gpt-oss-120b, 
2. Dense LoRA SFT done using DeepSpeed Zero3 HF Trainer while MoE LoRA SFT done using FSDP2 + EP.
3. Multipacking variable length 16384 context length, with global batch size of 8, so global total tokens is 65536.
4. All self attention linear layers with rank 256 with alpha multiply by 2.0 <sup> + </sup>
5. Liger fused cross entropy.
6. 1e-4 learning rate, 50 warmup, 3 epoch only.
7. Calculate accuracy for each epoch using non-reasoning and reasoning system prompt.

<sup> + </sup> with the rank of each equal to the total rank divided by the number of active experts, https://thinkingmachines.ai/blog/lora/

### WanDB

https://wandb.ai/aies-scicom-scicom-ai/malaysian-sft

### Benchmark

We benchmark using https://huggingface.co/datasets/UMxYTLAILabs/MalayMMLU

1. Calculate accuracy,

For Qwen/Qwen3-32B,

```bash
python3 malaymmlu.py --pattern "ds3-qwen3-32b-lora-256*" --num_gpus 8 --gpu_partition 2
```

For Qwen/Qwen2.5-72B-Instruct,

```bash
python3 malaymmlu.py --pattern "ds3-qwen2.5-72b-lora-256*" --num_gpus 8 --gpu_partition 4
```

For meta-llama/Llama-3.1-70B-Instruct,

```bash
python3 malaymmlu.py --pattern "ds3-llama3.1-70b-lora-256*" --num_gpus 8 --gpu_partition 4
```