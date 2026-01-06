# FSDP2 Fused MoE LoRA BF16 vs FP8

We should compare the loss using the FSDP2 Fused MoE with Qwen/Qwen3-30B-A3B-Instruct-2507, applying LoRA to all linear layers including the experts where the base layers in lower precision such as FP8, while keeping the global batch size the same.

## How to

1. Prepare the dataset, [../malaysian-sft/notebook/multipack-malaysian-instructions-glm3.ipynb](../malaysian-sft/notebook/multipack-malaysian-instructions-glm.ipynb)

2. Stack the experts,

For BF16,

```bash
python3 stack-checkpoint-transpose.py
```

For FP8,

```bash
python3 stack-checkpoint.py 
```

FP8 required to be in column major for the weight but major problem if move to column major it will become non-contiguous so FSDP2 cannot shard it, so what we can do, we save as row major contiguously during forward we will transpose to become column major.

3. Run the finetuning,

```bash
bash fsdp2-fused-moe.sh
```

## WanDB

https://wandb.ai/aies-scicom-scicom-ai/fsdp2-fused-moe-lora-bf16-vs-fp8

<img src="wandb.png" width="50%">