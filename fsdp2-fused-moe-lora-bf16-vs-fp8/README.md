# FSDP2 Fused MoE LoRA BF16 vs FP8

We should compare the loss using the FSDP2 Fused MoE with zai-org/GLM-4.5-Air, applying LoRA to all linear layers including the experts where the base layers in lower precision such as FP8, while keeping the global batch size the same.

## How to

1. Prepare the dataset,

```bash
HF_HUB_ENABLE_HF_TRANSFER=0 hf download Scicom-intl/Malaysian-Instructions-multipacking-glm --repo-type=dataset --local-dir=./multipacking-glm
```

Script to prepare the dataset at [../malaysian-sft/notebook/multipack-malaysian-instructions-glm.ipynb](../malaysian-sft/notebook/multipack-malaysian-instructions-glm.ipynb)

2. Stack the experts,

For BF16,

```bash
python3 stack-checkpoint-transpose.py
```

For FP8,

```bash
python3 stack-checkpoint.py 
```

FP8 required to be in column major for the weight but major problem if move to column major it will become non-contiguous so FSDP2 cannot shard it, so what we can do, we save as row major contiguouly during forward we will transpose to become column major.

3. Run the finetuning,

```bash
bash fsdp2-fused-moe.sh
```

## WanDB

https://wandb.ai/aies-scicom-scicom-ai/fsdp2-ep-vs-fsdp2-fused-moe

<img src="wandb.png" width="50%">