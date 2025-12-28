# FSDP1 MoE LoRA vs FSDP2 EP MoE LoRA

Because we build a lot our own FSDP2 EP MoE LORA script, I think we should compare the loss using the FSDP1 Hugging Face Trainer with Qwen3-30B-A3B, applying LoRA to all linear layers including the experts, while keeping the global batch size the same.

## How to

1. Prepare the dataset,

```bash
HF_HUB_ENABLE_HF_TRANSFER=0 hf download Scicom-intl/mosaic-ms-wikipedia-2023-10-01 --repo-type=dataset --local-dir=./multipacking
```

2. Run the finetuning,

```bash
bash fsdp2.sh
```

## WanDB

https://wandb.ai/aies-scicom-scicom-ai/fsdp1-peft-vs-fsdp2

<img src="wandb.png" width="50%">