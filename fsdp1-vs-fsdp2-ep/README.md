# FSDP1 vs FSDP2 EP

Comparing throughput FSDP1 vs FSDP2 EP on GPT OSS 20B LoRA on all linear layers including experts on the same global batch size.

## How to

1. Prepare the dataset,

```bash
HF_HUB_ENABLE_HF_TRANSFER=0 hf download Scicom-intl/malaysian-reasoning-16k-mosaic --repo-type=dataset --local-dir=./malaysian-reasoning-16k-mosaic
```

2. Run the finetuning,

```bash
bash fsdp1.sh
```

## WanDB

https://wandb.ai/aies-scicom-scicom-ai/fsdp1-vs-fsdp2-ep

<img src="wandb.png" width="50%">