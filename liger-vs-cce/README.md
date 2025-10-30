# Liger loss vs Apple CCE loss

Compare loss without torch compile due to dynamic length, we are going to use mosaic dataset prepared by https://huggingface.co/datasets/malaysia-ai/multipacking-multilingual-tts-10k-qwen3

## How to

1. Install dependencies,

```bash
bash install.sh
```

2. Clone the dataset,

```bash
hf download malaysia-ai/multipacking-multilingual-tts-10k-qwen3 --repo-type=dataset --local-dir=./multipacking
```

3. Run quick finetuning,

```bash
bash liger.sh
bash cce.sh
bash cce-kahan.sh
```