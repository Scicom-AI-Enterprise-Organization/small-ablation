# Multipacking Padding FA3 Compile

Compare speed up if we do multipacking padding for FA3 torch compile.

## How to

1. Install dependencies,

```bash
bash install.sh
```

2. Clone the dataset,

Non optimized,

```bash
HF_HUB_ENABLE_HF_TRANSFER=0 hf download malaysia-ai/multipacking-multilingual-tts-10k-qwen3 --repo-type=dataset --local-dir=./multipacking
```

Optimized,

```bash
HF_HUB_ENABLE_HF_TRANSFER=0 hf download Scicom-intl/multipacking-multilingual-tts-10k-qwen3-optimized --repo-type=dataset --local-dir=./multipacking-optimized
```