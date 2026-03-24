# Call Center Language Switching SLM

Train small language models (0.6B, 1.7B, 4B) for call center language switching detection using Qwen3 Base.

## Architecture

| Component | Detail |
|-----------|--------|
| Base Model | Qwen3 (0.6B / 1.7B / 4B) |
| Loss | Liger Fused Linear Cross Entropy (chunk CE) |
| Attention | Flash Attention 3/4 varlen (`cu_seqlens`) |
| Packing | Multipacking variable-length sequences |
| Precision | Pure bfloat16 |
| Data Format | chinidataset Parquet streaming (`uint32[]`) |
| Monitoring | WandB (loss, MFU, throughput) |

## Task

Binary signal - does the model predict `<|im_end|>` at the correct position?

- **Positive class**: conversation ends with `<|im_end|>` (complete turn)
- **Negative class**: conversation does NOT end with `<|im_end|>` (truncated/incomplete turn)

## Datasets

### Train

| Dataset | Source |
|---------|--------|
| Call Center Language Switching | `Scicom-intl/Call-Center-Language-Switching` |
| Function Call | `Scicom-intl/Function-Call` |
| Malaysian Multiturn Chat Assistant | `mesolitica/Malaysian-Multiturn-Chat-Assistant` |
| Malaysian Speech Instructions | `mesolitica/Malaysian-Speech-Instructions` |

### Test

50 conversations per language sampled from `Scicom-intl/Call-Center-Language-Switching`.

## Setup

### Install

```bash
bash install.sh
```

Or with pip:

```bash
pip install -r requirements.txt
```

> **Note:** Flash Attention requires CUDA. For B200 with FA4, install `flash-attn==4.0.0b1` instead.

## Steps

### 1. Prepare Dataset

**Option A – Python script (RunPod / headless):**

```bash
python prepare-dataset.py \
    --block_size 8192 \
    --n_workers 20 \
    --s3_bucket aies-research-datasets \
    --s3_prefix call-center-language-switching
```

**Option B – Notebook (interactive exploration):**

```bash
jupyter notebook prepare-dataset.ipynb
```

Both do the same 9 steps:
1. Download all 4 train datasets from HuggingFace
2. Convert to unified chat format
3. Build test set (50 conversations/language from Call-Center-Language-Switching)
4. Format with Qwen3 chat template:
   ```
   <|im_start|>system
   {system_message}<|im_end|>
   <|im_start|>user
   {user_message}<|im_end|>
   <|im_start|>assistant
   {assistant_message}<|im_end|>    <-- positive (has <|im_end|>)
   {partial_response}               <-- negative (no <|im_end|>)
   ```
5. Build positive (complete) and negative (truncated) samples
6. Tokenize and multipack into 8192-token blocks
7. Write to chinidataset Parquet format using `ParquetWriter`
8. Verify with `StreamingDataset`
9. Upload to S3 (`s3://aies-research-datasets/call-center-language-switching/`)

**Output:**

| Location | Train | Test |
|----------|-------|------|
| Local | `./parquet-train-merged/` | `./parquet-test/` |
| S3 | `s3://aies-research-datasets/call-center-language-switching/parquet-train-merged/` | `s3://aies-research-datasets/call-center-language-switching/parquet-test/` |

### 2. Train

```bash
# 0.6B on H100
bash train-0.6B.sh

# 1.7B on H100/H200
bash train-1.7B.sh

# 4B on H200/B200
bash train-4B.sh
```

### 3. Evaluate

Evaluate on the test set:
- Per-language accuracy for `<|im_end|>` prediction
- Precision / Recall / F1
- Cross-scale comparison (0.6B vs 1.7B vs 4B)

## Training Configs

| Scale | GPU | Batch Size | Grad Accum | LR | Block Size |
|-------|-----|-----------|-----------|-----|-----------|
| 0.6B | H100 (80GB) | 4 | 8 | 2e-5 | 8192 |
| 1.7B | H100/H200 | 2 | 16 | 2e-5 | 8192 |
| 4B | H200/B200 | 1 | 32 | 1e-5 | 8192 |

All configs use:
- `constant_with_warmup` scheduler, 100 warmup steps
- Gradient checkpointing
- 3 epochs
- WandB logging every step

### GPU-specific notes

| GPU | FA Version | Peak FLOPS (bf16) | Env Var |
|-----|-----------|-------------------|---------|
| H100 SXM | `flash_attention_3` | 989 TFLOPS | `GPU_PEAK_FLOPS=989e12` |
| H200 SXM | `flash_attention_3` | 989 TFLOPS | `GPU_PEAK_FLOPS=989e12` |
| B200 | `flash_attention_4` | 2250 TFLOPS | `GPU_PEAK_FLOPS=2250e12` |

## File Structure

```
call-center-language-switching/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── install.sh                 # Install script
├── prepare-dataset.py         # Dataset prep script (RunPod / headless)
├── prepare-dataset.ipynb      # Dataset prep notebook (interactive)
├── train.py                   # Training script
├── train-0.6B.sh              # 0.6B training config
├── train-1.7B.sh              # 1.7B training config
└── train-4B.sh                # 4B training config
```

## Key Design Decisions

### Multipacking + varlen FA

Multiple conversations are packed into a single 8192-token block. Each conversation maintains its own position IDs (reset per conversation) and the `attention_mask` field stores sequence lengths (not a 0/1 mask). During training, these lengths are converted to `cu_seqlens` for Flash Attention's variable-length kernel, ensuring no cross-contamination between packed conversations.

### Liger Chunk Cross Entropy

`LigerFusedLinearCrossEntropyLoss` fuses the final linear projection (`lm_head`) with the cross-entropy loss computation. This avoids materializing the full logits tensor (vocab_size x seq_len), significantly reducing peak memory - critical for 4B scale training on single GPU.

### chinidataset Parquet Format

Uses `chinidataset.ParquetWriter` with native `uint32[]` variable-length arrays instead of `mosaicml-streaming` MDS format. Benefits:
- 9.9x-128.9x faster reads
- Native variable-length array support (no custom encoding)
- Parquet columnar format with compression
- LRU caching and look-ahead prefetching via `StreamingDataset`

### Positive/Negative Class Construction

- **Positive**: full conversation with Qwen3 chat template, ending with `<|im_end|>`
- **Negative**: same conversation but truncated - last `<|im_end|>` removed, with 50% chance of further truncation at a random point in the last assistant response (keeps 20-80% of content)

This teaches the model to predict `<|im_end|>` for complete turns and recognize incomplete turns.
