# Turn Detector Qwen3-0.6B

Fine-tuned Qwen3-0.6B for turn-end detection in multilingual call center conversations. The model predicts `P(im_end)` - the probability that a speaker's turn is complete.

## How it works

- `P(im_end) > 0.5` → turn complete (positive)
- `P(im_end) < 0.5` → turn incomplete (negative)

## Eval Results

**Checkpoint:** `output-0.6B-v2/checkpoint-1300`

### Overall

| Metric | Score |
|--------|-------|
| Accuracy | 0.9700 (1164/1200) |
| Precision | 1.0000 |
| Recall | 0.9400 |
| F1 | 0.9691 |

### Per Class

| Class | Accuracy |
|-------|----------|
| Positive (turn complete) | 0.9400 (564/600) |
| Negative (turn incomplete) | 1.0000 (600/600) |

### Per Language

| Language | Overall | Positive | Negative |
|----------|---------|----------|----------|
| chinese-english | 0.9700 | 0.9400 | 1.0000 |
| chinese-malay | 0.9700 | 0.9400 | 1.0000 |
| chinese-tamil | 0.9700 | 0.9400 | 1.0000 |
| english-chinese | 1.0000 | 1.0000 | 1.0000 |
| english-malay | 0.9700 | 0.9400 | 1.0000 |
| english-tamil | 0.9400 | 0.8800 | 1.0000 |
| malay-chinese | 0.9700 | 0.9400 | 1.0000 |
| malay-english | 0.9300 | 0.8600 | 1.0000 |
| malay-tamil | 0.9500 | 0.9000 | 1.0000 |
| tamil-chinese | 1.0000 | 1.0000 | 1.0000 |
| tamil-english | 0.9800 | 0.9600 | 1.0000 |
| tamil-malay | 0.9900 | 0.9800 | 1.0000 |

### Threshold Sweep

| Threshold | Accuracy | Precision | Recall | F1 |
|-----------|----------|-----------|--------|-----|
| 0.1 | 0.9933 | 0.9966 | 0.9900 | 0.9933 |
| 0.2 | 0.9900 | 0.9966 | 0.9833 | 0.9899 |
| 0.3 | 0.9883 | 1.0000 | 0.9767 | 0.9882 |
| 0.4 | 0.9850 | 1.0000 | 0.9700 | 0.9848 |
| **0.5** | **0.9700** | **1.0000** | **0.9400** | **0.9691** |
| 0.6 | 0.9608 | 1.0000 | 0.9217 | 0.9592 |
| 0.7 | 0.9467 | 1.0000 | 0.8933 | 0.9437 |
| 0.8 | 0.9092 | 1.0000 | 0.8183 | 0.9001 |
| 0.9 | 0.8392 | 1.0000 | 0.6783 | 0.8083 |

### Probability Distribution

| Class | Mean | Median | Min | Max |
|-------|------|--------|-----|-----|
| Positive | 0.8817 | 0.9569 | 0.0046 | 0.9997 |
| Negative | 0.0010 | 0.0000 | 0.0000 | 0.2509 |

## WandB

- v1: https://wandb.ai/aies-scicom-scicom-ai/qwen3-0.6B-turn-detector?nw=nwuserhuseinzolkepli
- v2: https://wandb.ai/malaysia-ai/qwen3-0.6B-turn-detector-v2/runs/g4fhm4j6?nw=nwuserariffnzhn

## Dataset

- Train: positive samples only (complete conversations with `<|im_end|>`)
- Test: 1200 samples (600 positive + 600 negative), 50 conversations per language pair

### Sources

| Dataset | HuggingFace |
|---------|-------------|
| Call Center Language Switching | `Scicom-intl/Call-Center-Language-Switching` |
| Function Call | `Scicom-intl/Function-Call` |
| Malaysian Multiturn Chat Assistant | `mesolitica/Malaysian-Multiturn-Chat-Assistant` |
| Malaysian Speech Instructions | `mesolitica/Malaysian-Speech-Instructions` |

### S3

| Split | Path |
|-------|------|
| Train | `s3://aies-research-datasets/call-center-language-switching/parquet-train-merged-v2/` |
| Test | `s3://aies-research-datasets/call-center-language-switching/parquet-test-v2/` |
