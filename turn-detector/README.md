# Turn Detector Qwen3-0.6B

Fine-tuned Qwen3-0.6B for turn-end detection in multilingual call center conversations. The model predicts `P(im_end)` - the probability that a speaker's turn is complete.

## How it works

- `P(im_end) > 0.5` → turn complete (positive)
- `P(im_end) < 0.5` → turn incomplete (negative)

## Eval Session 1: Synthetic Test Set

**Test set:** [Scicom-intl/Evaluation-Malaysian-Turn-Detector](https://huggingface.co/datasets/Scicom-intl/Evaluation-Malaysian-Turn-Detector) — 238 synthetic samples (119 positive + 119 negative), 1-2 turns per conversation, 12 language pairs.

### Baseline vs Qwen3-0.6B

| Metric | Baseline ([livekit/turn-detector v0.4.1-intl](https://huggingface.co/livekit/turn-detector/tree/v0.4.1-intl)) | Qwen3-0.6B ([Scicom-intl/turn-detector-Qwen3-0.6B](https://huggingface.co/Scicom-intl/turn-detector-Qwen3-0.6B)) |
|--------|----------|------------|
| Accuracy | 45.80% | **85.29%** |
| Precision | 45.97% | **100.00%** |
| Recall | 47.90% | **70.59%** |
| F1 | 46.91% | **82.76%** |

### Per Class

| Class | Baseline | Qwen3-0.6B |
|-------|----------|------------|
| Positive (turn complete) | 47.90% | **70.59%** |
| Negative (turn incomplete) | 43.70% | **100.00%** |

### Per Language

| Language | Baseline Overall | Qwen3-0.6B Overall | Baseline Pos | Qwen3-0.6B Pos | Baseline Neg | Qwen3-0.6B Neg |
|----------|-----------------|-------------------|-------------|----------------|-------------|----------------|
| chinese-english | 55.00% | **85.00%** | 30.00% | **70.00%** | 80.00% | **100.00%** |
| chinese-malay | 55.00% | **85.00%** | 10.00% | **70.00%** | **100.00%** | **100.00%** |
| chinese-tamil | 50.00% | **90.00%** | 10.00% | **80.00%** | 90.00% | **100.00%** |
| english-chinese | 30.00% | **80.00%** | 20.00% | **60.00%** | 40.00% | **100.00%** |
| english-malay | 25.00% | **90.00%** | 30.00% | **80.00%** | 20.00% | **100.00%** |
| english-tamil | 35.00% | **90.00%** | **70.00%** | 80.00% | 0.00% | **100.00%** |
| malay-chinese | 50.00% | **100.00%** | 90.00% | **100.00%** | 10.00% | **100.00%** |
| malay-english | 35.00% | **100.00%** | 60.00% | **100.00%** | 10.00% | **100.00%** |
| malay-tamil | 40.00% | **85.00%** | **80.00%** | 70.00% | 0.00% | **100.00%** |
| tamil-chinese | 55.56% | **94.44%** | 22.22% | **88.89%** | 88.89% | **100.00%** |
| tamil-english | **70.00%** | 65.00% | **80.00%** | 30.00% | 60.00% | **100.00%** |
| tamil-malay | 50.00% | 60.00% | **70.00%** | 20.00% | 30.00% | **100.00%** |

### Threshold Sweep (Qwen3-0.6B)

| Threshold | Accuracy | Precision | Recall | F1 |
|-----------|----------|-----------|--------|-----|
| 0.1 | 95.80% | 100.00% | 91.60% | 95.61% |
| 0.2 | 92.86% | 100.00% | 85.71% | 92.31% |
| 0.3 | 91.60% | 100.00% | 83.19% | 90.83% |
| 0.4 | 87.82% | 100.00% | 75.63% | 86.12% |
| **0.5** | **85.29%** | **100.00%** | **70.59%** | **82.76%** |
| 0.6 | 82.77% | 100.00% | 65.55% | 79.19% |
| 0.7 | 79.83% | 100.00% | 59.66% | 74.74% |
| 0.8 | 74.79% | 100.00% | 49.58% | 66.29% |
| 0.9 | 68.49% | 100.00% | 36.97% | 53.99% |

### Probability Distribution (Qwen3-0.6B)

| Class | Mean | Median | Min | Max |
|-------|------|--------|-----|-----|
| Positive | 0.6775 | 0.7973 | 0.0001 | 0.9973 |
| Negative | 0.0000 | 0.0000 | 0.0000 | 0.0001 |

## WandB

- v1: https://wandb.ai/aies-scicom-scicom-ai/qwen3-0.6B-turn-detector?nw=nwuserhuseinzolkepli
- v2: https://wandb.ai/malaysia-ai/qwen3-0.6B-turn-detector-v2/runs/g4fhm4j6?nw=nwuserariffnzhn

## Dataset

- Train: positive samples only (complete conversations with `<|im_end|>`) — `s3://aies-research-datasets/call-center-language-switching/parquet-train-merged-v2/`
- Evaluation: [Scicom-intl/Evaluation-Malaysian-Turn-Detector](https://huggingface.co/datasets/Scicom-intl/Evaluation-Malaysian-Turn-Detector) — 238 synthetic samples (119 positive + 119 negative), 1-2 turns per conversation, 12 language pairs

### Sources

| Dataset | HuggingFace |
|---------|-------------|
| Call Center Language Switching | [Scicom-intl/Call-Center-Language-Switching](https://huggingface.co/datasets/Scicom-intl/Call-Center-Language-Switching) |
| Function Call | [Scicom-intl/Function-Call](https://huggingface.co/datasets/Scicom-intl/Function-Call) |
| Malaysian Multiturn Chat Assistant | [mesolitica/Malaysian-Multiturn-Chat-Assistant](https://huggingface.co/datasets/mesolitica/Malaysian-Multiturn-Chat-Assistant) |
| Malaysian Speech Instructions | [mesolitica/Malaysian-Speech-Instructions](https://huggingface.co/datasets/mesolitica/Malaysian-Speech-Instructions) |
