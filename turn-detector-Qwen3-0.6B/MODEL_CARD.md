---
license: apache-2.0
base_model: Qwen/Qwen3-0.6B
language:
  - ms
  - en
  - zh
  - ta
tags:
  - turn-detection
  - call-center
  - code-switching
  - multilingual
pipeline_tag: text-generation
---

# Turn Detector Qwen3-0.6B

Fine-tuned [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) for **real-time turn-end detection** in multilingual call center conversations.

The model predicts `P(<|im_end|>)` — the probability that a speaker has finished their turn. Designed for low-latency voice agent pipelines (e.g. LiveKit) to determine when to respond.

## How It Works

Given a conversation so far, the model outputs the probability of `<|im_end|>` as the next token:

- **P(im_end) > 0.5** → speaker is done talking (turn complete)
- **P(im_end) < 0.5** → speaker is still talking (turn incomplete)

## Usage

```python
import torch
import math
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Scicom-intl/turn-detector-Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).cuda().eval()

IM_END_ID = tokenizer.convert_tokens_to_ids("<|im_end|>")

def get_turn_end_prob(text):
    """Returns probability that the speaker's turn is complete."""
    # Strip trailing <|im_end|> so the model predicts whether to emit it
    if text.endswith("<|im_end|>"):
        text = text[:-len("<|im_end|>")]
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**inputs).logits
    prob = F.softmax(logits[0, -1], dim=-1)[IM_END_ID].item()
    return prob

# Complete turn - should be high probability
text = "<|im_start|>user\nHello, saya nak tanya pasal bil saya.<|im_end|>\n<|im_start|>assistant\nBoleh, sila berikan nombor akaun anda."
prob = get_turn_end_prob(text)
print(f"P(turn complete) = {prob:.4f}")  # ~0.74

# Incomplete turn - should be low probability
text = "<|im_start|>user\nHello, saya nak tanya pasal bil saya.<|im_end|>\n<|im_start|>assistant\nBoleh, sila berikan nombor"
prob = get_turn_end_prob(text)
print(f"P(turn complete) = {prob:.4f}")  # ~0.00
```

## Eval Results

**Test set:** 1200 samples (600 positive + 600 negative), 50 conversations per language pair.

### Overall (threshold = 0.5)

| Metric | Score |
|--------|-------|
| Accuracy | 97.00% |
| Precision | 100.00% |
| Recall | 94.00% |
| F1 | 96.91% |

### Per Language

| Language Pair | Overall | Positive | Negative |
|---------------|---------|----------|----------|
| chinese-english | 97.00% | 94.00% | 100.00% |
| chinese-malay | 97.00% | 94.00% | 100.00% |
| chinese-tamil | 97.00% | 94.00% | 100.00% |
| english-chinese | 100.00% | 100.00% | 100.00% |
| english-malay | 97.00% | 94.00% | 100.00% |
| english-tamil | 94.00% | 88.00% | 100.00% |
| malay-chinese | 97.00% | 94.00% | 100.00% |
| malay-english | 93.00% | 86.00% | 100.00% |
| malay-tamil | 95.00% | 90.00% | 100.00% |
| tamil-chinese | 100.00% | 100.00% | 100.00% |
| tamil-english | 98.00% | 96.00% | 100.00% |
| tamil-malay | 99.00% | 98.00% | 100.00% |

### Threshold Sweep

| Threshold | Accuracy | Precision | Recall | F1 |
|-----------|----------|-----------|--------|-----|
| 0.1 | 99.33% | 99.66% | 99.00% | 99.33% |
| 0.2 | 99.00% | 99.66% | 98.33% | 98.99% |
| 0.3 | 98.83% | 100.00% | 97.67% | 98.82% |
| 0.4 | 98.50% | 100.00% | 97.00% | 98.48% |
| **0.5** | **97.00%** | **100.00%** | **94.00%** | **96.91%** |
| 0.6 | 96.08% | 100.00% | 92.17% | 95.92% |
| 0.7 | 94.67% | 100.00% | 89.33% | 94.37% |
| 0.8 | 90.92% | 100.00% | 81.83% | 90.01% |
| 0.9 | 83.92% | 100.00% | 67.83% | 80.83% |

### Probability Distribution

| Class | Mean | Median | Min | Max |
|-------|------|--------|-----|-----|
| Positive (turn complete) | 0.8817 | 0.9569 | 0.0046 | 0.9997 |
| Negative (turn incomplete) | 0.0010 | 0.0000 | 0.0000 | 0.2509 |

## Training

- **Base model:** Qwen/Qwen3-0.6B
- **Training data:** Positive samples only (complete conversations ending with `<|im_end|>`)
- **Loss:** Liger Fused Linear Cross Entropy
- **Attention:** Flash Attention 2
- **Precision:** bfloat16
- **Block size:** 8192 (multipacked)
- **Batch size:** 4 x 8 gradient accumulation
- **Learning rate:** 2e-5 (constant)
- **Epochs:** 1

### Training Data Sources

| Dataset | Source |
|---------|--------|
| Call Center Language Switching | [Scicom-intl/Call-Center-Language-Switching](https://huggingface.co/datasets/Scicom-intl/Call-Center-Language-Switching) |
| Function Call | [Scicom-intl/Function-Call](https://huggingface.co/datasets/Scicom-intl/Function-Call) |
| Malaysian Multiturn Chat Assistant | [mesolitica/Malaysian-Multiturn-Chat-Assistant](https://huggingface.co/datasets/mesolitica/Malaysian-Multiturn-Chat-Assistant) |
| Malaysian Speech Instructions | [mesolitica/Malaysian-Speech-Instructions](https://huggingface.co/datasets/mesolitica/Malaysian-Speech-Instructions) |

## WandB

- [v1 training run](https://wandb.ai/aies-scicom-scicom-ai/qwen3-0.6B-turn-detector?nw=nwuserhuseinzolkepli)
- [v2 training run](https://wandb.ai/malaysia-ai/qwen3-0.6B-turn-detector-v2/runs/g4fhm4j6?nw=nwuserariffnzhn)
