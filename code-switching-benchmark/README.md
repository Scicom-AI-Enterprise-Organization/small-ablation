# Code-Switching Benchmark Results

Benchmark evaluates whether a model replies in the same language as the user's last message across English, Chinese, Malay, and Tamil.

All models are served via [vLLM](https://github.com/vllm-project/vllm) with **fp8 quantization** (`--quantization fp8`).

---

## How to Run

### 1. Install dependencies

```bash
bash install.sh
uv pip install vllm aiohttp tqdm
```

### 2. Run the benchmark

**Auto-launch vLLM and benchmark a model:**
```bash
python benchmark.py --model Qwen/Qwen3-4B -- --quantization fp8
```

**With multiple GPUs:**
```bash
python benchmark.py --model Qwen/Qwen3-4B --cuda-devices 0,1 -- --quantization fp8 --tensor-parallel-size 2
```

**Use an already-running vLLM server:**
```bash
python benchmark.py --model Qwen/Qwen3-4B --server-url http://localhost:8000
```

Results are saved to `results/<model-name>/` — `progress.jsonl` for per-sample records and `summary.json` for aggregated accuracy. The benchmark resumes automatically if interrupted.

---

## Summary

| Model | Overall | English | Chinese | Malay | Tamil |
|---|---|---|---|---|---|
| Qwen/Qwen3.6-27B | **67.7%** | 92.8% | 15.8% | 56.7% | 98.6% |
| mistralai/Mistral-Small-4-119B-2603 | 65.7% | 37.0% | 75.1% | 86.0% | 79.4% |
| MiniMaxAI/MiniMax-M2.7 | 64.7% | 92.3% | 27.8% | 32.1% | 90.2% |
| google/gemma-4-31B-it | 63.8% | 27.7% | 70.6% | 82.0% | 90.9% |
| Qwen/Qwen3.5-397B-A17B | 61.8% | 95.9% | 7.4% | 33.3% | 97.4% |
| Qwen/Qwen3.5-122B-A10B | 60.3% | 97.7% | 9.4% | 36.2% | 97.2% |
| Qwen/Qwen3.5-35B-A3B | 59.6% | 94.3% | 4.9% | 30.2% | 97.2% |
| zai-org/GLM-4.7-Flash | 53.0% | 69.2% | 22.5% | 29.4% | 80.5% |

---

## Per-Model Details

### Qwen/Qwen3.6-27B

| Language | Total | Matched | Accuracy |
|---|---|---|---|
| English | 960 | 891 | 92.8% |
| Chinese | 789 | 125 | 15.8% |
| Malay | 506 | 287 | 56.7% |
| Tamil | 725 | 715 | 98.6% |
| **Overall** | **2980** | **2018** | **67.7%** |

---

### mistralai/Mistral-Small-4-119B-2603

| Language | Total | Matched | Accuracy |
|---|---|---|---|
| English | 949 | 351 | 37.0% |
| Chinese | 776 | 583 | 75.1% |
| Malay | 500 | 430 | 86.0% |
| Tamil | 722 | 573 | 79.4% |
| **Overall** | **2947** | **1937** | **65.7%** |

---

### MiniMaxAI/MiniMax-M2.7

| Language | Total | Matched | Accuracy |
|---|---|---|---|
| English | 960 | 886 | 92.3% |
| Chinese | 788 | 219 | 27.8% |
| Malay | 505 | 162 | 32.1% |
| Tamil | 744 | 671 | 90.2% |
| **Overall** | **2997** | **1938** | **64.7%** |

---

### google/gemma-4-31B-it

| Language | Total | Matched | Accuracy |
|---|---|---|---|
| English | 960 | 266 | 27.7% |
| Chinese | 789 | 557 | 70.6% |
| Malay | 506 | 415 | 82.0% |
| Tamil | 746 | 678 | 90.9% |
| **Overall** | **3001** | **1916** | **63.8%** |

---

### Qwen/Qwen3.5-397B-A17B

| Language | Total | Matched | Accuracy |
|---|---|---|---|
| English | 910 | 873 | 95.9% |
| Chinese | 780 | 58 | 7.4% |
| Malay | 501 | 167 | 33.3% |
| Tamil | 718 | 699 | 97.4% |
| **Overall** | **2909** | **1797** | **61.8%** |

---

### Qwen/Qwen3.5-122B-A10B

| Language | Total | Matched | Accuracy |
|---|---|---|---|
| English | 853 | 833 | 97.7% |
| Chinese | 776 | 73 | 9.4% |
| Malay | 497 | 180 | 36.2% |
| Tamil | 531 | 516 | 97.2% |
| **Overall** | **2657** | **1602** | **60.3%** |

---

### Qwen/Qwen3.5-35B-A3B

| Language | Total | Matched | Accuracy |
|---|---|---|---|
| English | 922 | 869 | 94.3% |
| Chinese | 780 | 38 | 4.9% |
| Malay | 496 | 150 | 30.2% |
| Tamil | 674 | 655 | 97.2% |
| **Overall** | **2872** | **1712** | **59.6%** |

---

### zai-org/GLM-4.7-Flash

| Language | Total | Matched | Accuracy |
|---|---|---|---|
| English | 956 | 662 | 69.2% |
| Chinese | 786 | 177 | 22.5% |
| Malay | 504 | 148 | 29.4% |
| Tamil | 735 | 592 | 80.5% |
| **Overall** | **2981** | **1579** | **53.0%** |
