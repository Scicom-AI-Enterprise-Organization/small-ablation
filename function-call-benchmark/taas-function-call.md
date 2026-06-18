# Function-Call-TaaS Benchmark

**Dataset:** [Scicom-intl/Function-Call-TaaS](https://huggingface.co/datasets/Scicom-intl/Function-Call-TaaS)

100 multi-turn conversations across 9 telco B2B workflows (reconciliation, SD-WAN, fraud investigation, customer retention, and more), written in Manglish (Malay-English code-switching). Each conversation has 40+ available tools and 10–20 user turns.

---

## How the Evaluation Works

We simulate a real chat session with the model, **turn by turn**:

1. The user sends a message.
2. The model replies — it may call one or more tools, or just respond in text.
3. We **record what the model did** and compare it against the reference answer.
4. We inject the **reference tool results** into the conversation (since we don't have a real backend), so the model can continue the conversation.
5. Repeat from step 1 for the next user message.

The model sees its own previous replies throughout — it builds up context naturally, just like in a real deployment.

**Why reference tool results?** The conversations involve real telco systems (CRM, billing, inventory). Since we don't have access to those backends, we use the ground-truth tool responses from the dataset. They are attached to the model's own tool call IDs so the conversation stays coherent.

---

## Metrics

| Metric | Question it answers |
|--------|---------------------|
| **tool_call_f1** | When a tool should be called, does the model call one? And when it shouldn't, does it stay silent? (F1 of precision + recall) |
| **name_set_f1** | Did the model pick the right function(s)? (Set-level F1 over function names per turn) |
| **json_valid_rate** | Are the tool call arguments valid JSON? |
| **hallucination_rate** | Did the model call a function that doesn't exist in the schema? |
| **req_coverage** | Are the required parameters present in the arguments? |
| **type_accuracy** | Are the argument values the correct JSON type (string, int, array, etc.)? |
| **parallel_count_match** | When multiple tools should be called at once, did the model call the right number? |
| **id_propagation_rate** | Did the model correctly carry over IDs from tool results (e.g. a `reconciliation_run_id`) into its next tool call? |
| **refusal_rate** | When the user goes off-topic (e.g. mileage claims, lunch recommendations), does the model correctly stay silent instead of calling a tool? *(only available on `test-extra`)* |

---

## Run the Evaluation

```bash
pip install openai datasets python-dotenv

# Put your OpenRouter (or any OpenAI-compatible) API key in .env:
echo "OPENAI_API_KEY=sk-..." > .env

# Smoke test — 5 conversations, first model in model_list
python evaluate_function_calling.py --dry-run

# Evaluate one model
python evaluate_function_calling.py --model qwen/qwen3.6-27b

# Evaluate all models in model_list (3 in parallel, checkpointed)
python evaluate_function_calling.py --all-models

# Evaluate on the test-extra config (includes out-of-context turns / refusal metric)
python evaluate_function_calling.py --config test-extra --all-models

# Resume an interrupted run (reads checkpoints/ automatically)
python evaluate_function_calling.py --all-models
```

Checkpoints are saved to `checkpoints/<model>.jsonl` after every conversation. If the run is interrupted, re-running the same command picks up where it left off. When using `--config <name>`, checkpoints go to `checkpoints-<name>/` so different configurations don't collide.

---

## Function Calling Benchmark — `train` split

Dataset: [Scicom-intl/Function-Call-TaaS](https://huggingface.co/datasets/Scicom-intl/Function-Call-TaaS) — 100 multi-turn Manglish conversations across 9 telco B2B workflows, evaluated via full-replay mode (split: `train`).

| Model | Total Params | Active Params | Tool F1 | Name F1 | JSON Valid | Hallucination ↓ | Req Cov | Type Acc | Parallel | ID Prop |
|-------|-------------:|--------------:|--------:|--------:|-----------:|----------------:|--------:|---------:|---------:|--------:|
| minimax/minimax-m2.7 | 230B | 10B | **0.889** | 0.875 | 0.998 | **0.000** | 0.997 | 0.997 | **0.745** | 0.667 |
| qwen/qwen3.6-27b | 27.8B | 27.8B | 0.882 | 0.865 | **1.000** | **0.000** | **1.000** | 0.999 | 0.714 | 0.857 |
| google/gemma-4-31b-it | 31B | 31B | 0.878 | **0.952** | 0.962 | **0.000** | 0.964 | **1.000** | 0.731 | 0.188 |
| mistralai/mistral-small-2603 | 119B | 6.5B | 0.877 | 0.846 | 0.999 | 0.001 | 0.995 | **1.000** | 0.669 | 0.471 |
| z-ai/glm-4.7-flash | 31.2B | 3B | 0.871 | 0.866 | 0.989 | **0.000** | 0.985 | 0.988 | 0.716 | 0.292 |
| nvidia/nemotron-3-nano-30b-a3b | 31.6B | 3.2B | 0.815 | 0.694 | **1.000** | 0.004 | 0.975 | 0.986 | 0.406 | 0.667 |
| qwen/qwen3.6-35b-a3b | 35B | 3B | 0.791 | 0.773 | 0.992 | 0.001 | 0.987 | 0.998 | 0.529 | 0.667 |
| nvidia/nemotron-3-super-120b-a12b | 120B | 12B | 0.780 | 0.617 | **1.000** | 0.008 | 0.999 | 0.993 | 0.380 | **0.889** |
| google/gemma-4-26b-a4b-it | 26B | 4B | 0.668 | 0.904 | 0.920 | 0.004 | 0.915 | **1.000** | 0.454 | 0.312 |

---

## Function Calling Benchmark — `test-extra` config

Dataset: [Scicom-intl/Function-Call-TaaS](https://huggingface.co/datasets/Scicom-intl/Function-Call-TaaS) (config: `test-extra`) — 100 multi-turn Manglish conversations, each with intentional out-of-context user turns to also benchmark refusal behaviour. Evaluated via full-replay mode.

| Model | Total Params | Active Params | Tool F1 | Name F1 | JSON Valid | Hallucination ↓ | Req Cov | Type Acc | Parallel | ID Prop | Refusal ↑ |
|-------|-------------:|--------------:|--------:|--------:|-----------:|----------------:|--------:|---------:|---------:|--------:|----------:|
| mistralai/mistral-small-2603 | 119B | 6.5B | **0.831** | 0.822 | **1.000** | 0.004 | **0.999** | **1.000** | 0.798 | 0.379 | 0.537 |
| Qwen/Qwen3.5-122B-A10B | 122B | 10B | 0.809 | 0.804 | 0.996 | **0.000** | 0.996 | **1.000** | 0.786 | 0.356 | 0.477 |
| nvidia/nemotron-3-nano-30b-a3b | 31.6B | 3.2B | 0.793 | 0.755 | **1.000** | 0.013 | 0.974 | 0.982 | **0.813** | 0.462 | 0.394 |
| Qwen/Qwen3.5-397B-A17B | 397B | 17B | 0.787 | 0.802 | 0.993 | 0.001 | 0.992 | 0.996 | 0.753 | **0.523** | 0.500 |
| minimax/minimax-m2.7 | 230B | 10B | 0.782 | 0.794 | 0.999 | 0.001 | 0.994 | 0.998 | 0.724 | 0.454 | 0.569 |
| qwen/qwen3.6-27b | 27.8B | 27.8B | 0.782 | 0.805 | 0.998 | 0.001 | 0.993 | 0.998 | 0.762 | 0.345 | 0.455 |
| google/gemma-4-31b-it | 31B | 31B | 0.771 | **0.854** | 0.970 | 0.002 | 0.968 | **1.000** | 0.634 | 0.250 | 0.584 |
| z-ai/glm-4.7-flash | 31.2B | 3B | 0.767 | 0.779 | 0.998 | 0.005 | 0.992 | 0.984 | 0.706 | 0.447 | 0.576 |
| Qwen/Qwen3.5-35B-A3B | 35B | 3B | 0.756 | 0.784 | 0.989 | 0.001 | 0.986 | 0.995 | 0.755 | 0.405 | 0.406 |
| nvidia/nemotron-3-super-120b-a12b | 120B | 12B | 0.739 | 0.750 | 0.998 | 0.001 | 0.997 | 0.995 | 0.792 | 0.448 | 0.359 |
| google/gemma-4-26b-a4b-it | 26B | 4B | 0.707 | **0.854** | 0.955 | 0.004 | 0.949 | **1.000** | 0.563 | 0.261 | 0.657 |
| qwen/qwen3.6-35b-a3b | 35B | 3B | 0.337 | 0.742 | 0.994 | **0.000** | 0.988 | 0.999 | 0.206 | 0.435 | **0.838** |

**Notes:** `qwen/qwen3.6-35b-a3b` has the highest refusal rate (0.838) but the lowest Tool F1 — it tends to under-call across the board, not just on out-of-context turns. `mistralai/mistral-small-2603` is the most balanced overall, leading on Tool F1, JSON validity, required-param coverage, and type accuracy.

---

## Self-Hosting on H100 / H20 (vLLM)

### Node Specs

| GPU | VRAM per card | 8-GPU node total |
|-----|:-------------:|:----------------:|
| H100 SXM5 80 GB | 80 GB | 640 GB |
| H20 96 GB | 96 GB | 768 GB |

Both support FP8 natively. **FP8 vs BF16 on H100: ~2× throughput improvement and ~2× memory savings**, letting you fit larger models or run more replicas per node. [[1]](#ref1)[[2]](#ref2)

---

### VRAM Requirements & Deployment

All figures are approximate weights-only VRAM. Add ~10–15% for KV cache and activations. "Replicas" assumes 8 H100 80 GB cards.

| Model | BF16 VRAM | FP8 VRAM | Min GPUs (FP8) | Replicas / node |
|-------|----------:|----------:|:--------------:|:---------------:|
| Qwen/Qwen3.5-397B-A17B (397B MoE) | ~794 GB | ~397 GB | 8 (TP8+EP8) | 1 (node-filling) |
| minimax/minimax-m2.7 (230B MoE) | ~460 GB | ~230 GB | 4 (TP4+EP4) | 2 |
| mistralai/mistral-small-2603 (119B MoE) | ~238 GB | ~119 GB | 2 (TP2) | 4 |
| Qwen/Qwen3.5-122B-A10B (122B MoE) | ~244 GB | ~122 GB | 2 (TP2+EP2) | 4 |
| nvidia/nemotron-3-super-120b-a12b (120B MoE) | ~240 GB | ~120 GB | 3 (TP2–4) | 2 |
| qwen/qwen3.6-35b-a3b (35B MoE) | ~70 GB | ~35 GB | 1 | 8 |
| Qwen/Qwen3.5-35B-A3B (35B MoE) | ~70 GB | ~35 GB | 1 | 8 |
| google/gemma-4-31b-it (31B dense) | ~62 GB | ~31 GB | 1 | 8 |
| z-ai/glm-4.7-flash (31.2B MoE, 3B active) | ~62 GB | ~31 GB | 1 | 8 |
| nvidia/nemotron-3-nano-30b-a3b (31.6B MoE) | ~63 GB | ~32 GB | 1 | 8 |
| qwen/qwen3.6-27b (27.8B dense) | ~56 GB | ~28 GB | 1 | 8 |
| google/gemma-4-26b-a4b-it (26B MoE, 4B active) | ~52 GB | ~26 GB | 1 | 8 |

---

### vLLM Serve Commands

#### minimax/minimax-m2.7

Requires TP4 + expert parallelism. The compilation config fuses QK-norm for ~15% extra throughput. [[3]](#ref3)

```bash
vllm serve MiniMaxAI/MiniMax-M2.7 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --quantization fp8 \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2 \
  --compilation-config '{"mode":3,"pass_config":{"fuse_minimax_qk_norm":true}}' \
  --enable-auto-tool-choice
```

#### mistralai/mistral-small-2603

Uses MLA attention (same as DeepSeek v3). `FLASH_ATTN_MLA` backend is required for full throughput. [[4]](#ref4)

```bash
vllm serve mistralai/Mistral-Small-4-119B-2603 \
  --tensor-parallel-size 8 \
  --attention-backend FLASH_ATTN_MLA \
  --tool-call-parser mistral \
  --enable-auto-tool-choice
```

#### Qwen/Qwen3.5-122B-A10B

```bash
vllm serve Qwen/Qwen3.5-122B-A10B \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --quantization fp8 \
  --tool-call-parser hermes \
  --enable-auto-tool-choice
```

#### qwen/qwen3.6-27b (single GPU)

```bash
vllm serve Qwen/Qwen3.6-27B \
  --quantization fp8 \
  --tool-call-parser hermes \
  --enable-auto-tool-choice
```

---

### Throughput Estimates (H100, FP8)

Benchmarks measured on vLLM with H100 SXM cards, output-token throughput at typical batch sizes. [[5]](#ref5)[[6]](#ref6)

| Model | Serving config | Est. tok/s per replica |
|-------|---------------|:----------------------:|
| qwen/qwen3.6-27b | 1×H100, FP8 | ~2 200–2 500 |
| google/gemma-4-31b-it | 1×H100, FP8 | ~2 000–2 300 |
| z-ai/glm-4.7-flash | 1×H100, FP8 | ~3 000+ (3B active) |
| Qwen/Qwen3.5-35B-A3B | 1×H100, FP8 | ~2 800+ (3B active) |
| mistralai/mistral-small-2603 | 2×H100, FP8 | ~2 000–2 200 |
| Qwen/Qwen3.5-122B-A10B | 2×H100, FP8 | ~1 800–2 100 |
| minimax/minimax-m2.7 | 4×H100, FP8 | ~1 800–2 000 |
| Qwen/Qwen3.5-397B-A17B | 8×H100, FP8 | ~1 100–1 400 |

---

### Recommendations

| Goal | Pick | Why |
|------|------|-----|
| **Max throughput** | `qwen/qwen3.6-27b` | 8 replicas × 1 GPU; ~18 000 tok/s node-wide; strong Tool F1 across both splits |
| **Max balance (FP8 fits 4× per node)** | `mistralai/mistral-small-2603` | Best Tool F1 on `test-extra` (0.831); MLA attention; TP2 → 4 replicas |
| **Strong refusal + general competence** | `Qwen/Qwen3.5-122B-A10B` | Zero hallucination, balanced refusal (0.477) + Tool F1 (0.809) |
| **Ultra-low latency** | `z-ai/glm-4.7-flash` | Only 3B active params; fastest TTFT despite 31B total; Tool F1 = 0.767 on test-extra |
| **Highest ID propagation** | `Qwen/Qwen3.5-397B-A17B` | Best at carrying IDs across turns (0.523), but needs full 8-GPU node |

---

### References

<a id="ref1"></a>[1] NVIDIA H100 Tensor Core GPU Architecture Whitepaper — FP8 training and inference support: https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet

<a id="ref2"></a>[2] vLLM FP8 quantization docs — W8A8 FP8 throughput vs BF16: https://docs.vllm.ai/en/latest/quantization/fp8.html

<a id="ref3"></a>[3] MiniMax-M2.7 vLLM deployment guide (official HuggingFace README): https://huggingface.co/MiniMaxAI/MiniMax-M2.7

<a id="ref4"></a>[4] Mistral-Small-4-119B-2603 vLLM guide (MLA attention backend): https://huggingface.co/mistralai/Mistral-Small-4-119B-2603

<a id="ref5"></a>[5] vLLM throughput benchmark — Qwen3-32B on H100 (2352 tok/s BF16 baseline): https://github.com/vllm-project/vllm/discussions/13840

<a id="ref6"></a>[6] FP8 inference speedup benchmarks (2.2–2.3× over BF16): https://neuralmagic.com/blog/vllm-fp8/
