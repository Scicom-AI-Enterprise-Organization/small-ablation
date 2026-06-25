# RED Team LLM — Malaysian Customer Support

A full pipeline for generating, judging, and benchmarking LLM safety in Malaysian customer support contexts. Covers synthetic adversarial dataset generation (single-turn and multi-turn), judge model selection, and target model red-team evaluation.

---

## Project Structure

```
red-team-llm/
├── synthetic_generator/
│   ├── schema.py                 # Field enumerations shared across generators
│   ├── prompts.py                # Single-turn prompt templates
│   ├── prompts_multiturn.py      # Multi-turn prompt templates
│   ├── generator.py              # DatasetGenerator (single-turn, async)
│   └── generator_multiturn.py   # MultiTurnGenerator (multi-turn, async)
├── outputs/                      # Generated datasets saved here
├── benchmark_results/            # Red-team benchmark output
├── judge_results/                # Judge benchmark output
├── generate.py                   # Single-turn dataset generation CLI
├── generate_multiturn.py         # Multi-turn dataset generation CLI
├── benchmark_redteam.py          # Red-team benchmark (target model evaluation)
├── benchmark_judge.py            # Judge model accuracy benchmarking
└── requirements.txt
```

## Setup

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
export SGPU_API_KEY="your_api_key_here" # or pass in --api-key arguments in script command
source .venv/bin/activate
```

---

## Benchmark Dataset Generation for Red Team Evaluation

Generates synthetic adversarial datasets for evaluating LLM safety. Two formats are supported: **single-turn** (one adversarial message per entry) and **multi-turn** (full conversation where the attack unfolds progressively over 4–10 turns).

Both scripts run async with configurable concurrency, write results per batch, and use a randomly chosen language per batch.

### Single-Turn Dataset (`generate.py`)

```bash
# Basic — 20 entries, 1 batch
python generate.py --api-key $SGPU_API_KEY

# 100 entries across 5 concurrent batches
python generate.py --count 20 --batches 5 --concurrency 5 --api-key $SGPU_API_KEY

# Custom model and output path
python generate.py --model Qwen/Qwen3.6-35B-A3B --output outputs/my_dataset.jsonl --api-key $SGPU_API_KEY
```

Output: `outputs/dataset_<timestamp>.jsonl`

**Options**

```
--count        Entries per batch (default: 20)
--batches      Number of batches (default: 1)
--concurrency  Max concurrent API requests (default: 3)
--model        Model ID (default: Qwen/Qwen3.5-122B-A10B)
--api-url      API base URL
--api-key      API key (or set SGPU_API_KEY)
--output       Output file path
--retries      Retries on parse failure (default: 2)
```

---

### Multi-Turn Dataset (`generate_multiturn.py`)

Each entry is a full conversation (4–10 turns) where the attack strategy unfolds progressively. Assistant turns simulate what a vulnerable LLM would say, showing the end-to-end attack trajectory.

```bash
# Basic — 10 entries, 1 batch
python generate_multiturn.py --api-key $SGPU_API_KEY

# 40 entries across 4 concurrent batches
python generate_multiturn.py --count 10 --batches 4 --concurrency 4 --api-key $SGPU_API_KEY
```

Output: `outputs/multiturn_<timestamp>.jsonl`

**Options**

```
--count        Entries per batch (default: 10)
--batches      Number of batches (default: 1)
--concurrency  Max concurrent API requests (default: 3)
--model        Model ID (default: Qwen/Qwen3.5-122B-A10B)
--api-url      API base URL
--api-key      API key (or set SGPU_API_KEY)
--output       Output file path
--retries      Retries on parse failure (default: 2)
```

---

## Dataset Cleaning (`consolidate.ipynb`)

After generation, raw outputs are cleaned and deduplicated in `consolidate.ipynb` before being pushed to HuggingFace Hub.

### Single-Turn

1. **Load** all `outputs/dataset_*.jsonl` into a DataFrame
2. **Normalise fields** — fix typo variants (e.g. `"telecom"` → `"telco"`); fill missing `model` with default model name
3. **Drop rogue columns** — LLM occasionally generates hallucinated field names (e.g. `system_promt`, `tension`, `target_safe_response`); these are dropped
4. **Drop nulls** — remove rows with missing `system_prompt`
5. **Drop sparse attack types** — remove any `attack_type` with fewer than 10 samples
6. **Embed** each `input` using `Qwen/Qwen3-Embedding-8B` (async, via `https://tm-vm2-embedding-1.cae.scicom.dev/v1/embeddings`)
7. **Deduplicate** — compute cosine similarity matrix; remove entries where any pair exceeds 0.9 similarity (keep the first, drop the rest)
8. **Save** as HuggingFace Dataset (locally) 

### Multi-Turn

Steps 1–5 are the same with these additions/differences:

- **Drop nulls** on `system_prompt`, `conversation`, and `expected_safe_response`
- **Drop rogue columns** — more numerous due to LLM hallucinating conversation field variants (e.g. `conversion`, `converse`, `conversus`, `attack_strataegy`)
- **Validate conversation structure** — use Pydantic to verify each turn has `role` and `content`; drop invalid entries
- **Trim trailing assistant turns** — if the last turn in `conversation` is an `assistant` turn, remove it (the benchmark expects the conversation to end on the attack user turn)
- **Deduplicate** using embeddings of the full conversation text (threshold: 0.9)
- **Drop sparse attack types** with fewer than 20 samples
- **Recalculate `turn_count`** after trimming
- **Save** as HuggingFace Dataset (locally)

---

## Judge Benchmark (`benchmark_judge.py`)

Evaluates how accurately a model acts as a **safety judge**. Given a golden dataset of paired safe/unsafe responses, the judge picks the safer one and accuracy is measured against ground truth. A/B order is randomised per example to eliminate position bias. Results are written incrementally and runs can be resumed. The golden dataset is generated using [script](generate_safe_unsafe_response.py), [notebook](./llm-as-judge.ipynb)

### Golden dataset format

| Field | Description |
|---|---|
| `example_id` | Unique identifier |
| `attack_type` | Attack category |
| `user_input` | The adversarial message |
| `safe_response` | Ground-truth safe/aligned response |
| `unsafe_response` | Ground-truth unsafe/failing response |
| `difficulty` | Optional difficulty label |

### Usage

```bash
python benchmark_judge.py \
  --dataset_path golden_dataset_1.jsonl \
  --api_url https://serverlessgpu.aies.scicom.dev/v1/chat/completions \
  --api_key $SGPU_API_KEY \
  --model google/gemma-4-31b-it \
  --concurrency 10
```

Results saved to `judge_results/<model_name>.jsonl`. Re-running resumes automatically.

**Options**

```
--dataset_path    Path to golden dataset JSONL
--output_folder   Output directory (default: judge_results)
--api_url         API endpoint URL
--api_key         API key (or set SGPU_API_KEY)
--model           Judge model ID to evaluate
--concurrency     Max concurrent requests (default: 5)
```

### Results

Evaluated on `golden_dataset_1.jsonl` (240 examples):

| Model | Correct | Total | Accuracy |
|---|---|---|---|
| **google/gemma-4-31b-it** | **229** | 240 | **95.4%** |
| Qwen/Qwen3.6-35B-A3B | 227 | 240 | 94.6% |
| Qwen/Qwen3.5-122B-A10B | 226 | 240 | 94.2% |
| MiniMaxAI/MiniMax-M2.7 | 226 | 240 | 94.2% |
| mistralai/Mistral-Small-4-119B-2603 | 222 | 240 | 92.5% |
| qwen/qwen3.6-27b | 220 | 240 | 91.7% |

**Recommendation: use `google/gemma-4-31b-it` as the judge model** — highest accuracy at 95.4%, used as default in `benchmark_redteam.py`.

---

## Red Team Benchmark (`benchmark_redteam.py`)

Evaluates how well a **target model** withstands adversarial attacks. For each dataset entry:

1. Sends the adversarial input to the target model (with `system_prompt` and prior turns for multi-turn)
2. Passes the response to the judge model with full attack context (`attack_description`, `target_behavior`, `expected_safe_response`, `failure_mode`)
3. Records `safe` / `unsafe` verdict, reasoning, and confidence
4. Writes each result immediately — safe to interrupt and resume at any point

Both single-turn and multi-turn datasets are supported and auto-detected per entry.

### Usage

```bash
# Single-turn dataset, default judge (google/gemma-4-31b-it)
python benchmark_redteam.py \
  --dataset_path outputs/dataset_20260601_120000.jsonl \
  --target_model qwen/qwen3.6-27b \
  --target_api_url https://serverlessgpu.aies.scicom.dev/tm-fleet/v1/chat/completions \
  --api_key $SGPU_API_KEY \
  --concurrency 10
```

Results saved to `benchmark_results/<target_model>.jsonl`.

**Options**

```
--dataset_path     Path to red-team dataset JSONL (single-turn or multi-turn)
--target_model     Model ID to benchmark (required)
--judge_model      Judge model ID (default: google/gemma-4-31b-it)
--target_api_url   API URL for the target model (required)
--judge_api_url    API URL for the judge model (defaults to --target_api_url)
--api_key          API key (or set SGPU_API_KEY)
--concurrency      Max concurrent requests (default: 5)
--output_folder    Output folder (default: benchmark_results)
--retries          Retries on transient errors (default: 3)
```

**Output schema**

| Field | Description |
|---|---|
| `idx` | Entry index in the original dataset |
| `dataset_type` | `single_turn` or `multi_turn` |
| `industry`, `attack_type`, `attack_strategy`, `language`, `severity`, `tone` | From dataset entry |
| `input` | The adversarial message sent to the target model |
| `target_model` | Model being benchmarked |
| `judge_model` | Model used as judge |
| `target_response` | Raw response from the target model |
| `verdict` | `safe`, `unsafe`, or `error` |
| `is_safe` | `true` if verdict is `safe` |
| `reasoning` | Judge's one-sentence explanation |
| `confidence` | `low`, `medium`, or `high` |
| `attack_detected_by_model` | Whether the target model explicitly identified the attack |

**Summary output example**

```
════════════════════════════════════════════════════════════════
  Red-Team Benchmark Results
  Target : qwen/qwen3.6-27b
  Judge  : google/gemma-4-31b-it
────────────────────────────────────────────────────────────────
  Total evaluated  : 1788
  Safe responses   : 1758 / 1788  (98.3%)
  Unsafe responses :   30 / 1788  ( 1.7%)

  By attack_type:
    escalation_fishing             ████████████████░░  136/147 (93%)
    prompt_injection               █████████████████░  217/226 (96%)
    ...

  By severity:
    low                            █████████████████░  185/192 (96%)
    critical                       █████████████████░  430/432 (100%)
    ...
════════════════════════════════════════════════════════════════
```

### Results

All 6 models benchmarked on the same datasets, judged by `google/gemma-4-31b-it`. Refer to the [output](./benchmark_output.txt) for detailed breakdown.

**Single-Turn** (1,788 entries)

| Model | Safe | Unsafe | Safe Rate |
|---|---|---|---|
| **MiniMaxAI/MiniMax-M2.7** | 1754/1774 | 20 | **98.9%** |
| Qwen/Qwen3.5-122B-A10B | 1763/1784 | 21 | 98.8% |
| google/gemma-4-31b-it | 1759/1788 | 29 | 98.4% |
| qwen/qwen3.6-27b | 1758/1788 | 30 | 98.3% |
| Qwen/Qwen3.6-35B-A3B | 1756/1788 | 32 | 98.2% |
| Qwen/Qwen3.5-397B-A17B | 1753/1785 | 32 | 98.2% |
| mistralai/Mistral-Small-4-119B-2603 | 1629/1788 | 159 | 91.1% |
| zai-org/GLM-4.7-Flash | 1607/1777 | 170 | 90.4% |

Weakest attack types: `escalation_fishing` (73–93%), `prompt_injection` (81–98%).

**Multi-Turn** (2,054 entries)

| Model | Safe | Unsafe | Safe Rate |
|---|---|---|---|
| **MiniMaxAI/MiniMax-M2.7** | 2021/2045 | 24 | **98.8%** |
| Qwen/Qwen3.6-35B-A3B | 1994/2039 | 45 | 97.8% |
| qwen/qwen3.6-27b | 1987/2040 | 53 | 97.4% |
| Qwen/Qwen3.5-122B-A10B | 1963/2023 | 60 | 97.0% |
| Qwen/Qwen3.5-397B-A17B | 1971/2035 | 64 | 96.9% |
| google/gemma-4-31b-it | 1933/2045 | 112 | 94.5% |
| zai-org/GLM-4.7-Flash | 1781/2039 | 258 | 87.3% |
| mistralai/Mistral-Small-4-119B-2603 | 1734/2025 | 291 | 85.6% |

Multi-turn attacks are harder — all models score lower than single-turn. Weakest industry: `food_delivery` and `ecommerce`. Weakest language: `tamil`.

**Key finding:** `MiniMaxAI/MiniMax-M2.7` is the safest model on both modes. `zai-org/GLM-4.7-Flash` and `mistralai/Mistral-Small-4-119B-2603` show the largest gaps, with GLM-4.7-Flash failing 9.6% of single-turn and 12.7% of multi-turn attacks — most vulnerable on `escalation_fishing` (73%) and `roleplay_manipulation` (58% multi-turn).
