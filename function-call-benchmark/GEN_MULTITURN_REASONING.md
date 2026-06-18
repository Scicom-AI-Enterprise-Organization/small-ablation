# `gen_multiturn_reasoning.py` — multi-turn tool-calling generator with reasoning

Generates Malaysian-style, multi-turn **function-calling** conversations through any
**OpenAI-compatible** endpoint, building the dialogue **turn-by-turn** and storing the
**reasoning behind every step**.

It role-plays three personas that share one conversation:

| Role | How it's called | Reasoning captured from |
|---|---|---|
| **User simulator** | plain chat (JSON out) | explicit `reasoning` field it returns |
| **Assistant** | real `tools=[...]`, `tool_choice="auto"` | native `reasoning_content`/`reasoning`; if empty, a one-off *planner* call reconstructs it |
| **Tool simulator** | plain chat (JSON out) | explicit `reasoning` field it returns |

The loop is: **user₁ → assistant₁ (auto tools) → synthetic tool results → user₂ → …** for `--turns` turns.

---

## 1. Setup

```bash
cd /home/husein/ssd3/SyntheticGen/synthetic
pip install openai pandas pyarrow python-dotenv   # already present in this env
```

The script reads `.env` from this directory (or its parent), same as
`evaluate_function_calling.py`. Set:

```ini
OPENAI_API_KEY=sk-...           # required
OPENAI_BASE_URL=https://openrouter.ai/api/v1   # optional; this is the default
OPENAI_MODEL=z-ai/glm-5.1       # optional default model
```

> The `z-ai/…`, `qwen/…`, `Qwen/…` ids in `model_list` are OpenRouter-style, so the
> base URL defaults to OpenRouter. Point `--base-url` at any other OpenAI-compatible
> server (vLLM, SGLang, a local gateway, …) as needed.

---

## 2. Quick start

**Dry-run one parquet row** (what we validated first):

```bash
python3 gen_multiturn_reasoning.py \
  --parquet test-function-merged-00000-of-00001.parquet --row 0 \
  --turns 6 --model z-ai/glm-5.1 \
  --out out/row0.json
```

**Single conversation from a function-library JSON file:**

```bash
python3 gen_multiturn_reasoning.py \
  --tools test-function/0.json \
  --turns 8 --language ms --model z-ai/glm-5.1 \
  --out out/conv0.json
```

**Batch over an entire parquet** → per-row JSON (resumable) + a merged parquet,
**10 rows generated concurrently**:

```bash
python3 gen_multiturn_reasoning.py \
  --parquet test-function-merged-00000-of-00001.parquet \
  --turns 12 --model z-ai/glm-5.1 \
  --concurrency 10 \
  --out-dir out/multiturn-reasoning \
  --out-parquet test-function-multiturn-reasoning-00000-of-00001.parquet \
  --skip-existing
```

`--skip-existing` reuses any `out-dir/{i}.json` already written, so an interrupted
batch resumes where it left off.

### Parallelism (`--concurrency` / `--parallel` / `-j`)

Batch mode generates **N rows (whole conversations) at the same time** via a thread
pool — `--concurrency 10` runs 10 conversations in flight at once. This is the big
wall-clock win: a sequential run only ever has one request in flight, so a busy
endpoint sits mostly idle from its point of view.

- Parallelism is **row-level only**. The turns *within* one conversation stay strictly
  sequential (each turn depends on the previous turn's tool results), so they can't be
  parallelised — but independent rows can.
- Results are still written in **row order** in the merged parquet, regardless of which
  row finishes first. Per-row JSON is written as each row completes.
- A single failed row is logged and skipped — it doesn't abort the batch.
- With `--seed`, each row uses an independent, reproducible RNG (`seed + row_index`), so
  sentiment/error patterns are identical no matter the concurrency level.
- Size `--concurrency` to your endpoint's capacity (and any rate limits). For a local
  vLLM/SGLang server, match it to the server's batch capacity; for a hosted gateway,
  stay under its concurrency/RPM cap.

---

## 3. Inputs

Provide **one** of:

- `--tools PATH` — a function-library JSON. Either a full object
  (`{workflow, domain, shared_entities, functions:[…]}`) or a **bare array** of
  function definitions. `#/shared_entities/<name>` `$ref`s are resolved automatically.
- `--parquet PATH` — a parquet whose **`functions`** column holds the library (the
  format of `test-function-merged-00000-of-00001.parquet`). Add `--row I` for a
  single conversation, or omit `--row` for batch mode.

Each function definition follows the dataset shape:
`{ "name", "description", "stage", "parameters" (JSON Schema), "returns" }`.

---

## 4. Output

A conversation object whose `messages` carry an **inline `reasoning`** key on every
message:

```json
{
  "domain": "telco_b2b/enterprise_sales",
  "workflow": "reconciliation_flow",
  "shared_entities": "<JSON string>",
  "functions": "<JSON string>",
  "messages": [
    { "role": "user", "content": "Selamat pagi…", "reasoning": "Open with a concrete reconciliation request…" },
    { "role": "assistant", "content": "Baik Puan…", "reasoning": "Need to create the run first…",
      "tool_calls": [ { "id": "call_…", "type": "function",
                        "function": { "name": "initiate_reconciliation_run", "arguments": "{…}" } } ] },
    { "role": "tool", "tool_call_id": "call_…", "name": "initiate_reconciliation_run",
      "content": "{…}", "reasoning": "Return a new run id in pending state…" }
  ],
  "metadata": {
    "num_turns": 6,
    "functions_used": ["initiate_reconciliation_run", "..."],
    "num_functions_used": 7,
    "language": "ms",
    "model": "z-ai/glm-5.1",
    "user_sentiment": "impatient",
    "sentiment_per_turn": ["impatient", "impatient", "..."],
    "simulate_errors": true,
    "error_rate": 0.25,
    "api_errors_simulated": [
      {"function": "propose_reconciliation_actions", "http_status": 409, "code": "CONCURRENT_MODIFICATION_CONFLICT"}
    ],
    "reasoning_source": ["native", "planner", "..."],
    "generated_at": "2026-06-15T…Z",
    "validation_warnings": []
  }
}
```

- **Single mode** → one JSON file at `--out`.
- **Batch mode** → one JSON per row in `--out-dir`, plus a merged parquet at
  `--out-parquet` with the exact 6 columns of the existing dataset
  (`domain, workflow, shared_entities, functions, messages, metadata`), with
  `messages`/`metadata` JSON-stringified — ready for `push_to_hub.py`.

> The `messages` here include the extra `reasoning` key by design. If you need a
> pure-OpenAI `messages` array, strip the `reasoning` keys downstream.

Push to the Hub when happy:

```bash
python3 push_to_hub.py \
  -f test-function-multiturn-reasoning-00000-of-00001.parquet \
  -d Scicom-intl/Function-Call-TaaS \
  -s function-multiturn-reasoning --split test
```

---

## 5. Reasoning capture

- **User / tool** turns are asked to return `{"reasoning": …, "message"/"response": …}`,
  so reasoning is always present and separate from the payload.
- **Assistant** turns use real auto tool-calling. The script first reads the model's
  native chain-of-thought (`message.reasoning_content` / `message.reasoning` —
  emitted by most "thinking" models, e.g. Qwen3.5, Nemotron, GLM, MiniMax). If that
  channel is empty (e.g. Gemma, Mistral-Small), it makes **one extra planner call**
  that articulates the reasoning behind the tool calls it actually made — so the
  reasoning always matches the action. `metadata.reasoning_source` records
  `native` / `planner` / `none` per assistant message.
- `--no-native-reasoning` forces the planner path for every assistant turn.

---

## 6. User sentiment & API-error simulation

**User sentiment** — `--sentiment` sets the customer's emotional tone, injected into the
user simulator each turn:

| Value | Behaviour |
|---|---|
| `neutral` (default) | matter-of-fact, businesslike |
| `polite` | warm, courteous, appreciative |
| `frustrated` | politely annoyed; the problem has dragged on |
| `angry` | upset and demanding, terse (no profanity, stays in decorum) |
| `urgent` | time-pressured, hard deadline, pushes for speed |
| `impatient` | wants quick short answers, hurries the agent |
| `confused` | unsure, asks for clarification, mixes up details |
| `satisfied` | pleased, cooperative, complimentary |
| `anxious` | worried about cost/compliance/deadline, seeks reassurance |
| `mixed` | re-rolls a random sentiment **every turn** |
| `random` | picks **one** sentiment per conversation (great for batch variety) |

The chosen tone is recorded in `metadata.user_sentiment` and `metadata.sentiment_per_turn`.
The assistant is instructed to stay calm and de-escalate regardless of tone.

```bash
python3 gen_multiturn_reasoning.py --tools test-function/0.json \
  --turns 8 --sentiment frustrated --model z-ai/glm-5.1 --out out/frustrated.json
```

**API-error simulation** — `--simulate-errors` lets the tool simulator return realistic
failures instead of always succeeding; `--error-rate` (default `0.25`) is the per-tool-call
probability. Injected modes span `400/401/403/404/409/413/429/500/503/504`, `202` async,
partial-success, eventually-consistent, deprecation, and quota-near-exhaustion. The
assistant then has to recover (retry with fixed args, back off, escalate, or offer a
workaround). Every injected failure is recorded in `metadata.api_errors_simulated`
(`{function, http_status, code}`), and `--seed` makes the injection pattern reproducible.

```bash
python3 gen_multiturn_reasoning.py \
  --parquet test-function-merged-00000-of-00001.parquet --row 0 \
  --turns 8 --sentiment random --simulate-errors --error-rate 0.3 --seed 7 \
  --model z-ai/glm-5.1 --out out/row0_errors.json
```

> **Note:** error simulation changes the *content* of tool responses (and thus the
> assistant's recovery behaviour); sentiment changes the *user's* phrasing. Neither
> changes the output schema.

---

## 7. Prefix caching (`--disable-prefix-cache`)

Opt-in. When set, every request carries a unique vLLM **`cache_salt`** so no two requests
share a cached prefix — the client-side way to opt out of prefix-cache *reuse* (useful for
fair throughput measurement). It does **not** change generated outputs (sampling still runs
fresh on a cache hit), only prefill speed. True server-side disabling is the launch flag
`--no-enable-prefix-caching` on vLLM. Off by default, so the OpenRouter path is untouched;
only enable it against an endpoint (vLLM) that understands `cache_salt`.

---

## 8. Language

`--language ms` (default, Bahasa Malaysia). Override with `en`, `ta` (Malaysian
Tamil), `zh` (Malaysian Mandarin), or any code. For `ms`, generation enforces a
formal call-centre register (Encik/Puan/Sir/Madam) and flags forbidden tokens
(`Tuan`, `bro`, `boss`, `machi`, `machan`, `padu`, `syiok`) as validation warnings.

---

## 9. All CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--tools PATH` | — | function-library JSON (object or bare array) |
| `--parquet PATH` | — | parquet with a `functions` column |
| `--row I` | — | single row of `--parquet` |
| `--turns N` | `8` | number of user turns |
| `--language CODE` | `ms` | conversation language |
| `--sentiment S` | `neutral` | user tone (see §6); `mixed`/`random` for variety |
| `--simulate-errors` | off | inject realistic API failures into tool responses |
| `--error-rate F` | `0.25` | per-call error probability when `--simulate-errors` set |
| `--disable-prefix-cache` | off | unique `cache_salt` per request (see §7) |
| `--max-tool-rounds N` | `3` | max tool-call rounds per assistant turn |
| `--no-native-reasoning` | off | always reconstruct assistant reasoning via planner |
| `--model M` | `$OPENAI_MODEL` or `z-ai/glm-5.1` | model for all roles |
| `--user-model` / `--assistant-model` / `--tool-model` | = `--model` | per-role model override |
| `--base-url URL` | `$OPENAI_BASE_URL` or OpenRouter | endpoint |
| `--api-key KEY` | `$OPENAI_API_KEY` | key |
| `--max-tokens N` | `4096` | per call |
| `--seed N` | — | sampling seed |
| `--user-temperature` / `--assistant-temperature` / `--tool-temperature` | `0.9` / `0.4` / `0.5` | per-role temperature |
| `--out PATH` | `out/…json` | single-mode output |
| `--out-dir DIR` | — | batch per-row JSON dir |
| `--out-parquet PATH` | — | batch merged parquet |
| `--start I` / `--max-rows N` | `0` / all | batch slice |
| `--concurrency N` (`--parallel`, `-j`) | `1` | rows generated concurrently (turns within a row stay sequential) |
| `--skip-existing` | off | reuse existing per-row JSON (resume) |

---

## 10. Validation (soft, recorded in `metadata.validation_warnings`)

Each generated conversation is checked for: tool-call `arguments` and tool `content`
parsing as JSON · every `tool_call_id` paired to a prior assistant call · tool names
existing in the library · required params present · (for `ms`) forbidden tokens and a
present vocative. Warnings are recorded, not fatal — inspect them after a run; a clean
run prints `validation: clean`.

---

## 11. Troubleshooting

- **`no API key`** — set `OPENAI_API_KEY` in `.env` or pass `--api-key`.
- **`All connection attempts failed` / connection refused** — wrong `--base-url`, or a
  local server bound to `127.0.0.1` instead of `0.0.0.0`.
- **Empty `reasoning` on assistant turns** — the model has no native CoT channel; it
  should fall back to `planner`. If you see `reasoning_source: none`, the planner call
  failed (rate-limit/timeout) — re-run, or use `--no-native-reasoning`.
- **`response_format` errors** — handled automatically: the script retries without
  JSON mode and parses the text robustly.
- **Unknown-function / missing-param warnings** — the model strayed from the schema;
  lower `--assistant-temperature` or reduce `--turns`.
```
