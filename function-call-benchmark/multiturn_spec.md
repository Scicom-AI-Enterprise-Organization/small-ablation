# Multi-turn Conversation Generation Spec (Test-Function Library)

## Goal

For each row index `i` in [0..99], take the **expanded function library** at `/home/husein/ssd3/SyntheticGen/synthetic/test-function/{i}.json` and produce a **10-20 turn** Malaysian-style call-centre conversation that uses many of those functions naturally.

Save to `/home/husein/ssd3/SyntheticGen/synthetic/test-function-multiturn/{i}.json`.

Skip if file already exists.

## Inputs

- Expanded function libs: `/home/husein/ssd3/SyntheticGen/synthetic/test-function/{i}.json`
  - Schema: `{workflow_name, description, domain, shared_entities, functions: [ {name, description, stage, parameters, returns}, ... ]}`
  - Each row has 35-44 functions (vs the 12 originals).
- Canonical style reference: `/home/husein/ssd3/SyntheticGen/synthetic/dry_run3.json`
- Base style/format spec (read both): `/home/husein/ssd3/SyntheticGen/synthetic/gen_spec.md` — most rules carry over.

## What changes vs gen_spec.md

1. **Turn count: 10 to 20** (not 5-8). A "turn" = one user message + one assistant message (which may have tool calls + tool responses). Strive for natural-feeling progression, not padded turns.
2. **Function diversity: use 8-15 distinct functions** from the expanded library (not just 3-7). Mix lifecycle stages (init/extract/detect/analyze/plan/execute/monitor/recovery/close, plus bulk/audit/schedule/simulate/approval/policy/etc.).
3. **Parallel tool calls** are encouraged — multiple tool_calls in one assistant message when actions are independent.
4. **Multi-phase scenarios**: don't just do a single happy-path linearly. Include realistic sub-flows: a problem appears mid-conversation, an approval is required, a simulation/dry-run before execution, an audit query for context, a bulk operation, etc.
5. **Conversation narrative arcs** to consider (pick one or weave several):
   - Discover → investigate → diagnose → mitigate → verify → audit → close
   - Plan → simulate → approve → execute → monitor → rollback (if needed) → final report
   - Notice an anomaly → bulk-search → drill down → schedule remediation → export evidence
   - Listing/filtering → comparing → snapshotting → policy update → dependency check → execution

## Style (unchanged from gen_spec.md)

- Formal call-centre agent. Uses **Encik / Puan / Sir / Madam**. Never `Tuan`. Never `bro / boss / machi / machan / padu / syiok`.
- Polite Malaysian user, code-switches between Bahasa Malaysia, English, Manglish; occasional Tamil greeting (`Vanakkam`) ok.
- Malaysian context (MYR, +60 phone numbers, Touch n Go, Maybank, Cyberjaya, KL, Penang, Subang Jaya, etc.) where the domain fits.
- Pick persona by domain: end customer for billing/charging, NOC engineer / SRE / platform engineer for assurance/api/edge/netops, compliance officer for regulatory, partner admin for monetisation, etc.

## Output format (OpenAI-compatible)

```json
{
  "conversation_id": "myl-fnlib-{i}",
  "workflow_name": "<from library>",
  "domain": "<from library>",
  "messages": [ ... ],
  "metadata": {
    "num_turns": <int 10-20>,
    "functions_used": ["fn_a", "fn_b", ...],
    "language_style": "<short note>",
    "generated_at": "<ISO datetime>",
    "turn_details": [ {"turn":1, "intent":"...", "expected_functions":[...], "complexity":"simple|moderate|complex"}, ... ]
  }
}
```

Messages use:
- `{"role":"user","content":"..."}`
- `{"role":"assistant","content":"...", "tool_calls":[{"id":"call-...","type":"function","function":{"name":"...","arguments":"<JSON-stringified>"}}]}` — `content` may be a brief lead-in before the tool call.
- `{"role":"tool","tool_call_id":"call-...","name":"...","content":"<JSON-stringified plausible response>"}`

Every `tool_call_id` must match a prior assistant tool_call id. Parallel tool calls allowed.

## Validation before writing

1. JSON parses.
2. All `tool_calls[].function.arguments` parse as JSON.
3. All tool `content` parse as JSON.
4. Every `tool_call_id` matches a prior assistant tool call.
5. **Turn count (user-message count) between 10 and 20.**
6. **Number of distinct function names used: at least 8.**
7. No forbidden tokens (`Tuan`, `bro`, `boss`, `machi`, `machan`, `padu`, `syiok`).
8. At least one assistant message uses `Encik` / `Puan` / `Sir` / `Madam`.
9. Tool call `function.name` must exist in the library's `functions` array.
10. Tool call arguments respect the function's `parameters` schema (required fields, enums, correct nesting).
11. ID continuity: IDs returned by earlier tool responses are reused (not re-invented) in later tool calls.

## Per-row workflow

```python
import json, os
lib = json.load(open(f'/home/husein/ssd3/SyntheticGen/synthetic/test-function/{i}.json'))
fns = {f['name']: f for f in lib['functions']}
target = f'/home/husein/ssd3/SyntheticGen/synthetic/test-function-multiturn/{i}.json'
if os.path.exists(target): continue
# pick 8-15 functions covering a believable narrative arc through the workflow
# build 10-20 turn conversation, validate, write
```

## Reporting

- Indices written / skipped / failed
- Per-file: turn count + distinct function count
- One sample path for spot-check
