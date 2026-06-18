# Synthetic Conversation Generation — Spec for Sub-Agents

You are generating Malaysian-style synthetic call-centre conversations for function-calling training data.

## Working paths

- Train parquet: `/home/husein/ssd3/SyntheticGen/synthetic/train-00000-of-00001.parquet`
- Test parquet: `/home/husein/ssd3/SyntheticGen/synthetic/test-00000-of-00001.parquet`
- Reference example (canonical style + format): `/home/husein/ssd3/SyntheticGen/synthetic/dry_run3.json`
- Output dir for train: `/home/husein/ssd3/SyntheticGen/synthetic/train/{row_index}.json`
- Output dir for test: `/home/husein/ssd3/SyntheticGen/synthetic/test/{row_index}.json`

`{row_index}` is the integer index of the row in the parquet (0-based).

## Checkpointing / resume

Before generating row `i`, check if the output file already exists. If yes, SKIP — do not overwrite.

```python
import os
target = f'/home/husein/ssd3/SyntheticGen/synthetic/train/{i}.json'
if os.path.exists(target):
    continue
```

## Output format (OpenAI-compatible)

Top-level JSON object:

```json
{
  "conversation_id": "<unique e.g. myl-train-{i}>",
  "workflow_name": "<row['workflow']>",
  "domain": "<row['domain']>",
  "messages": [ ... ],
  "metadata": {
    "num_turns": <int>,
    "language_style": "<short note>",
    "generated_at": "<ISO datetime>",
    "turn_details": [ { "turn": 1, "intent": "...", "expected_functions": [...], "complexity": "simple|moderate|complex" }, ... ]
  }
}
```

Messages array uses these roles:

- `{"role": "user", "content": "..."}`
- `{"role": "assistant", "content": "...", "tool_calls": [ {"id": "<call-id>", "type": "function", "function": {"name": "...", "arguments": "<JSON-stringified args>"}} ]}` — `content` may be a brief lead-in string (`"Mohon tunggu sebentar."`) before the tool call. The `arguments` MUST be a JSON-stringified object.
- `{"role": "tool", "tool_call_id": "<call-id>", "name": "<function name>", "content": "<JSON-stringified plausible response>"}`

Every `tool_call_id` in a tool message must match a prior assistant `tool_calls[].id`. Parallel tool calls are allowed (one assistant message with multiple `tool_calls` followed by multiple tool messages).

## Conversation structure

- 5 to 8 logical turns. A "turn" = one user message + one assistant message (which may include one or more tool calls + tool responses + a final assistant summary).
- Use **3-7 distinct functions** drawn from the row's `functions` JSON. Choose functions that flow naturally for the named workflow (e.g., for `usage_to_payment` you might use `get_subscriber_context`, `apply_prepaid_topup`, `list_available_bundles`, `purchase_bundle`, `list_payment_transactions`).
- Tool call `arguments` MUST validate against the function's `parameters` schema (right field names, right enums, all required fields present).
- Tool response `content` MUST be plausible: realistic IDs, timestamps in 2026, MYR currency for telco/billing, fields that align with `returns` description.
- Numeric and date consistency: balance arithmetic must add up; timestamps must move forward; IDs referenced across turns must be the same.

## Personas & language

**Customer (user role):**
- Malaysian. Code-switches between Bahasa Malaysia, Manglish, sometimes English. Occasional polite Tamil greeting (`Vanakkam`) or expression (`Aiyo`, `Alamak`) is fine.
- POLITE customer to a hotline. NOT chummy with the agent.
- **FORBIDDEN tokens from user**: `bro`, `boss`, `machi`, `machan`, `padu`, `syiok` — these are too casual for a call-centre call.
- Phrasing typical of a Malaysian calling a hotline: "Hi, saya nak check…", "Boleh tolong…", "Sat tengok…", "One more thing sebelum saya tutup", "Vanakkam, please help…".
- The customer's spoken language may shift mid-conversation (Malay → English → Manglish) — natural code-switching.

**Agent (assistant role):**
- Formal call-centre agent. Polite, never casual.
- Vocative: **`Encik`** (male customer), **`Puan`** (female), **`Sir`** / **`Madam`** (when speaking English). Choose based on customer cues (e.g., "abang saya" → male customer → "Encik"; "kakak saya" → likely female).
- **FORBIDDEN tokens from agent**: `Tuan` (avoid — use Encik/Puan/Sir), `bro`, `boss`, `machi`, `machan`, `padu`, `syiok`, `bro`, `cuz`.
- Mirrors customer's primary language: if user opens in Bahasa, agent opens in Bahasa ("Selamat pagi/petang/malam Encik, terima kasih kerana menghubungi…"). If user switches to English, agent follows ("Certainly, Sir. I will…"). When user mixes Manglish, agent stays formal Bahasa.
- Standard agent phrases: "Mohon tunggu sebentar", "Terima kasih atas kesabaran", "Sila ambil maklum", "Adakah ada apa-apa lagi yang saya boleh bantu?"; English equivalent: "Please hold on for a moment", "Thank you for your patience", "Is there anything else I can help you with?"

## Malaysian context to use

- Currency: MYR (RM). Phone numbers: `+60xxxxxxxxx` (e.g. `60123456789`).
- Payment providers: Touch n Go eWallet, GrabPay, Boost, Maybank, CIMB, RHB, Bank Islam, BSN, 7-Eleven voucher, KK Mart voucher.
- Locations: KL, Selangor, Penang, Johor, Sabah, Sarawak, Cyberjaya, Putrajaya, Shah Alam, Subang Jaya, Klang, JB.
- Local names: Ahmad, Aishah, Faridah, Lim, Tan, Wong, Kumar, Ravi, Priya, Mohd Faiz, Siti Nur, Nurul.
- Local content: Astro, Maxis/Celcom/Digi/U Mobile (generic), TM Unifi, Mobile Legends, PUBG Mobile, TikTok, Spotify, Netflix, YouTube. Use these as **examples** unless the domain doesn't fit telco.

## Domain adaptation

The dataset spans many domains (`telco_billing`, `taas_cpaas`, `telco_b2b`, `taas_camara/device_swap`, `telco_5g_deep`, `telco_regulatory`, etc.). For each row:

1. **Read the `functions` JSON for that row**. Look at `workflow_name`, `description`, each function's name/description/parameters/returns.
2. **Pick a believable user role** for that workflow: end customer for billing/charging; sales ops / B2B account manager for enterprise sales; NOC engineer for assurance/alarm; compliance officer for lawful intercept; partner admin for API monetisation, etc. Adjust persona accordingly — **internal users (NOC, ops, compliance) are still Malaysian and still code-switch**, just with more technical jargon.
3. For non-end-customer personas, the agent may be an internal helpdesk / NOC tooling assistant rather than a retail call centre — but **language register stays formal and uses Encik/Puan/Sir**.

## Hard validation before writing

Run these checks mentally / programmatically before `Write`:

1. JSON parses (top-level object).
2. Every assistant `tool_calls[].function.arguments` is a JSON-stringified object that parses.
3. Every tool `content` parses as JSON.
4. Every tool message's `tool_call_id` matches a prior assistant tool call id.
5. No forbidden tokens (case-insensitive `\bbro\b`, `\bboss\b`, `\bmachi\b`, `\bmachan\b`, `\bpadu\b`, `\bsyiok\b`, `\bTuan\b`).
6. At least one assistant message uses `Encik` OR `Puan` OR `Sir` OR `Madam`.
7. Tool arguments respect the function's `parameters` schema (required fields present, enums valid).
8. Internal consistency: IDs referenced in turn N+1 are produced by tool calls in turn N or earlier.

If any check fails, fix and re-validate before writing the file.

## Per-row workflow (sub-agent loop)

```python
import pandas as pd, json, os, datetime
df = pd.read_parquet('/home/husein/ssd3/SyntheticGen/synthetic/train-00000-of-00001.parquet')
for i in YOUR_ASSIGNED_INDICES:
    target = f'/home/husein/ssd3/SyntheticGen/synthetic/train/{i}.json'
    if os.path.exists(target):
        continue
    row = df.iloc[i]
    spec = json.loads(row['functions'])
    # design a 5-8 turn conversation using 3-7 of spec['functions'], following rules above
    # build the JSON object, validate, then Write
```

## Diversity

Across the dataset, vary:
- Whether the user opens in Malay vs English vs Manglish
- Customer gender (Encik vs Puan)
- Time of day (pagi / petang / malam → match greeting)
- Whether code-switching happens mid-conversation
- Whether there's a happy-path or a problem-detected branch (suspicious charge, failed attempt, escalation)
- Number of tool calls per turn (single vs parallel)

## Reporting

After processing your slice, report:
- Indices completed (new files written)
- Indices skipped (files already existed)
- Any failures (with reason)
- Sample paths for spot-check

DO NOT over-explain. Just generate, validate, write, report.
