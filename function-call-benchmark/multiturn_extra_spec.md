# Multi-turn EXTRA Conversation Spec (Test-Function Library + Failure Patterns)

## Goal

For each row index `i` in [0..99], take the expanded function library at `/home/husein/ssd3/SyntheticGen/synthetic/test-function/{i}.json` and produce a **10-20 turn** Malaysian-style call-centre conversation that deliberately exercises:

- **(A)** Realistic API failure modes — HTTP 4xx + 5xx + async + partial-success
- **(B)** User-side messiness — typos, mind changes, urgency, undos
- **(D)** Agent-side behavioural edges — self-correction, verification, proactive warnings, pagination, admitting uncertainty
- **Malaysian-localised Tamil and Mandarin** code-switching alongside Bahasa/English/Manglish

Save to `/home/husein/ssd3/SyntheticGen/synthetic/test-function-multiturn-extra/{i}.json`. Skip if file exists.

This spec builds on `multiturn_spec.md` and `gen_spec.md` — most base rules still apply.

## What's the same as multiturn_spec.md

- 10-20 user turns
- Use 8+ distinct functions from the library
- OpenAI format: user / assistant / tool roles, parallel tool calls allowed, all `arguments` JSON-stringified, all tool `content` JSON-stringified
- All tool_call_ids paired
- All tool function names must exist in the library
- Formal call-centre agent: **Encik / Puan / Sir / Madam** (never `Tuan`). No forbidden tokens (`bro`/`boss`/`machi`/`machan`/`padu`/`syiok`).
- ID continuity across turns
- Reference example: `/home/husein/ssd3/SyntheticGen/synthetic/test-function-multiturn-extra/0.json` (dry run)

## What's NEW

### 1. API failure modes — pick 3-5 per conversation

Inject as `tool` message JSON content shaped like:

```json
{"error": {"http_status": <code>, "code": "<short>", "message": "<human>", ...}}
```

Or as partial-success / async (200 OK with warnings) or 202 Accepted. Mix codes across conversations.

**Required mix per file**: at least one **4xx** and one **5xx OR async**.

| Code | Pattern | Recovery the agent should demonstrate |
|---|---|---|
| 400 | Validation error (typo, bad enum, bad date format) | Agent surfaces the field path + suggests fix; asks user to confirm |
| 401 | Session/token expired or MFA required | Agent prompts re-auth (notes elevation_level), waits |
| 403 | Permission denied / scope insufficient | Agent escalates or routes to supervisor |
| 404 | ID not found, entity deleted, or wrong tenant | Agent disambiguates, suggests `search_*` or `list_*` |
| 409 | Version conflict, etag mismatch, idempotency-key reuse | Agent re-fetches latest, retries (may use compare_versions) |
| 413 | Payload too large | Agent splits into pages / batches |
| 429 | Rate limit | Agent backs off the indicated `retry_after_seconds` |
| 500 | Internal server error with `incident_id` | Agent flags incident, suggests fallback (cached data, alternative function) |
| 503 | Service unavailable / maintenance window | Agent waits, retries |
| 504 | Gateway timeout, partial extraction | Agent retries with `checkpoint_token` / narrower window |
| 202 + async | Long-running job, `polling_url` + `webhook_supported: true` | Agent offers polling OR webhook subscription |
| Partial success | Batch with mixed per-item statuses | Agent reports both halves, asks how to handle failures |
| Eventually consistent | 200 OK with `warning` about staleness | Agent waits / re-reads, explains gap |
| Deprecation | 200 OK with `X-API-Deprecated` warning + `sunset_date` | Agent flags it; suggests migration |
| Field redaction | 200 OK but specific fields are `***REDACTED***` | Agent works around without echoing PII |
| Quota near-exhaustion | 200 OK with `quota_remaining: <low>` | Agent warns proactively |

### 2. User-side messiness — pick 2-3 per conversation

- **Typo in ID** — user provides a malformed ID; agent surfaces 400 then asks confirmation
- **Ambiguous reference** — "yang baru tadi tu" when there are multiple; agent disambiguates
- **Mind change / pivot** — starts one task, switches scope mid-call ("tunggu, actually focus on X je dulu")
- **Forgets context after break** — "where were we?" — agent recaps without re-running tools
- **Misreads result** — thinks something failed when it succeeded; agent clarifies
- **Provides incomplete info** — agent must ask follow-up
- **Provides contradictory info** — agent flags inconsistency politely
- **Asks "why" / for rationale** — agent explains its tool-call choice
- **Urgency / impatience** — "boleh cepat sikit, customer tunggu / CFO demand"; agent acknowledges but maintains accuracy
- **Multi-tasking interrupt** — "sat sat, telefon masuk" / "kejap, anak panggil"; agent waits
- **Asks to repeat** — agent recaps from prior turn, no new tool calls
- **Asks to undo / rollback** — agent runs rollback or clone-and-revert
- **Spells ID phonetically with correction** — "9-9-8-2-1, sori 9-9-8-2-3"
- **Mid-sentence language switch** — Bahasa → English → Tamil → Mandarin in one paragraph
- **Says "tak faham"** — agent re-explains in simpler language
- **Sensitive info accidentally shared** — agent reminds to redact, does not echo

### 3. Agent-side behavioural edges — weave 2-3 per conversation

- **Self-correction** — realises earlier tool call used wrong scope / param; offers to redo
- **Proactive risk warning** — before destructive action, summarises blast radius and asks confirmation
- **Pagination handling** — response has `has_next_page: true`; agent offers to fetch next page
- **Verification before destructive action** — confirms intent on irreversible operations (archive, rollback, bulk apply)
- **Refuses unsafe action** — when user asks for something violating policy or auth scope; agent says no and explains
- **Admits uncertainty** — "saya tidak pasti, disyorkan eskalasi" — uses audit / dependency-graph functions to learn more
- **Suggests workaround** — when primary function fails, offers alternative (e.g., cached list when detect engine is down)
- **Flags deprecation warning** — proactively mentions sunset date when response includes deprecation header

### 4. Out-of-context user asides — keep 2-3 per conversation

User strays to unrelated personal/admin topics; agent declines politely with a one-sentence redirect, no echoing or speculating, then returns to the workflow. Examples:
- Asks about HR / leave / parking claim closing time
- Asks about internship / recruitment / career opportunities for relative
- Asks about lunch / makan recommendations
- Asks agent to translate or send a message for them to a third party
- Asks about office tour / visitor policy
- Asks for weather / traffic / personal admin help

Agent's redirect template (vary phrasing):
> "Maaf Encik/Puan, [topic] adalah di luar skop khidmat saya. Untuk hal tersebut, sila rujuk [right channel]. Kembali kepada operasi: [next step]?"

### 5. Malaysian-localised Tamil and Mandarin code-switching

Customers occasionally drop in **Malaysian Tamil** or **Malaysian Mandarin** phrases naturally — call-centre style, not literary. The agent acknowledges respectfully without forcing the third language back at the customer.

**Customer language profile rotation across the 100 rows** — pick roughly:
- ~40 rows: Malay-leading customer (Bahasa primary, English secondary)
- ~25 rows: English-leading customer (formal English / Manglish mix)
- ~20 rows: Tamil-leading customer (Tamil-Bahasa-English mix, Indian-Malaysian persona)
- ~15 rows: Mandarin-leading customer (Mandarin-English-Bahasa mix, Chinese-Malaysian persona)

**Malaysian Tamil markers** (use sparingly, sprinkled):
- Greetings: `Vanakkam` (hello, polite), `Nandri` / `Romba nandri` (thanks / many thanks)
- Exclamations: `Aiyo`, `Aiyoyo`, `Aiyaa`
- Affirmation/connector: `appadi-aa?` (is it so?), `seri` (ok)
- Persona names: Ravi, Kumar, Krishnan, Priya, Devi, Selvi, Anand, Kavitha, Sasikumar, Murugan, Ramesh
- Polite agent acknowledgement: agent may reply `Vanakkam, selamat petang Encik` for the first greeting only — don't overdo

**Malaysian Mandarin markers** (use sparingly):
- Greetings: `Ni hao` (hello), `Zao` (morning), `Wan an` (evening)
- Thanks: `Xie xie`, `Xie xie ni`
- Affirmation: `Hao de` (ok), `Dui` (yes)
- Negation: `Bu hao` (not good), `bu xing` (not workable)
- Concern: `Ai yaa` (Malaysian Mandarin/Hokkien-flavoured exclaim)
- Persona names: Lim Wei Jian, Tan Mei Ling, Wong Kah Yee, Chong, Cheah, Lee Hui Min, Ng Boon Keat, Yap Siew Lan, Goh, Teh
- Code-switch examples: "Wo nak topup la", "Ni guna which function?", "Ai yaa, again 500 error", "OK xie xie ya"
- Polite agent acknowledgement of greeting: agent may say `Ni hao, selamat petang Encik` for opener

**Do NOT** use over-the-top phrases (`wah lao eh`, `tak siao`, `kan ni nei`, etc.). Keep within polite call-centre register.

Across the conversation, the customer's three-language mix should feel natural — e.g., closing line could be `"Romba nandri Encik, terima kasih banyak, xie xie ya"` when persona is Tamil-Malaysian with Chinese-speaking colleague.

## Hard validation before writing

1. Top-level JSON parses; all `tool_calls[].function.arguments` parse; all tool `content` parses.
2. Every `tool_call_id` matches a prior assistant `tool_calls[].id`.
3. **10-20 user turns**.
4. **≥8 distinct function names**, all of which exist in the matched `test-function/{i}.json` library.
5. **≥3 API failures** simulated (mix of 4xx and 5xx / async / partial / eventually-consistent / deprecation / etc.). Each failure must be followed by an agent recovery move.
6. **≥2 out-of-context customer asides**, each met with a polite agent redirect that does not echo the off-topic content.
7. **≥2 agent-side behavioural edges** observable in the conversation (self-correction, proactive warning, verification, pagination, admit uncertainty, suggest workaround).
8. **No forbidden tokens** (`Tuan`, `bro`, `boss`, `machi`, `machan`, `padu`, `syiok`, `wah lao`, `tak siao`, `kan ni nei`).
9. **At least one `Encik`/`Puan`/`Sir`/`Madam`** in agent text.
10. **Language-profile signal**: if the customer is Tamil-leading or Mandarin-leading, at least 3 distinct Tamil/Mandarin tokens should appear in user messages.

## Output JSON shape

```json
{
  "conversation_id": "myl-fnlib-extra-{i}",
  "workflow_name": "<from library>",
  "domain": "<from library>",
  "messages": [ ... ],
  "metadata": {
    "num_turns": <int>,
    "language_profile": "malay|english|tamil|mandarin",
    "api_errors_simulated": [ {"http_status": 400, "service": "...", "recovered": true|false, "via": "..."} ],
    "out_of_context_turns": [ {"turn": <int>, "topic": "..."} ],
    "agent_edges_demonstrated": ["self_correction", "proactive_warning", "..."],
    "functions_used": [...],
    "generated_at": "<ISO>",
    "turn_details": [...]
  }
}
```

## Reporting

Per row written, log:
- Index, turn count, distinct fn count
- Number of API errors injected (with codes)
- Out-of-context count
- Language profile
- One-line narrative summary

Per slice: indices written / skipped / failed, and one sample path for spot-check.
