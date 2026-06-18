# Native-Script Multi-turn Conversation Spec

## Goal

Produce 50 conversations per native-script variant (Tamil and Chinese), where the **customer uses actual native-script characters** — not just romanised tokens. Two folders:

- `/home/husein/ssd3/SyntheticGen/synthetic/test-function-multiturn-extra-native-tamil/{i}.json`
- `/home/husein/ssd3/SyntheticGen/synthetic/test-function-multiturn-extra-native-chinese/{i}.json`

Indices `i` cover 0..49 of the test-function library: `/home/husein/ssd3/SyntheticGen/synthetic/test-function/{i}.json`.

## Patterns (mix both, ~25 each per folder)

### Pattern A — Mixed romanised + native script
Customer types most of their message in romanised Malaysian style (Manglish + Malay + romanised native tokens) but **drops in 1-3 full native-script phrases naturally**, like a real WhatsApp chat from a multilingual Malaysian.

Tamil example:
> "Vanakkam Encik, saya nak check balance. **என் டேட்டா மீதம் என்ன?** ada free voice minutes ke?"

Chinese example:
> "Ni hao Encik, Wo nak topup RM 50. **请帮我用 Touch n Go ya**, thanks."

### Pattern B — Fully native-script customer
Customer writes **the majority of their messages entirely in native script** (Tamil or Mandarin). Still Malaysian register — drop in occasional English/Malay tech terms (account_id values, function-y nouns, telco terms) as romanised inline because those don't translate.

Tamil example:
> "வணக்கம், என் பிரபெய்ட் பேலன்ஸ் செக் பண்ண முடியுமா? Number 60123456789, romba urgent."

Chinese example:
> "你好, 我的预付费余额还有多少? 我的号码 60123456789, 帮我查一下 lah."

**Agent mirrors the user's primary language.** A Tamil-speaking customer is routed to a Tamil-speaking call-centre agent; a Mandarin-speaking customer to a Mandarin-speaking agent — this is how multilingual call centres (Maxis, Astro, banks) actually staff their lines in Malaysia.

- **Tamil folder agent**: replies in formal **Tamil script + English** mix. Standard call-centre Tamil phrases like `வணக்கம் Encik`, `ஒரு நிமிஷம்` (one moment), `தயவுசெய்து` (please), `மிக்க நன்றி` (thank you very much). Keeps **English/romanised tech terms inline** (run ID, HTTP 401, MFA, retry, account_id, etc.). Honorific `Encik/Puan` preserved (these are Malay-borrow vocatives that are standard in Malaysian Tamil too).

- **Chinese folder agent**: replies in formal **Simplified Chinese + English** mix. Standard call-centre Mandarin phrases like `您好 Encik`, `请稍等` (please wait a moment), `感谢您的耐心` (thank you for your patience), `非常抱歉` (very sorry). Keeps **English/romanised tech terms inline**. Honorific `Encik/Puan/Sir` preserved.

The agent register is still **formal call-centre** — polite, professional, not chummy. No slang in Tamil (`நாஞ்சு`, `dey`, etc.) or Mandarin (`卧槽`, etc.).

## What's the same as multiturn_extra_spec.md

- 10-20 user turns
- 8+ distinct functions per file
- ≥3 API failure modes per conversation (4xx + 5xx + async + partial/eventually-consistent/deprecation/redaction/quota)
- ≥2 out-of-context customer asides with polite "di luar skop" agent redirects
- ≥2 agent-side behavioural edges
- OpenAI-compatible message structure
- Formal agent uses Encik/Puan/Sir/Madam (NEVER Tuan); no `bro`/`boss`/`machi`/`machan`/`padu`/`syiok`
- ID continuity across turns
- Tool args validate against schemas; tool responses plausible JSON
- All tool function names must exist in matched library

## Reference examples

- General style baseline: `/home/husein/ssd3/SyntheticGen/synthetic/test-function-multiturn-extra/0.json`
- Tamil dry run (this folder): `/home/husein/ssd3/SyntheticGen/synthetic/test-function-multiturn-extra-native-tamil/0.json`
- Chinese dry run (this folder): `/home/husein/ssd3/SyntheticGen/synthetic/test-function-multiturn-extra-native-chinese/0.json`

## Native-script content guidance

### Malaysian Tamil
- Use **Indian-Tamil-script** (`அ ஆ இ ஈ உ ஊ ...`) — the same script used in Malaysian Tamil schools, newspapers, and signage.
- Greetings: `வணக்கம்` (Vanakkam), `நன்றி` (Nandri), `ரொம்ப நன்றி` (Romba nandri)
- Common verbs/phrases:
  - "Can you check?" → `செக் பண்ண முடியுமா?`
  - "I want to..." → `நான் ... செய்ய வேண்டும்` / `வேணும்`
  - "Sorry" → `மன்னிக்கவும்` / `சாரி`
  - "How much?" → `எவ்வளவு?`
  - "Wait a moment" → `கொஞ்சம் பொறு` / `ஒரு நிமிஷம்`
  - "Thank you" → `நன்றி`
- Numbers: write digits as ASCII (`60123456789`, `RM 50`) — Tamil numerals are rare in modern Malaysia.
- Code-mix tech terms inline as romanised: `'account', 'balance', 'topup', 'API', 'webhook', 'reconciliation'` — leave these in English/romanised even inside Tamil sentences. Real Malaysian Tamil speakers do this.

### Malaysian Mandarin
- Use **Simplified Chinese** characters (`你 我 他 是 的`) — Malaysian standard. Avoid Traditional unless the persona is explicitly Taiwanese/HK-leaning.
- Greetings: `你好` (Ni hao), `早安` (zao an / morning), `晚安` (wan an / evening), `谢谢` (xie xie / thanks)
- Common phrases:
  - "Can you check?" → `可以帮我查吗?` / `可以检查一下吗?`
  - "I want to..." → `我要...` / `我想...`
  - "Sorry" → `不好意思` / `抱歉`
  - "How much?" → `多少钱?`
  - "Wait" → `等一下` / `稍等`
  - "Thank you" → `谢谢`
  - "OK / Got it" → `好的` / `知道了`
  - "Aiyaa expression" → `哎呀`
- Numbers: digits as ASCII.
- Code-mix tech terms inline as romanised — Malaysian Chinese speakers say `"我的 account 有 RM 50 balance"` not `"我的账户有五十令吉余额"`.

## Validation (in addition to base extra-spec checks)

1. Native script **must appear in user messages**:
   - Tamil folder: at least one Tamil-script character in **≥80% of user turns** (Pattern B) or **≥40% of user turns** (Pattern A).
   - Chinese folder: same.
2. Native script **must also appear in agent messages** (agent mirrors user's language):
   - Tamil folder: at least one Tamil-script character in **≥60% of assistant turns**.
   - Chinese folder: at least one Chinese-script character in **≥60% of assistant turns**.
3. ID values and JSON tool arguments stay ASCII (don't put Tamil/Chinese into JSON args or IDs).
4. All other extra-spec checks (10-20 turns, ≥8 fns, ≥3 API errors, ≥2 OOC, ≥2 edges, no forbidden tokens, vocative present).

## Output JSON shape

Same as multiturn_extra. Add to metadata:
```json
"native_script": "tamil" | "chinese",
"pattern": "A_mixed" | "B_fully_native"
```

## Reporting

Per row written, log:
- Index, pattern, turn count, distinct fn count, #failure-modes, #ooc, native-script coverage % of user turns
- One sample path
