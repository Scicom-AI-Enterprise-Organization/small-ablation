export const meta = {
  name: 'train-function-expand-remaining',
  description: 'LLM-expand the still-missing train rows (passed via args) per expand_spec.md',
  phases: [{ title: 'Expand', detail: 'agents x ~6 rows over the missing set, LLM expansion + self-validate' }],
}

const ROOT = '/home/husein/ssd3/SyntheticGen/synthetic'

// Self-target: a bootstrap agent computes which rows still lack a PASSing output.
phase('Scan')
const scan = await agent(
  `Run exactly this and return the result:\n` +
  `cd ${ROOT} && python3 - <<'PY'\n` +
  `import os,json\n` +
  `done=set(int(f[:-5]) for f in os.listdir('train-function') if f.endswith('.json'))\n` +
  `print(json.dumps([i for i in range(1142) if i not in done]))\n` +
  `PY\n` +
  `Return the JSON array printed (the missing row indices).`,
  { label: 'scan-missing', phase: 'Scan',
    schema: { type: 'object', additionalProperties: false,
      properties: { missing: { type: 'array', items: { type: 'integer' } } },
      required: ['missing'] } }
)
const missing = (scan && Array.isArray(scan.missing)) ? scan.missing.slice() : []
if (!missing.length) {
  log('no missing rows — all 1142 present')
  return { agents: 0, rows_written: 0, note: 'nothing missing' }
}

const SPEC = `
You are expanding a synthetic function-calling spec. For each assigned row index i:

FIRST, skip-if-done: if ${ROOT}/train-function/{i}.json already exists, run
  cd ${ROOT} && python3 validate_expansion.py {i}
and if it prints "{i}: PASS", DO NOT regenerate it — leave it and move on. Only (re)generate rows whose file is missing or FAILs.

INPUT: read ${ROOT}/train-function-src/{i}.json. Shape:
  { "workflow_name": "...", "description": "...", "domain": "...",
    "shared_entities": { "<key>": {<JSONSchema>}, ... },
    "functions": [ {"name","description","stage","parameters":{JSONSchema},"returns"}, ... ] }
There are ~12-15 original functions. ${ROOT}/train-function-src/_index.json lists each row's "complexity" hint — match that parameter complexity (deep nesting, arrays, polymorphic discriminators, 10-20 params per function where appropriate).

TASK: produce the SAME shape with the functions list EXPANDED to 30-50 entries (aim for ~40):
  - Keep ALL original functions verbatim (do not drop or rename any).
  - Keep workflow_name, description, domain EXACTLY as in the source.
  - Keep all original shared_entities keys. You MAY ADD new shared_entities (additive only, never rename existing) when new functions need them.
  - Add NEW functions that genuinely belong to THIS row's specific domain + workflow. Tailor them to the actual workflow_name/domain/entities — do NOT emit a generic one-size-fits-all template; phrase everything in this row's domain language. Reuse the row's real shared_entities via {"$ref":"#/shared_entities/<key>"} and reuse/extend existing stages; add new stages only when needed (audit, report, notify, simulate, bulk, schedule, policy, subscribe, query, evidence, escalate, recovery, monitor, etc.).
  - Each new function needs: snake_case unique "name"; action-oriented "description"; "stage"; full JSONSchema "parameters" (type:"object", properties, required where appropriate, $ref for shared entities, enums, nested objects, arrays of objects); one/two-sentence "returns".
  - No duplicates / no renamed copies of existing functions. Each adds real new capability. Cover more of the lifecycle: rich search/list/filter, bulk ops, notifications/subscriptions/webhooks, audit/history, schedule/cron, dry-run/simulate/preview, approval/review/escalation, comments/attachments, snapshot/restore/export/import, metrics/dashboards, policy/config, dependency/link mgmt, rollback/retry — always in this row's domain language.

OUTPUT: write the merged object to ${ROOT}/train-function/{i}.json. Use python to guarantee valid JSON:
  python3 - <<'PY'
  import json
  src=json.load(open('${ROOT}/train-function-src/{i}.json'))
  out=dict(src)
  out['shared_entities']=dict(src['shared_entities'])  # add keys as needed
  out['functions']=list(src['functions'])              # then append new func dicts
  json.dump(out, open('${ROOT}/train-function/{i}.json','w'), indent=2, ensure_ascii=False)
  PY

VALIDATE: after writing, run
  cd ${ROOT} && python3 validate_expansion.py <your indices space-separated>
Every line MUST say PASS. Fix any FAIL and re-run until all PASS. Do not finish with any failing index.
`

const REPORT_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    written: { type: 'array', items: { type: 'integer' } },
    skipped_existing: { type: 'array', items: { type: 'integer' } },
    all_passed: { type: 'boolean' },
    failures: { type: 'array', items: { type: 'string' } },
  },
  required: ['written','skipped_existing','all_passed','failures'],
}

const SIZE = 6
const slices = []
for (let s = 0; s < missing.length; s += SIZE) slices.push(missing.slice(s, s + SIZE))
log(`fanning out ${slices.length} agents over ${missing.length} missing rows (${SIZE} rows each)`)

phase('Expand')
const results = await parallel(slices.map((idxs) => () =>
  agent(
    `${SPEC}\n\nYOUR ASSIGNED ROW INDICES: ${idxs.join(', ')}.\nExpand each (skipping any already-PASSing file), write each output, then validate ALL assigned indices and ensure every line is PASS before returning.`,
    { label: `expand:${idxs[0]}-${idxs[idxs.length-1]}`, phase: 'Expand', schema: REPORT_SCHEMA }
  )
))

const ok = results.filter(Boolean)
return {
  agents: results.length,
  agents_ok: ok.length,
  agents_dead: results.length - ok.length,
  rows_written: ok.reduce((a, r) => a + r.written.length, 0),
  rows_skipped: ok.reduce((a, r) => a + r.skipped_existing.length, 0),
}
