# Function-Expansion Spec for Sub-Agents

## Goal

For each row of the TEST parquet, take the existing `functions` JSON (~12 functions) and **expand it to 30-50 functions** total, keeping the same style, domain, and workflow theme. Save to `/home/husein/ssd3/SyntheticGen/synthetic/test-function/{row_index}.json`.

## Inputs

- Test parquet: `/home/husein/ssd3/SyntheticGen/synthetic/test-00000-of-00001.parquet`
- Cols: `domain`, `complexity`, `workflow`, `functions` (str holding JSON)
- The `functions` JSON has shape:
  ```
  {
    "workflow_name": "...",
    "description": "...",
    "domain": "...",
    "shared_entities": { ... },
    "functions": [ { "name": "...", "description": "...", "stage": "...", "parameters": {...JSONSchema...}, "returns": "..." }, ... ]
  }
  ```

## Output format

Save **the same shape** to `/home/husein/ssd3/SyntheticGen/synthetic/test-function/{i}.json`, with:

- `workflow_name`, `description`, `domain` — unchanged.
- `shared_entities` — keep the originals; you MAY add new shared entities if the new functions need them (be additive, don't rename existing keys).
- `functions` — original list + new functions, totalling **30 to 50** entries.

`{i}` is the 0-based row index in the test parquet.

## Checkpoint / resume

Skip if `test-function/{i}.json` already exists.

```python
import os
if os.path.exists(f'/home/husein/ssd3/SyntheticGen/synthetic/test-function/{i}.json'):
    continue
```

## Function design rules

- **Stay in the workflow's domain.** If the workflow is `reconciliation_flow`, every new function should plausibly belong in that domain (sources, comparisons, audit, escalation, scheduling, reports, etc.). If it's `device_swap`, stay in the device-swap world. Etc.
- **Use existing stages** when possible (e.g., `init`, `precheck`, `extract`, `detect`, `analyze`, `plan`, `execute`, `monitor`, `recovery`, `close`) and feel free to add new stages if the originals miss something (`audit`, `report`, `notify`, `simulate`, `bulk`, `schedule`, `policy`, `subscribe`, `query`, `evidence`).
- **Use existing shared_entities** via `$ref` (e.g., `{"$ref": "#/shared_entities/account_id"}`) wherever appropriate — match the original style.
- **Match the parameter complexity** in the original row — deep nesting, arrays, polymorphic discriminators, 10-20 params per function is the complexity hint stored in the `complexity` column.
- Every new function must have:
  - `name` (snake_case, plausible, unique within the file)
  - `description` (one or two sentences, action-oriented)
  - `stage` (string)
  - `parameters` (full JSONSchema object: `type:"object"`, `properties:{...}`, `required:[...]` if needed; use `$ref` for shared entities; include enums, nested objects, arrays of objects, etc.)
  - `returns` (one-or-two-sentence description of what the function returns)
- **No duplicates.** Don't reproduce an existing function under a different name. Each new function should add genuine new capability (CRUD variations, batch operations, scheduling/cron, audit/history, notifications, dry-runs, simulations, search/list with rich filters, dependency lookups, escalations, exports, imports, policy management, role assignment, comments/annotations, attachments, subscriptions/webhooks, metrics, dashboards, snapshots, rollback, retries, partial commits, comparisons, transformations, bulk-import/export, etc.).
- **Coverage suggestion** (expand functions to cover more of the lifecycle):
  - **Search / list / filter** variants (richer filters, pagination, faceted search)
  - **Bulk** operations (bulk_create, bulk_update, bulk_delete, bulk_export)
  - **Notification / subscription / webhook** functions
  - **Audit / history / change-log** functions
  - **Schedule / cron / recurring** functions
  - **Dry-run / simulate / preview** functions
  - **Approval / review / escalation** functions
  - **Comments / annotations / attachments** on entities
  - **Snapshot / restore / export** functions
  - **Metrics / KPIs / dashboards** functions
  - **Policy / rule / config** management
  - **Dependency / link / association** management

## Validation before writing

1. Top-level JSON parses.
2. `functions` array has length **between 30 and 50** inclusive.
3. Every function has `name`, `description`, `stage`, `parameters` (with `type:"object"`), `returns`.
4. All function names are unique within the file.
5. All `$ref` strings point to keys that exist in `shared_entities`.
6. The original function names from the source are all still present in the output (don't drop originals).
7. Original `workflow_name`, `description`, `domain` are preserved verbatim.

## Reporting

After processing your slice, report:
- Indices written (new files)
- Indices skipped (files already existed)
- Per-index final function count (e.g., `"0: 36 functions"`)
- Any failures with reasons
- One sample file path for spot-check
