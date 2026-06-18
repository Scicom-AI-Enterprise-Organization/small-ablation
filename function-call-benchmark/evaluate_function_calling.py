#!/usr/bin/env python3
"""
Function calling evaluation for open source models via OpenAI-compatible API.
Loads the benchmark from HuggingFace: Scicom-intl/Function-Call-TaaS

Usage:
    python evaluate_function_calling.py \\
        --base-url http://localhost:8000/v1 \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --output results.json

    # Quick smoke test (5 conversations):
    python evaluate_function_calling.py \\
        --base-url http://localhost:8000/v1 \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --max-conversations 5

    # Use a specific HF split (default: train):
    python evaluate_function_calling.py ... --split test-basic

Metrics:
    tool_call_precision   when model calls a tool, was it appropriate?
    tool_call_recall      when a tool should be called, does the model call?
    tool_call_f1
    name_accuracy         fraction of expected tool names matched (per-turn)
    name_set_f1           set-level F1 for parallel tool call sets
    json_valid_rate       fraction of tool calls with parseable JSON args
    hallucination_rate    fraction of tool calls to non-existent functions
    req_coverage          avg fraction of required params present
    type_accuracy         avg fraction of top-level params with correct JSON type
    parallel_count_match  fraction of turns where call count matches reference
    id_propagation_rate   fraction of IDs from tool results re-used correctly
    refusal_rate          fraction of out-of-context turns where model made no tool calls
                          (only present when dataset has out_of_context_turns metadata)
"""

import argparse
import copy
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from openai import OpenAI

HF_DATASET = "Scicom-intl/Function-Call-TaaS"

# Load .env from script dir or its parent (silently if python-dotenv not installed)
def _load_dotenv():
    try:
        from dotenv import load_dotenv
        script_dir = Path(__file__).parent
        for candidate in (script_dir / ".env", script_dir.parent / ".env"):
            if candidate.exists():
                load_dotenv(candidate)
                return
    except ImportError:
        pass

_load_dotenv()


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def resolve_refs(schema: object, shared: dict) -> object:
    """Recursively inline $ref references of the form #/shared_entities/<name>."""
    if isinstance(schema, dict):
        if "$ref" in schema:
            parts = schema["$ref"].lstrip("#/").split("/")
            if len(parts) == 2 and parts[0] == "shared_entities":
                return resolve_refs(shared.get(parts[1], {}), shared)
            return schema
        return {k: resolve_refs(v, shared) for k, v in schema.items()}
    if isinstance(schema, list):
        return [resolve_refs(item, shared) for item in schema]
    return schema


def build_openai_tools(lib: dict) -> tuple[list[dict], dict, set]:
    """
    Convert function library to OpenAI tool list with $refs resolved.
    Returns (tools, fn_schema_map, fn_names).
    """
    shared = lib.get("shared_entities", {})
    tools = []
    fn_schema_map = {}
    for fn in lib["functions"]:
        resolved = resolve_refs(copy.deepcopy(fn), shared)
        params = resolved.get("parameters", {"type": "object", "properties": {}})
        tools.append({
            "type": "function",
            "function": {
                "name": resolved["name"],
                "description": resolved.get("description", ""),
                "parameters": params,
            },
        })
        fn_schema_map[resolved["name"]] = resolved
    fn_names = set(fn_schema_map.keys())
    return tools, fn_schema_map, fn_names


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def required_coverage(args: dict, fn_schema: dict) -> float:
    required = fn_schema.get("parameters", {}).get("required", [])
    if not required:
        return 1.0
    return sum(1 for r in required if r in args) / len(required)


_TYPE_MAP = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def type_accuracy(args: dict, fn_schema: dict) -> float:
    props = fn_schema.get("parameters", {}).get("properties", {})
    if not props or not args:
        return 1.0
    correct = total = 0
    for key, val in args.items():
        if key in props:
            expected = props[key].get("type")
            if expected and expected in _TYPE_MAP:
                total += 1
                if isinstance(val, _TYPE_MAP[expected]):
                    correct += 1
    return correct / total if total > 0 else 1.0


def extract_ids(content_str: str) -> set:
    """Extract ID-like strings from tool result JSON (hyphen-containing, no spaces)."""
    ids: set = set()
    try:
        def _walk(o):
            if isinstance(o, str):
                if "-" in o and 5 <= len(o) <= 80 and " " not in o:
                    ids.add(o)
            elif isinstance(o, dict):
                for v in o.values():
                    _walk(v)
            elif isinstance(o, list):
                for v in o:
                    _walk(v)
        _walk(json.loads(content_str))
    except Exception:
        pass
    return ids


def id_propagation(model_args: list[str], ref_args: list[str], available_ids: set) -> Optional[float]:
    """
    Of the IDs the reference passes to its next tool calls (from the available pool),
    what fraction does the model also pass?
    Returns None when there are no IDs to propagate.
    """
    all_ref = " ".join(ref_args)
    ref_uses = {i for i in available_ids if i in all_ref}
    if not ref_uses:
        return None
    all_model = " ".join(model_args)
    return len({i for i in ref_uses if i in all_model}) / len(ref_uses)


# ---------------------------------------------------------------------------
# Model call wrapper
# ---------------------------------------------------------------------------

def call_model(
    client: OpenAI, model: str, messages: list, tools: list, retries: int = 3
) -> Optional[object]:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=16384,
            )
            return resp.choices[0].message
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [error] {e}", file=sys.stderr)
                return None
            time.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------------------
# Per-turn metric record
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    turn: int
    ref_call_count: int
    model_call_count: int
    tp: bool = False
    fp: bool = False
    fn_miss: bool = False
    tn: bool = False
    name_matches: int = 0
    ref_name_set: list = field(default_factory=list)
    model_name_set: list = field(default_factory=list)
    json_valid: int = 0
    hallucinated: int = 0
    req_coverages: list = field(default_factory=list)
    type_accs: list = field(default_factory=list)
    id_prop: Optional[float] = None
    out_of_context: bool = False  # True when metadata marks this turn as off-topic


def compute_turn_result(
    turn: int,
    ref_tcs: list,
    model_tcs: list,
    fn_names: set,
    fn_schema_map: dict,
    available_ids: set,
) -> TurnResult:
    tr = TurnResult(
        turn=turn,
        ref_call_count=len(ref_tcs),
        model_call_count=len(model_tcs),
    )

    if tr.ref_call_count > 0 and tr.model_call_count > 0:
        tr.tp = True
    elif tr.ref_call_count == 0 and tr.model_call_count > 0:
        tr.fp = True
    elif tr.ref_call_count > 0 and tr.model_call_count == 0:
        tr.fn_miss = True
    else:
        tr.tn = True

    ref_names = [tc["function"]["name"] for tc in ref_tcs]
    ref_args_strs = [tc["function"].get("arguments", "{}") for tc in ref_tcs]
    model_names = []
    model_args_strs = []

    for mtc in model_tcs:
        fn_name = mtc.function.name
        model_names.append(fn_name)
        try:
            args = json.loads(mtc.function.arguments)
            tr.json_valid += 1
        except Exception:
            args = {}
        model_args_strs.append(mtc.function.arguments or "")

        if fn_name not in fn_names:
            tr.hallucinated += 1
            continue

        schema = fn_schema_map.get(fn_name, {})
        tr.req_coverages.append(required_coverage(args, schema))
        tr.type_accs.append(type_accuracy(args, schema))

    tr.name_matches = len(set(model_names) & set(ref_names))
    tr.ref_name_set = ref_names
    tr.model_name_set = model_names
    tr.id_prop = id_propagation(model_args_strs, ref_args_strs, available_ids)

    return tr


# ---------------------------------------------------------------------------
# Full-replay conversation evaluation
# ---------------------------------------------------------------------------

def evaluate_conversation(
    client: OpenAI,
    model: str,
    conv: dict,
    tools: list,
    fn_names: set,
    fn_schema_map: dict,
) -> tuple[list[TurnResult], Optional[str]]:
    """
    Simulates the conversation turn by turn, exactly like a real chat.

    - User messages are appended as-is.
    - At each assistant turn the model is called and its response goes into
      the history (so subsequent turns see the model's own prior replies).
    - Reference tool results are injected using the model's tool call IDs
      (since we have no real backend). Pairing is positional.
    - At each assistant turn the model's response is compared against the
      reference to compute metrics.
    """
    msgs = conv["messages"]
    history: list[dict] = []
    results: list[TurnResult] = []
    available_ids: set = set()
    turn_num = 0
    ref_idx = 0

    # out_of_context_turns uses 1-based turn indices; convert to 0-based set
    meta = conv.get("metadata", {})
    oot_turns: set = {
        t["turn"] - 1
        for t in meta.get("out_of_context_turns", [])
        if isinstance(t, dict) and "turn" in t
    }

    while ref_idx < len(msgs):
        msg = msgs[ref_idx]
        role = msg["role"]

        if role == "user":
            history.append({"role": "user", "content": msg["content"]})
            available_ids = set()
            ref_idx += 1

        elif role == "assistant":
            ref_tcs = msg.get("tool_calls") or []

            # Call the model with the conversation so far
            model_msg = call_model(client, model, history, tools)
            if model_msg is None:
                return results, "model call failed"

            model_tcs = model_msg.tool_calls or []

            # Evaluate this turn
            tr = compute_turn_result(
                turn=turn_num,
                ref_tcs=ref_tcs,
                model_tcs=model_tcs,
                fn_names=fn_names,
                fn_schema_map=fn_schema_map,
                available_ids=available_ids,
            )
            tr.out_of_context = turn_num in oot_turns
            results.append(tr)
            turn_num += 1

            # Append the model's own response to history (like a real conversation)
            assistant_hist: dict = {"role": "assistant", "content": model_msg.content or ""}
            if model_tcs:
                assistant_hist["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in model_tcs
                ]
            history.append(assistant_hist)
            ref_idx += 1

            # Collect the reference tool results that follow this assistant turn
            ref_tool_msgs: list[dict] = []
            while ref_idx < len(msgs) and msgs[ref_idx]["role"] == "tool":
                ref_tool_msgs.append(msgs[ref_idx])
                ref_idx += 1

            # Inject reference tool results using the model's tool call IDs (positional)
            for i, mtc in enumerate(model_tcs):
                content = (
                    ref_tool_msgs[i]["content"]
                    if i < len(ref_tool_msgs)
                    else json.dumps({"status": "no_reference_result"})
                )
                history.append({
                    "role": "tool",
                    "tool_call_id": mtc.id,
                    "name": mtc.function.name,
                    "content": content,
                })
                available_ids.update(extract_ids(content))

        elif role == "tool":
            ref_idx += 1  # already consumed above
        else:
            ref_idx += 1

    return results, None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(all_results: list[list[TurnResult]]) -> dict:
    tp = fp = fn_miss = tn = 0
    json_valid = json_total = 0
    hallucinated = hall_total = 0
    req_cov_vals: list[float] = []
    type_acc_vals: list[float] = []
    id_prop_vals: list[float] = []
    parallel_match = parallel_total = 0
    name_match_sum = name_match_n = 0
    name_set_tp = name_set_fp = name_set_fn = 0
    refusal_correct = refusal_total = 0

    for conv_turns in all_results:
        for tr in conv_turns:
            tp += tr.tp
            fp += tr.fp
            fn_miss += tr.fn_miss
            tn += tr.tn

            json_valid += tr.json_valid
            json_total += tr.model_call_count
            hallucinated += tr.hallucinated
            hall_total += tr.model_call_count

            req_cov_vals.extend(tr.req_coverages)
            type_acc_vals.extend(tr.type_accs)

            if tr.id_prop is not None:
                id_prop_vals.append(tr.id_prop)

            if tr.ref_call_count > 0:
                parallel_total += 1
                if tr.model_call_count == tr.ref_call_count:
                    parallel_match += 1

            if tr.ref_call_count > 0 and tr.model_call_count > 0:
                ref_set = set(tr.ref_name_set)
                model_set = set(tr.model_name_set)
                name_set_tp += len(ref_set & model_set)
                name_set_fp += len(model_set - ref_set)
                name_set_fn += len(ref_set - model_set)
                name_match_sum += tr.name_matches / max(tr.ref_call_count, tr.model_call_count)
                name_match_n += 1

            if tr.out_of_context:
                refusal_total += 1
                if tr.model_call_count == 0:
                    refusal_correct += 1

    def _div(a, b): return round(a / b, 4) if b > 0 else 0.0
    def _avg(lst): return round(sum(lst) / len(lst), 4) if lst else None

    prec = _div(tp, tp + fp)
    rec = _div(tp, tp + fn_miss)
    f1 = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) > 0 else 0.0

    stp, sfp, sfn = name_set_tp, name_set_fp, name_set_fn
    sprec = _div(stp, stp + sfp)
    srec = _div(stp, stp + sfn)
    sf1 = round(2 * sprec * srec / (sprec + srec), 4) if (sprec + srec) > 0 else 0.0

    return {
        "tool_call_precision": prec,
        "tool_call_recall": rec,
        "tool_call_f1": f1,
        "name_accuracy": _div(name_match_sum, name_match_n),
        "name_set_precision": sprec,
        "name_set_recall": srec,
        "name_set_f1": sf1,
        "json_valid_rate": _div(json_valid, json_total),
        "hallucination_rate": _div(hallucinated, hall_total),
        "req_coverage": _avg(req_cov_vals),
        "type_accuracy": _avg(type_acc_vals),
        "parallel_count_match": _div(parallel_match, parallel_total),
        "id_propagation_rate": _avg(id_prop_vals),
        "refusal_rate": _div(refusal_correct, refusal_total) if refusal_total > 0 else None,
        "_counts": {
            "tp": tp, "fp": fp, "fn": fn_miss, "tn": tn,
            "total_model_calls": json_total,
            "total_ref_calls": tp + fn_miss,
        },
    }


# ---------------------------------------------------------------------------
# Data loading from HuggingFace
# ---------------------------------------------------------------------------

def load_hf_data(split: str, max_n: int, config: Optional[str] = None) -> list[tuple[dict, dict]]:
    """Load (lib, conv) pairs from the HuggingFace dataset.

    config: HF dataset configuration/subset name (e.g. 'test-extra').
            When None the default configuration is used.
    split:  HF split name within that configuration (default 'train').
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install with: pip install datasets", file=sys.stderr)
        sys.exit(1)

    # Ensure HF cache is writable; fall back to a local path if needed.
    hf_home = os.environ.get("HF_HOME", "")
    if not hf_home:
        default_cache = Path.home() / ".cache" / "huggingface" / "hub"
        if not os.access(str(default_cache.parent.parent), os.W_OK):
            fallback = Path(os.environ.get("TMPDIR", "/tmp")) / "hf_cache"
            fallback.mkdir(parents=True, exist_ok=True)
            os.environ["HF_HOME"] = str(fallback)
            print(f"  [info] HF cache set to {fallback} (default was not writable)")

    label = f"{HF_DATASET}{'/' + config if config else ''} (split={split})"
    print(f"Loading {label} ...")
    if config:
        ds = load_dataset(HF_DATASET, config, split=split)
    else:
        ds = load_dataset(HF_DATASET, split=split)
    if max_n:
        ds = ds.select(range(min(max_n, len(ds))))

    pairs = []
    for i, row in enumerate(ds):
        try:
            lib = {
                "workflow_name": row["workflow"],
                "shared_entities": json.loads(row["shared_entities"]),
                "functions": json.loads(row["functions"]),
            }
            conv = {
                "conversation_id": row.get("conversation_id") or f"{row['workflow']}-{i}",
                "workflow_name": row["workflow"],
                "domain": row["domain"],
                "messages": json.loads(row["messages"]),
                "metadata": json.loads(row["metadata"]),
            }
            pairs.append((lib, conv))
        except Exception as e:
            print(f"  [warn] row {i} parse error: {e}", file=sys.stderr)

    print(f"Loaded {len(pairs)} conversations.")
    return pairs


# ---------------------------------------------------------------------------
# Single conversation evaluation (for thread pool)
# ---------------------------------------------------------------------------

def eval_single(args_tuple: tuple) -> dict:
    client, model, lib, conv = args_tuple
    tools, fn_schema_map, fn_names = build_openai_tools(lib)
    conv_id = conv.get("conversation_id", "?")

    turn_results, error = evaluate_conversation(client, model, conv, tools, fn_names, fn_schema_map)

    return {
        "conversation_id": conv_id,
        "workflow_name": conv.get("workflow_name"),
        "domain": conv.get("domain"),
        "error": error,
        "turns": [
            {
                "turn": tr.turn,
                "ref_call_count": tr.ref_call_count,
                "model_call_count": tr.model_call_count,
                "tp": tr.tp, "fp": tr.fp, "fn": tr.fn_miss, "tn": tr.tn,
                "name_matches": tr.name_matches,
                "ref_names": tr.ref_name_set,
                "model_names": tr.model_name_set,
                "json_valid": tr.json_valid,
                "hallucinated": tr.hallucinated,
                "req_coverages": tr.req_coverages,
                "type_accs": tr.type_accs,
                "id_prop": tr.id_prop,
                "out_of_context": tr.out_of_context,
            }
            for tr in turn_results
        ],
    }


def rebuild_turn_results(serialized: list[dict]) -> list[TurnResult]:
    out = []
    for t in serialized:
        tr = TurnResult(turn=t["turn"], ref_call_count=t["ref_call_count"], model_call_count=t["model_call_count"])
        tr.tp = t["tp"]; tr.fp = t["fp"]; tr.fn_miss = t["fn"]; tr.tn = t["tn"]
        tr.name_matches = t["name_matches"]
        tr.ref_name_set = t["ref_names"]
        tr.model_name_set = t["model_names"]
        tr.json_valid = t["json_valid"]
        tr.hallucinated = t["hallucinated"]
        tr.req_coverages = t["req_coverages"]
        tr.type_accs = t["type_accs"]
        tr.id_prop = t["id_prop"]
        tr.out_of_context = t.get("out_of_context", False)
        out.append(tr)
    return out


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(metrics: dict, n_convs: int, n_errors: int, model: str) -> None:
    m = metrics
    print("\n" + "=" * 60)
    print(f"  Function Calling Evaluation — {model}")
    print(f"  Conversations: {n_convs}  |  Errors: {n_errors}")
    print("=" * 60)
    print(f"  Tool Call Detection")
    print(f"    Precision:          {m['tool_call_precision']:.3f}")
    print(f"    Recall:             {m['tool_call_recall']:.3f}")
    print(f"    F1:                 {m['tool_call_f1']:.3f}")
    print(f"  Function Selection")
    print(f"    Name Accuracy:      {m['name_accuracy']:.3f}")
    print(f"    Name Set F1:        {m['name_set_f1']:.3f}  "
          f"(P={m['name_set_precision']:.3f}  R={m['name_set_recall']:.3f})")
    print(f"  Output Quality")
    print(f"    JSON Valid Rate:    {m['json_valid_rate']:.3f}")
    print(f"    Hallucination Rate: {m['hallucination_rate']:.3f}")
    print(f"  Argument Quality")
    rc = m['req_coverage']
    ta = m['type_accuracy']
    print(f"    Req Coverage:       {rc:.3f}" if rc is not None else "    Req Coverage:       n/a")
    print(f"    Type Accuracy:      {ta:.3f}" if ta is not None else "    Type Accuracy:       n/a")
    print(f"  Multi-turn")
    print(f"    Parallel Match:     {m['parallel_count_match']:.3f}")
    ip = m['id_propagation_rate']
    print(f"    ID Propagation:     {ip:.3f}" if ip is not None else "    ID Propagation:     n/a")
    rr = m.get('refusal_rate')
    if rr is not None:
        print(f"    Refusal Rate:       {rr:.3f}  (out-of-context turns correctly ignored)")
    c = m["_counts"]
    print(f"\n  Confusion: TP={c['tp']}  FP={c['fp']}  FN={c['fn']}  TN={c['tn']}")
    print(f"  Total model calls: {c['total_model_calls']}  |  ref calls: {c['total_ref_calls']}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _ckpt_path(model: str, ckpt_dir: Path) -> Path:
    return ckpt_dir / f"{model.replace('/', '_')}.jsonl"


def load_checkpoint(model: str, ckpt_dir: Path) -> dict:
    """Return {conv_id: result} for already-completed conversations."""
    path = _ckpt_path(model, ckpt_dir)
    done: dict = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done[r["conversation_id"]] = r
            except Exception:
                pass
    return done


def save_checkpoint(result: dict, model: str, ckpt_dir: Path) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(_ckpt_path(model, ckpt_dir), "a") as f:
        f.write(json.dumps(result) + "\n")


# ---------------------------------------------------------------------------
# Single-model eval loop (with checkpointing)
# ---------------------------------------------------------------------------

def run_model_eval(
    client: OpenAI,
    model: str,
    pairs: list,
    workers: int,
    ckpt_dir: Path,
    prefix: str = "",
) -> tuple[list[dict], dict]:
    """
    Run full eval for one model, resuming from checkpoint if available.
    Returns (conv_results, metrics).
    """
    done_map = load_checkpoint(model, ckpt_dir)
    remaining = [(lib, conv) for lib, conv in pairs
                 if conv["conversation_id"] not in done_map]

    tag = f"{prefix}[{model}]"
    if done_map:
        print(f"{tag} resuming — {len(done_map)} done, {len(remaining)} remaining")
    else:
        print(f"{tag} starting — {len(remaining)} conversations, {workers} workers")

    conv_results = list(done_map.values())

    if remaining:
        tasks = [(client, model, lib, conv) for lib, conv in remaining]
        total = len(done_map) + len(tasks)
        n_done = len(done_map)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(eval_single, t): t[3].get("conversation_id", i)
                       for i, t in enumerate(tasks)}
            for fut in as_completed(futures):
                cid = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    result = {
                        "conversation_id": str(cid),
                        "workflow_name": None,
                        "domain": None,
                        "error": str(e),
                        "turns": [],
                    }
                save_checkpoint(result, model, ckpt_dir)
                conv_results.append(result)
                n_done += 1
                err = f" [ERROR: {result['error']}]" if result.get("error") else ""
                n_turns = len(result["turns"])
                print(f"  {tag} [{n_done}/{total}] {result['conversation_id']}  "
                      f"turns={n_turns}{err}")

    all_turn_results = [rebuild_turn_results(cr["turns"]) for cr in conv_results]
    n_errors = sum(1 for cr in conv_results if cr.get("error"))
    metrics = aggregate(all_turn_results)
    print_summary(metrics, len(conv_results), n_errors, model)
    return conv_results, metrics


# ---------------------------------------------------------------------------
# README update
# ---------------------------------------------------------------------------

def update_readme(
    all_metrics: list[tuple[str, dict]],  # [(model, metrics), ...]
    readme_path: Path,
    split: str,
) -> None:
    # Sort by tool_call_f1 descending
    ranked = sorted(all_metrics, key=lambda x: x[1]["tool_call_f1"], reverse=True)

    has_refusal = any(m.get("refusal_rate") is not None for _, m in ranked)

    col_headers = "| Model | Tool F1 | Name F1 | JSON Valid | Hallucination ↓ | Req Cov | Type Acc | Parallel | ID Prop |"
    col_sep =     "|-------|--------:|--------:|-----------:|----------------:|--------:|---------:|---------:|--------:|"
    if has_refusal:
        col_headers += " Refusal ↑ |"
        col_sep     += "---------:|"

    header = (
        "\n## Function Calling Benchmark\n\n"
        f"Dataset: [Scicom-intl/Function-Call-TaaS](https://huggingface.co/datasets/Scicom-intl/Function-Call-TaaS)"
        f" — 100 multi-turn Manglish conversations across 9 telco B2B workflows, "
        f"evaluated via full-replay mode (split: `{split}`).\n\n"
        f"{col_headers}\n{col_sep}\n"
    )
    rows = []
    for model, m in ranked:
        ip = m["id_propagation_rate"]
        rr = m.get("refusal_rate")
        row = (
            f"| {model} "
            f"| {m['tool_call_f1']:.3f} "
            f"| {m['name_set_f1']:.3f} "
            f"| {m['json_valid_rate']:.3f} "
            f"| {m['hallucination_rate']:.3f} "
            f"| {m['req_coverage'] or 0.0:.3f} "
            f"| {m['type_accuracy'] or 0.0:.3f} "
            f"| {m['parallel_count_match']:.3f} "
            f"| {ip:.3f} |" if ip is not None else
            f"| {model} "
            f"| {m['tool_call_f1']:.3f} "
            f"| {m['name_set_f1']:.3f} "
            f"| {m['json_valid_rate']:.3f} "
            f"| {m['hallucination_rate']:.3f} "
            f"| {m['req_coverage'] or 0.0:.3f} "
            f"| {m['type_accuracy'] or 0.0:.3f} "
            f"| {m['parallel_count_match']:.3f} "
            f"| n/a |"
        )
        if has_refusal:
            row += f" {rr:.3f} |" if rr is not None else " n/a |"
        rows.append(row)
    table = header + "\n".join(rows) + "\n"

    existing = readme_path.read_text() if readme_path.exists() else ""

    # Replace existing benchmark section or append
    marker = "\n## Function Calling Benchmark"
    if marker in existing:
        existing = existing[:existing.index(marker)]

    readme_path.write_text(existing.rstrip() + table)
    print(f"\nREADME updated: {readme_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def read_model_file(path: str) -> list[str]:
    """Read non-empty, non-comment lines from a model list file."""
    models = []
    try:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                models.append(line)
    except FileNotFoundError:
        print(f"[error] model file not found: {path}", file=sys.stderr)
        sys.exit(1)
    return models


def main():
    script_dir = Path(__file__).parent
    default_model_file = str(script_dir / "model_list")
    # parse config early so we can derive the default checkpoint dir
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--config", default=None)
    _pre_args, _ = _pre.parse_known_args()
    _ckpt_suffix = f"checkpoints-{_pre_args.config}" if _pre_args.config else "checkpoints"
    default_ckpt_dir = str(script_dir / _ckpt_suffix)
    default_readme = str(script_dir / "README.md")

    parser = argparse.ArgumentParser(
        description="Evaluate function calling on Scicom-intl/Function-Call-TaaS"
    )
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "none"))
    parser.add_argument("--model", default=None, help="Single model to evaluate (overrides --model-file)")
    parser.add_argument("--model-file", default=default_model_file, help="File with one model per line")
    parser.add_argument("--all-models", action="store_true", help="Run all models in --model-file")
    parser.add_argument("--model-workers", type=int, default=3, help="Models running in parallel (default: 3)")
    parser.add_argument("--split", default="train", help="HuggingFace dataset split (default: train)")
    parser.add_argument("--config", default=None, help="HuggingFace dataset config/subset (e.g. test-extra)")
    parser.add_argument("--max-conversations", type=int, default=0, help="0 = all")
    parser.add_argument("--workers", type=int, default=4, help="Parallel conversations per model (default: 4)")
    parser.add_argument("--checkpoint-dir", default=default_ckpt_dir)
    parser.add_argument("--output", default=None, help="Output JSON (single-model only)")
    parser.add_argument("--readme", default=default_readme, help="README path to update after all-models run")
    parser.add_argument("--dry-run", action="store_true",
                        help="5 conversations, first model from --model-file")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    ckpt_dir = Path(args.checkpoint_dir)

    # ---- dry-run ----
    if args.dry_run:
        models = read_model_file(args.model_file)
        model = models[0]
        print(f"[dry-run] model={model}  conversations=5")
        pairs = load_hf_data(args.split, 5, args.config)
        conv_results, metrics = run_model_eval(client, model, pairs, args.workers, ckpt_dir)
        output_path = args.output or f"{model.replace('/', '_')}_results.json"
        Path(output_path).write_text(json.dumps({
            "model": model, "dry_run": True,
            "metrics": metrics, "conversations": conv_results,
        }, indent=2))
        print(f"Saved to {output_path}")
        return

    pairs = load_hf_data(args.split, args.max_conversations, args.config)

    # ---- all models ----
    if args.all_models:
        models = read_model_file(args.model_file)
        print(f"Running {len(models)} models  (model-workers={args.model_workers}  "
              f"conv-workers={args.workers})\n")

        all_metrics: list[tuple[str, dict]] = []
        lock = __import__("threading").Lock()

        def _eval_model(model: str) -> tuple[str, dict]:
            conv_results, metrics = run_model_eval(
                client, model, pairs, args.workers, ckpt_dir, prefix=f"  "
            )
            out = Path(f"{model.replace('/', '_')}_results.json")
            out.write_text(json.dumps({
                "model": model, "metrics": metrics, "conversations": conv_results,
            }, indent=2))
            return model, metrics

        with ThreadPoolExecutor(max_workers=args.model_workers) as pool:
            futures = {pool.submit(_eval_model, m): m for m in models}
            for fut in as_completed(futures):
                model_name = futures[fut]
                try:
                    m, metrics = fut.result()
                    with lock:
                        all_metrics.append((m, metrics))
                except Exception as e:
                    print(f"[error] {model_name}: {e}", file=sys.stderr)

        # Comparison table
        ranked = sorted(all_metrics, key=lambda x: x[1]["tool_call_f1"], reverse=True)
        print("\n" + "=" * 80)
        print(f"  All-model comparison (sorted by Tool F1)")
        print("=" * 80)
        print(f"  {'Model':<40} {'Tool F1':>8} {'Name F1':>8} {'JSON':>6} {'Hallu':>6} {'Parallel':>9}")
        print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*6} {'-'*6} {'-'*9}")
        for m, met in ranked:
            print(f"  {m:<40} {met['tool_call_f1']:>8.3f} {met['name_set_f1']:>8.3f} "
                  f"{met['json_valid_rate']:>6.3f} {met['hallucination_rate']:>6.3f} "
                  f"{met['parallel_count_match']:>9.3f}")
        print("=" * 80)

        update_readme(all_metrics, Path(args.readme), args.split)
        return

    # ---- single model ----
    if args.model:
        model = args.model
    else:
        models = read_model_file(args.model_file)
        model = models[0]
        print(f"[info] using first model: {model}")

    conv_results, metrics = run_model_eval(client, model, pairs, args.workers, ckpt_dir)
    output_path = args.output or f"{model.replace('/', '_')}_results.json"
    Path(output_path).write_text(json.dumps({
        "model": model, "split": args.split,
        "metrics": metrics, "conversations": conv_results,
    }, indent=2))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
