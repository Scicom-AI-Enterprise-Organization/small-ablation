#!/usr/bin/env python3
"""
Multi-turn, tool-calling synthetic-conversation generator with reasoning capture.

Drives a role-play loop through an OpenAI-compatible endpoint to build a
function-calling conversation turn-by-turn, storing the reasoning behind every
step:

    1. Given a function/tool library, a USER simulator writes user turn 1.
    2. An ASSISTANT (real `tools=[...]`, `tool_choice="auto"`) replies and may
       emit tool_calls.
    3. A TOOL simulator returns plausible synthetic JSON tool responses.
    4. The USER simulator writes the next user turn reacting to the assistant.
    5. Repeat for N turns.

Reasoning is captured for *every* message and stored inline as a `reasoning`
key on each message:
    - USER / TOOL roles: asked to return {"reasoning": ..., "message"/"response": ...}
    - ASSISTANT role: native `reasoning_content` / `reasoning` channel when the
      model exposes it; otherwise a one-off "planner" call reconstructs the
      reasoning behind the chosen tool_calls.

Default conversation language is Malay ("ms"); override with --language.

Reference conventions reused from evaluate_function_calling.py:
  * $ref resolution (#/shared_entities/<name>) when building OpenAI tools
  * .env loading from the script dir or its parent
  * OpenAI() client with --base-url / OPENAI_BASE_URL

Examples
--------
Single conversation from a function-library JSON file:

    python3 gen_multiturn_reasoning.py \
        --tools test-function/0.json \
        --turns 8 --language ms \
        --model z-ai/glm-5.1 \
        --out out/conv0.json

Single conversation from one parquet row (the `functions` column):

    python3 gen_multiturn_reasoning.py \
        --parquet test-function-merged-00000-of-00001.parquet --row 0 \
        --turns 8 --model z-ai/glm-5.1 --out out/row0.json

Batch over every row of a parquet -> per-row JSON + a merged parquet:

    python3 gen_multiturn_reasoning.py \
        --parquet test-function-merged-00000-of-00001.parquet \
        --turns 12 --model z-ai/glm-5.1 \
        --out-dir out/multiturn-reasoning \
        --out-parquet test-function-multiturn-reasoning-00000-of-00001.parquet \
        --skip-existing
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI


# ---------------------------------------------------------------------------
# .env loading (mirrors evaluate_function_calling.py)
# ---------------------------------------------------------------------------
def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    script_dir = Path(__file__).parent
    for candidate in (script_dir / ".env", script_dir.parent / ".env"):
        if candidate.exists():
            load_dotenv(candidate)
            return


_load_dotenv()


# ---------------------------------------------------------------------------
# Tool / schema helpers (ported from evaluate_function_calling.py)
# ---------------------------------------------------------------------------
def resolve_refs(schema: Any, shared: dict) -> Any:
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
    """Convert a function library to an OpenAI tool list with $refs resolved.

    Returns (tools, fn_schema_map, fn_names).
    """
    shared = lib.get("shared_entities", {}) or {}
    tools: list[dict] = []
    fn_schema_map: dict[str, dict] = {}
    for fn in lib["functions"]:
        resolved = resolve_refs(copy.deepcopy(fn), shared)
        params = resolved.get("parameters", {"type": "object", "properties": {}})
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": resolved["name"],
                    "description": resolved.get("description", ""),
                    "parameters": params,
                },
            }
        )
        fn_schema_map[resolved["name"]] = resolved
    return tools, fn_schema_map, set(fn_schema_map.keys())


_TYPE_MAP = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def required_coverage(args: dict, fn_schema: dict) -> float:
    required = fn_schema.get("parameters", {}).get("required", [])
    if not required:
        return 1.0
    return sum(1 for r in required if r in args) / len(required)


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
    return correct / total if total else 1.0


# ---------------------------------------------------------------------------
# Robust JSON extraction (some endpoints ignore response_format)
# ---------------------------------------------------------------------------
def _find_balanced_json(text: str, start: int) -> Optional[Any]:
    """Parse the first balanced {...} or [...] at/after `start`. String-aware."""
    n = len(text)
    i = start
    while i < n and text[i] not in "{[":
        i += 1
    if i >= n:
        return None
    open_ch = text[i]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_str = False
    esc = False
    j = i
    while j < n:
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[i : j + 1])
                    except json.JSONDecodeError:
                        return None
        j += 1
    return None


def extract_json(text: str) -> Optional[Any]:
    """Best-effort: parse `text` as JSON, tolerating ```json fences / prose."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t[:4].lower() == "json":
            t = t[4:]
        t = t.strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    return _find_balanced_json(t, 0)


# ---------------------------------------------------------------------------
# Language register guidance
# ---------------------------------------------------------------------------
LANGUAGE_NAMES = {
    "ms": "Bahasa Malaysia",
    "en": "English",
    "ta": "Malaysian Tamil",
    "zh": "Malaysian Mandarin",
}

FORBIDDEN_MS = ("Tuan", "bro", "boss", "machi", "machan", "padu", "syiok")


def lang_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


# ---------------------------------------------------------------------------
# User sentiment profiles (selectable via --sentiment)
# ---------------------------------------------------------------------------
SENTIMENTS = {
    "neutral": "matter-of-fact and businesslike; no strong emotion.",
    "polite": "warm, courteous and appreciative; thanks the agent for the help.",
    "frustrated": "politely annoyed; notes the problem has dragged on or recurred, but stays civil.",
    "angry": "upset and demanding; terse, wants it fixed NOW — but no profanity and still within call-centre decorum.",
    "urgent": "time-pressured; stresses a hard deadline and pushes the agent to move fast.",
    "impatient": "wants quick, short answers; lightly hurries or interrupts the agent.",
    "confused": "unsure and hesitant; asks for clarification and occasionally mixes up details.",
    "satisfied": "pleased and cooperative; compliments the agent's help.",
    "anxious": "worried about consequences (cost, compliance, deadline); seeks reassurance.",
}
REAL_SENTIMENTS = list(SENTIMENTS)

# Appended to the tool simulator instruction when an error is to be injected.
ERROR_INSTRUCTION = (
    "\n\nIMPORTANT: For THIS call, return a REALISTIC API FAILURE instead of a success. "
    "Pick a failure mode that fits this function and these arguments: "
    "400 validation (bad field/enum/date), 401 auth/MFA-required, 403 permission/scope, "
    "404 not-found/wrong-tenant, 409 version-conflict/etag/idempotency, 413 payload-too-large, "
    "429 rate-limit (include retry_after_seconds), 500 internal (include incident_id), "
    "503 maintenance, 504 gateway-timeout (suggest checkpoint_token/narrower window), "
    "202 async (include polling_url + webhook_supported), partial-success (mixed per-item statuses), "
    "eventually-consistent (200 + staleness warning), deprecation (200 + sunset_date), or "
    "quota-near-exhaustion (200 + low quota_remaining). "
    "Shape hard errors as {\"error\": {\"http_status\": <int>, \"code\": \"<short>\", \"message\": \"<human>\", ...}}; "
    "shape degraded successes as a normal object plus the relevant warning/async fields. "
    "Keep it plausible for this specific function."
)


def register_note(code: str) -> str:
    if code == "ms":
        return (
            "polite Malaysian register; you may code-switch with English technical "
            "terms (Manglish). Malaysian context: MYR, +60 phone numbers, 2026 dates"
        )
    if code == "en":
        return "polite Malaysian English / professional register"
    return f"polite Malaysian {lang_name(code)} register; code-switch English tech terms inline"


# ---------------------------------------------------------------------------
# OpenAI call wrappers (tenacity-style retry, like evaluate_function_calling.py)
# ---------------------------------------------------------------------------
def prefix_cache_extra(args) -> dict:
    """When --disable-prefix-cache is set, return a unique vLLM `cache_salt` so
    no two requests share a cached prefix (the client-side way to opt out of
    prefix-cache reuse). Empty otherwise so the default path is untouched."""
    if getattr(args, "disable_prefix_cache", False):
        return {"cache_salt": uuid.uuid4().hex}
    return {}


# Per-call resilience. The serverless gateway cold-starts (504) and rate-limits
# (429); we retry hard with capped exponential backoff. If a call still fails
# after this many attempts, the row-level retry loop re-attempts the whole row.
_RETRY_ATTEMPTS = 12
_BACKOFF_CAP_S = 30.0


def _retry_call(fn, *, retries: int = _RETRY_ATTEMPTS, what: str = "call"):
    last = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt == retries - 1:
                break
            time.sleep(min(_BACKOFF_CAP_S, 2 ** min(attempt, 5)))
    raise RuntimeError(f"{what} failed after {retries} attempts: {last}")


def chat_text(client: OpenAI, model: str, messages: list, *, temperature: float,
              max_tokens: int, seed: Optional[int] = None,
              extra_body: Optional[dict] = None) -> str:
    def _do():
        kw: dict = dict(model=model, messages=messages, temperature=temperature,
                        max_tokens=max_tokens)
        if seed is not None:
            kw["seed"] = seed
        if extra_body:
            kw["extra_body"] = extra_body
        resp = client.chat.completions.create(**kw)
        return resp.choices[0].message.content or ""
    return _retry_call(_do, what="chat_text")


def chat_json(client: OpenAI, model: str, messages: list, *, temperature: float,
              max_tokens: int, seed: Optional[int] = None,
              extra_body: Optional[dict] = None) -> Any:
    """Chat call expected to return a JSON object. Tries response_format json,
    falls back to plain + robust extraction."""
    def _do(use_rf: bool):
        kw: dict = dict(model=model, messages=messages, temperature=temperature,
                        max_tokens=max_tokens)
        if seed is not None:
            kw["seed"] = seed
        if extra_body:
            kw["extra_body"] = extra_body
        if use_rf:
            kw["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kw)
        return resp.choices[0].message.content or ""

    # First try with response_format; if the endpoint rejects it, retry without.
    try:
        text = _retry_call(lambda: _do(True), retries=2, what="chat_json(rf)")
    except Exception:
        text = _retry_call(lambda: _do(False), what="chat_json")
    obj = extract_json(text)
    if obj is None:
        # one more plain attempt
        text = _retry_call(lambda: _do(False), what="chat_json(plain)")
        obj = extract_json(text)
    return obj


def extract_reasoning(msg: Any) -> str:
    """Pull the native chain-of-thought off a chat message, if present."""
    for attr in ("reasoning_content", "reasoning"):
        val = getattr(msg, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    extra = getattr(msg, "model_extra", None) or {}
    for key in ("reasoning_content", "reasoning"):
        val = extra.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


# ---------------------------------------------------------------------------
# Conversation context: convo carries reasoning; API form strips it
# ---------------------------------------------------------------------------
def to_api_messages(system_prompt: str, convo: list[dict]) -> list[dict]:
    """Render the running conversation into clean OpenAI messages (no reasoning)."""
    out: list[dict] = [{"role": "system", "content": system_prompt}]
    for m in convo:
        role = m["role"]
        if role == "user":
            out.append({"role": "user", "content": m.get("content", "")})
        elif role == "assistant":
            am: dict = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("tool_calls"):
                am["tool_calls"] = m["tool_calls"]
            out.append(am)
        elif role == "tool":
            out.append({
                "role": "tool",
                "tool_call_id": m["tool_call_id"],
                "name": m.get("name", ""),
                "content": m.get("content", ""),
            })
    return out


def render_transcript(convo: list[dict]) -> str:
    """Compact human-readable transcript for the user/tool simulators."""
    lines: list[str] = []
    for m in convo:
        role = m["role"]
        if role == "user":
            lines.append(f"USER: {m.get('content', '')}")
        elif role == "assistant":
            txt = m.get("content") or ""
            if m.get("tool_calls"):
                names = ", ".join(tc["function"]["name"] for tc in m["tool_calls"])
                txt = (txt + f" [called tools: {names}]").strip()
            lines.append(f"ASSISTANT: {txt}")
        elif role == "tool":
            lines.append(f"TOOL[{m.get('name','')}]: {m.get('content','')[:600]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ID continuity: harvest id-like fields from tool responses so later turns can
# reuse the exact values instead of inventing fresh ones.
# ---------------------------------------------------------------------------
_ID_KEY_RE = re.compile(r"(^id$|_id$|_ref$|_token$|_number$|_no$)", re.IGNORECASE)


def extract_ids(obj: Any, pairs: Optional[list] = None) -> list:
    """Walk a parsed JSON tool response and collect (key, value) for id-like
    string fields, recursively."""
    if pairs is None:
        pairs = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and v and _ID_KEY_RE.search(k):
                pairs.append((k, v))
            else:
                extract_ids(v, pairs)
    elif isinstance(obj, list):
        for item in obj:
            extract_ids(item, pairs)
    return pairs


def update_known_ids(known_ids: dict, tool_content: str) -> None:
    """Merge ids found in one tool response (JSON string) into known_ids
    (key -> list of distinct values seen)."""
    try:
        payload = json.loads(tool_content)
    except (json.JSONDecodeError, TypeError):
        return
    for k, v in extract_ids(payload):
        vals = known_ids.setdefault(k, [])
        if v not in vals:
            vals.append(v)


def id_reuse_note(known_ids: dict, per_key: int = 6, max_keys: int = 40) -> str:
    """Render a 'reuse these exact ids' block for the assistant/tool prompts."""
    if not known_ids:
        return ""
    parts = [f"{k}={', '.join(vals[:per_key])}" for k, vals in list(known_ids.items())[:max_keys]]
    return ("\n\nEstablished entity IDs already in play — REUSE these EXACT values when "
            "the conversation refers to the same entity; do NOT invent a new id for an "
            "entity that already exists:\n" + "; ".join(parts))


def capability_summary(lib: dict, limit_chars: int = 160) -> str:
    out = []
    for fn in lib["functions"]:
        desc = (fn.get("description") or "").strip().replace("\n", " ")
        if len(desc) > limit_chars:
            desc = desc[:limit_chars].rstrip() + "…"
        out.append(f"- {fn['name']}: {desc}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Role prompts
# ---------------------------------------------------------------------------
def user_sim_system(lib: dict, language: str) -> str:
    return f"""You simulate a realistic {lang_name(language)} end-user / customer talking to an AI call-centre assistant.

Domain: "{lib.get('domain', '')}"   Workflow: "{lib.get('workflow', '')}"

The assistant has access to these backend tools/capabilities:
{capability_summary(lib)}

Your job: write the NEXT user message only.
- On the FIRST turn: open with a concrete, realistic request that naturally needs one or more of those tools. Give concrete details (IDs, amounts, dates) so the assistant can act. Use Malaysian context (MYR, +60 numbers, 2026 dates, real-ish names/companies).
- On LATER turns: react to the assistant's latest reply, then advance the workflow to a NEW sub-task (prefer a capability not used yet when natural). You may add light, realistic human messiness (a follow-up, a small correction, urgency) but stay coherent and goal-directed.
- Speak naturally in {lang_name(language)} ({register_note(language)}).
- You are the USER. Never call tools, never produce tool output, never speak as the assistant.

Respond with ONLY a JSON object:
{{"reasoning": "<1-3 sentences: why you say this next>", "message": "<the user's message text>"}}"""


def assistant_system(lib: dict, language: str) -> str:
    style = ""
    if language == "ms":
        style = " Use Encik/Puan/Sir/Madam. Never use 'Tuan' or slang (bro/boss/machi/machan/padu/syiok)."
    return f"""You are a professional Malaysian call-centre assistant for domain "{lib.get('domain','')}" (workflow "{lib.get('workflow','')}").

You have function tools available. Use them to fulfil the user's requests:
- Call the appropriate tool(s) with arguments that satisfy each function's schema: correct names, all required fields, valid enum values, correct nesting.
- PARALLEL TOOL CALLS: when the user asks for several things at once, or when multiple lookups/actions are independent of each other, issue them as MULTIPLE tool_calls in a SINGLE assistant message rather than one at a time. Only serialise when a later call genuinely needs an earlier call's result.
- REUSE ids and values returned by earlier tool results — never invent a new id for an entity that already exists; pass the exact id strings back into later tool calls.
- After tool results arrive, either call more tools or give the user a concise, accurate answer.
- If a tool returns an error (4xx/5xx) or a degraded/partial/async result, recover gracefully: surface the problem, fix bad arguments and retry, back off on rate limits, escalate on permission issues, or offer a workaround — don't pretend it succeeded.
- Stay calm, empathetic and professional regardless of the customer's tone; de-escalate frustration without losing accuracy.
- Register: formal and polite.{style}

Reply in {lang_name(language)}; technical terms and IDs may stay in English."""


def tool_sim_system(lib: dict) -> str:
    return f"""You simulate the backend system that EXECUTES function calls for domain "{lib.get('domain','')}".

Given one function call (name + arguments), its schema, and its documented "returns", produce a REALISTIC, plausible JSON response exactly as that backend tool would return.
- The response must be valid JSON and consistent with the function's "returns" description and the arguments passed.
- Keep continuity with the conversation: REUSE ids/values already established earlier; mint new realistic ids ONLY for newly-created entities; timestamps in 2026 moving forward; MYR currency; Malaysian context.
- Mostly success responses; you may include realistic warnings or partial fields when natural. Keep it concise — no prose outside the JSON.

Respond with ONLY a JSON object:
{{"reasoning": "<1-2 sentences: what you returned and why>", "response": <the tool's JSON response object>}}"""


# ---------------------------------------------------------------------------
# Role calls
# ---------------------------------------------------------------------------
def gen_user_turn(client, model, lib, language, convo, turn, sentiment, args) -> dict:
    sys_prompt = user_sim_system(lib, language)
    if turn == 1:
        instr = "Begin the conversation. Write the user's opening message."
    else:
        instr = ("Here is the conversation so far:\n\n"
                 + render_transcript(convo)
                 + "\n\nWrite the user's next message.")
    if sentiment in SENTIMENTS:
        instr += f"\n\nEmotional tone for this message: {sentiment} — {SENTIMENTS[sentiment]}"
    messages = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": instr}]
    obj = chat_json(client, model, messages, temperature=args.user_temperature,
                    max_tokens=args.max_tokens, seed=args.seed,
                    extra_body=prefix_cache_extra(args))
    if not isinstance(obj, dict) or "message" not in obj:
        # Last-resort: treat any text as the message with empty reasoning.
        text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
        return {"reasoning": "", "content": str(text)}
    return {"reasoning": str(obj.get("reasoning", "")), "content": str(obj["message"])}


def planner_reasoning(client, model, lib, language, convo, tool_calls, content, args) -> str:
    if tool_calls:
        desc = "call these tools: " + ", ".join(
            f"{tc['function']['name']}({tc['function']['arguments']})" for tc in tool_calls
        )
    else:
        desc = f"reply without calling any tool: {content!r}"
    messages = [
        {"role": "system", "content": "You explain a call-centre assistant's reasoning concisely and factually."},
        {"role": "user", "content": (
            f"Conversation so far:\n\n{render_transcript(convo)}\n\n"
            f"The assistant decided to {desc}.\n"
            "In 1-3 sentences, state the reasoning that led to this decision. "
            "Output ONLY the reasoning text, no preamble.")},
    ]
    try:
        return chat_text(client, model, messages, temperature=0.3,
                         max_tokens=400, seed=args.seed,
                         extra_body=prefix_cache_extra(args)).strip()
    except Exception:
        return ""


def gen_assistant_turn(client, model, lib, language, tools, convo, args, known_ids=None) -> dict:
    sys_prompt = assistant_system(lib, language) + id_reuse_note(known_ids or {})
    api_messages = to_api_messages(sys_prompt, convo)

    def _do():
        kw: dict = dict(model=model, messages=api_messages,
                        temperature=args.assistant_temperature,
                        max_tokens=args.max_tokens)
        # Only advertise tools when we actually have some. Sending tools=[] with
        # tool_choice="auto" makes some vLLM builds return an error body with no
        # `choices`, which crashed the final no-tools summary call.
        if tools:
            kw["tools"] = tools
            kw["tool_choice"] = "auto"
            if args.parallel_tools:
                # OpenAI-standard knob; servers that honour it will allow >1
                # tool_call per response. Off by default for compat.
                kw["parallel_tool_calls"] = True
        if args.seed is not None:
            kw["seed"] = args.seed
        eb = prefix_cache_extra(args)
        if eb:
            kw["extra_body"] = eb
        resp = client.chat.completions.create(**kw)
        if not getattr(resp, "choices", None):
            # Malformed/empty upstream body (e.g. a 200 carrying an error obj).
            raise RuntimeError(f"no choices in response: {str(resp)[:300]}")
        return resp

    # Let a persistently-failing call propagate: in batch mode the row-level retry
    # loop regenerates the whole conversation rather than baking an empty turn
    # (honours "always retry until success"). _do already raises on an empty body.
    resp = _retry_call(_do, what="assistant")
    msg = resp.choices[0].message
    content = msg.content or ""
    tool_calls = []
    for tc in (msg.tool_calls or []):
        tool_calls.append({
            "id": tc.id,
            "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
        })

    reasoning = extract_reasoning(msg)
    source = "native"
    if not reasoning and not args.no_native_reasoning:
        # Native channel empty for this model -> reconstruct via planner.
        reasoning = planner_reasoning(client, model, lib, language, convo, tool_calls, content, args)
        source = "planner" if reasoning else "none"
    elif args.no_native_reasoning:
        reasoning = planner_reasoning(client, model, lib, language, convo, tool_calls, content, args)
        source = "planner" if reasoning else "none"

    return {"role": "assistant", "content": content, "tool_calls": tool_calls,
            "reasoning": reasoning, "_reasoning_source": source}


def gen_tool_response(client, model, lib, fn_schema_map, convo, tool_call, args,
                      force_error: bool = False, known_ids=None) -> dict:
    name = tool_call["function"]["name"]
    raw_args = tool_call["function"]["arguments"]
    schema = fn_schema_map.get(name, {})
    sys_prompt = tool_sim_system(lib) + id_reuse_note(known_ids or {})
    instr = (
        f"Conversation so far:\n\n{render_transcript(convo)}\n\n"
        f"Now simulate the response for this function call.\n"
        f"Function name: {name}\n"
        f"Arguments (JSON): {raw_args}\n"
        f"Function schema: {json.dumps(schema, ensure_ascii=False)[:4000]}\n"
        f"Documented returns: {schema.get('returns', '')}\n"
    )
    if force_error:
        instr += ERROR_INSTRUCTION
    messages = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": instr}]
    obj = chat_json(client, model, messages, temperature=args.tool_temperature,
                    max_tokens=args.max_tokens, seed=args.seed,
                    extra_body=prefix_cache_extra(args))
    if isinstance(obj, dict) and "response" in obj:
        reasoning = str(obj.get("reasoning", ""))
        response = obj["response"]
    elif isinstance(obj, dict):
        reasoning = str(obj.get("reasoning", ""))
        # Some models put the payload at top level without a "response" key.
        response = {k: v for k, v in obj.items() if k != "reasoning"} or obj
    else:
        reasoning = ""
        response = obj if obj is not None else {"status": "ok"}
    content = json.dumps(response, ensure_ascii=False)
    return {"role": "tool", "tool_call_id": tool_call["id"], "name": name,
            "content": content, "reasoning": reasoning}


# ---------------------------------------------------------------------------
# Conversation driver
# ---------------------------------------------------------------------------
def generate_conversation(client, lib, args, rng, tag: str = "") -> dict:
    tools, fn_schema_map, fn_names = build_openai_tools(lib)
    language = args.language
    convo: list[dict] = []
    reasoning_sources: list[str] = []
    known_ids: dict[str, list] = {}   # id-like values harvested from tool responses
    max_parallel = 0                  # most tool_calls in any one assistant message

    user_model = args.user_model or args.model
    asst_model = args.assistant_model or args.model
    tool_model = args.tool_model or args.model

    # Resolve the conversation-level sentiment ('random' picks one per conversation).
    # `rng` is a per-conversation random.Random so concurrent rows stay reproducible.
    if args.sentiment == "random":
        conv_sentiment = rng.choice(REAL_SENTIMENTS)
    else:
        conv_sentiment = args.sentiment
    sentiments_used: list[str] = []

    for turn in range(1, args.turns + 1):
        # Per-turn sentiment: 'mixed' rerolls every turn, otherwise it's constant.
        if conv_sentiment == "mixed":
            turn_sentiment = rng.choice(REAL_SENTIMENTS)
        else:
            turn_sentiment = conv_sentiment if conv_sentiment in SENTIMENTS else "neutral"
        sentiments_used.append(turn_sentiment)

        # 1. USER
        u = gen_user_turn(client, user_model, lib, language, convo, turn, turn_sentiment, args)
        convo.append({"role": "user", "content": u["content"], "reasoning": u["reasoning"]})

        # 2. ASSISTANT (+ tool rounds)
        rounds = 0
        while True:
            a = gen_assistant_turn(client, asst_model, lib, language, tools, convo, args, known_ids)
            reasoning_sources.append(a.pop("_reasoning_source"))
            convo.append(a)
            if not a["tool_calls"]:
                break
            max_parallel = max(max_parallel, len(a["tool_calls"]))
            # 3. TOOL responses (synthetic), one per call. With --simulate-errors,
            # each call independently has --error-rate chance of a failure mode.
            for tc in a["tool_calls"]:
                force_err = args.simulate_errors and rng.random() < args.error_rate
                t = gen_tool_response(client, tool_model, lib, fn_schema_map, convo, tc, args,
                                      force_error=force_err, known_ids=known_ids)
                convo.append(t)
                update_known_ids(known_ids, t["content"])   # harvest ids for later reuse
            rounds += 1
            if rounds >= args.max_tool_rounds:
                # Give the assistant one final chance to summarise without tools.
                final = gen_assistant_turn(client, asst_model, lib, language, [], convo, args, known_ids)
                reasoning_sources.append(final.pop("_reasoning_source"))
                final["tool_calls"] = []
                convo.append(final)
                break
        print(f"  {tag}turn {turn}/{args.turns} done "
              f"({sum(1 for m in convo if m['role']=='assistant')} asst, "
              f"{sum(1 for m in convo if m['role']=='tool')} tool msgs)", file=sys.stderr)

    metadata = build_metadata(lib, convo, fn_names, fn_schema_map, args, reasoning_sources,
                              conv_sentiment, sentiments_used, max_parallel, known_ids)
    return {
        "domain": lib.get("domain", ""),
        "workflow": lib.get("workflow", ""),
        "shared_entities": json.dumps(lib.get("shared_entities", {}), ensure_ascii=False),
        "functions": json.dumps(lib.get("functions", []), ensure_ascii=False),
        "messages": convo,
        "metadata": metadata,
    }


def build_metadata(lib, convo, fn_names, fn_schema_map, args, reasoning_sources,
                   conv_sentiment, sentiments_used, max_parallel=0, known_ids=None) -> dict:
    used: list[str] = []
    warnings: list[str] = []
    api_errors: list[dict] = []
    call_ids: set[str] = set()

    for m in convo:
        if m["role"] == "assistant":
            for tc in m.get("tool_calls", []):
                call_ids.add(tc["id"])
                name = tc["function"]["name"]
                if name not in used:
                    used.append(name)
                if name not in fn_names:
                    warnings.append(f"unknown function: {name}")
                try:
                    parsed = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    warnings.append(f"bad JSON args for {name}")
                    parsed = {}
                if isinstance(parsed, dict) and name in fn_schema_map:
                    if required_coverage(parsed, fn_schema_map[name]) < 1.0:
                        warnings.append(f"missing required params for {name}")
        elif m["role"] == "tool":
            if m["tool_call_id"] not in call_ids:
                warnings.append(f"unpaired tool_call_id: {m['tool_call_id']}")
            try:
                payload = json.loads(m.get("content", ""))
            except json.JSONDecodeError:
                warnings.append(f"bad JSON tool content for {m.get('name','')}")
                payload = None
            if isinstance(payload, dict):
                err = payload.get("error")
                status = payload.get("http_status") or payload.get("status_code")
                top_is_err = str(status).isdigit() and int(status) >= 400
                if isinstance(err, dict):
                    api_errors.append({"function": m.get("name", ""),
                                       "http_status": err.get("http_status") or status,
                                       "code": err.get("code")})
                elif top_is_err:
                    api_errors.append({"function": m.get("name", ""), "http_status": status,
                                       "code": payload.get("code")})

    if args.language == "ms":
        text = " ".join((m.get("content") or "") for m in convo)
        for tok in FORBIDDEN_MS:
            if tok.lower() in text.lower():
                warnings.append(f"forbidden token: {tok}")
        if not any(v in text for v in ("Encik", "Puan", "Sir", "Madam")):
            warnings.append("no Encik/Puan/Sir/Madam vocative")

    # ID reuse evidence: how many distinct harvested id values get passed back
    # into a later tool call's arguments.
    known_ids = known_ids or {}
    all_id_values = {v for vals in known_ids.values() for v in vals}
    call_args_blob = " ".join(
        tc["function"].get("arguments", "")
        for m in convo if m["role"] == "assistant"
        for tc in m.get("tool_calls", [])
    )
    ids_reused = sorted(v for v in all_id_values if v in call_args_blob)

    return {
        "num_turns": sum(1 for m in convo if m["role"] == "user"),
        "functions_used": used,
        "num_functions_used": len(used),
        "language": args.language,
        "model": args.model,
        "user_sentiment": conv_sentiment,
        "sentiment_per_turn": sentiments_used,
        "simulate_errors": args.simulate_errors,
        "error_rate": args.error_rate if args.simulate_errors else 0.0,
        "api_errors_simulated": api_errors,
        "max_parallel_tool_calls": max_parallel,
        "distinct_ids_seen": len(all_id_values),
        "ids_reused_in_calls": ids_reused,
        "reasoning_source": reasoning_sources,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "validation_warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------
def load_library_from_file(path: str) -> dict:
    data = json.load(open(path))
    if isinstance(data, list):
        return {"functions": data}
    if "functions" not in data:
        raise ValueError(f"{path}: no 'functions' key and not a list")
    return data


def load_library_from_parquet_row(row) -> dict:
    def _maybe_json(v, default):
        if v is None:
            return default
        if isinstance(v, (dict, list)):
            return v
        try:
            return json.loads(v)
        except (json.JSONDecodeError, TypeError):
            return default

    return {
        "domain": row.get("domain", "") if hasattr(row, "get") else row["domain"],
        "workflow": row.get("workflow", "") if hasattr(row, "get") else row["workflow"],
        "shared_entities": _maybe_json(row.get("shared_entities"), {}),
        "functions": _maybe_json(row.get("functions"), []),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def make_rng(seed: Optional[int], offset: int = 0) -> random.Random:
    """Per-conversation RNG. With --seed, row i is reproducible regardless of
    concurrency (seed+i); without it, draws fresh system entropy."""
    return random.Random(seed + offset) if seed is not None else random.Random()


def write_single(conv: dict, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(conv, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"[wrote] {out_path}")


def conv_to_parquet_row(conv: dict) -> dict:
    return {
        "domain": conv["domain"],
        "workflow": conv["workflow"],
        "shared_entities": conv["shared_entities"],
        "functions": conv["functions"],
        "messages": json.dumps(conv["messages"], ensure_ascii=False),
        "metadata": json.dumps(conv["metadata"], ensure_ascii=False),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multi-turn tool-calling synthetic conversation generator with reasoning capture.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_argument_group("input (choose --tools OR --parquet)")
    src.add_argument("--tools", help="Path to a function-library JSON (object with functions/shared_entities, or a bare functions array).")
    src.add_argument("--parquet", help="Path to a parquet whose `functions` column holds the library.")
    src.add_argument("--row", type=int, help="Single row index of --parquet -> one conversation.")

    gen = p.add_argument_group("generation")
    gen.add_argument("--turns", type=int, default=8, help="Number of user turns.")
    gen.add_argument("--language", default="ms", help="Conversation language code (ms, en, ta, zh, ...).")
    gen.add_argument("--max-tool-rounds", type=int, default=3, help="Max tool-call rounds per assistant turn.")
    gen.add_argument("--no-native-reasoning", action="store_true", help="Always reconstruct assistant reasoning via a planner call.")
    gen.add_argument("--sentiment", default="neutral",
                     help="User emotional tone: one of %s, or 'mixed' (reroll per turn) / 'random' (one per conversation)."
                          % ", ".join(REAL_SENTIMENTS))
    gen.add_argument("--simulate-errors", action="store_true",
                     help="Let the tool simulator inject realistic API failure modes (4xx/5xx/async/partial/...).")
    gen.add_argument("--error-rate", type=float, default=0.25,
                     help="Per tool-call probability of an injected error when --simulate-errors is set.")
    gen.add_argument("--parallel-tools", action="store_true",
                     help="Send the OpenAI `parallel_tool_calls=True` knob so the model may emit >1 tool_call per response (the prompt encourages parallel calls regardless). Off by default for endpoint compatibility.")

    mdl = p.add_argument_group("model / endpoint")
    mdl.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "z-ai/glm-5.1"))
    mdl.add_argument("--user-model", default=None, help="Override model for the user simulator.")
    mdl.add_argument("--assistant-model", default=None, help="Override model for the assistant.")
    mdl.add_argument("--tool-model", default=None, help="Override model for the tool-response simulator.")
    mdl.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"))
    mdl.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    mdl.add_argument("--max-tokens", type=int, default=4096)
    mdl.add_argument("--seed", type=int, default=None)
    mdl.add_argument("--user-temperature", type=float, default=0.9)
    mdl.add_argument("--assistant-temperature", type=float, default=0.4)
    mdl.add_argument("--tool-temperature", type=float, default=0.5)

    out = p.add_argument_group("output")
    out.add_argument("--out", help="Single-mode output JSON path.")
    out.add_argument("--out-dir", help="Batch per-row JSON output dir.")
    out.add_argument("--out-parquet", help="Batch merged parquet output path.")
    out.add_argument("--start", type=int, default=0, help="Batch start row index.")
    out.add_argument("--max-rows", type=int, default=None, help="Batch: max rows to process.")
    out.add_argument("--concurrency", "--parallel", "-j", type=int, default=1, dest="concurrency",
                     help="Batch: number of ROWS (conversations) to generate concurrently. Turns within a row stay sequential.")
    out.add_argument("--skip-existing", action="store_true", help="Batch: reuse an existing per-row JSON instead of regenerating.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if not args.api_key:
        print("error: no API key (set OPENAI_API_KEY or pass --api-key)", file=sys.stderr)
        return 2
    if not args.tools and not args.parquet:
        print("error: provide --tools or --parquet", file=sys.stderr)
        return 2

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    print(f"[config] base_url={args.base_url} model={args.model} "
          f"language={args.language} turns={args.turns} sentiment={args.sentiment} "
          f"simulate_errors={args.simulate_errors}"
          + (f" error_rate={args.error_rate}" if args.simulate_errors else ""),
          file=sys.stderr)

    # ---- single conversation from a tools file ----
    if args.tools:
        lib = load_library_from_file(args.tools)
        conv = generate_conversation(client, lib, args, make_rng(args.seed))
        out = args.out or "out/conversation.json"
        write_single(conv, out)
        _print_summary(conv)
        return 0

    # ---- parquet input ----
    import pandas as pd
    df = pd.read_parquet(args.parquet)

    # single row
    if args.row is not None:
        lib = load_library_from_parquet_row(df.iloc[args.row].to_dict())
        conv = generate_conversation(client, lib, args, make_rng(args.seed, args.row))
        out = args.out or f"out/row{args.row}.json"
        write_single(conv, out)
        _print_summary(conv)
        return 0

    # batch
    if not args.out_dir and not args.out_parquet:
        print("error: batch mode needs --out-dir and/or --out-parquet", file=sys.stderr)
        return 2
    end = len(df) if args.max_rows is None else min(len(df), args.start + args.max_rows)
    indices = list(range(args.start, end))

    def process_row(i: int) -> tuple[int, Optional[dict]]:
        per_row_path = os.path.join(args.out_dir, f"{i}.json") if args.out_dir else None
        if args.skip_existing and per_row_path and os.path.exists(per_row_path):
            print(f"[skip] row {i} (exists)")
            return i, json.load(open(per_row_path))
        print(f"[gen] row {i}")
        try:
            lib = load_library_from_parquet_row(df.iloc[i].to_dict())
            conv = generate_conversation(client, lib, args, make_rng(args.seed, i),
                                         tag=f"row {i} ")
        except Exception as e:  # noqa: BLE001 — one bad row shouldn't kill the batch
            print(f"[error] row {i}: {type(e).__name__}: {e}", file=sys.stderr)
            return i, None
        if per_row_path:
            write_single(conv, per_row_path)
        return i, conv

    results: dict[int, dict] = {}

    def run_pass(todo: list[int]) -> None:
        if args.concurrency > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                futs = [ex.submit(process_row, i) for i in todo]
                for fut in as_completed(futs):
                    i, conv = fut.result()
                    if conv is not None:
                        results[i] = conv
        else:
            for i in todo:
                _, conv = process_row(i)
                if conv is not None:
                    results[i] = conv

    # Retry until success: re-attempt any unfinished row pass after pass, never
    # dropping a row. A pass that makes no progress (e.g. gateway down) sleeps
    # then retries rather than giving up.
    pending = list(indices)
    pass_num = 0
    while pending:
        pass_num += 1
        print(f"[batch] pass {pass_num}: {len(pending)} row(s) to do, concurrency={args.concurrency}")
        before = len(results)
        run_pass(pending)
        pending = [i for i in indices if i not in results]
        if pending:
            gained = len(results) - before
            if gained == 0:
                print(f"[batch] pass {pass_num} made NO progress; {len(pending)} row(s) still "
                      f"failing: {pending}. Retrying in 60s...", file=sys.stderr)
                time.sleep(60)
            else:
                print(f"[batch] pass {pass_num}: +{gained} done, {len(pending)} remaining; retrying...")
    print(f"[batch] all {len(indices)} rows complete after {pass_num} pass(es)")

    # Preserve row order in the merged parquet regardless of completion order.
    rows = [results[i] for i in indices]
    if args.out_parquet:
        pd.DataFrame([conv_to_parquet_row(c) for c in rows]).to_parquet(args.out_parquet, index=False)
        print(f"[wrote] {args.out_parquet} ({len(rows)} rows)")
    return 0


def _print_summary(conv: dict) -> None:
    meta = conv["metadata"]
    print(f"  turns={meta['num_turns']} functions_used={meta['num_functions_used']} "
          f"sentiment={meta['user_sentiment']} "
          f"api_errors={len(meta['api_errors_simulated'])} "
          f"max_parallel_tool_calls={meta.get('max_parallel_tool_calls')} "
          f"ids_reused={len(meta.get('ids_reused_in_calls', []))}/{meta.get('distinct_ids_seen')} "
          f"reasoning_sources={set(meta['reasoning_source'])}")
    if meta["api_errors_simulated"]:
        codes = [e.get("http_status") for e in meta["api_errors_simulated"]]
        print(f"  injected errors: {codes}")
    if meta["validation_warnings"]:
        print(f"  WARNINGS ({len(meta['validation_warnings'])}):")
        for w in meta["validation_warnings"][:20]:
            print(f"    - {w}")
    else:
        print("  validation: clean")


if __name__ == "__main__":
    sys.exit(main())
