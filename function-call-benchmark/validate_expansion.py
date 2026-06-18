"""
Validate LLM-expanded function files against expand_spec.md.

Usage:
    python3 validate_expansion.py 0 1 2 ...      # validate specific row indices
    python3 validate_expansion.py --range 0 12   # validate [0, 12)
    python3 validate_expansion.py --all          # validate every output file present

Checks per row (output train-function/{i}.json vs source train-function-src/{i}.json):
  1. Output JSON parses.
  2. functions length in [30, 50].
  3. Each function has name, description, stage, parameters(type=object), returns.
  4. Function names unique within the file.
  5. No dangling $ref (resolved by the shared-entity KEY, the segment right
     after `#/shared_entities/`; deep refs into sub-properties are allowed).
  6. Every ORIGINAL function name is still present.
  7. workflow_name / description / domain preserved verbatim.
  8. Every original shared_entities KEY preserved (additions allowed, no renames).

Prints one line per row (PASS / FAIL: reasons) and a summary. Exit code is the
number of failing rows (0 = all good), so callers can branch on success.
"""

import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "train-function")
SRC_DIR = os.path.join(HERE, "train-function-src")


def check(i):
    """Return a list of problem strings for row i ([] == pass)."""
    problems = []
    out_path = os.path.join(OUT_DIR, f"{i}.json")
    src_path = os.path.join(SRC_DIR, f"{i}.json")
    if not os.path.exists(out_path):
        return [f"output file missing: {out_path}"]
    try:
        with open(out_path) as fh:
            d = json.load(fh)
    except Exception as e:
        return [f"output JSON does not parse: {e}"]
    with open(src_path) as fh:
        src = json.load(fh)

    fs = d.get("functions")
    if not isinstance(fs, list):
        return ["'functions' is not a list"]
    if not (30 <= len(fs) <= 50):
        problems.append(f"function count {len(fs)} not in [30,50]")

    se = d.get("shared_entities", {})
    if not isinstance(se, dict):
        problems.append("'shared_entities' is not an object")
    se_keys = set(se.keys()) if isinstance(se, dict) else set()

    names = []
    for f in fs:
        if not isinstance(f, dict):
            problems.append("a function is not an object")
            continue
        for k in ("name", "description", "stage", "parameters", "returns"):
            if k not in f:
                problems.append(f"function {f.get('name')!r} missing {k}")
        names.append(f.get("name"))
        params = f.get("parameters")
        if not isinstance(params, dict) or params.get("type") != "object":
            problems.append(f"function {f.get('name')!r} parameters.type != 'object'")

    dup = {n for n in names if names.count(n) > 1}
    if dup:
        problems.append(f"duplicate function names: {sorted(dup)}")

    # dangling $ref (deep-ref aware)
    for ref in set(re.findall(r"#/shared_entities/[A-Za-z0-9_/\-]+", json.dumps(d))):
        parts = ref.split("/")
        key = parts[2] if len(parts) > 2 else ""
        if key not in se_keys:
            problems.append(f"dangling $ref {ref}")

    # originals preserved
    name_set = set(names)
    for o in src["functions"]:
        if o["name"] not in name_set:
            problems.append(f"original function lost: {o['name']}")

    # header verbatim
    for k in ("workflow_name", "description", "domain"):
        if d.get(k) != src.get(k):
            problems.append(f"header field {k!r} changed")

    # original shared_entities keys preserved
    for k in src.get("shared_entities", {}):
        if k not in se_keys:
            problems.append(f"original shared_entity key lost: {k}")

    return problems


def main():
    args = sys.argv[1:]
    if not args:
        print("give indices, --range A B, or --all", file=sys.stderr)
        sys.exit(2)
    if args[0] == "--all":
        indices = sorted(int(os.path.basename(p)[:-5]) for p in glob.glob(os.path.join(OUT_DIR, "*.json")))
    elif args[0] == "--range":
        indices = list(range(int(args[1]), int(args[2])))
    else:
        indices = [int(a) for a in args]

    failed = []
    counts = []
    for i in indices:
        problems = check(i)
        out_path = os.path.join(OUT_DIR, f"{i}.json")
        if os.path.exists(out_path):
            try:
                counts.append(len(json.load(open(out_path))["functions"]))
            except Exception:
                pass
        if problems:
            failed.append(i)
            print(f"{i}: FAIL: " + " | ".join(problems))
        else:
            print(f"{i}: PASS")
    print(f"--- {len(indices)} checked, {len(failed)} failed"
          + (f", func counts {min(counts)}-{max(counts)}" if counts else ""))
    if failed:
        print("FAILED INDICES:", failed)
    sys.exit(len(failed))


if __name__ == "__main__":
    main()
