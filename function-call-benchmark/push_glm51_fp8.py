"""Pack the per-conversation JSON files in out/glm51-fp8 into a parquet and
push it to the Hugging Face Hub.

Each file under --in-dir is one conversation produced by
gen_multiturn_reasoning.py, with the shape:
    {domain, workflow, shared_entities(str), functions(str), messages(list), metadata(dict)}
We serialise `messages` and `metadata` to JSON strings so the parquet schema
matches the other Function-Call-TaaS subsets (all six columns are strings).

Default run (convert + push):
    python3 push_glm51_fp8.py

Just build the parquet, skip the upload:
    python3 push_glm51_fp8.py --no-push

Override anything:
    python3 push_glm51_fp8.py \
        --in-dir out/glm51-fp8 \
        --out glm5.1-fp8-test-00000-of-00001.parquet \
        --dataset Scicom-intl/Function-Call-TaaS \
        --subset glm5.1-fp8-test --split test
"""

import argparse
import glob
import json
import os
import re
import sys

import pandas as pd

COLUMNS = ["domain", "workflow", "shared_entities", "functions", "messages", "metadata"]


def _numeric_key(path: str):
    """Sort 0.json, 1.json, 2.json ... numerically (fall back to name)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.fullmatch(r"\d+", stem)
    return (0, int(stem)) if m else (1, stem)


def conv_to_row(conv: dict, src: str) -> dict:
    """One conversation dict -> one all-strings parquet row.

    `shared_entities` and `functions` are already JSON strings on disk;
    `messages` (list) and `metadata` (dict) are serialised here.
    """
    return {
        "domain": conv.get("domain", ""),
        "workflow": conv.get("workflow", ""),
        "shared_entities": conv["shared_entities"]
        if isinstance(conv["shared_entities"], str)
        else json.dumps(conv["shared_entities"], ensure_ascii=False),
        "functions": conv["functions"]
        if isinstance(conv["functions"], str)
        else json.dumps(conv["functions"], ensure_ascii=False),
        "messages": json.dumps(conv["messages"], ensure_ascii=False),
        "metadata": json.dumps(conv["metadata"], ensure_ascii=False),
    }


def build_dataframe(in_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(in_dir, "*.json")), key=_numeric_key)
    if not files:
        sys.exit(f"error: no .json files found in {in_dir!r}")
    rows, skipped = [], []
    for f in files:
        try:
            conv = json.load(open(f, encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            skipped.append((f, str(e)))
            continue
        missing = [c for c in COLUMNS if c not in conv]
        if missing:
            skipped.append((f, f"missing keys: {missing}"))
            continue
        rows.append(conv_to_row(conv, f))
    if skipped:
        print(f"[warn] skipped {len(skipped)} file(s):", file=sys.stderr)
        for f, why in skipped[:20]:
            print(f"  {f}: {why}", file=sys.stderr)
    print(f"[ok] packed {len(rows)} conversation(s) from {len(files)} file(s) in {in_dir}")
    return pd.DataFrame(rows, columns=COLUMNS)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--in-dir", default="out/glm51-fp8",
                   help="Directory of per-conversation JSON files.")
    p.add_argument("--out", default="glm5.1-fp8-test-00000-of-00001.parquet",
                   help="Local parquet output path.")
    p.add_argument("-d", "--dataset", default="Scicom-intl/Function-Call-TaaS",
                   help="Target Hub dataset repo id.")
    p.add_argument("-s", "--subset", default="glm5.1-fp8-test",
                   help="Dataset subset / config name.")
    p.add_argument("--split", default="test", help="Split name to write rows under.")
    p.add_argument("--private", action="store_true", help="Create the repo as private.")
    p.add_argument("--token", default=None,
                   help="HF token (defaults to cached login / HF_TOKEN).")
    p.add_argument("--no-push", action="store_true",
                   help="Only build the parquet; do not upload.")
    args = p.parse_args()

    df = build_dataframe(args.in_dir)
    df.to_parquet(args.out, index=False)
    print(f"[wrote] {args.out} ({len(df)} rows, {len(df.columns)} cols)")

    if args.no_push:
        print("[skip] --no-push set, not uploading.")
        return

    from datasets import Dataset

    dataset = Dataset.from_pandas(df, preserve_index=False)
    dataset.push_to_hub(
        args.dataset,
        config_name=args.subset,
        split=args.split,
        private=args.private,
        token=args.token,
    )
    print(
        f"[pushed] {len(dataset)} rows -> "
        f"https://huggingface.co/datasets/{args.dataset} "
        f"(subset={args.subset!r}, split={args.split!r})"
    )


if __name__ == "__main__":
    main()
