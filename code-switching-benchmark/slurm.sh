#!/bin/bash
#SBATCH --job-name=code-switching-benchmark-gemma-4-31b
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=0
#SBATCH --chdir=/tmp

export HOME=/aura-usrdata/$USER
export XDG_CACHE_HOME=/share/.cache
export TRITON_CACHE_DIR=/share/triton_cache
export TORCHINDUCTOR_CACHE_DIR=/share/torchinductor_cache
export FLASHINFER_WORKSPACE_DIR=/share/flashinfer_cache
export HF_HOME=/share/huggingface
export TRANSFORMERS_CACHE=/share/huggingface
export VLLM_CACHE_ROOT=/share/vllm_cache
export CUDA_CACHE_PATH=/share/nv_cache
export NUMBA_CACHE_DIR=/share/numba_cache

WORK_DIR=/aura-usrdata/$USER
cd "$WORK_DIR"

echo 'CURRENT DIR: ' $PWD

cat > benchmark.py << 'EOF'
import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import aiohttp
import fasttext
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio

SCRIPT_DIR = Path(__file__).parent

LABEL_LANG_MAP = {
    "__label__en": "english",
    "__label__zh": "chinese",
    "__label__ta": "tamil",
    "__label__ms": "malay",
    "__label__id": "malay",
}
LANG_LABEL_MAP = {v: k for k, v in LABEL_LANG_MAP.items()}

SYSTEM_PROMPT = (
    "You are a helpful customer service assistant. "
    "Always detect the language of the user's last message and reply strictly in that same language. "  
    "Do not mix languages in your response."
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Code-switching benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", required=True, help="Model name/path (e.g. Qwen/Qwen3-4B)")
    parser.add_argument("--host", default="localhost", help="vLLM server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port (default: 8000)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: results/<sanitized-model-name>)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=20, help="Concurrent requests (default: 20)"
    )
    parser.add_argument(
        "--server-timeout",
        type=int,
        default=600,
        help="Seconds to wait for vLLM server readiness (default: 600)",
    )
    parser.add_argument(
        "--server-url",
        default=None,
        help="Skip launching vLLM; use this existing server URL instead",
    )
    parser.add_argument(
        "--fasttext-model",
        default=str(SCRIPT_DIR / "lid.176.bin"),
        help="Path to fasttext language ID model (default: lid.176.bin beside this script)",
    )
    parser.add_argument(
        "--cuda-devices",
        default=None,
        help="GPU indices to expose to vLLM via CUDA_VISIBLE_DEVICES (e.g. '0,1'). "
             "Has no effect when --server-url is used.",
    )
    # Everything after '--' is forwarded verbatim to the vLLM server process
    parser.add_argument(
        "vllm_extra",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to vLLM server (place after --)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# vLLM process management
# ---------------------------------------------------------------------------

_vllm_proc: subprocess.Popen | None = None


def _shutdown(signum=None, frame=None):
    if _vllm_proc and _vllm_proc.poll() is None:
        print("\nShutting down vLLM server...")
        _vllm_proc.terminate()
        try:
            _vllm_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _vllm_proc.kill()
    sys.exit(0)


def start_vllm(model: str, host: str, port: int, extra: list[str], cuda_devices: str | None = None) -> subprocess.Popen:
    global _vllm_proc
    cmd = [
        "vllm", "serve", model,
        "--host", host,
        "--port", str(port),
    ] + extra
    env = os.environ.copy()
    if cuda_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    prefix = f"CUDA_VISIBLE_DEVICES={cuda_devices} " if cuda_devices is not None else ""
    print(f"Starting vLLM: {prefix}{' '.join(cmd)}")
    _vllm_proc = subprocess.Popen(cmd, stdout=None, stderr=None, env=env)
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    return _vllm_proc


def wait_for_server(base_url: str, timeout: int, proc: subprocess.Popen | None = None):
    print(f"Waiting for server at {base_url} (timeout {timeout}s)...", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc and proc.poll() is not None:
            raise RuntimeError(f"vLLM process exited unexpectedly (code {proc.returncode})")
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=3):
                print("\nServer is ready!")
                return
        except Exception:
            sys.stdout.write(".")
            sys.stdout.flush()
            time.sleep(5)
    raise TimeoutError(f"Server did not become ready within {timeout}s")


# ---------------------------------------------------------------------------
# Dataset loading & cleaning
# ---------------------------------------------------------------------------


def load_and_clean_dataset(lang_model: fasttext.FastText._FastText):
    print("Loading dataset...")
    # from datasets import load_from_disk
    # ds = load_from_disk(
    #     "cleaned_dataset_non_indonesian"
    # )
    ds = load_dataset("Scicom-intl/Malaysian-Call-Center-Language-Switching", split="train")
    print(f"Cleaned dataset: {len(ds)} samples")
    return ds


# ---------------------------------------------------------------------------
# Async inference
# ---------------------------------------------------------------------------


async def _call_model(
    base_url: str,
    model: str,
    sample: dict,
    semaphore: asyncio.Semaphore,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *sample["messages"][: sample["code_switching_turn"] + 1],
    ]
    async with semaphore:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
            try:
                async with session.post(
                    f"{base_url}/v1/chat/completions",
                    json={"model": model, "messages": messages},
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"Non-200 response: {resp.status}")

                    data = await resp.json(content_type=None)
                    return data["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"\nRequest error (sample skipped): {e}")
                return ""

async def run_benchmark(
    base_url: str,
    model: str,
    samples,
    concurrency: int,
    lang_model: fasttext.FastText._FastText,
    progress_file: Path,
    completed: set[int],
) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)

    async def process(idx: int, sample: dict):
        response = await _call_model(base_url, model, sample, semaphore)
        return idx, sample, response

    with open(progress_file, "a", encoding="utf-8") as prog_f:
        tasks = [
            asyncio.create_task(process(i, sample))
            for i, sample in enumerate(samples)
            if i not in completed
        ]

        pending = len(tasks)
        print(
            f"Benchmarking {pending} samples "
            f"({len(completed)} already completed, {len(samples)} total)..."
        )

        new_results: list[dict] = []
        for coro in tqdm_asyncio.as_completed(tasks, total=pending):
            idx, sample, response = await coro

            if response.strip():
                detected, _ = lang_model.predict(response.replace("\n", " "))
                resp_lang = LABEL_LANG_MAP.get(detected[0], "unknown")
            else:
                resp_lang = "unknown"
                continue # skip empty responses in evaluation and summary

            match = resp_lang == sample["switch_language"]
            record = {
                "index": idx,
                "industry": sample["industry"],
                "topic": sample["topic"],
                "source_language": sample["source_language"],
                "switch_language": sample["switch_language"],
                "code_switching_turn": sample["code_switching_turn"],
                "messages": sample["messages"],
                "response": response,
                "response_lang": resp_lang,
                "match": match,
            }
            prog_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            prog_f.flush()
            new_results.append(record)

    return new_results


# ---------------------------------------------------------------------------
# Progress / summary helpers
# ---------------------------------------------------------------------------


def load_progress(progress_file: Path) -> tuple[set[int], list[dict]]:
    completed: set[int] = set()
    records: list[dict] = []
    if not progress_file.exists():
        return completed, records
    with open(progress_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                completed.add(rec["index"])
                records.append(rec)
            except json.JSONDecodeError:
                pass
    return completed, records


def print_summary(all_records: list[dict], model: str) -> dict:
    # filter any record with empty response 
    all_records = [r for r in all_records if r["response"].strip()]
    total = len(all_records)
    if total == 0:
        print("No results.")
        return {}

    matched = sum(1 for r in all_records if r["match"])
    acc = matched / total

    print("\n" + "=" * 60)
    print(f"Model    : {model}")
    print(f"Total    : {total}")
    print(f"Match    : {matched}")
    print(f"Accuracy : {acc:.4f}  ({acc * 100:.2f}%)")

    per_lang: dict[str, list[bool]] = defaultdict(list)
    for r in all_records:
        per_lang[r["switch_language"]].append(r["match"])

    print("\nPer switch-language accuracy:")
    for lang, matches in sorted(per_lang.items()):
        n = len(matches)
        m = sum(matches)
        print(f"  {lang:<12}: {m}/{n} = {m / n:.4f}")
    print("=" * 60)

    return {
        "total": total,
        "matched": matched,
        "accuracy": acc,
        "per_language": {
            lang: {
                "total": len(v),
                "matched": sum(v),
                "accuracy": sum(v) / len(v),
            }
            for lang, v in per_lang.items()
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def sanitize(name: str) -> str:
    return name.replace("/", "__").replace(":", "_")


def main():
    args = parse_args()

    vllm_extra = [a for a in args.vllm_extra if a != "--"]

    # Resolve server URL
    if args.server_url:
        base_url = args.server_url.rstrip("/")
        managed_proc = None
    else:
        base_url = f"http://{args.host}:{args.port}"
        managed_proc = start_vllm(args.model, args.host, args.port, vllm_extra, args.cuda_devices)

    # Block until the server accepts requests
    wait_for_server(base_url, args.server_timeout, managed_proc)

    # Resolve output directory
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else SCRIPT_DIR / "results" / sanitize(args.model)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_file = out_dir / "progress.jsonl"
    summary_file = out_dir / "summary.json"
    print(f"Output directory: {out_dir}")

    # Load fasttext language-ID model
    if not Path(args.fasttext_model).exists():
        sys.exit(f"fasttext model not found: {args.fasttext_model}")
    print(f"Loading fasttext model from {args.fasttext_model}...")
    lang_model = fasttext.load_model(args.fasttext_model)

    # Load & clean HuggingFace dataset
    samples = load_and_clean_dataset(lang_model)

    # Restore any previous progress
    completed, existing_records = load_progress(progress_file)
    if completed:
        print(f"Resuming: {len(completed)} samples already done.")

    # Run async benchmark
    new_records = asyncio.run(
        run_benchmark(
            base_url=base_url,
            model=args.model,
            samples=samples,
            concurrency=args.concurrency,
            lang_model=lang_model,
            progress_file=progress_file,
            completed=completed,
        )
    )

    all_records = existing_records + new_records

    # Print and persist summary
    stats = print_summary(all_records, args.model)
    if stats:
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump({"model": args.model, **stats}, f, indent=2, ensure_ascii=False)
        print(f"\nSummary saved to {summary_file}")
        print(f"Full results in   {progress_file}")

    # Clean up vLLM if we launched it
    if managed_proc and managed_proc.poll() is None:
        print("Stopping vLLM server...")
        managed_proc.terminate()
        managed_proc.wait(timeout=15)


if __name__ == "__main__":
    main()
EOF

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

uv venv --python 3.12 --allow-existing
uv pip install huggingface_hub==1.16.1 ipykernel datasets==4.8.5 vllm==0.19.1
uv pip install git+https://github.com/tchiayan/fastText.git

# download language identification model
wget -nc https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

source .venv/bin/activate

export MODEL=google/gemma-4-31B-it
hf download $MODEL
python benchmark.py \
    --model $MODEL \
    --port 1234 \
    --server-timeout 3600 \
    --cuda-devices 0 \
    --output-dir "results/$(echo $MODEL | tr '/' '_')_$(date +%m%d)" \
    --concurrency 50 \
    -- --max-model-len 12000 --port 1234 --quantization fp8