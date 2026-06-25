#!/usr/bin/env python3
"""CLI runner for synthetic multi-turn red-team dataset generation."""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import aiohttp
from tqdm.asyncio import tqdm

from synthetic_generator.generator_multiturn import MultiTurnGenerator, DEFAULT_API_URL, DEFAULT_MODEL

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic multi-turn Malaysian LLM red-team dataset")
    parser.add_argument("--count", type=int, default=10, help="Number of entries per batch (default: 10)")
    parser.add_argument("--batches", type=int, default=1, help="Number of generation batches (default: 1)")
    parser.add_argument("--concurrency", type=int, default=3, help="Max concurrent API requests (default: 3)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model ID to use")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="API base URL")
    parser.add_argument("--api-key", default=None, help="API key (or set SGPU_API_KEY env var)")
    parser.add_argument(
        "--output", default=None,
        help="Output file path (default: outputs/multiturn_<timestamp>.jsonl)"
    )
    parser.add_argument("--retries", type=int, default=2, help="Retries per batch on parse failure")
    return parser.parse_args()


async def run_batch(
    generator: MultiTurnGenerator,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    batch_idx: int,
    args,
    f,
    pbar: tqdm,
    file_lock: asyncio.Lock,
) -> int:
    async with semaphore:
        entries, language = await generator.generate(session, count=args.count, retries=args.retries)
        async with file_lock:
            for entry in entries:
                entry["model"] = args.model
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
        pbar.update(len(entries))
        pbar.set_postfix(done=f"{batch_idx + 1}/{args.batches}", lang=language)
        return len(entries)


async def main_async(args, api_key: str):
    generator = MultiTurnGenerator(
        api_url=args.api_url,
        model=args.model,
        api_key=api_key,
        timeout=180,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) if args.output else Path("outputs") / f"multiturn_{timestamp}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(args.concurrency)
    file_lock = asyncio.Lock()

    with open(output_path, "w", encoding="utf-8") as f:
        with tqdm(total=args.batches * args.count, unit="entry", desc="Generating multi-turn") as pbar:
            async with aiohttp.ClientSession() as session:
                tasks = [
                    run_batch(generator, session, semaphore, i, args, f, pbar, file_lock)
                    for i in range(args.batches)
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

    total_written = 0
    failed_batches = 0
    for r in results:
        if isinstance(r, Exception):
            logging.error("Batch failed permanently: %s", r)
            failed_batches += 1
        else:
            total_written += r

    if failed_batches:
        print(f"Warning: {failed_batches}/{args.batches} batch(es) failed after all retries.", file=sys.stderr)
    print(f"Saved {total_written} multi-turn entries to {output_path}")


def main():
    args = parse_args()

    api_key = args.api_key or os.environ.get("SGPU_API_KEY")
    if not api_key:
        print("ERROR: Provide --api-key or set SGPU_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    asyncio.run(main_async(args, api_key))


if __name__ == "__main__":
    main()
