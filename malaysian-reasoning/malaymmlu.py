"""
MalayMMLU Evaluation Script

Evaluates language models on MalayMMLU benchmark using vLLM serving.
Supports parallel evaluation across multiple GPUs.
"""

import itertools
import json
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Optional
import click
import requests
from multiprocess import Pool
from tqdm import tqdm


@dataclass
class Config:
    """Configuration for evaluation."""
    gpu_memory_utilization: float = 0.95
    max_tokens: int = 4096
    num_repeats: int = 5
    max_workers: int = 10
    max_retries: int = 3
    health_check_timeout: float = 5.0
    health_check_interval: float = 5.0
    health_check_max_attempts: int = 1000
    process_cleanup_timeout: int = 10
    inter_model_delay: float = 10.0
    system_prompt: str = (
        "First, you try to think step-by-step in {{lang}}, "
        "after that, put your final answer within $\\boxed{}$."
    )
    language: str = "malay"
    benchmark_file: str = "MalayMMLU_0shot.json"


class VLLMServer:
    """Manages vLLM server lifecycle."""

    def __init__(self, model: str, port: int, gpu_id: int, config: Config):
        self.model = model
        self.port = port
        self.gpu_id = gpu_id
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.base_url = f"http://localhost:{port}"

    def start(self) -> bool:
        """Start the vLLM server and wait for it to be healthy."""
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        cmd = [
            "/root/.venv/bin/vllm", "serve", self.model,
            "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
            "--port", str(self.port),
        ]
        print(cmd)

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        return self._wait_for_health()

    def _wait_for_health(self) -> bool:
        health_url = f"{self.base_url}/docs"
        pbar = tqdm(
            range(self.config.health_check_max_attempts),
            desc=f"Waiting for {health_url}",
        )

        for attempt in pbar:
            try:
                response = requests.get(
                    health_url,
                    timeout=self.config.health_check_timeout,
                )
                if response.status_code == 200:
                    return True
            except requests.RequestException:
                pass

            pbar.set_description(f"Health check {health_url} attempt {attempt + 1}")
            time.sleep(self.config.health_check_interval)

        return False

    def generate(self, messages: list[dict]) -> Optional[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
        }

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["reasoning_content"].strip()
        except (requests.RequestException, KeyError, IndexError):
            return None

    def stop(self):
        if self.process is None or self.process.poll() is not None:
            return

        print(f"Stopping vLLM server on port {self.port}...")
        self.process.send_signal(signal.SIGINT)

        try:
            self.process.wait(timeout=self.config.process_cleanup_timeout)
        except subprocess.TimeoutExpired:
            print(f"Force killing vLLM server on port {self.port}...")
            self.process.kill()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class MalayMMLUEvaluator:

    ANSWER_PATTERN = re.compile(r"\$boxed\{(.*?)\}\$")

    def __init__(self, server: VLLMServer, output_dir: Path, config: Config):
        self.server = server
        self.output_dir = output_dir
        self.config = config
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_questions(self) -> list[tuple[int, str]]:
        with open(self.config.benchmark_file) as f:
            data = json.load(f)
        return [(i, item["prompt"]) for i, item in enumerate(data)]

    def _get_output_path(self, question_id: int, repeat_id: int) -> Path:
        return self.output_dir / f"{question_id}-{repeat_id}.json"

    def _result_exists(self, question_id: int, repeat_id: int) -> bool:
        path = self._get_output_path(question_id, repeat_id)
        if not path.exists():
            return False

        try:
            with open(path) as f:
                json.load(f)
            return True
        except (json.JSONDecodeError, IOError):
            return False

    def _save_result(self, question_id: int, repeat_id: int, answer: str):
        path = self._get_output_path(question_id, repeat_id)
        with open(path, "w") as f:
            json.dump(answer, f)

    def _build_messages(self, question: str) -> list[dict]:
        system_content = self.config.system_prompt.replace(
            "{{lang}}", self.config.language
        )
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": question},
        ]

    def _extract_answer(self, response: str) -> Optional[str]:
        matches = self.ANSWER_PATTERN.findall(response)
        return matches[0] if len(matches) == 1 else None

    def generate_answer(self, question_id: int, question: str):
        for repeat_id in range(self.config.num_repeats):
            if self._result_exists(question_id, repeat_id):
                continue

            messages = self._build_messages(question)

            for _ in range(self.config.max_retries):
                response = self.server.generate(messages)
                if response is None:
                    continue

                answer = self._extract_answer(response)
                if answer is not None:
                    self._save_result(question_id, repeat_id, answer)
                    break

    def run_parallel(self, questions: list[tuple[int, str]]):
        queue: Queue = Queue()
        for item in questions:
            queue.put(item)

        total = queue.qsize()

        def worker(worker_id: int):
            while not queue.empty():
                try:
                    question_id, question = queue.get_nowait()
                    self.generate_answer(question_id, question)
                except Exception:
                    break

        self.generate_answer(questions[0][0], questions[0][1])

        workers = [
            Thread(target=worker, args=(i,))
            for i in range(self.config.max_workers)
        ]
        for w in workers:
            w.start()

        pbar = tqdm(total=total, desc=f"Generating answers for {self.output_dir}")
        last_remaining = total

        while not queue.empty():
            remaining = queue.qsize()
            progress = last_remaining - remaining
            if progress > 0:
                pbar.update(progress)
                last_remaining = remaining
            time.sleep(0.1)

        pbar.update(last_remaining)
        pbar.close()

        for w in workers:
            w.join()

    def run_sequential_cleanup(self, questions: list[tuple[int, str]]):
        for question_id, question in tqdm(questions, desc="Cleanup pass"):
            self.generate_answer(question_id, question)


def evaluate_model(model: str, port: int, output_dir: str, gpu_id: int, config: Config):
    output_path = Path(output_dir)

    with VLLMServer(model, port, gpu_id, config) as server:
        evaluator = MalayMMLUEvaluator(server, output_path, config)
        questions = evaluator.load_questions()
        evaluator.run_parallel(questions)
        evaluator.run_sequential_cleanup(questions)

    time.sleep(config.inter_model_delay)


def process_batch(args: tuple[list[tuple], int]):
    rows, gpu_id = args
    config = Config()

    for model, port, output_dir in rows:
        try:
            evaluate_model(model, port, output_dir, gpu_id, config)
        except Exception as e:
            print(f"Error evaluating {model}: {e}")

    return []


def run_multiprocess(items: list, func, num_workers: int = 6):
    if not items:
        return

    chunk_size = max(1, len(items) // num_workers)
    chunks = [
        (items[i:i + chunk_size], i // chunk_size)
        for i in range(0, len(items), chunk_size)
    ]

    with Pool(num_workers) as pool:
        pool.map(func, chunks)

@click.command()
@click.option("--pattern", default="nfs/nfs/*-merged", help="checkpoint glob pattern, can split by comma.")
@click.option("--num_gpus", default=8, help="number of gpus")
def main(pattern, num_gpus):
    merged_dirs = []
    for p in pattern.split(','):
        merged_dirs.extend(glob(p))
    merged_dirs = [d for d in merged_dirs if "malaymmlu" not in d]

    tasks = [
        (path, 8000 + idx, f"malaymmlu-{Path(path).name}")
        for idx, path in enumerate(merged_dirs)
    ]

    print(f"Found {len(tasks)} models to evaluate:")
    for model, port, output_dir in tasks:
        print(f"  - {model} -> {output_dir}")

    run_multiprocess(tasks, process_batch, num_workers=num_gpus)


if __name__ == "__main__":
    main()