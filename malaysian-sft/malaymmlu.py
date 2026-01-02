"""
MalayMMLU Evaluation Script

Evaluates language models on MalayMMLU benchmark using vLLM serving.
Supports parallel evaluation across multiple GPUs.
"""

import json
import os
import re
import select
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from queue import Queue, Empty
from threading import Thread
from typing import Optional

import click
import requests
from multiprocess import Pool
from tqdm import tqdm


@dataclass
class Config:
    """Configuration for evaluation."""
    gpu_memory_utilization: float = 0.9
    max_tokens: int = 4096
    num_repeats: int = 3
    max_workers: int = 50
    max_retries: int = 3
    health_check_timeout: float = 5.0
    health_check_interval: float = 5.0
    health_check_max_attempts: int = 1000
    process_cleanup_timeout: int = 10
    inter_model_delay: float = 10.0
    request_timeout: float = 60
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
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        cmd = [
            "/root/.venv/bin/vllm", "serve", self.model,
            "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
            "--port", str(self.port),
            "--max-model-len", "8000",
            "--tensor-parallel-size", str(self.gpu_id.count(',') + 1),
        ]

        print(f"Starting vLLM: {' '.join(cmd)}")
        print(f"CUDA_VISIBLE_DEVICES={self.gpu_id}")

        self.process = subprocess.Popen(
            cmd,
            env=env,
            text=True,
            bufsize=1,
        )

        return self._wait_for_health()

    def _read_output(self) -> Optional[str]:
        """Read available output from process without blocking."""
        if self.process is None or self.process.stdout is None:
            return None
        
        ready, _, _ = select.select([self.process.stdout], [], [], 0.1)
        if ready:
            line = self.process.stdout.readline()
            return line if line else None
        return None

    def _wait_for_health(self) -> bool:
        """Wait for server to become healthy."""
        health_url = f"{self.base_url}/docs"
        
        print(f"Waiting for {health_url} to become healthy...")

        for attempt in range(self.config.health_check_max_attempts):

            try:
                response = requests.get(
                    health_url,
                    timeout=self.config.health_check_timeout,
                )
                if response.status_code == 200:
                    print(f"Server healthy after {attempt + 1} attempts")
                    return True
            except requests.RequestException:
                pass

            if attempt % 10 == 0:
                print(f"Health check attempt {attempt + 1}...")
            
            time.sleep(self.config.health_check_interval)

        print(f"Server failed to become healthy after {self.config.health_check_max_attempts} attempts")
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
                timeout=self.config.request_timeout,
            )
            if response.status_code != 200:
                print(f"[WARN] {self.model} HTTP {response.status_code}: {response.text[:500]}")
                return None
            r = response.json()["choices"][0]["message"]
            if isinstance(["reasoning_content"], str) and len(r["reasoning_content"]) > 10:
                t = r["reasoning_content"]
            else:
                t = r["content"]
            return t.strip()
        except requests.Timeout:
            print(f"[WARN] Request timed out after {self.config.request_timeout}s")
            return None
        except (requests.RequestException, KeyError, IndexError, TypeError) as e:
            print(f"[WARN] Generate error: {e}")
            return None

    def stop(self):
        """Stop the vLLM server gracefully."""
        if self.process is None or self.process.poll() is not None:
            return

        print(f"Stopping vLLM server on port {self.port}...")
        self.process.send_signal(signal.SIGINT)

        try:
            self.process.wait(timeout=self.config.process_cleanup_timeout)
        except subprocess.TimeoutExpired:
            print(f"Force killing vLLM server on port {self.port}...")
            self.process.kill()
            self.process.wait()

        print(f"Waiting for port {self.port} to be released...")
        for _ in range(30):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", self.port))
                print(f"Port {self.port} released")
                break
            except OSError:
                time.sleep(1.0)
        else:
            print(f"Warning: Port {self.port} may still be in use")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class MalayMMLUEvaluator:

    ANSWER_PATTERN = re.compile(r"boxed\{([^}]*)\}")

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
        # print('------')
        # print(response)
        # print('------\n\n')
        matches = self.ANSWER_PATTERN.findall(response)
        return matches[0] if len(matches) == 1 else None

    def generate_answer(self, question_id: int, question: str):
        for repeat_id in range(self.config.num_repeats):
            if self._result_exists(question_id, repeat_id):
                continue

            messages = self._build_messages(question)

            for attempt in range(self.config.max_retries):
                
                answer = None
                for _ in range(3):
                    response = self.server.generate(messages)
                    if response is None:
                        continue

                    answer = self._extract_answer(response)
                    if answer is not None and answer not in 'ABCDEF':
                        continue
                    break
                    
                if answer is not None:
                    self._save_result(question_id, repeat_id, answer)
                    break

    def run_parallel(self, questions: list[tuple[int, str]]):
        queue: Queue = Queue()
        for item in questions:
            queue.put(item)

        total = queue.qsize()
        completed = [0]  # Use list to allow modification in nested function

        def worker(worker_id: int):
            while True:
                try:
                    question_id, question = queue.get(timeout=5.0)
                except Empty:
                    break
                
                try:
                    self.generate_answer(question_id, question)
                    completed[0] += 1
                except Exception as e:
                    print(f"[Worker {worker_id}] Error processing q{question_id}: {e}")
                finally:
                    queue.task_done()

        # Warm up
        print(f"Warming up with first question...")
        self.generate_answer(questions[0][0], questions[0][1])

        workers = [
            Thread(target=worker, args=(i,), daemon=True)
            for i in range(self.config.max_workers)
        ]
        for w in workers:
            w.start()

        pbar = tqdm(total=total, desc=f"Generating answers for {self.output_dir}")
        last_completed = 0
        stall_count = 0

        while True:
            alive_workers = sum(1 for w in workers if w.is_alive())
            
            current_completed = completed[0]
            progress = current_completed - last_completed
            
            if progress > 0:
                pbar.update(progress)
                last_completed = current_completed
                stall_count = 0
            else:
                stall_count += 1
            
            remaining = queue.qsize()
            
            # Debug every 100 stalls (~10 seconds)
            if stall_count > 0 and stall_count % 100 == 0:
                print(f"\n[DEBUG] Stalled for {stall_count * 0.1:.1f}s - "
                      f"queue: {remaining}, completed: {current_completed}/{total}, "
                      f"alive workers: {alive_workers}")
            
            # Exit conditions
            if remaining == 0 and current_completed >= total - 1:  # -1 for warmup
                break
            if alive_workers == 0:
                print(f"\n[WARNING] All workers died with {remaining} items remaining")
                break
            # Timeout after 5 minutes of no progress
            if stall_count > 3000:
                print(f"\n[ERROR] Timeout - no progress for 5 minutes")
                break
                
            time.sleep(0.1)

        pbar.close()

        for w in workers:
            w.join(timeout=2.0)

    def run_sequential_cleanup(self, questions: list[tuple[int, str]]):
        for question_id, question in tqdm(questions, desc="Cleanup pass"):
            self.generate_answer(question_id, question)


def evaluate_model(model: str, port: int, output_dir: str, gpu_id: int, config: Config):
    output_path = Path(output_dir)

    with VLLMServer(model, port, gpu_id, config) as server:
        evaluator = MalayMMLUEvaluator(server, output_path, config)
        questions = evaluator.load_questions()
        evaluator.run_parallel(questions)
        evaluator.run_parallel(questions)

    time.sleep(config.inter_model_delay)


def process_batch(args: tuple[list[tuple], int]):
    rows, gpu_id = args
    config = Config()

    for model, port, output_dir, done_folder in rows:
        try:
            evaluate_model(model, port, output_dir, gpu_id, config)
            # with open(os.path.join(done_folder, model), 'w') as fopen:
            #     json.dump(done, fopen)
        except Exception as e:
            import traceback
            print(f"Error evaluating {model}: {e}")
            traceback.print_exc()

    return []


def run_multiprocess(items: list, func, num_workers: int = 6, partition=1):
    """Run function across multiple processes."""
    if not items:
        return

    gpus = []
    for gpu_id in range(0, num_workers, partition):
        gpu_id_ = list(range(gpu_id, gpu_id + partition))
        gpu_id_ = ','.join([str(i) for i in gpu_id_])
        gpus.append(gpu_id_)

    import itertools
    from collections import defaultdict
    
    gpus_cycle = itertools.cycle(gpus)
    work = defaultdict(list)
    for item in items:
        work[next(gpus_cycle)].append(item)

    work = [(v, k) for k, v in work.items()]

    print(f"Distributing {len(items)} items across {len(work)} GPUs")
    for chunk, gpu_id in work:
        models = [m[0] for m in chunk]
        print(f"  GPU {gpu_id}: {len(chunk)} items - {models}")

    with Pool(len(work)) as pool:
        pool.map(func, work)


@click.command()
@click.option("--pattern", default="nfs/nfs/*-merged", help="checkpoint glob pattern, can split by comma.")
@click.option("--num_gpus", default=8, help="number of gpus")
@click.option("--gpu_partition", default=1, help="number of gpus per process")
@click.option("--done_folder", default="done-malaymmlu", help="done folder malaymmlu")
def main(pattern, num_gpus, gpu_partition, done_folder):

    os.makedirs(done_folder, exist_ok=True)

    merged_dirs = []
    for p in pattern.split(','):
        merged_dirs.extend(glob(p))
    merged_dirs = [d for d in merged_dirs if "malaymmlu" not in d]

    tasks = [
        (path, 8000 + idx, f"malaymmlu-{Path(path).name}", done_folder)
        for idx, path in enumerate(merged_dirs)
    ]
    filtered_tasks = []
    for t in tasks:
        try:
            with open(os.path.join(done_folder, t[2])) as fopen:
                json.load(fopen)
            continue
        except:
            filtered_tasks.append(t)

    print(f"Found {len(filtered_tasks)} models to evaluate:")
    for model, port, output_dir, done_folder in filtered_tasks:
        print(f"  - {model} -> {output_dir}")

    run_multiprocess(filtered_tasks, process_batch, num_workers=num_gpus, partition=gpu_partition)


if __name__ == "__main__":
    main()