#!/usr/bin/env python3
# Train Qwen3 MoE (fused Triton kernels + LoRA) with a multipacked MDS dataset

import os
import math
import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import numpy as np
import torch
from torch.utils.data import Dataset

rank = os.getenv("LOCAL_RANK", os.getenv("RANK", "0"))
os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/triton_cache_{rank}")
os.environ["TRITON_PRINT_AUTOTUNING"] = os.environ.get("TRITON_PRINT_AUTOTUNING", "1")

torch._dynamo.config.optimize_ddp = False  # same as your second script

logger = logging.getLogger("train_qwen3_moe_fused_multipack")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    set_seed,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model

from qwen3_moe_fused.fast_lora import patch_Qwen3MoeFusedSparseMoeBlock_forward
from qwen3_moe_fused.lora import patch_lora_config
from qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedForCausalLM
from qwen3_moe_fused.quantize.quantizer import patch_bnb_quantizer

from streaming import LocalDataset
from streaming.base.format.mds.encodings import Encoding, _encodings



@dataclass
class ModelArgs:
    model_name_or_path: str = field(
        default="bash99/Qwen3-30B-A3B-Instruct-2507-fused-bnb-4bit",
        metadata={"help": "HF model repo or local path"}
    )
    use_rslora: bool = field(default=True)
    qkv_o_rank: int = field(default=16)
    moe_ffn_rank: int = field(default=4)
    gate_rank: int = field(default=16)  # can be unstable; keep but you may disable


@dataclass
class DataArgs:
    train_file: str = field(
        default=None,
        metadata={"help": "Path to the MDS LocalDataset directory (multipacked)"},
    )
    num_proc: int = field(default=1)


@dataclass
class RunArgs(TrainingArguments):
    seed: int = field(default=3407)
    bf16: bool = field(default=True)
    report_to: Optional[str] = field(default="none")



def print_trainable_parameters(model):
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    pct = 100 * trainable / total if total else 0.0
    logger.info(f"Trainable params: {trainable} || All params: {total} || {pct:.4f}% trainable")


def block_diagonal_concat_inverted(*masks, dtype=torch.bfloat16):

    total = sum(m.size(0) for m in masks)
    combined = torch.zeros(total, total, dtype=dtype)
    cur = 0
    for m in masks:
        L = m.size(0)
        combined[cur:cur+L, cur:cur+L] = m.to(dtype)
        cur += L
    min_val = torch.finfo(dtype).min if dtype.is_floating_point else torch.iinfo(dtype).min
    additive = torch.where(combined == 1, torch.tensor(0, dtype=dtype), torch.tensor(min_val, dtype=dtype))
    return additive.unsqueeze(0)  # [1, T, T]


class UInt32(Encoding):
    def encode(self, obj) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes):
        return np.frombuffer(data, np.uint32)


_encodings["uint32"] = UInt32


class MultipackMDS(Dataset):

    def __init__(self, local_dir: str, use_bf16: bool = True):
        super().__init__()
        self.ds = LocalDataset(local=local_dir)
        self.use_bf16 = use_bf16

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.ds[idx]
        input_ids = item["input_ids"].astype(np.int64)
        position_ids = item["position_ids"].astype(np.int64)


        seg_lens = item["attention_mask"]  # uint32 lengths
        seg_masks: List[torch.Tensor] = []
        for L in seg_lens:
            L = int(L)
            seg_masks.append(torch.tril(torch.ones(L, L)))

        dtype = torch.bfloat16 if self.use_bf16 else torch.float16
        additive_mask = block_diagonal_concat_inverted(*seg_masks, dtype=dtype)  # [1, T, T]

        out = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "position_ids": torch.tensor(position_ids, dtype=torch.long),
            "attention_mask": additive_mask,                 # [1, T, T]
            "labels": torch.tensor(input_ids, dtype=torch.long),
        }
        return out



def main():
    parser = HfArgumentParser((ModelArgs, DataArgs, RunArgs))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, run_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, run_args = parser.parse_args_into_dataclasses()

    if run_args.should_log:
        logging.getLogger().setLevel(logging.INFO)

    set_seed(run_args.seed)

    if not data_args.train_file or not os.path.isdir(data_args.train_file):
        raise ValueError("--train_file must be a directory containing your MDS shards.")

    patch_bnb_quantizer()
    patch_lora_config(
        rank_pattern={
            "q_proj": model_args.qkv_o_rank,
            "k_proj": model_args.qkv_o_rank,
            "v_proj": model_args.qkv_o_rank,
            "o_proj": model_args.qkv_o_rank,
            "gate":  model_args.gate_rank,   # you can comment this if unstable
            "gate_proj": model_args.moe_ffn_rank,
            "up_proj":   model_args.moe_ffn_rank,
            "down_proj": model_args.moe_ffn_rank,
        }
    )
    patch_Qwen3MoeFusedSparseMoeBlock_forward()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)

    model = Qwen3MoeFusedForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16 if run_args.bf16 else torch.float16,
        attn_implementation="sdpa",
    )

    lora_cfg = LoraConfig(
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate", "gate_proj", "up_proj", "down_proj",
        ],
        rank_pattern={
            "q_proj": model_args.qkv_o_rank,
            "k_proj": model_args.qkv_o_rank,
            "v_proj": model_args.qkv_o_rank,
            "o_proj": model_args.qkv_o_rank,
            "gate":  model_args.gate_rank,
            "gate_proj": model_args.moe_ffn_rank,
            "up_proj":   model_args.moe_ffn_rank,
            "down_proj": model_args.moe_ffn_rank,
        },
        lora_alpha=1 if model_args.use_rslora else (2 * model_args.qkv_o_rank),
        use_rslora=model_args.use_rslora,
        bias="none",
    )

    model = get_peft_model(model, lora_cfg)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print_trainable_parameters(model)

    ds = MultipackMDS(local_dir=data_args.train_file, use_bf16=run_args.bf16)

    trainer = Trainer(
        model=model,
        args=run_args,
        train_dataset=ds,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    checkpoint = None
    if run_args.resume_from_checkpoint is not None:
        checkpoint = run_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    trainer.save_model()
    trainer.save_state()


if __name__ == "__main__":
    main()
