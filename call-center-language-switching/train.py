#!/usr/bin/env python
# coding=utf-8
"""
Call Center Language Switching SLM Training
- Qwen3 Base with Liger Fused Linear Cross Entropy (chunk CE)
- Multipacking with Flash Attention varlen (cu_seqlens)
- Pure bfloat16 training
- WandB MFU tracking
- chinidataset ParquetWriter/StreamingDataset for data loading
"""

import torch
from torch import nn
import torch.nn.functional as F

import logging
import os
import sys
import wandb
import time
from dataclasses import dataclass, field
from typing import Optional

import transformers
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers import Qwen3ForCausalLM
from transformers import TrainerCallback, TrainerState, TrainerControl
import numpy as np
from chinidataset import StreamingDataset
from liger_kernel.transformers import apply_liger_kernel_to_qwen3, LigerFusedLinearCrossEntropyLoss

apply_liger_kernel_to_qwen3(
    rope=True,
    swiglu=True,
    rms_norm=True,
    cross_entropy=False,
    fused_linear_cross_entropy=False,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    model_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Model dtype: bfloat16"},
    )
    flash_attn_version: str = field(
        default="flash_attention_3",
        metadata={"help": "Flash attention version: flash_attention_3 (H100/H200) or flash_attention_4 (B200)"},
    )


@dataclass
class DataTrainingArguments:
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to chinidataset parquet streaming dataset directory"},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to chinidataset parquet test dataset directory"},
    )
    block_size: Optional[int] = field(
        default=8192,
        metadata={"help": "Multipacking block size (sequence length)"},
    )


class Model(Qwen3ForCausalLM):
    """
    Qwen3 with Liger Fused Linear Cross Entropy Loss.
    Uses chunk cross entropy for memory-efficient training.
    """
    def __init__(self, config):
        super().__init__(config)
        self.loss = LigerFusedLinearCrossEntropyLoss(reduction="sum")

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                labels=None, num_items_in_batch=None, **kwargs):
        super_out = self.model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        if labels is not None:
            embeddings = super_out.last_hidden_state
            embeddings = embeddings[:, :-1].reshape(-1, embeddings.shape[-1])
            labels = labels[..., 1:].contiguous()
            labels = labels.reshape(-1)
            loss = self.loss(self.lm_head.weight, embeddings, labels)
            if num_items_in_batch is not None:
                num_items_in_batch = num_items_in_batch.to(loss.device)
                loss = loss / num_items_in_batch
            else:
                # eval mode: normalize by number of non-ignored tokens
                loss = loss / (labels != -100).sum()
            return {'loss': loss}
        return super_out


class WandbMFUCallback(TrainerCallback):
    """Track Model FLOPs Utilization and throughput on WandB."""
    def __init__(self, num_flops_per_token, gpu_flops=989e12):
        self.num_flops_per_token = num_flops_per_token
        self.gpu_flops = gpu_flops
        self.previous = 0
        self.begin_step = None
        self.end_step = None

    def on_step_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.begin_step = time.time()

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.end_step = time.time()

    def on_log(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        dt = self.end_step - self.begin_step
        throughput_per_sec = (state.num_input_tokens_seen - self.previous) / dt
        flops_per_sec = self.num_flops_per_token * throughput_per_sec
        mfu = 100 * flops_per_sec / self.gpu_flops
        wandb.log({
            "train_mfu": mfu,
            "train_throughput_per_sec": throughput_per_sec,
        }, step=state.global_step)
        self.previous = state.num_input_tokens_seen


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    sequence_length = data_args.block_size

    class DatasetFixed(torch.utils.data.Dataset):
        def __init__(self, local):
            self.dataset = StreamingDataset(local=local)

        def __getitem__(self, idx):
            data = self.dataset[idx]
            data.pop('text', None)

            if data['attention_mask'].max() >= sequence_length:
                return None

            for k in data.keys():
                data[k] = data[k].astype(np.int64)

            return data

        def __len__(self):
            return len(self.dataset)

    model = Model.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation=model_args.flash_attn_version,
        torch_dtype=model_args.model_dtype,
    )
    print(model)

    dataset = DatasetFixed(data_args.train_file)
    print('dataset', len(dataset), dataset[0]['attention_mask'].shape)

    eval_dataset = None
    if data_args.test_file:
        eval_dataset = DatasetFixed(data_args.test_file)
        print('eval_dataset', len(eval_dataset))

    def collator(batch):
        batch = [b for b in batch if b is not None]
        input_ids = [b['input_ids'] for b in batch]
        position_ids = [b['position_ids'] for b in batch]
        labels = [b['input_ids'].copy() for b in batch]
        attention_mask = [b['attention_mask'] for b in batch]

        input_ids = np.concatenate(input_ids)
        position_ids = np.concatenate(position_ids)
        labels = np.concatenate(labels)
        query_lens = np.concatenate(attention_mask)

        cumsum = [0] + np.cumsum(query_lens).tolist()
        cu_seq_lens_q = torch.tensor(cumsum, dtype=torch.int32)
        cu_seq_lens_k = torch.tensor(cumsum, dtype=torch.int32)
        max_seqlen_q = np.max(query_lens)

        return {
            'input_ids': torch.tensor(input_ids)[None],
            'position_ids': torch.tensor(position_ids)[None],
            'labels': torch.tensor(labels)[None],
            'cu_seq_lens_q': cu_seq_lens_q,
            'cu_seq_lens_k': cu_seq_lens_k,
            'max_length_q': max_seqlen_q,
            'max_length_k': max_seqlen_q,
        }

    nparams = sum(p.numel() for p in model.parameters())
    nparams_embedding = model.model.embed_tokens.weight.numel()
    l = model.config.num_hidden_layers
    h = model.config.num_attention_heads
    q = model.config.head_dim
    num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * sequence_length

    # GPU peak FLOPS: H100=989e12, H200=989e12, B200=2250e12 (bf16)
    gpu_flops = float(os.environ.get('GPU_PEAK_FLOPS', '989e12'))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=None,
        preprocess_logits_for_metrics=None,
        callbacks=[WandbMFUCallback(num_flops_per_token, gpu_flops=gpu_flops)],
    )
    resume = training_args.resume_from_checkpoint
    if resume == "true":
        # Find the latest checkpoint in output_dir
        import glob
        checkpoints = sorted(glob.glob(os.path.join(training_args.output_dir, "checkpoint-*")),
                             key=lambda x: int(x.split("-")[-1]))
        resume = checkpoints[-1] if checkpoints else None
    trainer.train(resume_from_checkpoint=resume)


if __name__ == "__main__":
    main()
