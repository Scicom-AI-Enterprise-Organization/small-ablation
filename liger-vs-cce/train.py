#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

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
    AddedToken,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers import Qwen3ForCausalLM
from transformers import TrainerCallback, TrainerState, TrainerControl
import streaming
import json
import numpy as np
from streaming import LocalDataset
from streaming.base.format.mds.encodings import Encoding, _encodings
from cut_cross_entropy import linear_cross_entropy
from liger_kernel.transformers import apply_liger_kernel_to_qwen3, LigerFusedLinearCrossEntropyLoss

apply_liger_kernel_to_qwen3(
    rope=True,
    swiglu=True,
    rms_norm=True,
    cross_entropy=False,
    fused_linear_cross_entropy=False,
)

torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    use_liger: bool = field(default=True, metadata={"help": "Use liger loss"}, )
    cce_impl: str = field(default="cce_kahan_full_c", metadata={"help": "CCE implementation"}, )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: Optional[str] = field(
        default=None, metadata={
            "help": "The input training data file (a text file)."})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )

class ModelCCE(Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        
    def forward(self, input_ids, attention_mask=None, position_ids=None, labels=None, num_items_in_batch=None, **kwargs):
        super_out = self.model.forward(
            input_ids = input_ids, 
            position_ids = position_ids, 
            attention_mask = attention_mask, 
            output_hidden_states = True,
            **kwargs,
        )
        if labels is not None:
            embeddings = super_out.last_hidden_state
            reduction = "sum" if num_items_in_batch is not None else "mean"
            loss = linear_cross_entropy(
                embeddings, 
                self.lm_head.weight, 
                labels, 
                shift=True,
                impl=self.config.cce_impl,
                reduction=reduction,
            )
            if reduction == "sum":
                if torch.is_tensor(num_items_in_batch):
                    num_items_in_batch = num_items_in_batch.to(loss.device)
                loss = loss / num_items_in_batch
            return {'loss': loss}
        return super_out

class Model(Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.loss = LigerFusedLinearCrossEntropyLoss(reduction="sum")
        
    def forward(self, input_ids, attention_mask=None, position_ids=None, labels=None, num_items_in_batch=None, **kwargs):
        super_out = self.model.forward(
            input_ids = input_ids,
            position_ids = position_ids, 
            attention_mask = attention_mask, 
            output_hidden_states = True,
            **kwargs,
        )
        if labels is not None:
            embeddings = super_out.last_hidden_state
            embeddings = embeddings[:,:-1].reshape(-1, embeddings.shape[-1])
            labels = labels[..., 1:].contiguous()
            labels = labels.reshape(-1)
            loss = self.loss(self.lm_head.weight, embeddings, labels)
            num_items_in_batch = num_items_in_batch.to(loss.device)
            loss = loss / num_items_in_batch
            return {'loss': loss}
        return super_out

class WandbMFUCallback(TrainerCallback):
    def __init__(self, num_flops_per_token):
        self.num_flops_per_token = num_flops_per_token
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
        mfu = 100 * flops_per_sec / 989e12
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
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}" +
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}")
    logger.info(f"Training/evaluation parameters {training_args}")

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    extra = [AddedToken('<|speech_start|>')]
    for i in range(65536):
        extra.append(AddedToken(f'<|s_{i}|>'))
    tokenizer.add_tokens(extra)
    
    sequence_length = data_args.block_size

    class UInt32(Encoding):
        def encode(self, obj) -> bytes:
            return obj.tobytes()

        def decode(self, data: bytes):
            return np.frombuffer(data, np.uint32)

    _encodings['uint32'] = UInt32

    class DatasetFixed(torch.utils.data.Dataset):
        def __init__(self, local):
            self.dataset = LocalDataset(local=local)

        def __getitem__(self, idx):
            data = self.dataset[idx]
            data.pop('audio', None)
            data.pop('text', None)
            data.pop('token_type_ids', None)

            if data['attention_mask'].max() >= sequence_length:
                print(data)
                return

            for k in data.keys():
                data[k] = data[k].astype(np.int64)
        
            return data

        def __len__(self):
            return len(self.dataset)

    if model_args.use_liger:
        model_class = Model
    else:
        model_class = ModelCCE

    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation = 'flash_attention_3',
        torch_dtype = torch.bfloat16,
    )
    if not model_args.use_liger:
        model.config.cce_impl = model_args.cce_impl
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False, pad_to_multiple_of=8)
    print(model)

    dataset = DatasetFixed(data_args.train_file)
    print('dataset', len(dataset), dataset[0]['attention_mask'].shape)

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
        max_cumsum = int(np.max(cumsum))
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
            'max_length_k': max_seqlen_q
        }

    nparams = sum(p.numel() for p in model.parameters())
    nparams_embedding = model.model.embed_tokens.weight.numel()
    l, h, q = model.config.num_hidden_layers, model.config.num_attention_heads, model.config.head_dim
    num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * sequence_length

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=None,
        preprocess_logits_for_metrics=None,
        callbacks=[WandbMFUCallback(num_flops_per_token)],
    )
    trainer.train()


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()