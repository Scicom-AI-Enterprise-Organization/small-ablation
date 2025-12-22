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
from transformers import LlamaForCausalLM
from transformers import TrainerCallback, TrainerState, TrainerControl
import streaming
import json
import numpy as np
from streaming import LocalDataset
from streaming.base.format.mds.encodings import Encoding, _encodings
from peft import LoraConfig, get_peft_model
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
    model_dtype: str = field(default="bfloat16", metadata={"help": "model dtype"}, )
    rank: int = field(default=16, metadata={"help": "rank"}, )
    alpha: int = field(default=32, metadata={"help": "alpha"}, )

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

class Model(LlamaForCausalLM):
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

    model = Model.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation = 'flash_attention_3',
        torch_dtype = model_args.model_dtype,
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    if training_args.fsdp is not None and len(training_args.fsdp):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})

    peft_config = LoraConfig(
        r=model_args.rank,
        lora_alpha=model_args.alpha,
        target_modules="all-linear",
    )
    model = get_peft_model(model, peft_config)

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=None,
        preprocess_logits_for_metrics=None,
    )
    trainer.train()


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()