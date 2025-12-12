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
import math
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
from transformers import GptOssForCausalLM, Mxfp4Config
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts
from transformers.models.gpt_oss.modeling_gpt_oss import load_balancing_loss_func, GptOssDecoderLayer
from peft import LoraConfig, get_peft_model
import peft.utils.other as peft_other
import deepspeed
import streaming
import json
import numpy as np
from streaming import LocalDataset
from streaming.base.format.mds.encodings import Encoding, _encodings
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

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
    rank: int = field(default=8, metadata={"help": "rank"}, )
    alpha: int = field(default=16, metadata={"help": "alpha"}, )
    include_expert_lora: bool = field(default=False, metadata={"help": "expert lora"}, )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: Optional[str] = field(
        default=None, metadata={
            "help": "The input training data file (a text file)."})

class Model(GptOssForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.loss = LigerFusedLinearCrossEntropyLoss(reduction="sum")
        
    def forward(
        self, 
        input_ids, 
        attention_mask=None, 
        position_ids=None, 
        labels=None, 
        num_items_in_batch=None, 
        logits_to_keep=0,
        output_router_logits=None,
        **kwargs,
    ):
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        super_out = self.model.forward(
            input_ids = input_ids,
            position_ids = position_ids, 
            attention_mask = attention_mask, 
            output_router_logits=output_router_logits,
            **kwargs,
        )
        if labels is not None:
            embeddings = super_out.last_hidden_state
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            embeddings = embeddings[:, slice_indices, :]
            embeddings = embeddings[:,:-1].reshape(-1, embeddings.shape[-1])
            labels = labels[..., 1:].contiguous()
            labels = labels.reshape(-1)
            loss = self.loss(self.lm_head.weight, embeddings, labels)
            num_items_in_batch = num_items_in_batch.to(loss.device)
            
            if output_router_logits:
                aux_loss = load_balancing_loss_func(
                    super_out.router_logits,
                    self.num_experts,
                    self.num_experts_per_tok,
                    attention_mask,
                )
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)

            loss = loss / num_items_in_batch
            return {'loss': loss}
        return super_out

class ExpertLoRA(nn.Module):
    def __init__(self, experts_module, r=4, alpha=1.0):
        super().__init__()
        self.m = experts_module
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        E = self.m.num_experts
        self.E = E
        self.H = self.m.hidden_size
        self.D = 2 * self.m.expert_dim
        self.F = self.m.expert_dim

        device = self.m.gate_up_proj.device
        dtype = self.m.gate_up_proj.dtype

        self.lora_gate_up_A = nn.ModuleDict({})
        self.lora_gate_up_B = nn.ModuleDict({})
        self.lora_down_A = nn.ModuleDict({})
        self.lora_down_B = nn.ModuleDict({})

        self.lora_gate_up_A['e'] = nn.Embedding(E, self.H * r, device=device, dtype=dtype)
        self.lora_gate_up_B['e'] = nn.Embedding(E, r * self.D, device=device, dtype=dtype)
        self.lora_down_A['e'] = nn.Embedding(E, self.F * r, device=device, dtype=dtype)
        self.lora_down_B['e'] = nn.Embedding(E, r * self.H, device=device, dtype=dtype)

        with torch.no_grad():
            init.kaiming_uniform_(self.lora_gate_up_A['e'].weight.view(E, self.H, r), a=math.sqrt(5))
            init.kaiming_uniform_(self.lora_down_A['e'].weight.view(E, self.F, r), a=math.sqrt(5))
            self.lora_gate_up_B['e'].weight.zero_()
            self.lora_down_B['e'].weight.zero_()
    
    def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None):
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.H)
        next_states = torch.zeros_like(hidden_states)
        
        with torch.no_grad():
            expert_mask = F.one_hot(router_indices, num_classes=self.E)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit_set = set(expert_mask.sum(dim=(-1, -2)).nonzero().squeeze(-1).tolist())
        
        all_indices = torch.arange(self.E, device=hidden_states.device)
        all_A = self.lora_gate_up_A['e'](all_indices)      # [E, H*r]
        all_B = self.lora_gate_up_B['e'](all_indices)      # [E, r*D]
        all_A2 = self.lora_down_A['e'](all_indices)        # [E, F*r]
        all_B2 = self.lora_down_B['e'](all_indices)        # [E, r*H]
        
        # Dummy op to ensure gradient flows to all params
        dummy = (all_A.sum() + all_B.sum() + all_A2.sum() + all_B2.sum()) * 0.0
        next_states = next_states + dummy
        
        for expert_idx in expert_hit_set:
            if expert_idx >= self.E:
                continue
            
            with torch.no_grad():
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            
            if len(token_idx) == 0:
                continue
            
            current_state = hidden_states[token_idx]
            
            A = all_A[expert_idx].view(self.H, self.r)
            B = all_B[expert_idx].view(self.r, self.D)
            
            lora_update = (current_state @ A) @ B * self.scaling
            
            gate_up = (current_state @ self.m.gate_up_proj[expert_idx] + lora_update) + self.m.gate_up_proj_bias[expert_idx]
            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            gate = gate.clamp(max=self.m.limit)
            up = up.clamp(min=-self.m.limit, max=self.m.limit)
            glu = gate * torch.sigmoid(gate * self.m.alpha)
            gated_output = (up + 1) * glu
            
            A2 = all_A2[expert_idx].view(self.F, self.r)
            B2 = all_B2[expert_idx].view(self.r, self.H)
            
            lora_update2 = (gated_output @ A2) @ B2 * self.scaling
            
            out = (gated_output @ self.m.down_proj[expert_idx] + lora_update2) + self.m.down_proj_bias[expert_idx]
            weighted_output = out * routing_weights[token_idx, top_k_pos, None]
            next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))
        
        return next_states.view(batch_size, -1, self.H)

class LinearLoRA(nn.Module):
    def __init__(self, linear: nn.Linear, r=4, alpha=1.0):
        super().__init__()
        self.linear = linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = linear.in_features
        out_features = linear.out_features
        
        device = self.linear.weight.device
        dtype = self.linear.weight.dtype

        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        
        self.lora_A['e'] = nn.Linear(
            in_features, r, bias=False, 
            device = device,
            dtype = dtype,
        )
        self.lora_B['e'] = nn.Linear(
            r, out_features, bias=False, 
            device = device,
            dtype = dtype,
        )

        for param in self.lora_A['e'].parameters():
            param.requires_grad = True
        for param in self.lora_B['e'].parameters():
            param.requires_grad = True

        init.kaiming_uniform_(self.lora_A['e'].weight, a=math.sqrt(5))
        init.zeros_(self.lora_B['e'].weight)

    def forward(self, x):
        out = self.linear(x)
        lora_update = self.lora_B['e'](self.lora_A['e'](x)) * self.scaling
        return out + lora_update

class ExtendTrainer(Trainer):
    def _fsdp_qlora_plugin_updates(self):
        from peft import PeftConfig
        from peft.utils.other import fsdp_auto_wrap_policy
        
        self.accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(self.model)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        import safetensors

        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        state_dict = {k:v for k, v in state_dict.items() if '.e.' in k}

        safetensors.torch.save_file(
            state_dict, os.path.join(output_dir, 'weight.pt'), metadata={"format": "pt"}
        )

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

            for k in data.keys():
                data[k] = data[k].astype(np.int64)
        
            return data

        def __len__(self):
            return len(self.dataset)

    model_kwargs = dict(
        attn_implementation="kernels-community/vllm-flash-attn3",
        torch_dtype=model_args.model_dtype,
        use_cache=False,
    )
    if 'bf16' not in model_args.model_name_or_path.lower():
        quantization_config = Mxfp4Config(dequantize=True)
        model_kwargs['quantization_config'] = quantization_config

    model = Model.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})

    if model_args.include_expert_lora:

        for name, param in model.named_parameters():
            param.requires_grad = False

        selected = [
            "q_proj", 
            "k_proj", 
            "v_proj", 
            "o_proj",
        ]

        for name, module in model.named_modules():
            for child_name, child in module.named_children():
                if len(child_name) and any([a in child_name for a in selected]) and isinstance(child, nn.Linear):
                    lora = LinearLoRA(child, r=model_args.rank, alpha=model_args.alpha)
                    setattr(module, child_name, lora)

                if child_name == 'experts' and isinstance(child, GptOssExperts):
                    lora = ExpertLoRA(child, r=model_args.rank, alpha=model_args.alpha)
                    setattr(module, child_name, lora)
    
    else:
        peft_config = LoraConfig(
            r=model_args.rank,
            lora_alpha=model_args.alpha,
            target_modules="all-linear",
        )
        model = get_peft_model(model, peft_config)

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

    trainer = ExtendTrainer(
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