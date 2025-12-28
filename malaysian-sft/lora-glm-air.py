import torch

torch._dynamo.config.capture_scalar_outputs = True

import os
import math
import time
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
from functools import partial
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch import distributed as dist
from torch.distributed.tensor import distribute_tensor
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy
from transformers import (
    set_seed,
    get_linear_schedule_with_warmup,
    AutoConfig,
    AutoTokenizer,
    Glm4MoeForCausalLM,
)
from transformers.models.glm4_moe.modeling_glm4_moe import (
    Glm4MoeMLP,
    Glm4MoeDecoderLayer,
    Glm4MoeTopkRouter,
)
from transformers.models.glm4_moe import modeling_glm4_moe
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from streaming import LocalDataset
from streaming.base.format.mds.encodings import Encoding, _encodings
from tqdm import tqdm
import numpy as np
import wandb

class UInt32(Encoding):
    def encode(self, obj) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes):
        return np.frombuffer(data, np.uint32)

_encodings['uint32'] = UInt32

class Dataset(Dataset):
    def __init__(self, folder, sequence_length=16384):
        self.dataset = LocalDataset(local=folder)
        self.sequence_length = sequence_length
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        data.pop('audio', None)
        data.pop('text', None)
        data.pop('token_type_ids', None)

        for k in data.keys():
            data[k] = data[k].astype(np.int64)

        data['labels'] = data['input_ids'].copy()
        attention_mask_sum = data['attention_mask'].sum()
        
        if attention_mask_sum < self.sequence_length:
            balance = self.sequence_length - attention_mask_sum
            data['input_ids'] = np.concatenate([data['input_ids'], np.array([151329] * balance)])
            data['position_ids'] = np.concatenate([data['position_ids'], np.array([0] * balance)])
            data['labels'] = np.concatenate([data['labels'], np.array([-100] * balance)])
            data['attention_mask'] = np.concatenate([data['attention_mask'], np.array([balance])])
    
        return data
    
    def __len__(self):
        return len(self.dataset)

def collator(batch):
    batch = [b for b in batch if b is not None]
    input_ids = [b['input_ids'] for b in batch]
    position_ids = [b['position_ids'] for b in batch]
    labels = [b['labels'] for b in batch]
    attention_mask = [b['attention_mask'] for b in batch]
    input_ids = np.concatenate(input_ids)
    position_ids = np.concatenate(position_ids)
    labels = np.concatenate(labels)
    query_lens = np.concatenate(attention_mask)
    cumsum = [0] + np.cumsum(query_lens).tolist()
    max_cumsum = int(np.max(cumsum))
    cu_seq_lens_q = torch.tensor(cumsum, dtype=torch.int32)
    cu_seq_lens_k = torch.tensor(cumsum, dtype=torch.int32)
    max_seqlen_q = int(np.max(query_lens))
    return {
        'input_ids': torch.tensor(input_ids)[None],
        'position_ids': torch.tensor(position_ids)[None],
        'labels': torch.tensor(labels)[None],
        'cu_seq_lens_q': cu_seq_lens_q,
        'cu_seq_lens_k': cu_seq_lens_k,
        'max_length_q': max_seqlen_q,
        'max_length_k': max_seqlen_q
    }

class Model(Glm4MoeForCausalLM):
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

            loss = loss / num_items_in_batch
            return {'loss': loss}
        return super_out

class Glm4MoeMoEExpertParallel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device_mesh = self.config.device_mesh
        self.world_size = self.device_mesh.size()
        self.rank = self.device_mesh.get_local_rank()
        
        self.total_experts = config.n_routed_experts
        self.experts_per_rank = self.total_experts // self.world_size
        assert self.total_experts % self.world_size == 0, \
            f"n_routed_experts ({self.total_experts}) must be divisible by world_size ({self.world_size})"
        
        self.local_expert_start = self.rank * self.experts_per_rank
        self.local_expert_end = (self.rank + 1) * self.experts_per_rank
        self._is_sharded = False

        self.experts = nn.ModuleList(
            [
                Glm4MoeMLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = Glm4MoeTopkRouter(config)
        self.shared_experts = Glm4MoeMLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )

    def shard_experts(self):
        if self._is_sharded:
            return

        local_experts = nn.ModuleList([
            self.experts[i] for i in range(self.local_expert_start, self.local_expert_end)
        ])
        del self.experts
        torch.cuda.empty_cache()
        self.experts = local_experts
        self._is_sharded = True

    def moe(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor):
        if self._is_sharded:
            return self._moe_sharded(hidden_states, topk_indices, topk_weights)
        else:
            return self._moe_local(hidden_states, topk_indices, topk_weights)
    
    def _moe_local(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor):
        final_hidden_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype)
    
        num_experts = len(self.experts)
        expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=num_experts).permute(2, 1, 0)

        for expert_idx in range(num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            
            if top_x.numel() > 0:
                current_state = hidden_states[top_x]
                current_hidden_states = expert_layer(current_state) * topk_weights[top_x, idx, None]
                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
            else:
                dummy = expert_layer(hidden_states[:1]) * 0
                final_hidden_states = final_hidden_states + dummy.sum() * 0
        
        return final_hidden_states.type(hidden_states.dtype)
    
    def _moe_sharded(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor):
        final_hidden_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype)
        
        expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=self.total_experts).permute(2, 1, 0)
        
        for local_expert_idx in range(self.experts_per_rank):
            global_expert_idx = self.local_expert_start + local_expert_idx
            expert_layer = self.experts[local_expert_idx]
            
            idx, top_x = torch.where(expert_mask[global_expert_idx])
            
            if top_x.numel() > 0:
                current_state = hidden_states[top_x]
                current_hidden_states = expert_layer(current_state) * topk_weights[top_x, idx, None]
                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
            else:
                dummy = expert_layer(hidden_states[:1]) * 0
                final_hidden_states = final_hidden_states + dummy.sum() * 0
                
        torch.distributed.all_reduce(final_hidden_states, group=self.device_mesh.get_group())
        return final_hidden_states.type(hidden_states.dtype)

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states

modeling_glm4_moe.Glm4MoeMoE = Glm4MoeMoEExpertParallel

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
            dtype = torch.float32,
        )
        self.lora_B['e'] = nn.Linear(
            r, out_features, bias=False, 
            device = device,
            dtype = torch.float32,
        )

        for param in self.lora_A['e'].parameters():
            param.requires_grad = True
        for param in self.lora_B['e'].parameters():
            param.requires_grad = True

        # https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py#L260
        init.kaiming_uniform_(self.lora_A['e'].weight, a=math.sqrt(5))
        init.zeros_(self.lora_B['e'].weight)

    def forward(self, x):
        out = self.linear(x)
        lora_update = self.lora_B['e'](self.lora_A['e'](x.to(self.lora_A['e'].weight.dtype))) * self.scaling
        out = out + lora_update.to(x.dtype)
        return out

def check_fn(module):
    return isinstance(module, (Glm4MoeDecoderLayer, Glm4MoeMoEExpertParallel))

non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)

@torch.no_grad()
def clip_grad_norm_(parameters, max_norm, norm_type=2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)
    
    grads = [p.grad for p in parameters if p.grad is not None]
    
    norms = []
    for grad in grads:
        if hasattr(grad, 'full_tensor'):  # DTensor
            grad_full = grad.full_tensor()
        else:
            grad_full = grad
        
        if norm_type == float('inf'):
            norms.append(grad_full.abs().max())
        else:
            norms.append(grad_full.norm(norm_type))
    
    if len(norms) == 0:
        return torch.tensor(0.0)
    
    total_norm = torch.stack(norms).norm(norm_type)
    
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    
    for grad in grads:
        if hasattr(grad, 'full_tensor'):  # DTensor
            grad.mul_(clip_coef_clamped)
        else:
            grad.mul_(clip_coef_clamped)
    
    return total_norm

def main():
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    device_type = torch.accelerator.current_accelerator()
    device = torch.device(f"{device_type}:{rank}")
    torch.accelerator.device_index(rank)
    torch.cuda.set_device(rank)
    print(f"Running on rank {rank} on device {device}")
    
    backend = torch.distributed.get_default_backend_for_device(device)
    torch.distributed.init_process_group("cuda:nccl,cpu:gloo")
    num_threads = os.cpu_count() // (
        torch.cuda.device_count() if torch.cuda.is_available() else 1
    )
    torch.set_num_threads(num_threads)
    device_mesh = init_device_mesh(device_type.type, (world_size,), mesh_dim_names=("dp", "ep"))
    tp_mesh = device_mesh["dp"]
    dp_mesh = device_mesh["dp"]
    dp_rank = dp_mesh.get_local_rank()
    dp_world_size = dp_mesh.size()

    set_seed(42)
    model_name = "ramdisk/GLM-4.5-Air"
    checkpoint_dir = os.path.join('nfs/nfs', model_name.replace('/', '-'))
    warmup_steps = 50
    learning_rate = 1e-4
    num_epoch = 3
    dataset = 'multipacking-glm'
    batch_size = 2
    grad_accumulation = 16

    os.makedirs(checkpoint_dir, exist_ok=True)

    config = AutoConfig.from_pretrained(model_name)
    config.device_mesh = tp_mesh
    
    model = Model.from_pretrained(
        model_name, 
        attn_implementation="kernels-community/vllm-flash-attn3",
        torch_dtype=torch.bfloat16,
        config=config,
    )

    for module in model.modules():
        if isinstance(module, Glm4MoeMoEExpertParallel):
            module.shard_experts()

    for name, param in model.named_parameters():
        param.requires_grad = False

    selected = [
        "q_proj", 
        "k_proj", 
        "v_proj", 
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ]

    rank_lora = 256
    alpha_lora = 512
    top_k = model.config.num_experts_per_tok
    r = rank_lora // top_k
    alpha = alpha_lora // top_k

    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if len(child_name) and any([a in child_name for a in selected]) and isinstance(child, nn.Linear):
                if 'mlp.experts' in name:
                    selected_rank = r
                    selected_alpha = alpha
                else:
                    selected_rank = rank_lora
                    selected_alpha = alpha_lora

                lora = LinearLoRA(child, r=selected_rank, alpha=selected_alpha)
                setattr(module, child_name, lora)

    lora_mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
    )

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    ignored_params = set()
    for module in model.modules():
        if isinstance(module, Glm4MoeMoEExpertParallel):
            for param in module.experts.parameters():
                ignored_params.add(param)

    lora_mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
    )

    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if '.lora' in name and isinstance(child, nn.Linear):
                fully_shard(child, mp_policy=lora_mp_policy)

    fsdp_kwargs = {}
    fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    fsdp_kwargs["mesh"] = dp_mesh
    fsdp_kwargs["ignored_params"] = ignored_params

    for module in model.modules():
        if isinstance(module, Glm4MoeDecoderLayer):
            fully_shard(module, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)

    for module in model.modules():
        if isinstance(module, Glm4MoeMoEExpertParallel):
            module.experts = module.experts.to(device)

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=check_fn,
    )
    # model = torch.compile(model)

    dataset = Dataset(dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=5,
        prefetch_factor=5,
        pin_memory=True,
        collate_fn=collator,
    )
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=True)
    steps_per_epoch = len(train_loader) // grad_accumulation
    total_steps = steps_per_epoch * num_epoch
    scheduler = get_linear_schedule_with_warmup(
        optim, 
        warmup_steps, 
        num_training_steps=total_steps
    )

    if rank == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.size(), param.dtype)

    step = 0
    pbar = tqdm(total=total_steps, initial=step)
    iter_train_loader = iter(train_loader)
    if rank == 0:
        wandb.init()
    
    total_global_total_tokens = 0
    while step < total_steps:
        batches = []
        total_tokens = 0
        for _ in range(grad_accumulation):
            try:
                batch = next(iter_train_loader)
            except StopIteration:
                iter_train_loader = iter(train_loader)
                batch = next(iter_train_loader)
            valid_tokens = (batch['labels'] != -100).sum().item()
            total_tokens += valid_tokens
            batches.append(batch)

        token_tensor = torch.tensor([total_tokens], dtype=torch.long, device=device)
        dp_group = dp_mesh.get_group()
        dist.all_reduce(token_tensor, op=dist.ReduceOp.SUM, group=dp_group)
        global_total_tokens = token_tensor.item()
        total_global_total_tokens += global_total_tokens
        
        torch.cuda.synchronize()
        t0 = time.time()

        loss_sum = 0.0
        for b in batches:
            for k in b.keys():
                if isinstance(b[k], torch.Tensor):
                    b[k] = b[k].to(device, non_blocking=True)
            
            b['num_items_in_batch'] = torch.tensor(global_total_tokens)
            out = model(**b, use_cache=False)
            loss = out["loss"] * dp_world_size
            loss.backward()
            loss_sum += loss

        grad_norm = clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        scheduler.step()
        optim.zero_grad()

        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0

        throughput_per_sec = global_total_tokens / dt

        if rank == 0:
            scalar_dict = {
                "train/grad_norm": grad_norm,
                "train/learning_rate": scheduler.get_last_lr()[0],
                "train/loss": loss_sum,
                "train/global_step": step,
                "train/train_tokens_per_second": throughput_per_sec,
                "train/num_input_tokens_seen": total_global_total_tokens,
            }
            print(scalar_dict)
            try:
                wandb.log(scalar_dict)
            except Exception as e:
                print('failed pushed to wandb', e)

        if (step + 1) % steps_per_epoch == 0:
            print(f'saving checkpoint at {step}')
            sharded_sd = model.state_dict()
            cpu_state_dict = {}
            
            for param_name, sharded_param in sharded_sd.items():
                try:
                    full_param = sharded_param.full_tensor()
                except:
                    full_param = sharded_param
                    
                if '.lora' in param_name:
                    cpu_state_dict[param_name] = full_param.cpu()
                else:
                    del full_param
            
            torch.save(cpu_state_dict, os.path.join(checkpoint_dir, f'{step}-{rank}-model_state_dict.pt'))

        step += 1
        pbar.update(1)

if __name__ == "__main__":
    main()
