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
from torch.distributed._tensor import Shard, Replicate
from transformers import (
    set_seed,
    get_linear_schedule_with_warmup,
    GptOssForCausalLM,
)
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts
from transformers.models.gpt_oss.modeling_gpt_oss import load_balancing_loss_func, GptOssDecoderLayer
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
    def __init__(self, folder):
        self.dataset = LocalDataset(local=folder)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        data.pop('text', None)
        data.pop('token_type_ids', None)

        for k in data.keys():
            data[k] = data[k].astype(np.int64)
    
        return data
    
    def __len__(self):
        return len(self.dataset)

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
    def __init__(self, experts_module, tp_mesh, r=4, alpha=1.0):
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

        lora_gate_up_A = torch.zeros(E, self.H, r, dtype=dtype)
        lora_gate_up_B = torch.zeros(E, r, self.D, dtype=dtype)
        lora_down_A = torch.zeros(E, self.F, r, dtype=dtype)
        lora_down_B = torch.zeros(E, r, self.H, dtype=dtype)

        with torch.no_grad():
            # https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py#L260
            init.kaiming_uniform_(lora_gate_up_A, a=math.sqrt(5))
            init.kaiming_uniform_(lora_down_A, a=math.sqrt(5))

        a = self.m.gate_up_proj
        del self.m.gate_up_proj
        self.m.gate_up_proj = distribute_tensor(
            a,
            device_mesh=tp_mesh,
            placements=[Shard(0)]
        )

        a = self.m.gate_up_proj_bias
        del self.m.gate_up_proj_bias
        self.m.gate_up_proj_bias = distribute_tensor(
            a,
            device_mesh=tp_mesh,
            placements=[Shard(0)]
        )

        a = self.m.down_proj
        del self.m.down_proj
        self.m.down_proj = distribute_tensor(
            a,
            device_mesh=tp_mesh,
            placements=[Shard(0)]
        )

        a = self.m.down_proj_bias
        del self.m.down_proj_bias
        self.m.down_proj_bias = distribute_tensor(
            a,
            device_mesh=tp_mesh,
            placements=[Shard(0)]
        )

        self.lora_gate_up_A = nn.Parameter(distribute_tensor(
            lora_gate_up_A,
            device_mesh=tp_mesh,
            placements=[Shard(0)]
        ))
        self.lora_gate_up_B = nn.Parameter(distribute_tensor(
            lora_gate_up_B,
            device_mesh=tp_mesh,
            placements=[Shard(0)]
        ))
        self.lora_down_A = nn.Parameter(distribute_tensor(
            lora_down_A,
            device_mesh=tp_mesh,
            placements=[Shard(0)]
        ))
        self.lora_down_B = nn.Parameter(distribute_tensor(
            lora_down_B,
            device_mesh=tp_mesh,
            placements=[Shard(0)]
        ))

        self.tp_mesh = tp_mesh
        self.world_size = tp_mesh.size()
        self.experts_per_rank = E // self.world_size

        del lora_gate_up_A, lora_gate_up_B, lora_down_A, lora_down_B

    def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None):
        """Optimized forward using batched operations - no expert loop"""
        print(f"SHAPES - hidden_states: {hidden_states.shape}, router_indices: {router_indices.shape}, routing_weights: {routing_weights.shape}")
        original_shape = hidden_states.shape  # [batch, seq, H] or [seq, H]
        batch_size = hidden_states.shape[0]
        
        # Flatten hidden states to 2D: [total_tokens, H]
        hidden_states = hidden_states.reshape(-1, self.H)
        num_tokens = hidden_states.shape[0]
        
        # Get top_k from the last dimension of router_indices
        top_k = router_indices.shape[-1]
        
        # Flatten router_indices and routing_weights to 2D: [total_tokens, top_k]
        router_indices = router_indices.reshape(-1, top_k)  # [num_tokens, top_k]
        routing_weights = routing_weights.reshape(-1, top_k)  # [num_tokens, top_k]
        
        # Verify shapes match
        assert router_indices.shape[0] == num_tokens, \
            f"Shape mismatch: hidden_states has {num_tokens} tokens but router_indices has {router_indices.shape[0]}"
        
        local_rank = self.tp_mesh.get_local_rank()
        local_start = local_rank * self.experts_per_rank
        local_end = local_start + self.experts_per_rank
        
        # Get local weights
        gate_up_proj_local = self.m.gate_up_proj.to_local()      
        gate_up_bias_local = self.m.gate_up_proj_bias.to_local() 
        down_proj_local = self.m.down_proj.to_local()            
        down_bias_local = self.m.down_proj_bias.to_local()       
        
        lora_gate_up_A_local = self.lora_gate_up_A.to_local()    
        lora_gate_up_B_local = self.lora_gate_up_B.to_local()    
        lora_down_A_local = self.lora_down_A.to_local()          
        lora_down_B_local = self.lora_down_B.to_local()          
        
        # Now flatten to 1D for processing all token-expert pairs
        # [num_tokens, top_k] -> [num_tokens * top_k]
        flat_expert_indices = router_indices.reshape(-1)  # [num_tokens * top_k]
        flat_routing_weights = routing_weights.reshape(-1)  # [num_tokens * top_k]
        total_pairs = flat_expert_indices.shape[0]  # num_tokens * top_k
        
        # Token indices: which original token does each pair belong to
        # [0,0,0,...,1,1,1,...,2,2,2,...] for top_k repeats
        token_indices = torch.arange(num_tokens, device=hidden_states.device)
        token_indices = token_indices.unsqueeze(1).expand(-1, top_k).reshape(-1)  # [num_tokens * top_k]
        
        # Expand hidden states for all top_k selections
        expanded_hidden = hidden_states[token_indices]  # [num_tokens * top_k, H]
        
        # Filter to local experts - mask should be [num_tokens * top_k]
        local_mask = (flat_expert_indices >= local_start) & (flat_expert_indices < local_end)
        
        next_states = torch.zeros_like(hidden_states)  # [num_tokens, H]
        
        if not local_mask.any():
            dist.all_reduce(next_states, group=self.tp_mesh.get_group())
            return next_states.view(original_shape)
        
        # Apply mask - all these are [M] where M = number of local expert assignments
        local_expert_indices = flat_expert_indices[local_mask] - local_start
        local_hidden = expanded_hidden[local_mask]            # [M, H]
        local_weights = flat_routing_weights[local_mask]      # [M]
        local_token_indices = token_indices[local_mask]       # [M]
        
        # Sort by expert for better memory access
        sorted_order = torch.argsort(local_expert_indices, stable=True)
        sorted_expert_indices = local_expert_indices[sorted_order]
        sorted_hidden = local_hidden[sorted_order]
        sorted_weights = local_weights[sorted_order]
        sorted_token_indices = local_token_indices[sorted_order]
        
        # Gather expert weights
        sel_gate_up = gate_up_proj_local[sorted_expert_indices]
        sel_gate_up_bias = gate_up_bias_local[sorted_expert_indices]
        sel_lora_A = lora_gate_up_A_local[sorted_expert_indices]
        sel_lora_B = lora_gate_up_B_local[sorted_expert_indices]
        
        # Batched gate_up
        sorted_hidden_3d = sorted_hidden.unsqueeze(1)  # [M, 1, H]
        gate_up_main = torch.bmm(sorted_hidden_3d, sel_gate_up).squeeze(1)
        lora_inter = torch.bmm(sorted_hidden_3d, sel_lora_A)
        lora_gate_up = torch.bmm(lora_inter, sel_lora_B).squeeze(1) * self.scaling
        gate_up = gate_up_main + lora_gate_up + sel_gate_up_bias
        
        # Activation
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(max=self.m.limit)
        up = up.clamp(min=-self.m.limit, max=self.m.limit)
        glu = gate * torch.sigmoid(gate * self.m.alpha)
        gated_output = (up + 1) * glu
        
        # Batched down projection
        sel_down = down_proj_local[sorted_expert_indices]
        sel_down_bias = down_bias_local[sorted_expert_indices]
        sel_lora_down_A = lora_down_A_local[sorted_expert_indices]
        sel_lora_down_B = lora_down_B_local[sorted_expert_indices]
        
        gated_output_3d = gated_output.unsqueeze(1)
        down_main = torch.bmm(gated_output_3d, sel_down).squeeze(1)
        lora_down_inter = torch.bmm(gated_output_3d, sel_lora_down_A)
        lora_down = torch.bmm(lora_down_inter, sel_lora_down_B).squeeze(1) * self.scaling
        out = down_main + lora_down + sel_down_bias
        
        # Apply routing weights and scatter
        weighted_output = out * sorted_weights.unsqueeze(1)
        next_states.index_add_(0, sorted_token_indices, weighted_output.to(hidden_states.dtype))
        
        dist.all_reduce(next_states, group=self.tp_mesh.get_group())
        
        return next_states.view(original_shape)

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
        
        self.lora_A = nn.Linear(
            in_features, r, bias=False, 
            device = device,
            dtype = dtype,
        )
        self.lora_B = nn.Linear(
            r, out_features, bias=False, 
            device = device,
            dtype = dtype,
        )

        for param in self.lora_A.parameters():
            param.requires_grad = True
        for param in self.lora_B.parameters():
            param.requires_grad = True

        # https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py#L260
        init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        init.zeros_(self.lora_B.weight)

    def forward(self, x):
        out = self.linear(x)
        lora_update = self.lora_B(self.lora_A(x.to(self.lora_A.weight.dtype))) * self.scaling
        return out + lora_update.to(x.dtype)

def check_fn(module):
    return isinstance(module, GptOssDecoderLayer) 

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
    print(f"Running on rank {rank} on device {device}")
    
    backend = torch.distributed.get_default_backend_for_device(device)
    torch.distributed.init_process_group("cuda:nccl,cpu:gloo")
    num_threads = os.cpu_count() // (
        torch.cuda.device_count() if torch.cuda.is_available() else 1
    )
    torch.set_num_threads(num_threads)
    device_mesh = init_device_mesh(device_type.type, (4, 2), mesh_dim_names=("dp", "tp"))
    tp_mesh = device_mesh["tp"]
    dp_mesh = device_mesh["dp"]
    dp_rank = dp_mesh.get_local_rank()
    dp_world_size = dp_mesh.size()

    set_seed(42)
    model_name = "unsloth/gpt-oss-20b-BF16"
    warmup_steps = 50
    learning_rate = 1e-4
    total_steps = 200
    dataset = 'malaysian-reasoning-16k-mosaic'
    batch_size = 4
    grad_accumulation = 4
    
    model = Model.from_pretrained(
        model_name, 
        attn_implementation="kernels-community/vllm-flash-attn3",
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )

    for name, param in model.named_parameters():
        param.requires_grad = False

    selected = [
        "q_proj", 
        "k_proj", 
        "v_proj", 
        "o_proj",
    ]

    rank_lora = 256
    alpha_lora = 512
    top_k = model.config.num_experts_per_tok
    r = rank_lora // top_k
    alpha = alpha_lora // top_k

    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if len(child_name) and any([a in child_name for a in selected]) and isinstance(child, nn.Linear):
                lora = LinearLoRA(child, r=rank_lora, alpha=alpha_lora)
                setattr(module, child_name, lora)

            if child_name == 'experts' and isinstance(child, GptOssExperts):
                lora = ExpertLoRA(child, tp_mesh=tp_mesh, r=r, alpha=alpha)
                setattr(module, child_name, lora)

    fsdp_kwargs = {}
    fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    fsdp_kwargs["mesh"] = dp_mesh
    # only save some memory
    # but do not forgot torch.distributed.init_process_group("cuda:nccl,cpu:gloo")
    # check the comment https://github.com/axolotl-ai-cloud/axolotl/issues/3058#issuecomment-3177615390
    # fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

    for module in model.modules():
        if isinstance(module, GptOssDecoderLayer):
            fully_shard(module, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=check_fn,
    )
    # model = torch.compile(model)

    dataset = Dataset(dataset)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dp_world_size,
        rank=dp_rank,
        shuffle=True,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=5,
        prefetch_factor=5,
        pin_memory=True,
        collate_fn=collator,
    )
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=True)
    scheduler = get_linear_schedule_with_warmup(
        optim, 
        warmup_steps, 
        num_training_steps=total_steps
    )

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.size(), param.dtype)

    step = 0
    pbar = tqdm(total=total_steps, initial=step)
    iter_train_loader = iter(train_loader)
    if rank == 0:
        wandb.init()

    # if rank == 0:
    #     torch.cuda.memory._record_memory_history(max_entries=100000)
    
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
            total_tokens += batch['input_ids'].shape[1]
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
            try:
                wandb.log(scalar_dict)
            except Exception as e:
                print('failed pushed to wandb', e)
        
        step += 1
        pbar.update(1)

    # if rank == 0:
    #     try:
    #         torch.cuda.memory._dump_snapshot(f"fsdp2-ep.pickle")
    #     except Exception as e:
    #        print(f"Failed to capture memory snapshot {e}")
    
    #     # Stop recording memory snapshot history.
    #     torch.cuda.memory._record_memory_history(enabled=None)

if __name__ == "__main__":
    main()
