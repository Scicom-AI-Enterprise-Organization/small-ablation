import os
import math
import time
import torch
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

        lora_gate_up_A = nn.Parameter(torch.zeros(E, self.H, r, dtype=dtype))
        lora_gate_up_B = nn.Parameter(torch.zeros(E, r, self.D, dtype=dtype))

        lora_down_A = nn.Parameter(torch.zeros(E, self.F, r, dtype=dtype))
        lora_down_B = nn.Parameter(torch.zeros(E, r, self.H, dtype=dtype))

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

        self.lora_gate_up_A = distribute_tensor(
            lora_gate_up_A,
            device_mesh=tp_mesh,
            placements=[Shard(0)]
        )
        self.lora_gate_up_B = distribute_tensor(
            lora_gate_up_B,
            device_mesh=tp_mesh,
            placements=[Shard(0)]
        )
        self.lora_down_A = distribute_tensor(
            lora_down_A,
            device_mesh=tp_mesh,
            placements=[Shard(0)]
        )
        self.lora_down_B = distribute_tensor(
            lora_down_B,
            device_mesh=tp_mesh,
            placements=[Shard(0)]
        )

        del lora_gate_up_A, lora_gate_up_B, lora_down_A, lora_down_B

    def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None):
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.H)
        next_states = torch.zeros_like(hidden_states)
        
        with torch.no_grad():
            expert_mask = F.one_hot(router_indices, num_classes=self.E)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        
        for expert_idx in expert_hit[:]:
            expert_idx = expert_idx[0]

            if expert_idx == self.m.num_experts:
                continue
            
            with torch.no_grad():
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            
            current_state = hidden_states[token_idx]
            A = self.lora_gate_up_A[expert_idx].to_local()
            B = self.lora_gate_up_B[expert_idx].to_local()
            
            lora_update = (current_state @ A) @ B * self.scaling
            lora_update = lora_update
            
            gate_up = (current_state @ self.m.gate_up_proj[expert_idx].to_local() + lora_update) + self.m.gate_up_proj_bias[expert_idx].to_local()
            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            gate = gate.clamp(max=self.m.limit)
            up = up.clamp(min=-self.m.limit, max=self.m.limit)
            glu = gate * torch.sigmoid(gate * self.m.alpha)
            gated_output = (up + 1) * glu
            
            A = self.lora_down_A[expert_idx].to_local()
            B = self.lora_down_B[expert_idx].to_local()
            
            lora_update = (gated_output @ A) @ B * self.scaling
            
            out = (gated_output @ self.m.down_proj[expert_idx].to_local() + lora_update) + self.m.down_proj_bias[expert_idx].to_local()
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
    print(len(grads))
    
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
    checkpoint_dir = model_name.replace('/', '-')
    warmup_steps = 50
    learning_rate = 1e-4
    num_epoch = 3
    dataset = 'multipacking'
    batch_size = 1
    grad_accumulation = 8

    os.makedirs(checkpoint_dir, exist_ok=True)
    
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

    rank = 256
    alpha = 512
    top_k = model.config.num_experts_per_tok
    r = rank // top_k
    alpha = alpha // top_k

    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if len(child_name) and any([a in child_name for a in selected]) and isinstance(child, nn.Linear):
                lora = LinearLoRA(child, r=rank, alpha=alpha)
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
    model = torch.compile(model)

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

    steps_per_epoch = len(train_loader) // grad_accumulation
    total_steps = steps_per_epoch * num_epoch
    scheduler = get_linear_schedule_with_warmup(
        optim, 
        warmup_steps, 
        num_training_steps=total_steps
    )

    step = 0
    pbar = tqdm(total=total_steps, initial=step)
    iter_train_loader = iter(train_loader)
    if rank == 0:
        wandb.init()
    
    while step < total_steps:
        batches = []
        for _ in range(grad_accumulation):
            try:
                batch = next(iter_train_loader)
            except StopIteration:
                iter_train_loader = iter(loader)
                batch = next(iter_train_loader)
            batches.append(batch)
        
        torch.cuda.synchronize()
        t0 = time.time()

        for b in batches:
            for k in b.keys():
                if isinstance(b[k], torch.Tensor):
                    b[k] = b[k].to(device, non_blocking=True)
                
            out = model(**b, use_cache=False)
            loss = out["loss"] / grad_accumulation
            loss.backward()

        grad_norm = clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        scheduler.step()
        optim.zero_grad()

        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0

        throughput_per_sec = len(batches) * batch_size * 4096 / dt

        if (step + 1) % log_interval == 0 and rank == 0:
            scalar_dict = {
                "grad_norm": grad_norm,
                "lr_g": scheduler.get_last_lr()[0],
                "loss": loss.item() * grad_accumulation,
                "global_step": step,
                "throughput_per_sec": throughput_per_sec * world_size,
            }
            try:
                wandb.log(scalar_dict)
            except:
                pass

        if (step + 1) % steps_per_epoch == 0 and rank == 0:
            sharded_sd = model.state_dict()
            cpu_state_dict = {}
            for param_name, sharded_param in sharded_sd.items():
                full_param = sharded_param.full_tensor()
                if torch.distributed.get_rank() == 0:
                    cpu_state_dict[param_name] = full_param.cpu()
                else:
                    del full_param
            
            torch.save(cpu_state_dict, f"{checkpoint_dir}/model_state_dict-{step}.pt")
        
        step += 1
        pbar.update(1)

        

if __name__ == "__main__":
    main()
