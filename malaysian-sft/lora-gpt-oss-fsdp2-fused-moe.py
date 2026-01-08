import torch

torch._dynamo.config.capture_scalar_outputs = True
torch.set_float32_matmul_precision('high')

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
    Mxfp4Config,
)
from transformers.models.gpt_oss import modeling_gpt_oss
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssExperts,
    load_balancing_loss_func,
    GptOssDecoderLayer,
)
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from streaming import LocalDataset
from streaming.base.format.mds.encodings import Encoding, _encodings
from tqdm import tqdm
import numpy as np
import wandb
import click

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
        data.pop('audio', None)
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

class ExpertLoRAWeights(nn.Module):
    """Wrapper to make expert LoRA weights more FSDP-friendly"""
    def __init__(self, num_experts, in_dim, out_dim, r, dtype=torch.bfloat16):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(num_experts, in_dim, r, dtype=dtype))
        self.B = nn.Parameter(torch.zeros(num_experts, r, out_dim, dtype=dtype))
        
        with torch.no_grad():
            init.kaiming_uniform_(self.A, a=math.sqrt(5))

class GptOssExpertsParallel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim))
        self.gate_up_proj_bias = nn.Parameter(torch.empty(self.num_experts, 2 * self.expert_dim))
        self.down_proj = nn.Parameter(torch.empty((self.num_experts, self.expert_dim, self.hidden_size)))
        self.down_proj_bias = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))
        self.swiglu_alpha = 1.702
        self.limit = 7.0
        self._is_stacked = False
    
    def apply_lora_stack(self, r, alpha):
        if self._is_stacked:
            return

        self.r = r
        self.alpha = alpha
        self.alpha = alpha / r
        
        self._is_stacked = True

        self.gate_lora = ExpertLoRAWeights(
            self.num_experts, self.gate_up_proj.shape[1], self.gate_up_proj.shape[2], r
        )
        self.down_lora = ExpertLoRAWeights(
            self.num_experts, self.down_proj.shape[1], self.down_proj.shape[2], r
        )
    
    def gpt_oss_swiglu(self, x):
        """GptOss-style activation with interleaved gate/up and clamping"""
        gate, up = x[..., ::2], x[..., 1::2]
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.swiglu_alpha)
        return (up + 1) * glu

    def moe(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor, top_k: int):
        M = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[-1]

        sort_indices = topk_indices.view(-1).argsort()
        sorted_pos = sort_indices // top_k
        grouped_inputs = hidden_states[sorted_pos]

        experts_count = topk_indices.view(-1).bincount(minlength=self.num_experts)
        cu_experts_count = experts_count.cumsum(dim=0).to(torch.int32)

        gate_up_out = torch._grouped_mm(
            grouped_inputs,
            self.gate_up_proj,
            cu_experts_count,
        )
        
        if self._is_stacked and hasattr(self, 'gate_up_lora'):
            lora_A = torch._grouped_mm(grouped_inputs, self.gate_up_lora.A, cu_experts_count)
            lora_B = torch._grouped_mm(lora_A, self.gate_up_lora.B, cu_experts_count)
            gate_up_out = gate_up_out + lora_B * self.alpha

        # Add bias - need to repeat for each token assigned to each expert
        bias = self.gate_up_proj_bias.repeat_interleave(experts_count.long(), dim=0)
        # Handle padding if total assigned tokens < M * top_k
        tail_slack = grouped_inputs.shape[0] - int(cu_experts_count[-1])
        if tail_slack > 0:
            bias = torch.cat([bias, bias.new_zeros((tail_slack, bias.shape[-1]))], dim=0)
        gate_up_out = gate_up_out + bias.to(gate_up_out.dtype)

        intermediate = self.gpt_oss_swiglu(gate_up_out)

        down_out = torch._grouped_mm(
            intermediate,
            self.down_proj,
            cu_experts_count,
        )
        
        if self._is_stacked and hasattr(self, 'down_lora'):
            lora_A = torch._grouped_mm(intermediate, self.down_lora.A, cu_experts_count)
            lora_B = torch._grouped_mm(lora_A, self.down_lora.B, cu_experts_count)
            down_out = down_out + lora_B * self.alpha

        down_bias = self.down_proj_bias.repeat_interleave(experts_count.long(), dim=0)
        if tail_slack > 0:
            down_bias = torch.cat([down_bias, down_bias.new_zeros((tail_slack, down_bias.shape[-1]))], dim=0)
        down_out = down_out + down_bias.to(down_out.dtype)

        down_out = down_out * topk_weights.view(-1)[sort_indices].unsqueeze(-1)

        outputs = hidden_states.new_zeros(M, hidden_dim)
        sorted_pos_expanded = sorted_pos.unsqueeze(-1).expand(-1, hidden_dim)
        outputs.scatter_add_(0, sorted_pos_expanded, down_out.to(outputs.dtype))

        return outputs
    
    def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states_flat = hidden_states.reshape(-1, self.hidden_size)
        
        # router_indices shape: (num_tokens, top_k)
        
        # If routing_weights is (num_tokens, num_experts), we need to gather top_k weights
        if routing_weights.shape[1] == self.num_experts:
            topk_weights = torch.gather(routing_weights, 1, router_indices)
        else:
            topk_weights = routing_weights
        
        top_k = router_indices.shape[1]
        
        final_hidden_states = self.moe(hidden_states_flat, router_indices, topk_weights, top_k)
        final_hidden_states = final_hidden_states.view(batch_size, -1, self.hidden_size)
        
        return final_hidden_states

modeling_gpt_oss.GptOssExperts = GptOssExpertsParallel

class LinearLoRA(nn.Module):
    def __init__(self, linear: nn.Linear, r=4, alpha=1.0):
        super().__init__()
        self.linear = linear
        self.scaling = alpha / r

        in_features = linear.in_features
        out_features = linear.out_features
        
        self.lora_A = nn.Parameter(torch.zeros(r, in_features, dtype=torch.bfloat16))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, dtype=torch.bfloat16))
        
        with torch.no_grad():
            init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            # lora_B stays zero

    def forward(self, x):
        out = self.linear(x)
        lora_out = F.linear(F.linear(x.to(self.lora_A.dtype), self.lora_A), self.lora_B) * self.scaling
        return out + lora_out.to(out.dtype)

def check_fn(module):
    return isinstance(module, GptOssDecoderLayer) 

non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)

@click.command()
@click.option('--model_name', default='gfs/01be5b33/gpt-oss-20b-BF16', help='model name')
@click.option('--batch_size', default=4, help='batch size')
@click.option('--grad_accumulation', default=1, help='gradient accumulation')
def main(model_name, batch_size, grad_accumulation):
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
    device_mesh = init_device_mesh(device_type.type, (world_size,), mesh_dim_names=("dp",))
    dp_mesh = device_mesh["dp"]
    dp_rank = dp_mesh.get_local_rank()
    dp_world_size = dp_mesh.size()

    set_seed(42)
    checkpoint_dir = model_name.replace('/', '-')
    warmup_steps = 50
    learning_rate = 1e-4
    num_epoch = 3
    dataset = 'multipacking-gpt-oss'

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

    rank_lora = 256
    alpha_lora = 512

    for name, module in tqdm(model.named_modules(), desc="LoRA linear"):
        for child_name, child in module.named_children():
            if len(child_name) and any([a in child_name for a in selected]) and isinstance(child, nn.Linear):
                lora = LinearLoRA(child, r=rank_lora, alpha=alpha_lora)
                setattr(module, child_name, lora)
    
    top_k = model.config.num_experts_per_tok
    r = rank_lora // top_k
    alpha = alpha_lora // top_k
    
    for module in tqdm(model.modules(), desc="apply lora stack"):
        if isinstance(module, GptOssExpertsParallel):
            module.apply_lora_stack(r=r, alpha=alpha)

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

    for module in tqdm(model.modules(), desc="FSDP layer"):
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

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

            dist.barrier()

            sharded_sd = model.state_dict()
            cpu_state_dict = {}
                
            for param_name, sharded_param in sharded_sd.items():
                full_param = sharded_param.full_tensor()
                if rank == 0 and 'lora' in param_name:
                    cpu_state_dict[param_name] = full_param.cpu()
                else:
                    del full_param
            
            if rank == 0:
                torch.save(cpu_state_dict, os.path.join(checkpoint_dir, f'{step}-model_state_dict.pt'))

            dist.barrier()

        step += 1
        pbar.update(1)

if __name__ == "__main__":
    main()
