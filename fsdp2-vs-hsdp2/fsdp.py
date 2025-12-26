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
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    set_seed,
    get_wsd_schedule,
    Qwen3ForCausalLM,
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy
from liger_kernel.transformers import apply_liger_kernel_to_qwen3
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

        # https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py#L260
        init.kaiming_uniform_(self.lora_A['e'].weight, a=math.sqrt(5))
        init.zeros_(self.lora_B['e'].weight)

    def forward(self, x):
        out = self.linear(x)
        lora_update = self.lora_B['e'](self.lora_A['e'](x.to(self.lora_A['e'].weight.dtype))) * self.scaling
        return out + lora_update.to(x.dtype)
        
def check_fn(module):
    return isinstance(module, Qwen3DecoderLayer) 

non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)

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
    device_mesh = init_device_mesh(device_type.type, (world_size,), mesh_dim_names=("dp",))
    dp_mesh = device_mesh["dp"]
    dp_rank = dp_mesh.get_local_rank()
    dp_world_size = dp_mesh.size()

    apply_liger_kernel_to_qwen3(
        rope=False,
        swiglu=True,
        rms_norm=True,
        cross_entropy=False,
        fused_linear_cross_entropy=True,
    )

    set_seed(42)
    model_name = "Qwen/Qwen3-14B"
    warmup_steps = 20
    learning_rate = 1e-4
    log_interval = 1
    total_steps = 200
    dataset = 'multipacking'
    batch_size = 8
    grad_accumulation = 1

    model = Qwen3ForCausalLM.from_pretrained(
        model_name, 
        attn_implementation='flash_attention_3',
        torch_dtype=torch.bfloat16,
    )

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
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if len(child_name) and any([a in child_name for a in selected]) and isinstance(child, nn.Linear):
                lora = LinearLoRA(child, r=128, alpha=256)
                setattr(module, child_name, lora)

    fsdp_kwargs = {}
    fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    # only save some memory
    # but do not forgot torch.distributed.init_process_group("cuda:nccl,cpu:gloo")
    # check the comment https://github.com/axolotl-ai-cloud/axolotl/issues/3058#issuecomment-3177615390
    # fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()
    for module in model.modules():
        if isinstance(module, Qwen3DecoderLayer):
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
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    scheduler = get_wsd_schedule(
        optim, 
        warmup_steps, 
        int(total_steps * 0.2), 
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
                iter_train_loader = iter(train_loader)
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

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        
        step += 1
        pbar.update(1)

if __name__ == "__main__":
    main()
