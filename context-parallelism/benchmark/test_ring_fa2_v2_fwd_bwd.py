import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from ring_fa2_v2 import ring_flash_attn
from flash_attn import flash_attn_func
import torch
import torch.distributed as dist
import time

def run():
    world_size = int(os.environ['LOCAL_WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    batch_size = 1
    seqlen = 8192
    nheads = 16
    d = 128

    device = torch.device(f'cuda:{local_rank}')
    dtype = torch.bfloat16

    assert seqlen % world_size == 0
    assert d % 8 == 0

    qkv = torch.randn(
        3, batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True,
    )
    dist.broadcast(qkv, src=0)

    dout = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
    dist.broadcast(dout, src=0)

    local_qkv = qkv.chunk(world_size, dim=-3)[local_rank]
    local_dout = dout.chunk(world_size, dim=-3)[local_rank].clone().detach()

    local_q = local_qkv[0].clone().detach().requires_grad_(True)
    local_k = local_qkv[1].clone().detach().requires_grad_(True)
    local_v = local_qkv[2].clone().detach().requires_grad_(True)

    local_out, local_lse = ring_flash_attn(q=local_q, k=local_k, v=local_v, causal=True)
    local_out.backward(local_dout)

if __name__ == "__main__":
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    
    warmup = 3
    repeat = 10
    for _ in range(warmup):
        run()
        torch.cuda.synchronize()
        dist.barrier()
    
    torch.cuda.synchronize()
    dist.barrier()
    start = time.time()
    for _ in range(repeat):
        run()
        torch.cuda.synchronize()
        dist.barrier()

    end = time.time()
    avg_time = (end - start) / repeat
    if local_rank == 0:
        print(f"Average step time: {avg_time:.4f} sec")

    dist.destroy_process_group()

"""
torchrun \
--nproc_per_node 4 \
benchmark/test_ring_fa2_v2_fwd_bwd.py
"""