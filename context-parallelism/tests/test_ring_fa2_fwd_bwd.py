import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import torch
import torch.distributed as dist
from ring_fa2 import ring_flash_attn
from flash_attn import flash_attn_func

if __name__ == "__main__":
    world_size = int(os.environ['LOCAL_WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(backend='nccl')

    batch_size = 1
    seqlen = 100 * world_size
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
    q = qkv[0].clone().detach().requires_grad_(True)
    k = qkv[1].clone().detach().requires_grad_(True)
    v = qkv[2].clone().detach().requires_grad_(True)

    dout = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
    dist.broadcast(dout, src=0)

    out = flash_attn_func(q, k, v, causal=True)
    out.backward(dout)

    local_qkv = qkv.chunk(world_size, dim=-3)[local_rank]
    local_dout = dout.chunk(world_size, dim=-3)[local_rank].clone().detach()

    local_q = local_qkv[0].clone().detach().requires_grad_(True)
    local_k = local_qkv[1].clone().detach().requires_grad_(True)
    local_v = local_qkv[2].clone().detach().requires_grad_(True)

    local_out, local_lse = ring_flash_attn(q=local_q, k=local_k, v=local_v, causal=True)
    local_out.backward(local_dout)

    length = local_q.shape[-3]
    start_length = int(local_rank * length)
    end_length = int((local_rank + 1) * length)

    assert torch.allclose(local_out, out[:,start_length:end_length], atol=0.125, rtol=0)
    assert torch.allclose(local_q.grad, q.grad[:,start_length:end_length], atol=0.125, rtol=0)
    assert torch.allclose(local_k.grad, k.grad[:,start_length:end_length], atol=0.125, rtol=0)
    assert torch.allclose(local_v.grad, v.grad[:,start_length:end_length], atol=0.125, rtol=0)

"""
torchrun \
--nproc_per_node 8 \
tests/test_ring_fa2_fwd_bwd.py
"""