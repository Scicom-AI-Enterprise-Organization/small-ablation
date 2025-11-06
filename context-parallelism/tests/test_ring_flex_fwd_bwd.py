import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from ring_flex import ring_flex_attn
from utils import causal_mask
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
import torch
import torch.distributed as dist
import os
import math

if __name__ == "__main__":
    world_size = int(os.environ['LOCAL_WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(backend='nccl')

    batch_size = 1
    seqlen = 150
    nheads = 16
    d = 128

    device = torch.device(f'cuda:{local_rank}')
    dtype = torch.bfloat16

    assert seqlen % world_size == 0
    assert d % 8 == 0

    qkv = torch.randn(
        3, batch_size, nheads, seqlen, d, device=device, dtype=dtype, requires_grad=True,
    )
    dist.broadcast(qkv, src=0)
    q = qkv[0].clone().detach().requires_grad_(True)
    k = qkv[1].clone().detach().requires_grad_(True)
    v = qkv[2].clone().detach().requires_grad_(True)

    dout = torch.randn(batch_size, nheads, seqlen, d, device=device, dtype=dtype)
    dist.broadcast(dout, src=0)

    scale = q.shape[-1] ** (-0.5)
    block_mask = create_block_mask(causal_mask, None, None, q.shape[-2], q.shape[-2], device = local_rank)

    out, lse = flex_attention(q, k, v, block_mask=block_mask, scale=scale, return_lse = True)
    out.backward(dout)

    local_qkv = qkv.chunk(world_size, dim=-2)[local_rank]
    local_dout = dout.chunk(world_size, dim=2)[local_rank].clone().detach()

    local_q = local_qkv[0].clone().detach().requires_grad_(True)
    local_k = local_qkv[1].clone().detach().requires_grad_(True)
    local_v = local_qkv[2].clone().detach().requires_grad_(True)

    local_out, local_lse = ring_flex_attn(q=local_q, k=local_k, v=local_v, causal=True, _compile=True)
    local_out.backward(local_dout)

    length = local_q.shape[-2]
    start_length = int(local_rank * length)
    end_length = int((local_rank + 1) * length)

    assert torch.allclose(local_out, out[:,:,start_length:end_length], atol=0.125, rtol=0)
    assert torch.allclose(local_lse, lse[:,:,start_length:end_length], atol=0.125, rtol=0)
    assert torch.allclose(local_q.grad, q.grad[:,:,start_length:end_length], atol=0.125, rtol=0)
    assert torch.allclose(local_k.grad, k.grad[:,:,start_length:end_length], atol=0.125, rtol=0)
    assert torch.allclose(local_v.grad, v.grad[:,:,start_length:end_length], atol=0.125, rtol=0)

"""
torchrun \
--nproc_per_node 3 \
tests/test_ring_flex_fwd_bwd.py
"""