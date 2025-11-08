"""
Heavily borrow from https://github.com/zhuzilin/ring-flash-attention/blob/main/ring_flash_attn/ring_flash_attn.py
but update latest parameters for FA2 2.8.3
"""

import torch
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from utils import merge_attention, RingComm

def _forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    causal: bool = True,
):
    """
    q: [B, L, H, D]
    k: [B, L, H, D]
    v: [B, L, H, D]
    """
    comm = RingComm(process_group)
    out = None
    lse = None
    next_k, next_v = None, None

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k, next_v = comm.send_recv_kv(k, v)

        if not causal or step <= comm.rank:
            with torch.no_grad():
                fwd_causal = causal and step == 0

                outputs = _flash_attn_forward(
                    q, 
                    k,
                    v,
                    dropout_p=0, 
                    softmax_scale=scale, 
                    causal=fwd_causal,
                    window_size_left=-1,
                    window_size_right=-1,
                    softcap=0.0,
                    alibi_slopes=None,
                    return_softmax=False,
                )
                block_out, block_lse, _, _ = outputs
                block_lse = block_lse.transpose(1, 2)
                out, lse = merge_attention(out, lse, block_out, block_lse)
                out = out.to(q.dtype)
        
        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_k, next_v
    
    return out, lse

def _backward(
    process_group,
    dout: torch.Tensor,
    dlse: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    scale: float,
    causal: bool = True,
):
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    dq, dk, dv = None, None, None

    block_dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    block_dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    block_dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)

    next_dk, next_dv = None, None
    next_k, next_v = None, None

    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k, next_v = kv_comm.send_recv_kv(k, v)

        if not causal or step <= kv_comm.rank:
            bwd_causal = causal and step == 0

            _flash_attn_backward(
                dout=dout,
                q=q,
                k=k,
                v=v,
                out=out,
                softmax_lse=lse,
                dq=block_dq_buffer,
                dk=block_dk_buffer,
                dv=block_dv_buffer,
                dropout_p=0, 
                softmax_scale=scale, 
                causal=bwd_causal,
                window_size_left=-1,
                window_size_right=-1,
                softcap=0.0,
                alibi_slopes=None,
                deterministic=False,
            )

            if dq is None:
                dq = block_dq_buffer.to(torch.float32)
                dk = block_dk_buffer.to(torch.float32)
                dv = block_dv_buffer.to(torch.float32)
            else:
                dq += block_dq_buffer
                d_kv_comm.wait()
                dk = block_dk_buffer + next_dk
                dv = block_dv_buffer + next_dv
        
        elif step != 0:
            d_kv_comm.wait()
            dk, dv = next_dk, next_dv

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k, v = next_k, next_v
        
        next_dk, next_dv = d_kv_comm.send_recv_kv(dk, dv)
    
    d_kv_comm.wait()
    
    return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)

class RingFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        scale,
        causal,
        group,
    ):
        if scale is None:
            scale = q.shape[-1] ** (-0.5)

        k = k.contiguous()
        v = v.contiguous()
        out, lse = _forward(group, q, k, v, scale=scale)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.scale = scale
        ctx.causal = causal
        ctx.group = group
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse, *args):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _backward(
            ctx.group,
            dout,
            dlse,
            q,
            k,
            v,
            out,
            lse,
            scale=ctx.scale,
            causal=ctx.causal,
        )
        return dq, dk, dv, None, None, None, None

def ring_flash_attn(
    q,
    k,
    v,
    scale=None,
    causal=False,
    group=None,
):
    return RingFlashAttnFunc.apply(q, k, v, scale, causal, group)