import torch
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from utils import merge_attention
from utils_v2 import RingComm

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
    Overlap pattern:
      - post recv of next_k/next_v early (post_kv)
      - compute attention on current k/v (flash_attn)
      - wait for transfer to complete only when we need to swap in the received k/v
      - repeat
    """
    comm = RingComm(process_group)
    out = None
    lse = None

    k_buffer = torch.empty_like(k)
    v_buffer = torch.empty_like(v)

    next_k = next_v = None
    if comm.world_size > 1:
        next_k, next_v = comm.post_kv(k, v, k_buffer, v_buffer)

    for step in range(comm.world_size):
        do_compute = (not causal) or (step <= comm.rank)

        if do_compute:
            fwd_causal = causal and step == 0
            with torch.no_grad():
                block_out, block_lse, _, _ = _flash_attn_forward(
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
                block_lse = block_lse.transpose(1, 2)
                out, lse = merge_attention(out, lse, block_out, block_lse)
                out = out.to(q.dtype)

        if step + 1 != comm.world_size:
            comm.wait()

            k, v = next_k, next_v
            k_buffer = torch.empty_like(k)
            v_buffer = torch.empty_like(v)

            next_k, next_v = comm.post_kv(k, v, k_buffer, v_buffer)

    comm.wait()
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
    """
    Backward using same overlap/double-buffering idea for dk/dv reductions.
    We'll:
      - post recv of next k/v early
      - compute local grads (dq, dk_local, dv_local) via _flash_attn_backward into per-block buffers
      - accumulate local dq immediately
      - for dk/dv across blocks, use a second ring to send partial dk/dv around the ring.
    """

    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)

    dq = None
    block_dq_buffer = torch.empty_like(q)
    block_dk_buffer = torch.empty_like(k)
    block_dv_buffer = torch.empty_like(v)

    k_buffer = torch.empty_like(k)
    v_buffer = torch.empty_like(v)

    dk_buffer = torch.empty_like(k)
    dv_buffer = torch.empty_like(v)

    next_k = next_v = None
    if kv_comm.world_size > 1:
        next_k, next_v = kv_comm.post_kv(k, v, k_buffer, v_buffer)

    next_dk = next_dv = None

    for step in range(kv_comm.world_size):
        do_compute = (not causal) or (step <= kv_comm.rank)
        if do_compute:
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
                # use float32 accumulation for stability then cast back later
                dq = block_dq_buffer.to(torch.float32)
            else:
                dq += block_dq_buffer.to(torch.float32)

            local_dk = block_dk_buffer.to(torch.float32)
            local_dv = block_dv_buffer.to(torch.float32)
        else:
            local_dk = torch.zeros_like(k, dtype=torch.float32)
            local_dv = torch.zeros_like(v, dtype=torch.float32)

        next_dk_recv = d_kv_comm.send_recv(local_dk, dk_buffer)
        next_dv_recv = d_kv_comm.send_recv(local_dv, dv_buffer)
        d_kv_comm.post()

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k, v = next_k, next_v
            k_buffer = torch.empty_like(k)
            v_buffer = torch.empty_like(v)
            next_k, next_v = kv_comm.post_kv(k, v, k_buffer, v_buffer)

        d_kv_comm.wait()

        if next_dk is None:
            next_dk = next_dk_recv.to(torch.float32)
            next_dv = next_dv_recv.to(torch.float32)
        else:
            next_dk = next_dk + next_dk_recv.to(torch.float32)
            next_dv = next_dv + next_dv_recv.to(torch.float32)

    kv_comm.wait()
    d_kv_comm.wait()

    if dq is None:
        dq = torch.zeros_like(q)
    dq = dq.to(q.dtype)
    dk = next_dk.to(q.dtype) if next_dk is not None else torch.zeros_like(k)
    dv = next_dv.to(q.dtype) if next_dv is not None else torch.zeros_like(v)

    return dq, dk, dv


class RingFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale, causal, group):
        if scale is None:
            scale = q.shape[-1] ** (-0.5)

        k = k.contiguous()
        v = v.contiguous()
        out, lse = _forward(group, q, k, v, scale=scale, causal=causal)
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
        return dq, dk, dv, None, None, None


def ring_flash_attn(q, k, v, scale=None, causal=False, group=None):
    return RingFlashAttnFunc.apply(q, k, v, scale, causal, group)
