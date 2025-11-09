import torch
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from utils import merge_attention, RingComm

# Helper to allocate persistent double buffers
def _alloc_double_buffers_like(x):
    return [torch.empty_like(x), torch.empty_like(x)]

def _forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    causal: bool = True,
):
    """
    Overlap scheme:
      - kv_comm (comm stream) posts recv of next (k,v) while compute stream runs FlashAttention
      - compute stream only waits right before swapping in next buffers
      - Persistent double buffers avoid per-iteration allocations
    """
    device = q.device
    kv_comm = RingComm(process_group, device=device)  # its own comm stream

    out = None
    lse = None

    # Persistent double-buffers for incoming K/V
    k_bufs = _alloc_double_buffers_like(k)
    v_bufs = _alloc_double_buffers_like(v)
    buf_idx = 0

    world_size = kv_comm.world_size

    # Pre-post for step 0 (if more than 1 rank)
    next_k = next_v = None
    if world_size > 1:
        next_k, next_v = kv_comm.post_kv(k, v, k_bufs[buf_idx], v_bufs[buf_idx])

    for step in range(world_size):
        do_compute = (not causal) or (step <= kv_comm.rank)
        if do_compute:
            fwd_causal = causal and step == 0
            with torch.no_grad():
                block_out, block_lse, _, _ = _flash_attn_forward(
                    q, k, v,
                    dropout_p=0.0,
                    softmax_scale=scale,
                    causal=fwd_causal,
                    window_size_left=-1,
                    window_size_right=-1,
                    softcap=0.0,
                    alibi_slopes=None,
                    return_softmax=False,
                )
                block_lse = block_lse.transpose(1, 2)  # [B,H,Lq] -> consistent with merge
                out, lse = merge_attention(out, lse, block_out, block_lse)
                # Keep output dtype same as q (bf16/fp16 OK)
                out = out.to(q.dtype)

        # Prepare next iteration
        if step + 1 != world_size:
            # Wait for arrival of next_k/next_v (device-side)
            kv_comm.wait()

            # Swap in newly received tensors
            k, v = next_k, next_v

            # Flip buffer index and post next transfer immediately
            buf_idx ^= 1
            next_k, next_v = kv_comm.post_kv(k, v, k_bufs[buf_idx], v_bufs[buf_idx])

    # Ensure any pending comm is observed before returning
    kv_comm.wait()
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
    Backward overlap:
      - kv_comm (comm stream A): circulate K/V for per-step local grads
      - dkv_comm (comm stream B): circulate and *accumulate* dk/dv partials (reduce-in-ring)
      - Persistent per-block grad buffers; FP32 accumulation for dq/dk/dv, cast back at end
    """
    device = q.device
    kv_comm = RingComm(process_group, device=device)     # KV movement
    dkv_comm = RingComm(process_group, device=device)    # dK/dV reduction

    world_size = kv_comm.world_size

    # Persistent per-step buffers (match dtypes of inputs)
    block_dq = torch.empty_like(q)
    block_dk = torch.empty_like(k)
    block_dv = torch.empty_like(v)

    # FP32 accumulators for stability
    dq_accum = torch.zeros_like(q, dtype=torch.float32)
    dk_accum = None  # built via ring accumulation
    dv_accum = None

    # Double buffers for next K/V receives
    k_bufs = _alloc_double_buffers_like(k)
    v_bufs = _alloc_double_buffers_like(v)
    kv_buf_idx = 0

    # Double buffers for next dK/dV receives (float32 accum transport)
    dk_bufs = [torch.empty_like(k, dtype=torch.float32), torch.empty_like(k, dtype=torch.float32)]
    dv_bufs = [torch.empty_like(v, dtype=torch.float32), torch.empty_like(v, dtype=torch.float32)]
    dkv_buf_idx = 0

    # Pre-post next K/V for first iteration
    next_k = next_v = None
    if world_size > 1:
        next_k, next_v = kv_comm.post_kv(k, v, k_bufs[kv_buf_idx], v_bufs[kv_buf_idx])

    for step in range(world_size):
        do_compute = (not causal) or (step <= kv_comm.rank)

        # 1) Compute local per-step grads (dq_local, dk_local, dv_local)
        if do_compute:
            bwd_causal = causal and step == 0
            _flash_attn_backward(
                dout=dout,
                q=q,
                k=k,
                v=v,
                out=out,
                softmax_lse=lse,
                dq=block_dq,
                dk=block_dk,
                dv=block_dv,
                dropout_p=0.0,
                softmax_scale=scale,
                causal=bwd_causal,
                window_size_left=-1,
                window_size_right=-1,
                softcap=0.0,
                alibi_slopes=None,
                deterministic=False,
            )
            dq_accum.add_(block_dq.float())
            local_dk = block_dk.float()
            local_dv = block_dv.float()
        else:
            # No contribution this step
            local_dk = torch.zeros_like(k, dtype=torch.float32)
            local_dv = torch.zeros_like(v, dtype=torch.float32)

        # 2) Post dK/dV ring exchange for this step on dkv_comm stream
        recv_dk = dkv_comm.send_recv(local_dk, dk_bufs[dkv_buf_idx])
        recv_dv = dkv_comm.send_recv(local_dv, dv_bufs[dkv_buf_idx])
        dkv_comm.post()

        # 3) Prepare next K/V for next iteration (post now, compute keeps running)
        if step + 1 != world_size:
            kv_comm.wait()  # wait for next_k/next_v arrival (device-side)
            k, v = next_k, next_v
            kv_buf_idx ^= 1
            next_k, next_v = kv_comm.post_kv(k, v, k_bufs[kv_buf_idx], v_bufs[kv_buf_idx])

        # 4) Finish this step's dK/dV exchange and accumulate into running totals
        dkv_comm.wait()
        if dk_accum is None:
            dk_accum = recv_dk
            dv_accum = recv_dv
        else:
            dk_accum.add_(recv_dk)
            dv_accum.add_(recv_dv)

        # 5) Flip dkv receive buffers for next iteration
        dkv_buf_idx ^= 1

    # Ensure comms fully done
    kv_comm.wait()
    dkv_comm.wait()

    # Cast back to input dtype (bf16/fp16) at the end
    dq = dq_accum.to(q.dtype)
    dk = (dk_accum if dk_accum is not None else torch.zeros_like(k, dtype=torch.float32)).to(k.dtype)
    dv = (dv_accum if dv_accum is not None else torch.zeros_like(v, dtype=torch.float32)).to(v.dtype)
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
