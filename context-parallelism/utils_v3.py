"""
ChatGPT 5 kinda banger tho, at this point it much better than me for overlapping.
"""

import torch
import torch.distributed as dist
from typing import Optional, Tuple

causal_mask = lambda b, h, q_idx, kv_idx: q_idx >= kv_idx

def is_compiled_module(module):
    if not hasattr(torch, "_dynamo"):
        return False
    return isinstance(module, torch._dynamo.eval_frame.OptimizedModule)

@torch.jit.ignore
def _to_fp32(x):
    return x.float() if x.dtype in (torch.bfloat16, torch.float16) else x

def merge_attention(a, lse_a, b, lse_b):
    """
    Numerically stable merge in FP32, return original dtypes.
    a,b: [B, Lq, H, D], lse_*: [B, H, Lq]
    """
    if a is None:
        return b, lse_b
    a32, b32 = _to_fp32(a), _to_fp32(b)
    lse_a32, lse_b32 = _to_fp32(lse_a), _to_fp32(lse_b)
    max_lse = torch.maximum(lse_a32, lse_b32)
    lse_a_exp = torch.exp(lse_a32 - max_lse)
    lse_b_exp = torch.exp(lse_b32 - max_lse)
    denom = lse_a_exp + lse_b_exp
    out32 = (a32 * lse_a_exp[..., None] + b32 * lse_b_exp[..., None]) / denom[..., None]
    lse_out32 = torch.log(denom) + max_lse
    return out32.to(a.dtype), lse_out32.to(lse_a.dtype)

class RingComm:
    """
    NCCL ring communicator with its own CUDA stream and event-based synchronization.
    - send/recv are posted on self.stream
    - wait() makes the *current* stream wait on the posted ops (no host-block)
    """
    def __init__(self, process_group: dist.ProcessGroup, device: torch.device = None, stream: torch.cuda.Stream = None):
        self._pg = process_group
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)

        self._ops = []
        self._reqs = None
        self.stream = stream if stream is not None else torch.cuda.Stream(device=device)
        self._evt = torch.cuda.Event()

        # neighbors in logical ring (local indices)
        send_local = (self.rank + 1) % self.world_size
        recv_local = (self.rank - 1) % self.world_size
        # convert to global ranks for subgroup
        self.send_rank = dist.get_global_rank(process_group, send_local)
        self.recv_rank = dist.get_global_rank(process_group, recv_local)

    def send_recv(self, to_send: torch.Tensor, recv_buf: Optional[torch.Tensor]) -> torch.Tensor:
        if recv_buf is None:
            recv_buf = torch.empty_like(to_send)
        # enqueue ops (they'll be committed in post())
        self._ops.append(dist.P2POp(dist.isend, to_send, self.send_rank, group=self._pg))
        self._ops.append(dist.P2POp(dist.irecv, recv_buf, self.recv_rank, group=self._pg))
        return recv_buf

    def post(self):
        # push comm ops on our comm stream
        if not self._ops:
            return
        with torch.cuda.stream(self.stream):
            self._reqs = dist.batch_isend_irecv(self._ops)
            self._ops = []
            # record event after ops are launched; completing this event means comm finished
            self._evt.record(self.stream)

    def wait(self):
        """
        Make the current stream wait for posted comm to complete (device-side). Non-blocking on host.
        """
        # If nothing was posted, nothing to wait on.
        if self._reqs is None:
            return
        torch.cuda.current_stream().wait_event(self._evt)
        # Finalize reqs lazily (they should already be done once event is passed).
        for r in self._reqs:
            r.wait()
        self._reqs = None

    def post_kv(self, k: torch.Tensor, v: torch.Tensor, k_buf: torch.Tensor, v_buf: torch.Tensor):
        nk = self.send_recv(k, k_buf)
        nv = self.send_recv(v, v_buf)
        self.post()
        return nk, nv
