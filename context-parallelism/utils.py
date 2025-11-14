import torch
import torch.distributed as dist
import torch.nn.functional as F
from typing import Optional, Tuple


causal_mask = lambda b, h, q_idx, kv_idx: q_idx >= kv_idx

def is_compiled_module(module):
    """
    Check whether the module was compiled with torch.compile()
    """
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

# https://github.com/zhuzilin/ring-flash-attention/blob/main/ring_flash_attn/utils.py#L53

class RingComm:
    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self._ops = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None

        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)

    def send_recv(
        self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = recv_tensor

        send_op = dist.P2POp(
            dist.isend, to_send, self.send_rank, group=self._process_group
        )
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []

    def send_recv_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_buffer: Optional[torch.Tensor] = None,
        v_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        next_k, next_v = self.send_recv(k, k_buffer), self.send_recv(v, v_buffer)
        self.commit()
        return next_k, next_v