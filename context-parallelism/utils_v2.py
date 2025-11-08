import torch
import torch.distributed as dist
from typing import Optional, Tuple

causal_mask = lambda b, h, q_idx, kv_idx: q_idx >= kv_idx

def is_compiled_module(module):
    """
    Check whether the module was compiled with torch.compile()
    """
    if not hasattr(torch, "_dynamo"):
        return False
    return isinstance(module, torch._dynamo.eval_frame.OptimizedModule)

def merge_attention(a, lse_a, b, lse_b):
    if a is None:
        return b, lse_b
    max_lse = torch.maximum(lse_a, lse_b)
    lse_a_exp = torch.exp(lse_a - max_lse)
    lse_b_exp = torch.exp(lse_b - max_lse)
    out = ((a * lse_a_exp[..., None] + b * lse_b_exp[..., None]) / (lse_a_exp + lse_b_exp)[..., None])
    return out, torch.log(lse_a_exp + lse_b_exp) + max_lse

class RingComm:
    """
    Ring communicator which allows explicit post (enqueue+commit) and wait to enable overlap.
    Usage pattern (double-buffering):
      buf_next_k, buf_next_v = ring.post_kv(k, v, k_buffer, v_buffer)  # starts async send/recv
      # do local compute on current k/v while transfer happens
      ring.wait()  # wait for the outstanding transfer if/when we need the received tensors
      k, v = buf_next_k, buf_next_v
    """

    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self._ops = []
        self._reqs = None
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)

        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)

    def _append_p2p(self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor]) -> torch.Tensor:
        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = recv_tensor

        send_op = dist.P2POp(dist.isend, to_send, self.send_rank, group=self._process_group)
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def post(self):
        """
        Commit any ops appended so far (start the async send/recv).
        After post() returns, requests are running asynchronously.
        """
        if self._reqs is not None:
            raise RuntimeError("post called while there are outstanding requests")

        self._reqs = dist.batch_isend_irecv(self._ops)
        self._ops = []

    def wait(self):
        """
        Wait for outstanding requests (if any) to complete.
        """
        if self._reqs is None:
            return
        for r in self._reqs:
            r.wait()
        self._reqs = None

    def send_recv(self, to_send: torch.Tensor, recv_buf: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Append a single pair of send/recv ops, but DO NOT commit. This returns the recv buffer
        object which will be filled once you call post() and then wait().
        """
        return self._append_p2p(to_send, recv_buf)

    def post_kv(self,
                k: torch.Tensor,
                v: torch.Tensor,
                k_buffer: Optional[torch.Tensor] = None,
                v_buffer: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append send/recv for k and v, then commit them immediately (post).
        Returns recv buffers (these will be valid only after wait()).
        """
        next_k = self.send_recv(k, k_buffer)
        next_v = self.send_recv(v, v_buffer)
        self.post()
        return next_k, next_v