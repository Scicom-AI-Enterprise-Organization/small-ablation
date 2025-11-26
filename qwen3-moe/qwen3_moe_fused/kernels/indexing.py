from functools import partial

import torch


# Assume s is sorted
@partial(torch.compile, fullgraph=True, mode="max-autotune-no-cudagraphs")
@torch.no_grad()
def get_batch_begins_ends(s: torch.Tensor, E: int) -> torch.Tensor:
    arange = torch.arange(E, device=s.device, dtype=s.dtype)
    s_begins = (arange[:, None] > s[None, :]).sum(dim=1, dtype=torch.int32)
    s_ends = (arange[:, None] >= s[None, :]).sum(dim=1, dtype=torch.int32)
    s_begins_ends = torch.stack([s_begins, s_ends], dim=1)
    return s_begins_ends


# Faster than torch.histc when each element of s is an int in [0, E)
@partial(torch.compile, fullgraph=True, mode="max-autotune-no-cudagraphs")
@torch.no_grad()
def get_expert_counts(s: torch.Tensor, E: int) -> torch.Tensor:
    arange = torch.arange(E, device=s.device, dtype=s.dtype)
    counts = (arange[:, None] == s[None, :]).sum(dim=1, dtype=torch.int32)
    return counts


# Faster than torch.sort when each element of s is an int in [0, E)
@partial(torch.compile, fullgraph=True, mode="max-autotune-no-cudagraphs")
@torch.no_grad()
def sort_experts(s: torch.Tensor, E: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    E_arange = torch.arange(E, device=s.device, dtype=s.dtype)
    compare = E_arange[:, None] == s[None, :]
    counts = compare.sum(dim=1, dtype=torch.int32)
    s_sorted = torch.repeat_interleave(counts, output_size=s.numel())  # int32

    s_arange = torch.arange(s.numel(), device=s.device, dtype=s.dtype)
    ranks_in_bin = compare.cumsum(dim=1, dtype=torch.int32)
    ranks_in_bin = ranks_in_bin[s, s_arange]
    offsets = counts.cumsum(dim=0, dtype=torch.int32) - counts
    idx = ranks_in_bin + offsets[s] - 1  # int32

    inv_idx = torch.empty_like(idx)  # int32
    inv_idx[idx] = s_arange.to(inv_idx.dtype)

    # The above definition of idx is the opposite of torch.sort
    return s_sorted, inv_idx, idx


@partial(torch.compile, fullgraph=True, mode="max-autotune-no-cudagraphs")
@torch.no_grad()
def get_expert_counts_and_idx(s: torch.Tensor, E: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    E_arange = torch.arange(E, device=s.device, dtype=s.dtype)
    compare = E_arange[:, None] == s[None, :]
    counts = compare.sum(dim=1, dtype=torch.int32)

    s_arange = torch.arange(s.numel(), device=s.device, dtype=s.dtype)
    ranks_in_bin = compare.cumsum(dim=1, dtype=torch.int32)
    ranks_in_bin = ranks_in_bin[s, s_arange]
    offsets = counts.cumsum(dim=0, dtype=torch.int32) - counts
    idx = ranks_in_bin + offsets[s] - 1  # int32

    inv_idx = torch.empty_like(idx)  # int32
    inv_idx[idx] = s_arange.to(inv_idx.dtype)

    # The above definition of idx is the opposite of torch.sort
    return counts, inv_idx, idx