import torch
import triton
import triton.language as tl
import torch.utils.benchmark as benchmark

def pad_for_alignment(grouped_inputs, experts_count, alignment=16):
    """Pad inputs so each expert group is aligned. Returns padding context for reuse."""
    experts_count_padded = ((experts_count + alignment - 1) // alignment) * alignment
    
    # cu_experts_count = experts_count.cumsum(dim=0).to(torch.int32)
    cu_original = torch.cat([torch.zeros(1, dtype=torch.int32, device=experts_count.device), 
                             experts_count.cumsum(0).to(torch.int32)])
    cu_padded = torch.cat([torch.zeros(1, dtype=torch.int32, device=experts_count.device), 
                           experts_count_padded.cumsum(0).to(torch.int32)])
    
    total_tokens = grouped_inputs.shape[0]
    token_indices = torch.arange(total_tokens, device=grouped_inputs.device)
    expert_ids = torch.searchsorted(cu_original[1:], token_indices, right=True)
    position_in_group = token_indices - cu_original[expert_ids]
    dest_indices = cu_padded[expert_ids] + position_in_group
    
    total_padded = cu_padded[-1]
    padded_inputs = torch.zeros(total_padded, grouped_inputs.shape[1], 
                                dtype=grouped_inputs.dtype, device=grouped_inputs.device)
    padded_inputs[dest_indices] = grouped_inputs
    
    ctx = {
        'cu_original': cu_original[1:],
        'cu_padded': cu_padded[1:],
        'dest_indices': dest_indices,
        'total_padded': total_padded,
        'total_tokens': total_tokens,
    }
    return padded_inputs, ctx
    
def pad_for_alignment_fast(grouped_inputs, experts_count, alignment=16):
    """Faster padding with minimal allocations."""
    num_experts = experts_count.shape[0]
    device = grouped_inputs.device
    
    # Compute padded counts - vectorized
    experts_count_padded = (experts_count + alignment - 1) // alignment * alignment
    
    # Cumsum once, prepend zero
    cu_original = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    cu_padded = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    cu_original[1:] = experts_count.cumsum(0).to(torch.int32)
    cu_padded[1:] = experts_count_padded.cumsum(0).to(torch.int32)
    
    total_tokens = grouped_inputs.shape[0]
    total_padded = cu_padded[-1].item()
    
    # Create destination indices without searchsorted
    # Build expert_ids using repeat_interleave (faster than searchsorted)
    expert_ids = torch.repeat_interleave(
        torch.arange(num_experts, device=device),
        experts_count
    )
    
    # Position within each expert group
    position_in_group = torch.arange(total_tokens, device=device) - cu_original[expert_ids]
    
    # Destination in padded tensor
    dest_indices = cu_padded[expert_ids] + position_in_group
    
    # Allocate and scatter
    padded_inputs = torch.zeros(
        total_padded, grouped_inputs.shape[1],
        dtype=grouped_inputs.dtype, device=device
    )
    padded_inputs[dest_indices] = grouped_inputs
    
    ctx = {
        'cu_padded': cu_padded[1:],
        'dest_indices': dest_indices,
        'total_padded': total_padded,
    }
    return padded_inputs, ctx


def pad_for_alignment_v2(grouped_inputs, experts_count, alignment=16):
    """Even faster - fused operations, minimal intermediate tensors."""
    num_experts = experts_count.shape[0]
    device = grouped_inputs.device
    dtype = grouped_inputs.dtype
    hidden_dim = grouped_inputs.shape[1]
    
    # Padded counts
    experts_count_padded = ((experts_count + (alignment - 1)) & ~(alignment - 1))  # Bitwise for power-of-2
    
    # Single cumsum call, then slice
    cu_original = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=device),
        experts_count.to(torch.int32).cumsum(0)
    ])
    cu_padded = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=device),
        experts_count_padded.to(torch.int32).cumsum(0)
    ])
    
    total_padded = cu_padded[-1].item()
    
    # Use bucketize instead of searchsorted (can be faster)
    total_tokens = grouped_inputs.shape[0]
    token_indices = torch.arange(total_tokens, device=device, dtype=torch.int32)
    expert_ids = torch.bucketize(token_indices, cu_original[1:], right=True)
    
    # Compute dest indices
    dest_indices = cu_padded[expert_ids] + (token_indices - cu_original[expert_ids])
    
    # Scatter into padded tensor
    padded_inputs = grouped_inputs.new_zeros(total_padded, hidden_dim)
    padded_inputs.index_copy_(0, dest_indices.long(), grouped_inputs)
    
    return padded_inputs, {
        'cu_padded': cu_padded[1:],
        'dest_indices': dest_indices.long(),
        'total_padded': total_padded,
    }


def pad_for_alignment_v3(grouped_inputs, experts_count, alignment=16):
    """Precompute-friendly version - separate index computation."""
    device = grouped_inputs.device
    num_experts = experts_count.shape[0]
    
    # All integer math on GPU
    experts_count_padded = ((experts_count + alignment - 1) // alignment) * alignment
    
    cu_original = experts_count.to(torch.int64).cumsum(0)
    cu_padded = experts_count_padded.to(torch.int64).cumsum(0)
    
    # Prepend zeros
    cu_original = torch.cat([torch.zeros(1, dtype=torch.int64, device=device), cu_original])
    cu_padded = torch.cat([torch.zeros(1, dtype=torch.int64, device=device), cu_padded])
    
    total_tokens = grouped_inputs.shape[0]
    total_padded = cu_padded[-1]
    
    # Build mapping with repeat_interleave - avoids search entirely
    expert_ids = torch.repeat_interleave(experts_count)  # Auto 0..N-1
    local_pos = torch.cat([
        torch.arange(c, device=device, dtype=torch.int64) 
        for c in experts_count.tolist()
    ]) if total_tokens < 50000 else _compute_local_pos_large(experts_count, cu_original, total_tokens, device)
    
    dest_indices = cu_padded[expert_ids] + local_pos
    
    # Scatter
    padded = grouped_inputs.new_zeros(total_padded, grouped_inputs.shape[1])
    padded[dest_indices] = grouped_inputs
    
    return padded, {
        'cu_padded': cu_padded[1:].to(torch.int32),
        'dest_indices': dest_indices,
        'total_padded': total_padded.item(),
    }


def _compute_local_pos_large(experts_count, cu_original, total_tokens, device):
    """For large token counts, avoid list comprehension."""
    token_indices = torch.arange(total_tokens, device=device, dtype=torch.int64)
    expert_ids = torch.searchsorted(cu_original[1:], token_indices, right=True)
    return token_indices - cu_original[expert_ids]


# Fastest unpad - just index
def unpad_tensor_fast(padded_tensor, ctx):
    return padded_tensor[ctx['dest_indices']]


# Fastest pad helper - reuse ctx
def pad_tensor_fast(tensor, ctx):
    padded = tensor.new_zeros(ctx['total_padded'], tensor.shape[1])
    padded[ctx['dest_indices']] = tensor
    return padded

@triton.jit
def pad_kernel(
    input_ptr, output_ptr,
    cu_original_ptr, cu_padded_ptr,
    num_experts: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    expert_id = tl.program_id(1)
    
    # Load boundaries for this expert
    start_orig = tl.load(cu_original_ptr + expert_id)
    end_orig = tl.load(cu_original_ptr + expert_id + 1)
    start_pad = tl.load(cu_padded_ptr + expert_id)
    
    count = end_orig - start_orig
    
    # Each program handles BLOCK_SIZE tokens
    token_offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = token_offset < count
    
    # Copy hidden_dim elements per token
    for d in range(0, hidden_dim, BLOCK_SIZE):
        d_offset = d + tl.arange(0, BLOCK_SIZE)
        d_mask = d_offset < hidden_dim
        
        src_idx = (start_orig + token_offset[:, None]) * hidden_dim + d_offset[None, :]
        dst_idx = (start_pad + token_offset[:, None]) * hidden_dim + d_offset[None, :]
        
        vals = tl.load(input_ptr + src_idx, mask=mask[:, None] & d_mask[None, :], other=0.0)
        tl.store(output_ptr + dst_idx, vals, mask=mask[:, None] & d_mask[None, :])


def pad_for_alignment_triton(grouped_inputs, experts_count, alignment=16):
    device = grouped_inputs.device
    num_experts = experts_count.shape[0]
    hidden_dim = grouped_inputs.shape[1]
    
    experts_count_padded = ((experts_count + alignment - 1) // alignment) * alignment
    
    cu_original = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=device),
        experts_count.to(torch.int32).cumsum(0)
    ])
    cu_padded = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=device),
        experts_count_padded.to(torch.int32).cumsum(0)
    ])
    
    total_padded = cu_padded[-1].item()
    max_tokens_per_expert = experts_count.max().item()
    
    padded = torch.zeros(total_padded, hidden_dim, dtype=grouped_inputs.dtype, device=device)
    
    BLOCK_SIZE = 128
    grid = ((max_tokens_per_expert + BLOCK_SIZE - 1) // BLOCK_SIZE, num_experts)
    
    pad_kernel[grid](
        grouped_inputs, padded,
        cu_original, cu_padded,
        num_experts, hidden_dim,
        BLOCK_SIZE,
    )
    
    return padded, {'cu_padded': cu_padded[1:], 'total_padded': total_padded}

# Test
experts_count = torch.randint(100, 500, (64,), device='cuda')
total = experts_count.sum().item()
grouped_inputs = torch.randn(total, 4096, device='cuda', dtype=torch.bfloat16)

t0 = benchmark.Timer(
    stmt='pad_for_alignment(grouped_inputs, experts_count)',
    globals={'pad_for_alignment': pad_for_alignment, 'grouped_inputs': grouped_inputs, 'experts_count': experts_count}
)
t1 = benchmark.Timer(
    stmt='pad_for_alignment_fast(grouped_inputs, experts_count)',
    globals={'pad_for_alignment_fast': pad_for_alignment_fast, 'grouped_inputs': grouped_inputs, 'experts_count': experts_count}
)
t2 = benchmark.Timer(
    stmt='pad_for_alignment_v2(grouped_inputs, experts_count)',
    globals={'pad_for_alignment_v2': pad_for_alignment_v2, 'grouped_inputs': grouped_inputs, 'experts_count': experts_count}
)

print("Original:", t0.timeit(100))
print("Fast:", t1.timeit(100))
print("V2:", t2.timeit(100))

"""
Original: <torch.utils.benchmark.utils.common.Measurement object at 0x7f0b8abf13c0>
pad_for_alignment(grouped_inputs, experts_count)
  689.17 us
  1 measurement, 100 runs , 1 thread
Fast: <torch.utils.benchmark.utils.common.Measurement object at 0x7f0b8abf1c90>
pad_for_alignment_fast(grouped_inputs, experts_count)
  734.95 us
  1 measurement, 100 runs , 1 thread
V2: <torch.utils.benchmark.utils.common.Measurement object at 0x7f0b8abf13c0>
pad_for_alignment_v2(grouped_inputs, experts_count)
  635.32 us
  1 measurement, 100 runs , 1 thread
"""