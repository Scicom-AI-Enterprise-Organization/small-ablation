import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _online_softmax_log_prob_kernel(
    inputs_ptr,
    weight_ptr,
    targets_ptr,
    output_ptr,
    current_chunk_size,
    H,
    V,
    ignore_index,
    BLOCK_H: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """
    Memory-efficient kernel using online softmax algorithm.
    Computes log_prob for each token without materializing full logits.
    
    Each program handles one token and processes vocabulary in tiles.
    Uses online algorithm to maintain running max and sum_exp.
    """
    pid = tl.program_id(0)
    
    # Critical: check against actual chunk size, not the constant
    if pid >= current_chunk_size:
        return
    
    # Load target
    target = tl.load(targets_ptr + pid)
    
    # If ignore_index, return 0
    if target == ignore_index:
        tl.store(output_ptr + pid, 0.0)
        return
    
    # Online softmax: maintain running max and sum
    running_max = -float('inf')
    running_sum = 0.0
    target_logit = 0.0
    
    # Process vocabulary in tiles
    for v_start in range(0, V, BLOCK_V):
        v_end = min(v_start + BLOCK_V, V)
        v_size = v_end - v_start
        
        # Compute logits for this vocabulary tile
        # logits[v] = sum_h(input[h] * weight[v, h])
        logits_tile = tl.zeros([BLOCK_V], dtype=tl.float32)
        
        for h_start in range(0, H, BLOCK_H):
            h_end = min(h_start + BLOCK_H, H)
            h_offs = h_start + tl.arange(0, BLOCK_H)
            h_mask = h_offs < H
            
            # Load input chunk
            input_ptrs = inputs_ptr + pid * H + h_offs
            input_vals = tl.load(input_ptrs, mask=h_mask, other=0.0)
            
            # Compute contribution to logits for each vocab item in tile
            v_offs = tl.arange(0, BLOCK_V)
            v_mask = (v_start + v_offs) < V
            
            for v_idx in range(BLOCK_V):
                v_abs = v_start + v_idx
                if v_abs < V:
                    weight_ptrs = weight_ptr + v_abs * H + h_offs
                    weight_vals = tl.load(weight_ptrs, mask=h_mask, other=0.0)
                    dot_prod = tl.sum(input_vals * weight_vals)
                    # Accumulate into logits_tile
                    logits_tile = tl.where(v_offs == v_idx, logits_tile + dot_prod, logits_tile)
        
        # Update running max and sum using online algorithm
        v_offs = tl.arange(0, BLOCK_V)
        v_mask = (v_start + v_offs) < V
        logits_tile = tl.where(v_mask, logits_tile, -float('inf'))
        
        tile_max = tl.max(logits_tile)
        new_max = tl.maximum(running_max, tile_max)
        
        # Rescale previous sum and add new contributions
        # sum_exp_new = sum_exp_old * exp(old_max - new_max) + sum(exp(logits_tile - new_max))
        running_sum = running_sum * tl.exp(running_max - new_max)
        running_sum += tl.sum(tl.where(v_mask, tl.exp(logits_tile - new_max), 0.0))
        running_max = new_max
        
        # Check if target is in this tile and extract its logit
        # We need to use a mask instead of direct indexing
        v_offs = tl.arange(0, BLOCK_V)
        is_target = (v_start + v_offs) == target
        # Extract target logit using mask (will be 0 for non-target positions)
        target_contrib = tl.sum(tl.where(is_target, logits_tile, 0.0))
        # Only update if target is actually in this tile
        target_logit = tl.where((target >= v_start) & (target < v_end), target_contrib, target_logit)
    
    # Compute final log probability
    log_sum_exp = tl.log(running_sum) + running_max
    log_prob = target_logit - log_sum_exp
    
    tl.store(output_ptr + pid, log_prob)


@triton.jit
def _fused_log_prob_kernel(
    inputs_ptr,
    weight_ptr,
    targets_ptr,
    output_ptr,
    current_chunk_size,
    H,
    V,
    ignore_index,
    stride_input_batch,
    stride_weight_vocab,
    BLOCK_H: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """
    Fused kernel: computes logits and log_softmax in single pass with tiling.
    Processes small tiles of [BLOCK_TOKENS, BLOCK_V] at a time.
    """
    pid_m = tl.program_id(0)  # token dimension
    
    token_idx = pid_m
    if token_idx >= current_chunk_size:
        return
    
    # Load target
    target = tl.load(targets_ptr + token_idx)
    
    if target == ignore_index:
        tl.store(output_ptr + token_idx, 0.0)
        return
    
    # Step 1: Compute all logits in tiles while tracking max (for numerical stability)
    max_logit = -float('inf')
    target_logit = 0.0
    
    # First pass: compute max
    for v_start in range(0, V, BLOCK_V):
        v_offs = v_start + tl.arange(0, BLOCK_V)
        v_mask = v_offs < V
        
        # Compute logits for this vocab tile
        logits = tl.zeros([BLOCK_V], dtype=tl.float32)
        
        for h_start in range(0, H, BLOCK_H):
            h_offs = h_start + tl.arange(0, BLOCK_H)
            h_mask = h_offs < H
            
            # Load input
            input_ptrs = inputs_ptr + token_idx * stride_input_batch + h_offs
            inputs_vals = tl.load(input_ptrs, mask=h_mask, other=0.0)
            
            # Load weight tile and compute
            for i in range(BLOCK_V):
                if v_start + i < V:
                    weight_ptrs = weight_ptr + (v_start + i) * stride_weight_vocab + h_offs
                    weight_vals = tl.load(weight_ptrs, mask=h_mask, other=0.0)
                    logit_contrib = tl.sum(inputs_vals * weight_vals)
                    logits = tl.where(tl.arange(0, BLOCK_V) == i, logits + logit_contrib, logits)
        
        # Update max
        tile_max = tl.max(tl.where(v_mask, logits, -float('inf')))
        max_logit = tl.maximum(max_logit, tile_max)
        
        # Save target logit if in this tile using mask
        v_offs = tl.arange(0, BLOCK_V)
        is_target = (v_start + v_offs) == target
        target_contrib = tl.sum(tl.where(is_target, logits, 0.0))
        target_logit = tl.where((target >= v_start) & (target < v_start + BLOCK_V), target_contrib, target_logit)
    
    # Second pass: compute sum of exp
    sum_exp = 0.0
    for v_start in range(0, V, BLOCK_V):
        v_offs = v_start + tl.arange(0, BLOCK_V)
        v_mask = v_offs < V
        
        # Recompute logits for this vocab tile
        logits = tl.zeros([BLOCK_V], dtype=tl.float32)
        
        for h_start in range(0, H, BLOCK_H):
            h_offs = h_start + tl.arange(0, BLOCK_H)
            h_mask = h_offs < H
            
            input_ptrs = inputs_ptr + token_idx * stride_input_batch + h_offs
            inputs_vals = tl.load(input_ptrs, mask=h_mask, other=0.0)
            
            for i in range(BLOCK_V):
                if v_start + i < V:
                    weight_ptrs = weight_ptr + (v_start + i) * stride_weight_vocab + h_offs
                    weight_vals = tl.load(weight_ptrs, mask=h_mask, other=0.0)
                    logit_contrib = tl.sum(inputs_vals * weight_vals)
                    logits = tl.where(tl.arange(0, BLOCK_V) == i, logits + logit_contrib, logits)
        
        # Accumulate exp sum
        exp_logits = tl.exp(logits - max_logit)
        sum_exp += tl.sum(tl.where(v_mask, exp_logits, 0.0))
    
    # Compute log probability
    log_sum_exp = tl.log(sum_exp) + max_logit
    log_prob = target_logit - log_sum_exp
    
    tl.store(output_ptr + token_idx, log_prob)


def get_sum_logprob_triton_online(inputs, targets, weight, chunk_size=512, ignore_index=-100):
    """
    Memory-efficient version using online softmax algorithm.
    Never materializes full [chunk_size, V] logits matrix.
    """
    sum_log_prob = 0.0
    BT, H = inputs.shape
    V = weight.shape[0]
    
    for start_idx in range(0, BT, chunk_size):
        end_idx = min(start_idx + chunk_size, BT)
        current_chunk_size = end_idx - start_idx
        
        _inputs_chunk = inputs[start_idx:end_idx]
        _targets_chunk = targets[start_idx:end_idx]
        
        # Allocate output
        chunk_log_probs = torch.zeros(current_chunk_size, device=inputs.device, dtype=torch.float32)
        
        # Tune block sizes based on H and V
        BLOCK_H = min(triton.next_power_of_2(H), 128)
        BLOCK_V = min(triton.next_power_of_2(V // 32), 256)  # Process vocab in reasonable chunks
        
        grid = (current_chunk_size,)
        
        _online_softmax_log_prob_kernel[grid](
            _inputs_chunk,
            weight,
            _targets_chunk,
            chunk_log_probs,
            current_chunk_size,
            H,
            V,
            ignore_index,
            BLOCK_H=BLOCK_H,
            BLOCK_V=BLOCK_V,
        )
        
        sum_log_prob += chunk_log_probs.sum()
    
    num_valid_tokens = (targets != ignore_index).sum()
    return sum_log_prob / num_valid_tokens


def get_sum_logprob_triton_fused(inputs, targets, weight, chunk_size=512, ignore_index=-100):
    """
    Fused kernel version with tiled computation.
    """
    sum_log_prob = 0.0
    BT, H = inputs.shape
    V = weight.shape[0]
    
    for start_idx in range(0, BT, chunk_size):
        end_idx = min(start_idx + chunk_size, BT)
        current_chunk_size = end_idx - start_idx
        
        _inputs_chunk = inputs[start_idx:end_idx]
        _targets_chunk = targets[start_idx:end_idx]
        
        chunk_log_probs = torch.zeros(current_chunk_size, device=inputs.device, dtype=torch.float32)
        
        BLOCK_H = min(triton.next_power_of_2(H), 128)
        BLOCK_V = min(256, triton.next_power_of_2(V // 16))
        
        grid = (current_chunk_size,)
        
        _fused_log_prob_kernel[grid](
            _inputs_chunk,
            weight,
            _targets_chunk,
            chunk_log_probs,
            current_chunk_size,
            H,
            V,
            ignore_index,
            _inputs_chunk.stride(0),
            weight.stride(0),
            BLOCK_H=BLOCK_H,
            BLOCK_V=BLOCK_V,
        )
        
        sum_log_prob += chunk_log_probs.sum()
    
    num_valid_tokens = (targets != ignore_index).sum()
    return sum_log_prob / num_valid_tokens


# Original implementation for comparison
def get_sum_logprob_original(inputs, targets, weight, chunk_size=512, ignore_index=-100):
    sum_log_prob = 0.0
    BT, H = inputs.shape
    
    for start_idx in range(0, BT, chunk_size):
        end_idx = min(start_idx + chunk_size, BT)
        _inputs_chunk = inputs[start_idx:end_idx]
        _targets_chunk = targets[start_idx:end_idx]
    
        logits = _inputs_chunk.to(torch.float32) @ weight.T
        log_probs_chunk = F.log_softmax(logits.float(), dim=-1)
        
        loss_mask = _targets_chunk != ignore_index
        label_chunk = torch.where(loss_mask, _targets_chunk, 0)
        per_token_logps = log_probs_chunk.gather(-1, label_chunk.unsqueeze(-1)).squeeze(-1)
        log_prob = (per_token_logps * loss_mask).sum(-1)
        sum_log_prob += log_prob
    
    return sum_log_prob / (targets != ignore_index).sum()


# Example usage and testing
if __name__ == "__main__":
    # Test parameters
    BT, H, V = 1025, 4096, 32000
    chunk_size = 256
    
    device = torch.device("cuda")
    inputs = torch.randn(BT, H, device=device, dtype=torch.float32)
    targets = torch.randint(0, V, (BT,), device=device)
    weight = torch.randn(V, H, device=device, dtype=torch.float32)
    
    # Compare implementations
    print("Running original PyTorch version...")
    result_original = get_sum_logprob_original(inputs, targets, weight, chunk_size)
    print(f"Original result: {result_original.item():.6f}")
    
    print("\nRunning online softmax Triton version...")
    result_online = get_sum_logprob_triton_online(inputs, targets, weight, chunk_size)
    print(f"Online result: {result_online.item():.6f}")
    
    print("\nRunning fused Triton version...")
    result_fused = get_sum_logprob_triton_fused(inputs, targets, weight, chunk_size)
    print(f"Fused result: {result_fused.item():.6f}")
    
    print(f"\nDifference (online vs original): {abs(result_online.item() - result_original.item()):.6e}")
    print(f"Difference (fused vs original): {abs(result_fused.item() - result_original.item()):.6e}")