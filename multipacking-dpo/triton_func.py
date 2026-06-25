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


def get_sum_logprob_triton_online(inputs, targets, weight, chunk_size=4096, ignore_index=-100, average_log_prob=False):
    """
    Memory-efficient version using online softmax algorithm.
    Never materializes full [chunk_size, V] logits matrix.
    """
    device = inputs.device
    sum_log_prob = torch.tensor(0.0, device=device, dtype=torch.float32)
    BT, H = inputs.shape
    V = weight.shape[0]
    
    # Ensure inputs are float32 for numerical stability
    inputs = inputs.float()
    weight = weight.float()
    
    for start_idx in range(0, BT, chunk_size):
        end_idx = min(start_idx + chunk_size, BT)
        current_chunk_size = end_idx - start_idx
        
        _inputs_chunk = inputs[start_idx:end_idx].contiguous()
        _targets_chunk = targets[start_idx:end_idx].contiguous()
        
        # Allocate output
        chunk_log_probs = torch.zeros(current_chunk_size, device=device, dtype=torch.float32)
        
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
    
    if average_log_prob:
        num_valid_tokens = (targets != ignore_index).sum()
        sum_log_prob = sum_log_prob / num_valid_tokens
    return sum_log_prob

def get_sum_logprob_triton_fused(inputs, targets, weight, chunk_size=4096, ignore_index=-100, average_log_prob=False):
    """
    Fused kernel version with tiled computation.
    """
    device = inputs.device
    sum_log_prob = torch.tensor(0.0, device=device, dtype=torch.float32)
    BT, H = inputs.shape
    V = weight.shape[0]
    
    # Ensure inputs are float32 for numerical stability
    inputs = inputs.float()
    weight = weight.float()
    
    for start_idx in range(0, BT, chunk_size):
        end_idx = min(start_idx + chunk_size, BT)
        current_chunk_size = end_idx - start_idx
        
        _inputs_chunk = inputs[start_idx:end_idx].contiguous()
        _targets_chunk = targets[start_idx:end_idx].contiguous()
        
        chunk_log_probs = torch.zeros(current_chunk_size, device=device, dtype=torch.float32)
        
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

    if average_log_prob:
        num_valid_tokens = (targets != ignore_index).sum()
        sum_log_prob = sum_log_prob / num_valid_tokens
    return sum_log_prob


# Original implementation for comparison
def get_sum_logprob_original(inputs, targets, weight, chunk_size=512, ignore_index=-100, average_log_prob=False):
    sum_log_prob = 0.0
    BT, H = inputs.shape

    # Ensure inputs are float32 for numerical stability
    inputs = inputs.float()
    weight = weight.float()
    
    for start_idx in range(0, BT, chunk_size):
        end_idx = min(start_idx + chunk_size, BT)
        _inputs_chunk = inputs[start_idx:end_idx]
        _targets_chunk = targets[start_idx:end_idx]
    
        logits = _inputs_chunk @ weight.T
        log_probs_chunk = F.log_softmax(logits.float(), dim=-1)
        
        loss_mask = _targets_chunk != ignore_index
        label_chunk = torch.where(loss_mask, _targets_chunk, 0)
        per_token_logps = log_probs_chunk.gather(-1, label_chunk.unsqueeze(-1)).squeeze(-1)
        log_prob = (per_token_logps * loss_mask).sum(-1)
        sum_log_prob += log_prob

    if average_log_prob:
        sum_log_prob = sum_log_prob / (targets != ignore_index).sum()
    return sum_log_prob


def pad_dim1(tensors, padding_value=0):
    tensors = [t.unsqueeze(0) if t.dim() == 2 else t for t in tensors]

    max_len = max(t.shape[1] for t in tensors)

    padded = []
    for t in tensors:
        pad_len = max_len - t.shape[1]
        if pad_len > 0:
            t = F.pad(t, (0, 0, 0, pad_len), value=padding_value)
        padded.append(t)

    return torch.cat(padded, dim=0)

def get_logprobs(
    inputs_chosen, 
    inputs_rejected, 
    refs_chosen, 
    refs_rejected, 
    targets_chosen, 
    targets_rejected, 
    inputs_weight, 
    refs_weight,
):

    logpprob_inputs_chosen = get_sum_logprob_triton_online(inputs_chosen, targets_chosen, inputs_weight)
    logpprob_inputs_rejected = get_sum_logprob_triton_online(inputs_rejected, targets_rejected, inputs_weight)
    logpprob_refs_chosen = get_sum_logprob_triton_online(refs_chosen, targets_chosen, refs_weight)
    logpprob_refs_rejected = get_sum_logprob_triton_online(refs_rejected, targets_rejected, refs_weight)

    return logpprob_inputs_chosen, logpprob_inputs_rejected, logpprob_refs_chosen, logpprob_refs_rejected
    
if __name__ == "__main__":
    # Test parameters
    BT, H, V = 1000, 4096, 32000
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

    # https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json#L11
    inputs_chosen = torch.randn(10000, 5120, dtype=torch.bfloat16).cuda()
    inputs_rejected = torch.randn(5000, 5120, dtype=torch.bfloat16).cuda()
    refs_chosen = torch.randn(10000, 5120, dtype=torch.bfloat16).cuda()
    refs_rejected = torch.randn(5000, 5120, dtype=torch.bfloat16).cuda()
    
    # https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json#L29
    targets_chosen = torch.randint(low=0, high=151936, size=(10000,)).cuda()
    targets_rejected = torch.randint(low=0, high=151936, size=(5000,)).cuda()

    from torch.nn.utils.rnn import pad_sequence
    from liger_kernel.chunked_loss import LigerFusedLinearDPOLoss
    
    targets = pad_sequence([targets_chosen, targets_rejected], batch_first=True, padding_value=-100).cuda()
    
    # assumed packing 1 sequences
    # liger divide by batch size // 2
    num_seqs = 1
    
    inputs_padded = pad_dim1([inputs_chosen, inputs_rejected])
    refs_padded = pad_dim1([refs_chosen, refs_rejected])

    inputs_weight = torch.nn.Linear(5120, 151936).cuda()
    refs_weight = torch.nn.Linear(5120, 151936).cuda()

    out = get_logprobs(
        inputs_chosen=inputs_chosen,
        inputs_rejected=inputs_rejected,
        refs_chosen=refs_chosen,
        refs_rejected=refs_rejected,
        targets_chosen=targets_chosen,
        targets_rejected=targets_rejected,
        inputs_weight=inputs_weight.weight,
        refs_weight=refs_weight.weight,
    )

    logpprob_inputs_chosen, logpprob_inputs_rejected, logpprob_refs_chosen, logpprob_refs_rejected = out

    beta = 0.1
    chosen_logratios = logpprob_inputs_chosen - logpprob_refs_chosen
    rejected_logratios = logpprob_inputs_rejected - logpprob_refs_rejected
    
    chosen_rewards = beta * chosen_logratios
    rejected_rewards = beta * rejected_logratios
    logits_diff = beta * (chosen_logratios - rejected_logratios)
    loss = -F.logsigmoid(logits_diff)

    liger_loss = LigerFusedLinearDPOLoss()

    out = liger_loss(
        inputs_weight.weight,
        inputs_padded.to(torch.float32),
        targets,
        ref_input=refs_padded.to(torch.float32),
        ref_weight=refs_weight.weight
    )
    print(f"\nDifference loss (online vs liger dpo): {abs(loss.item() - out[0].item()):.6e}")
    