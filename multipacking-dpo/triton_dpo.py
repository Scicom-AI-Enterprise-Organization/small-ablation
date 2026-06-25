import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _log_prob_backward_kernel(
    grad_output_ptr,
    inputs_ptr,
    weight_ptr,
    targets_ptr,
    grad_inputs_ptr,
    num_tokens,
    H,
    V,
    ignore_index,
    stride_input_batch,
    stride_weight_vocab,
    stride_grad_input_batch,
    BLOCK_H: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """
    Memory-efficient backward pass using online softmax.
    
    Computes: grad_input = grad_output * (weight[target] - sum_v(softmax[v] * weight[v]))
    
    This avoids materializing the full [V] softmax vector.
    """
    pid = tl.program_id(0)
    token_idx = pid
    
    if token_idx >= num_tokens:
        return
    
    # Load target and grad_output
    target = tl.load(targets_ptr + token_idx)
    grad_out = tl.load(grad_output_ptr + token_idx)
    
    if target == ignore_index:
        return
    
    # Step 1: Compute softmax statistics (max and sum) using online algorithm
    running_max = float('-inf')
    running_sum = 0.0
    
    # Load input once for this token
    input_vals_full = tl.zeros([BLOCK_H], dtype=tl.float32)
    
    for v_start in range(0, V, BLOCK_V):
        v_offs = v_start + tl.arange(0, BLOCK_V)
        v_mask = v_offs < V
        
        logits_tile = tl.zeros([BLOCK_V], dtype=tl.float32)
        
        # Compute logits for this vocab tile
        for h_start in range(0, H, BLOCK_H):
            h_offs = h_start + tl.arange(0, BLOCK_H)
            h_mask = h_offs < H
            
            input_ptrs = inputs_ptr + token_idx * stride_input_batch + h_offs
            input_vals = tl.load(input_ptrs, mask=h_mask, other=0.0)
            
            for v_idx in range(BLOCK_V):
                v_abs = v_start + v_idx
                if v_abs < V:
                    weight_ptrs = weight_ptr + v_abs * stride_weight_vocab + h_offs
                    weight_vals = tl.load(weight_ptrs, mask=h_mask, other=0.0)
                    logit = tl.sum(input_vals * weight_vals)
                    logits_tile = tl.where(tl.arange(0, BLOCK_V) == v_idx, logits_tile + logit, logits_tile)
        
        # Update running statistics
        logits_tile = tl.where(v_mask, logits_tile, float('-inf'))
        tile_max = tl.max(logits_tile)
        new_max = tl.maximum(running_max, tile_max)
        
        running_sum = running_sum * tl.exp(running_max - new_max)
        running_sum += tl.sum(tl.where(v_mask, tl.exp(logits_tile - new_max), 0.0))
        running_max = new_max
    
    # Step 2: Compute grad_input in tiles to avoid materializing full gradient
    # Process H dimension in blocks
    for h_start in range(0, H, BLOCK_H):
        h_offs = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offs < H
        
        # Load input for this H block
        input_ptrs = inputs_ptr + token_idx * stride_input_batch + h_offs
        input_vals = tl.load(input_ptrs, mask=h_mask, other=0.0)
        
        # Accumulator for: sum_v(softmax[v] * weight[v, h])
        weighted_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
        
        # Process vocabulary in tiles
        for v_start in range(0, V, BLOCK_V):
            for v_idx in range(BLOCK_V):
                v_abs = v_start + v_idx
                if v_abs < V:
                    # Compute logit for this vocab item (need to recompute)
                    logit = 0.0
                    for h_inner in range(0, H, BLOCK_H):
                        h_inner_offs = h_inner + tl.arange(0, BLOCK_H)
                        h_inner_mask = h_inner_offs < H
                        
                        input_inner_ptrs = inputs_ptr + token_idx * stride_input_batch + h_inner_offs
                        input_inner_vals = tl.load(input_inner_ptrs, mask=h_inner_mask, other=0.0)
                        
                        weight_inner_ptrs = weight_ptr + v_abs * stride_weight_vocab + h_inner_offs
                        weight_inner_vals = tl.load(weight_inner_ptrs, mask=h_inner_mask, other=0.0)
                        
                        logit += tl.sum(input_inner_vals * weight_inner_vals)
                    
                    # Compute softmax probability for this vocab item
                    softmax_v = tl.exp(logit - running_max) / running_sum
                    
                    # Load weight[v, h_start:h_end] and accumulate
                    weight_ptrs = weight_ptr + v_abs * stride_weight_vocab + h_offs
                    weight_vals = tl.load(weight_ptrs, mask=h_mask, other=0.0)
                    
                    weighted_sum += softmax_v * weight_vals
        
        # Load target weight
        target_weight_ptrs = weight_ptr + target * stride_weight_vocab + h_offs
        target_weight = tl.load(target_weight_ptrs, mask=h_mask, other=0.0)
        
        # Compute final gradient: grad_output * (target_weight - weighted_sum)
        grad_input_h = grad_out * (target_weight - weighted_sum)
        
        # Store gradient
        grad_input_ptrs = grad_inputs_ptr + token_idx * stride_grad_input_batch + h_offs
        tl.store(grad_input_ptrs, grad_input_h, mask=h_mask)


@triton.jit
def _log_prob_weight_backward_kernel(
    grad_output_ptr,
    inputs_ptr,
    weight_ptr,
    targets_ptr,
    grad_weight_ptr,
    num_tokens,
    H,
    V,
    ignore_index,
    stride_input_batch,
    stride_weight_vocab,
    stride_grad_weight_vocab,
    BLOCK_H: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """
    Compute gradient w.r.t. weight.
    
    For each vocab item v:
        grad_weight[v] = sum_tokens(grad_output * (indicator(target==v) - softmax[v]) * input)
    
    This kernel processes one vocab slice at a time.
    """
    pid = tl.program_id(0)
    v_idx = pid
    
    if v_idx >= V:
        return
    
    # Process all tokens for this vocab item
    for h_start in range(0, H, BLOCK_H):
        h_offs = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offs < H
        
        grad_accum = tl.zeros([BLOCK_H], dtype=tl.float32)
        
        # Iterate over all tokens
        for token_idx in range(num_tokens):
            target = tl.load(targets_ptr + token_idx)
            
            if target == ignore_index:
                continue
            
            grad_out = tl.load(grad_output_ptr + token_idx)
            
            # Load input
            input_ptrs = inputs_ptr + token_idx * stride_input_batch + h_offs
            input_vals = tl.load(input_ptrs, mask=h_mask, other=0.0)
            
            # Compute softmax for this token (need full pass over vocab)
            running_max = float('-inf')
            running_sum = 0.0
            this_logit = 0.0
            
            # Pass 1: compute softmax statistics
            for v_start in range(0, V, BLOCK_V):
                for v_inner in range(BLOCK_V):
                    v_abs = v_start + v_inner
                    if v_abs < V:
                        logit = 0.0
                        for h_inner in range(0, H, BLOCK_H):
                            h_inner_offs = h_inner + tl.arange(0, BLOCK_H)
                            h_inner_mask = h_inner_offs < H
                            
                            inp_ptrs = inputs_ptr + token_idx * stride_input_batch + h_inner_offs
                            inp_vals = tl.load(inp_ptrs, mask=h_inner_mask, other=0.0)
                            
                            w_ptrs = weight_ptr + v_abs * stride_weight_vocab + h_inner_offs
                            w_vals = tl.load(w_ptrs, mask=h_inner_mask, other=0.0)
                            
                            logit += tl.sum(inp_vals * w_vals)
                        
                        # Save logit for current v_idx
                        if v_abs == v_idx:
                            this_logit = logit
                        
                        # Update running stats
                        new_max = tl.maximum(running_max, logit)
                        running_sum = running_sum * tl.exp(running_max - new_max) + tl.exp(logit - new_max)
                        running_max = new_max
            
            # Compute softmax for this vocab item
            softmax_v = tl.exp(this_logit - running_max) / running_sum
            
            # Compute gradient contribution
            is_target = (target == v_idx)
            grad_coeff = grad_out * (tl.where(is_target, 1.0, 0.0) - softmax_v)
            
            grad_accum += grad_coeff * input_vals
        
        # Store accumulated gradient for this vocab item
        grad_weight_ptrs = grad_weight_ptr + v_idx * stride_grad_weight_vocab + h_offs
        tl.atomic_add(grad_weight_ptrs, grad_accum, mask=h_mask)


class LogProbFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, weight, ignore_index=-100):
        """
        Forward pass: compute mean log probability.
        
        Args:
            inputs: [N, H]
            targets: [N]
            weight: [V, H]
        
        Returns:
            mean_log_prob: scalar
        """
        device = inputs.device
        N, H = inputs.shape
        V = weight.shape[0]
        
        # Ensure float32
        inputs = inputs.float()
        weight = weight.float()
        
        # Compute log probs using existing kernel
        log_probs = torch.zeros(N, device=device, dtype=torch.float32)
        
        # Use online softmax kernel
        BLOCK_H = min(triton.next_power_of_2(H), 256)
        BLOCK_V = min(1024, triton.next_power_of_2(V // 16))
        
        from triton_func import _online_softmax_log_prob_kernel
        
        grid = (N,)
        _online_softmax_log_prob_kernel[grid](
            inputs,
            weight,
            targets,
            log_probs,
            N,
            H,
            V,
            ignore_index,
            BLOCK_H=BLOCK_H,
            BLOCK_V=BLOCK_V,
        )
        
        # Compute mean
        valid_mask = (targets != ignore_index)
        num_valid = valid_mask.sum()
        mean_log_prob = log_probs.sum() / num_valid
        
        # Save for backward
        ctx.save_for_backward(inputs, targets, weight, valid_mask, num_valid)
        ctx.ignore_index = ignore_index
        
        return mean_log_prob
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Memory-efficient backward using Triton kernels.
        """
        inputs, targets, weight, valid_mask, num_valid = ctx.saved_tensors
        ignore_index = ctx.ignore_index
        
        N, H = inputs.shape
        V = weight.shape[0]
        device = inputs.device
        
        # Scale grad_output by 1/num_valid
        grad_output_per_token = torch.full((N,), grad_output.item() / num_valid.item(), 
                                           device=device, dtype=torch.float32)
        
        # Zero out gradients for ignored tokens
        grad_output_per_token = grad_output_per_token * valid_mask.float()
        
        # Allocate gradient tensors
        grad_inputs = torch.zeros_like(inputs, dtype=torch.float32)
        grad_weight = torch.zeros_like(weight, dtype=torch.float32)
        
        # Tune block sizes
        BLOCK_H = min(triton.next_power_of_2(H), 256)
        BLOCK_V = min(1024, triton.next_power_of_2(V // 16))
        
        # Compute grad_inputs using Triton
        grid_inputs = (N,)
        _log_prob_backward_kernel[grid_inputs](
            grad_output_per_token,
            inputs,
            weight,
            targets,
            grad_inputs,
            N,
            H,
            V,
            ignore_index,
            inputs.stride(0),
            weight.stride(0),
            grad_inputs.stride(0),
            BLOCK_H=BLOCK_H,
            BLOCK_V=BLOCK_V,
        )
        
        grid_weight = (V,)
        _log_prob_weight_backward_kernel[grid_weight](
            grad_output_per_token,
            inputs,
            weight,
            targets,
            grad_weight,
            N,
            H,
            V,
            ignore_index,
            inputs.stride(0),
            weight.stride(0),
            grad_weight.stride(0),
            BLOCK_H=BLOCK_H,
            BLOCK_V=BLOCK_V,
        )
        
        return grad_inputs, None, grad_weight, None


def log_prob_with_grad(inputs, targets, weight, ignore_index=-100):
    """
    Compute mean log probability with gradient support.
    
    Args:
        inputs: [N, H] - input features
        targets: [N] - target token ids
        weight: [V, H] - output weight matrix
        ignore_index: token id to ignore
    
    Returns:
        mean_log_prob: scalar tensor with gradient
    """
    return LogProbFunction.apply(inputs, targets, weight, ignore_index)


def dpo_loss_fused(
    inputs_chosen,
    inputs_rejected,
    refs_chosen,
    refs_rejected,
    targets_chosen,
    targets_rejected,
    inputs_weight,
    refs_weight,
    beta=0.1,
    ignore_index=-100,
):
    """
    Compute DPO loss with backward support.
    
    Args:
        inputs_chosen: [N_chosen, H]
        inputs_rejected: [N_rejected, H]
        refs_chosen: [N_chosen, H]
        refs_rejected: [N_rejected, H]
        targets_chosen: [N_chosen]
        targets_rejected: [N_rejected]
        inputs_weight: [V, H] - policy model weights
        refs_weight: [V, H] - reference model weights
        beta: DPO temperature parameter
    
    Returns:
        loss: scalar tensor with gradients
        chosen_rewards: for logging
        rejected_rewards: for logging
    """
    # Compute log probabilities with gradients
    logprob_inputs_chosen = log_prob_with_grad(
        inputs_chosen, targets_chosen, inputs_weight, ignore_index
    )
    logprob_inputs_rejected = log_prob_with_grad(
        inputs_rejected, targets_rejected, inputs_weight, ignore_index
    )
    
    # Reference model (no gradients needed)
    with torch.no_grad():
        logprob_refs_chosen = log_prob_with_grad(
            refs_chosen, targets_chosen, refs_weight, ignore_index
        )
        logprob_refs_rejected = log_prob_with_grad(
            refs_rejected, targets_rejected, refs_weight, ignore_index
        )
    
    # Compute DPO loss
    chosen_logratios = logprob_inputs_chosen - logprob_refs_chosen
    rejected_logratios = logprob_inputs_rejected - logprob_refs_rejected
    
    chosen_rewards = beta * chosen_logratios
    rejected_rewards = beta * rejected_logratios
    
    logits_diff = beta * (chosen_logratios - rejected_logratios)
    loss = -F.logsigmoid(logits_diff)
    
    return loss, chosen_rewards, rejected_rewards


# Example usage
if __name__ == "__main__":
    # Test gradients
    inputs_chosen = torch.randn(100, 5120, dtype=torch.float32).cuda()
    inputs_rejected = torch.randn(50, 5120, dtype=torch.float32).cuda()
    refs_chosen = torch.randn(100, 5120, dtype=torch.float32).cuda()
    refs_rejected = torch.randn(50, 5120, dtype=torch.float32).cuda()
    
    targets_chosen = torch.randint(0, 151936, (100,)).cuda()
    targets_rejected = torch.randint(0, 151936, (50,)).cuda()

    inputs_weight = torch.nn.Linear(5120, 151936).cuda()
    refs_weight = torch.nn.Linear(5120, 151936).cuda()
    
    # Compute loss
    loss, chosen_rewards, rejected_rewards = dpo_loss_fused(
        inputs_chosen=inputs_chosen,
        inputs_rejected=inputs_rejected,
        refs_chosen=refs_chosen,
        refs_rejected=refs_rejected,
        targets_chosen=targets_chosen,
        targets_rejected=targets_rejected,
        inputs_weight=inputs_weight.weight,
        refs_weight=refs_weight.weight,
        beta=0.1,
    )
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Chosen rewards: {chosen_rewards.item():.4f}")
    print(f"Rejected rewards: {rejected_rewards.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    print(f"\nGradient norms:")
    print(f"  inputs_weight: {inputs_weight.weight.grad.norm().item():.4f}")