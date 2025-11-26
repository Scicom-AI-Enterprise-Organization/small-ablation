import torch

from .grouped_gemm.interface import grouped_gemm


# output[b, o] = sum_i weight[selected_experts[b], o, i] * input[b, i]
def _moe_fused_linear_naive_fwd(
    input: torch.Tensor, weight: torch.Tensor, selected_experts: torch.Tensor
) -> torch.Tensor:
    """
    Computes a MoE linear operation.

    The operation is defined as:
    `output[b, o] = sum_i weight[selected_experts[b], o, i] * input[b, i]`

    Args:
        input (`torch.FloatTensor`): input tensor of shape `(batch_size, in_features)`.
        weight (`torch.FloatTensor`): weight tensor of shape `(num_experts, out_features, in_features)`.
        selected_experts (`torch.LongTensor`): tensor of selected expert indices in shape `(batch_size,)`.
            Each element is in the range `[0, num_experts)`.

    Returns:
        output (`torch.FloatTensor`): output tensor of shape `(batch_size, out_features)`.
    """
    batch_size, in_features = input.shape
    num_experts, out_features, _ = weight.shape

    output = torch.empty(batch_size, out_features, device=input.device, dtype=input.dtype)
    for b in range(batch_size):
        _weight = weight[selected_experts[b], :, :]
        _input = input[b, :]
        output[b, :] = _weight @ _input
    return output


# grad_input[b, i] = sum_o weight[selected_experts[b], o, i] * grad_output[b, o]
def _moe_fused_linear_naive_bwd_input(
    grad_output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, selected_experts: torch.Tensor
) -> torch.Tensor:
    batch_size, in_features = input.shape
    num_experts, out_features, _ = weight.shape

    grad_input = torch.empty_like(input)
    for b in range(batch_size):
        _weight = weight[selected_experts[b], :, :]
        _grad_output = grad_output[b, :]
        grad_input[b, :] = _grad_output @ _weight
    return grad_input


# grad_weight[e, o, i] = sum_b if(selected_experts[b] == e) grad_output[b, o] * input[b, i]
def _moe_fused_linear_naive_bwd_weight(
    grad_output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, selected_experts: torch.Tensor
) -> torch.Tensor:
    batch_size, in_features = input.shape
    num_experts, out_features, _ = weight.shape

    grad_weight = torch.zeros_like(weight)
    for b in range(batch_size):
        grad_weight[selected_experts[b], :, :] += grad_output[b, :, None] * input[b, None, :]
    return grad_weight


def moe_fused_linear(input: torch.Tensor, weight: torch.Tensor, m_sizes: torch.Tensor) -> torch.Tensor:
    """
    Computes a MoE linear operation using grouped GEMM.

    The operation is defined as:
    `output[b, o] = sum_i weight[selected_experts[b], o, i] * input[b, i]`

    Args:
        input (`torch.FloatTensor`): input tensor of shape `(batch_size, in_features)`.
        weight (`torch.FloatTensor`): weight tensor of shape `(num_experts, out_features, in_features)`.
        m_sizes (`torch.LongTensor`): counts of selected experts in shape `(num_experts)`. Should sum to `batch_size`.

    Returns:
        output (`torch.FloatTensor`): output tensor of shape `(batch_size, out_features)`.
    """
    return grouped_gemm(input, weight, m_sizes)