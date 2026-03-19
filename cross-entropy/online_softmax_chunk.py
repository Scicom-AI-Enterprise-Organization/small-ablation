import torch
import triton
import triton.language as tl
import time
from typing import Callable
from dataclasses import dataclass


# =============================================================================
# Implementation 1: No chunking (baseline)
# =============================================================================

def logprobs_no_chunk(
    hidden: torch.Tensor,      # [N, H]
    weight: torch.Tensor,      # [V, H]
    labels: torch.Tensor,      # [N]
    inv_temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Baseline: materialize full [N, V] logits."""
    logits = (hidden @ weight.t()).float() * inv_temperature  # [N, V]
    logz = torch.logsumexp(logits, dim=-1)  # [N]
    target_logits = logits.gather(1, labels.unsqueeze(-1)).squeeze(-1)  # [N]
    logprobs = target_logits - logz
    
    # Entropy: H = log(Z) - E[x] = logZ - sum(p * x)
    probs = torch.softmax(logits, dim=-1)
    entropy = logz - (probs * logits).sum(dim=-1)
    
    return logprobs, entropy


# =============================================================================
# Implementation 2: V-chunked (PyTorch online softmax)
# =============================================================================

def _online_logsumexp_and_weighted_update(
    m: torch.Tensor, s: torch.Tensor, t: torch.Tensor, chunk_logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Online logsumexp + weighted-sum accumulator for entropy.
    """
    chunk_m = torch.amax(chunk_logits, dim=-1)
    m_new = torch.maximum(m, chunk_m)
    exp_old = torch.exp(m - m_new)

    chunk_exp = torch.exp(chunk_logits - m_new.unsqueeze(-1))
    s_new = s * exp_old + chunk_exp.sum(dim=-1)
    t_new = t * exp_old + (chunk_exp * chunk_logits).sum(dim=-1)
    return m_new, s_new, t_new


def logprobs_v_chunk_pytorch(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    inv_temperature: float = 1.0,
    chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """V-chunked with PyTorch online softmax."""
    device = hidden.device
    n = hidden.shape[0]
    vocab = weight.shape[0]

    m = torch.full((n,), float("-inf"), device=device, dtype=torch.float32)
    s = torch.zeros((n,), device=device, dtype=torch.float32)
    t = torch.zeros((n,), device=device, dtype=torch.float32)
    target_logits = torch.zeros((n,), device=device, dtype=torch.float32)

    for start in range(0, vocab, chunk_size):
        end = min(start + chunk_size, vocab)
        w_chunk = weight[start:end]
        logits = (hidden @ w_chunk.t()).float() * inv_temperature

        m, s, t = _online_logsumexp_and_weighted_update(m, s, t, logits)

        mask = (labels >= start) & (labels < end)
        if torch.any(mask):
            idx = (labels[mask] - start).to(torch.long)
            target_logits[mask] = logits[mask, idx]

    logz = m + torch.log(s)
    logprobs = target_logits - logz
    entropy = logz - (t / s)

    return logprobs, entropy


# =============================================================================
# Implementation 3: V-chunked Triton
# =============================================================================

@triton.jit
def _online_logsumexp_weighted_update_with_targets_kernel(
    m_ptr,
    s_ptr,
    t_ptr,
    target_logits_ptr,
    chunk_logits_ptr,
    labels_ptr,
    vocab_start,
    N,
    C,
    stride_logits_n,
    stride_logits_c,
    BLOCK_C: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    if row_idx >= N:
        return
    
    m_old = tl.load(m_ptr + row_idx)
    s_old = tl.load(s_ptr + row_idx)
    t_old = tl.load(t_ptr + row_idx)
    
    label = tl.load(labels_ptr + row_idx)
    local_label = label - vocab_start

    # First pass: find chunk max
    chunk_m = float("-inf")
    for c_start in tl.range(0, C, BLOCK_C):
        c_offsets = c_start + tl.arange(0, BLOCK_C)
        mask = c_offsets < C
        
        logits = tl.load(
            chunk_logits_ptr + row_idx * stride_logits_n + c_offsets * stride_logits_c,
            mask=mask,
            other=float("-inf"),
        )
        chunk_m = tl.maximum(chunk_m, tl.max(logits, axis=0))
    
    m_new = tl.maximum(m_old, chunk_m)
    exp_old = tl.exp(m_old - m_new)
    
    # Second pass: compute sums and extract target logit
    chunk_s = 0.0
    chunk_t = 0.0
    target_logit = 0.0
    
    for c_start in tl.range(0, C, BLOCK_C):
        c_offsets = c_start + tl.arange(0, BLOCK_C)
        mask = c_offsets < C
        
        logits = tl.load(
            chunk_logits_ptr + row_idx * stride_logits_n + c_offsets * stride_logits_c,
            mask=mask,
            other=float("-inf"),
        )
        
        chunk_exp = tl.exp(logits - m_new)
        chunk_exp = tl.where(mask, chunk_exp, 0.0)
        
        chunk_s += tl.sum(chunk_exp, axis=0)
        chunk_t += tl.sum(chunk_exp * logits, axis=0)
        
        target_mask = (c_offsets == local_label) & (local_label >= 0) & (local_label < C)
        target_logit += tl.sum(tl.where(target_mask, logits, 0.0), axis=0)
    
    s_new = s_old * exp_old + chunk_s
    t_new = t_old * exp_old + chunk_t
    
    tl.store(m_ptr + row_idx, m_new)
    tl.store(s_ptr + row_idx, s_new)
    tl.store(t_ptr + row_idx, t_new)
    
    label_in_chunk = (label >= vocab_start) & (label < vocab_start + C)
    if label_in_chunk:
        tl.store(target_logits_ptr + row_idx, target_logit)


def logprobs_v_chunk_triton(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    inv_temperature: float = 1.0,
    chunk_size: int = 1024,
    block_c: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """V-chunked with Triton online softmax."""
    device = hidden.device
    n = hidden.shape[0]
    vocab = weight.shape[0]
    
    m = torch.full((n,), float("-inf"), device=device, dtype=torch.float32)
    s = torch.zeros((n,), device=device, dtype=torch.float32)
    t = torch.zeros((n,), device=device, dtype=torch.float32)
    target_logits = torch.zeros((n,), device=device, dtype=torch.float32)
    
    for start in range(0, vocab, chunk_size):
        end = min(start + chunk_size, vocab)
        w_chunk = weight[start:end]
        
        logits = (hidden @ w_chunk.t()).float() * inv_temperature
        logits = logits.contiguous()
        
        C = logits.shape[1]
        grid = (n,)
        
        _online_logsumexp_weighted_update_with_targets_kernel[grid](
            m, s, t, target_logits,
            logits, labels,
            start,
            n, C,
            logits.stride(0),
            logits.stride(1),
            BLOCK_C=block_c,
        )
    
    logz = m + torch.log(s)
    logprobs = target_logits - logz
    entropy = logz - (t / s)
    
    return logprobs, entropy


# =============================================================================
# Custom autograd: V-chunk cross entropy — PyTorch backward
# =============================================================================

def _fwd_logsumexp_update(
    m: torch.Tensor, s: torch.Tensor, chunk_logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Online logsumexp accumulator (no entropy tracking)."""
    chunk_m = torch.amax(chunk_logits, dim=-1)
    m_new = torch.maximum(m, chunk_m)
    exp_old = torch.exp(m - m_new)
    chunk_exp = torch.exp(chunk_logits - m_new.unsqueeze(-1))
    s_new = s * exp_old + chunk_exp.sum(dim=-1)
    return m_new, s_new


class _VChunkCrossEntropyFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,   # [N, H]
        weight: torch.Tensor,   # [V, H]
        labels: torch.Tensor,   # [N]
        chunk_size: int,
    ) -> torch.Tensor:
        device = hidden.device
        n, vocab = hidden.shape[0], weight.shape[0]

        m = torch.full((n,), float("-inf"), device=device, dtype=torch.float32)
        s = torch.zeros((n,), device=device, dtype=torch.float32)
        target_logits = torch.zeros((n,), device=device, dtype=torch.float32)

        hidden_f32 = hidden.float()
        for v_start in range(0, vocab, chunk_size):
            v_end = min(v_start + chunk_size, vocab)
            logits = hidden_f32 @ weight[v_start:v_end].float().t()
            m, s = _fwd_logsumexp_update(m, s, logits)
            mask = (labels >= v_start) & (labels < v_end)
            if torch.any(mask):
                idx = (labels[mask] - v_start).long()
                target_logits[mask] = logits[mask, idx]

        logz = m + torch.log(s)
        logprobs = target_logits - logz

        ctx.save_for_backward(hidden, weight, labels, logz)
        ctx.chunk_size = chunk_size
        return logprobs

    @staticmethod
    def backward(ctx, grad_logprobs: torch.Tensor):
        hidden, weight, labels, logz = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        vocab = weight.shape[0]

        # grad_hidden: float32 accumulator (small tensor, high precision)
        # grad_weight: bfloat16 accumulator (large [V,H] tensor, saves ~1 GB vs float32)
        grad_hidden = torch.zeros(hidden.shape, device=hidden.device, dtype=torch.float32)
        grad_weight = torch.zeros_like(weight)
        g = grad_logprobs.float()
        hidden_f32 = hidden.float()

        for v_start in range(0, vocab, chunk_size):
            v_end = min(v_start + chunk_size, vocab)
            w_f32 = weight[v_start:v_end].float()
            logits = hidden_f32 @ w_f32.t()
            probs = torch.exp(logits - logz.unsqueeze(-1))

            # grad_logit[i,c] = g[i] * (1[c==target] - softmax[i,c])
            grad_logits = (-g).unsqueeze(-1) * probs
            mask = (labels >= v_start) & (labels < v_end)
            if torch.any(mask):
                idx = (labels[mask] - v_start).long()
                grad_logits[mask, idx] += g[mask]

            grad_hidden.add_(grad_logits @ w_f32)
            grad_weight[v_start:v_end].add_((grad_logits.t() @ hidden_f32).to(weight.dtype))

        return grad_hidden.to(hidden.dtype), grad_weight, None, None


def loss_v_chunk_pytorch(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """V-chunk cross entropy with custom PyTorch backward. Returns scalar loss (sum)."""
    logprobs = _VChunkCrossEntropyFn.apply(hidden, weight, labels, chunk_size)
    return -logprobs.sum()


# =============================================================================
# Custom autograd: V-chunk cross entropy — Triton backward
# =============================================================================

@triton.jit
def _compute_grad_logits_kernel(
    logits_ptr,        # [N, C] in-place: logits in, grad_logits out
    logz_ptr,          # [N]
    grad_logprob_ptr,  # [N]
    labels_ptr,        # [N]
    vocab_start,
    N, C,
    stride_n, stride_c,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= N:
        return

    logz_val = tl.load(logz_ptr + row)
    g_val = tl.load(grad_logprob_ptr + row)
    label = tl.load(labels_ptr + row)
    local_label = label - vocab_start

    for c_start in tl.range(0, C, BLOCK_C):
        c_offs = c_start + tl.arange(0, BLOCK_C)
        mask = c_offs < C

        logit = tl.load(
            logits_ptr + row * stride_n + c_offs * stride_c,
            mask=mask, other=float("-inf"),
        )
        prob = tl.exp(logit - logz_val)
        grad = -g_val * prob

        is_target = (c_offs == local_label) & (local_label >= 0) & (local_label < C)
        grad = tl.where(is_target, grad + g_val, grad)

        tl.store(
            logits_ptr + row * stride_n + c_offs * stride_c,
            grad, mask=mask,
        )


class _VChunkCrossEntropyTritonFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,   # [N, H]
        weight: torch.Tensor,   # [V, H]
        labels: torch.Tensor,   # [N]
        chunk_size: int,
        block_c: int,
    ) -> torch.Tensor:
        device = hidden.device
        n, vocab = hidden.shape[0], weight.shape[0]

        m = torch.full((n,), float("-inf"), device=device, dtype=torch.float32)
        s = torch.zeros((n,), device=device, dtype=torch.float32)
        t = torch.zeros((n,), device=device, dtype=torch.float32)
        target_logits = torch.zeros((n,), device=device, dtype=torch.float32)

        hidden_f32 = hidden.float()
        for v_start in range(0, vocab, chunk_size):
            v_end = min(v_start + chunk_size, vocab)
            logits = (hidden_f32 @ weight[v_start:v_end].float().t()).contiguous()
            C = logits.shape[1]
            _online_logsumexp_weighted_update_with_targets_kernel[(n,)](
                m, s, t, target_logits,
                logits, labels,
                v_start, n, C,
                logits.stride(0), logits.stride(1),
                BLOCK_C=block_c,
            )

        logz = m + torch.log(s)
        logprobs = target_logits - logz

        ctx.save_for_backward(hidden, weight, labels, logz)
        ctx.chunk_size = chunk_size
        ctx.block_c = block_c
        return logprobs

    @staticmethod
    def backward(ctx, grad_logprobs: torch.Tensor):
        hidden, weight, labels, logz = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        block_c = ctx.block_c
        n, vocab = hidden.shape[0], weight.shape[0]

        # grad_hidden: float32 accumulator (small tensor, high precision)
        # grad_weight: bfloat16 accumulator (large [V,H] tensor, saves ~1 GB vs float32)
        grad_hidden = torch.zeros(hidden.shape, device=hidden.device, dtype=torch.float32)
        grad_weight = torch.zeros_like(weight)
        g = grad_logprobs.float().contiguous()
        logz_c = logz.contiguous()
        hidden_f32 = hidden.float()

        for v_start in range(0, vocab, chunk_size):
            v_end = min(v_start + chunk_size, vocab)
            w_f32 = weight[v_start:v_end].float()
            # Recompute logits in float32 (no storage needed from forward)
            logits = (hidden_f32 @ w_f32.t()).contiguous()
            C = logits.shape[1]

            # Triton kernel converts logits -> grad_logits in-place
            _compute_grad_logits_kernel[(n,)](
                logits, logz_c, g, labels,
                v_start, n, C,
                logits.stride(0), logits.stride(1),
                BLOCK_C=block_c,
            )

            grad_hidden.add_(logits @ w_f32)
            grad_weight[v_start:v_end].add_((logits.t() @ hidden_f32).to(weight.dtype))

        return grad_hidden.to(hidden.dtype), grad_weight, None, None, None


def loss_v_chunk_triton(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 1024,
    block_c: int = 128,
) -> torch.Tensor:
    """V-chunk cross entropy with custom Triton backward. Returns scalar loss (sum)."""
    logprobs = _VChunkCrossEntropyTritonFn.apply(hidden, weight, labels, chunk_size, block_c)
    return -logprobs.sum()


# =============================================================================
# Liger and CCE wrappers
# =============================================================================

def loss_liger(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Liger fused linear cross entropy. Returns scalar loss (sum)."""
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    return LigerFusedLinearCrossEntropyLoss(reduction="sum")(weight, hidden, labels)


def loss_cce(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    impl: str = "cce_kahan_full_c",
) -> torch.Tensor:
    """Apple Cut Cross-Entropy. Returns scalar loss (sum)."""
    from cut_cross_entropy import linear_cross_entropy
    return linear_cross_entropy(hidden, weight, labels, shift=False, impl=impl, reduction="sum")


# =============================================================================
# Forward+backward benchmark utilities
# =============================================================================

@dataclass
class FwdBwdResult:
    name: str
    loss_match: bool
    loss_diff: float
    grad_match: bool
    grad_diff: float
    peak_memory_mb: float
    avg_time_ms: float
    std_time_ms: float


def measure_fwd_bwd(
    fn: Callable,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    warmup: int = 3,
    iterations: int = 10,
    **kwargs,
) -> tuple[float, torch.Tensor, float, float, float]:
    """Run fn(hidden, weight, labels, **kwargs).backward() and measure peak memory and time."""

    def run():
        h = hidden.detach().clone().requires_grad_(True)
        w = weight.detach().clone().requires_grad_(True)
        loss = fn(h, w, labels, **kwargs)
        loss.backward()
        torch.cuda.synchronize()
        return loss.item(), h.grad.clone()

    for _ in range(warmup):
        torch.cuda.reset_peak_memory_stats()
        run()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    loss_val, grad_h = run()
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        run()
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5

    return loss_val, grad_h, peak_memory, avg, std


def _baseline_loss(hidden, weight, labels):
    logits = hidden.float() @ weight.float().t()
    logprobs = torch.log_softmax(logits, dim=-1)
    return -logprobs.gather(1, labels.unsqueeze(1)).squeeze(1).sum()


def run_fwd_bwd_benchmark(
    N: int,
    H: int,
    V: int,
    chunk_size: int = 1024,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
) -> list[FwdBwdResult]:
    torch.manual_seed(seed)
    device = "cuda"
    hidden = torch.randn(N, H, device=device, dtype=dtype)
    weight = torch.randn(V, H, device=device, dtype=dtype)
    labels = torch.randint(0, V, (N,), device=device, dtype=torch.long)

    results = []

    # Baseline (no chunking, standard autograd)
    torch.cuda.empty_cache()
    try:
        loss_ref, grad_ref, mem, avg_t, std_t = measure_fwd_bwd(_baseline_loss, hidden, weight, labels)
        results.append(FwdBwdResult(
            name="No Chunk (Baseline)",
            loss_match=True, loss_diff=0.0,
            grad_match=True, grad_diff=0.0,
            peak_memory_mb=mem, avg_time_ms=avg_t, std_time_ms=std_t,
        ))
        baseline_ok = True
    except torch.cuda.OutOfMemoryError:
        print(f"  [No Chunk] OOM at N={N}, V={V}")
        loss_ref, grad_ref, baseline_ok = None, None, False

    def _add_result(name, fn, ref_loss, ref_grad, **kwargs):
        torch.cuda.empty_cache()
        try:
            lv, gh, mem, avg_t, std_t = measure_fwd_bwd(fn, hidden, weight, labels, **kwargs)
            if ref_loss is not None:
                lm = abs(lv - ref_loss) < 1.0
                ld = abs(lv - ref_loss)
                gm, gd = compare_tensors(gh.float(), ref_grad.float())
            else:
                lm, ld, gm, gd = True, 0.0, True, 0.0
            results.append(FwdBwdResult(
                name=name,
                loss_match=lm, loss_diff=ld,
                grad_match=gm, grad_diff=gd,
                peak_memory_mb=mem, avg_time_ms=avg_t, std_time_ms=std_t,
            ))
        except Exception as e:
            print(f"  [{name}] skipped: {e}")

    _add_result("V-Chunk PyTorch", loss_v_chunk_pytorch, loss_ref, grad_ref, chunk_size=chunk_size)
    _add_result("V-Chunk Triton", loss_v_chunk_triton, loss_ref, grad_ref, chunk_size=chunk_size)

    try:
        _add_result("Liger", loss_liger, loss_ref, grad_ref)
    except ImportError:
        print("  [Liger] skipped: liger-kernel not installed")

    try:
        _add_result("CCE (kahan_full_c)", loss_cce, loss_ref, grad_ref, impl="cce_kahan_full_c")
    except ImportError:
        print("  [CCE] skipped: cut-cross-entropy not installed")

    return results


def print_fwd_bwd_results(results: list[FwdBwdResult], config: str):
    print(f"\n{'='*80}")
    print(f"Config: {config}")
    print(f"{'='*80}")
    print(f"{'Implementation':<22} {'Loss Match':<12} {'Loss Diff':<12} {'Grad Match':<12} {'Grad Diff':<12} {'Memory (MB)':<12} {'Time (ms)':<15}")
    print(f"{'-'*105}")
    for r in results:
        lm = "✓" if r.loss_match else "✗"
        gm = "✓" if r.grad_match else "✗"
        print(f"{r.name:<22} {lm:<12} {r.loss_diff:<12.2e} {gm:<12} {r.grad_diff:<12.2e} {r.peak_memory_mb:<12.1f} {r.avg_time_ms:.2f} ± {r.std_time_ms:.2f}")


def test_fwd_bwd_performance():
    print("\n" + "="*80)
    print("FORWARD+BACKWARD BENCHMARK (V-Chunk vs Liger vs CCE)")
    print("="*80)

    configs = [
        (2048, 4096, 65536,  8192, "Medium Vocab (64K)"),
        (2048, 4096, 128000, 8192, "Large Vocab (128K)"),
        (4096, 4096, 128000, 8192, "Batch 4K, Vocab 128K"),
        (10240, 4096, 128000, 8192, "Batch 10K, Vocab 128K"),
    ]

    for n, h, v, chunk, desc in configs:
        results = run_fwd_bwd_benchmark(n, h, v, chunk_size=chunk)
        print_fwd_bwd_results(results, f"{desc}: N={n}, H={h}, V={v}, chunk={chunk}")


# =============================================================================
# Testing utilities
# =============================================================================

@dataclass
class BenchmarkResult:
    name: str
    logprobs_match: bool
    entropy_match: bool
    logprobs_max_diff: float
    entropy_max_diff: float
    peak_memory_mb: float
    avg_time_ms: float
    std_time_ms: float


def measure_memory_and_time(
    fn: Callable,
    *args,
    warmup: int = 3,
    iterations: int = 10,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, float, float, float]:
    """Run function and measure peak memory and execution time."""
    
    # Warmup
    for _ in range(warmup):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        _ = fn(*args, **kwargs)
        torch.cuda.synchronize()
    
    # Measure memory (single run)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    logprobs, entropy = fn(*args, **kwargs)
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    
    # Measure time (multiple runs)
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = fn(*args, **kwargs)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms
    
    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    
    return logprobs, entropy, peak_memory, avg_time, std_time


def compare_tensors(a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-4, atol: float = 1e-5) -> tuple[bool, float]:
    """Compare two tensors and return (match, max_diff)."""
    max_diff = (a - b).abs().max().item()
    match = torch.allclose(a, b, rtol=rtol, atol=atol)
    return match, max_diff


def run_benchmark(
    N: int,
    H: int,
    V: int,
    chunk_size: int = 1024,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
) -> list[BenchmarkResult]:
    """Run full benchmark comparing all implementations."""
    
    torch.manual_seed(seed)
    device = "cuda"
    
    # Generate test data
    hidden = torch.randn(N, H, device=device, dtype=dtype)
    weight = torch.randn(V, H, device=device, dtype=dtype)
    labels = torch.randint(0, V, (N,), device=device, dtype=torch.long)
    
    results = []
    
    # 1. Baseline (no chunking)
    torch.cuda.empty_cache()
    try:
        logprobs_ref, entropy_ref, mem_ref, time_ref, std_ref = measure_memory_and_time(
            logprobs_no_chunk, hidden, weight, labels
        )
        baseline_success = True
    except torch.cuda.OutOfMemoryError:
        print(f"  [No Chunk] OOM at N={N}, V={V}")
        baseline_success = False
        logprobs_ref = None
        entropy_ref = None
        mem_ref = float("inf")
        time_ref = float("inf")
        std_ref = 0.0
    
    if baseline_success:
        results.append(BenchmarkResult(
            name="No Chunk (Baseline)",
            logprobs_match=True,
            entropy_match=True,
            logprobs_max_diff=0.0,
            entropy_max_diff=0.0,
            peak_memory_mb=mem_ref,
            avg_time_ms=time_ref,
            std_time_ms=std_ref,
        ))
    
    # 2. V-chunked PyTorch
    torch.cuda.empty_cache()
    logprobs_pt, entropy_pt, mem_pt, time_pt, std_pt = measure_memory_and_time(
        logprobs_v_chunk_pytorch, hidden, weight, labels, chunk_size=chunk_size
    )
    
    if baseline_success:
        lp_match, lp_diff = compare_tensors(logprobs_pt, logprobs_ref)
        ent_match, ent_diff = compare_tensors(entropy_pt, entropy_ref)
    else:
        lp_match, lp_diff = True, 0.0
        ent_match, ent_diff = True, 0.0
    
    results.append(BenchmarkResult(
        name="V-Chunk PyTorch",
        logprobs_match=lp_match,
        entropy_match=ent_match,
        logprobs_max_diff=lp_diff,
        entropy_max_diff=ent_diff,
        peak_memory_mb=mem_pt,
        avg_time_ms=time_pt,
        std_time_ms=std_pt,
    ))
    
    # 3. V-chunked Triton
    torch.cuda.empty_cache()
    logprobs_tr, entropy_tr, mem_tr, time_tr, std_tr = measure_memory_and_time(
        logprobs_v_chunk_triton, hidden, weight, labels, chunk_size=chunk_size
    )
    
    if baseline_success:
        lp_match, lp_diff = compare_tensors(logprobs_tr, logprobs_ref)
        ent_match, ent_diff = compare_tensors(entropy_tr, entropy_ref)
    else:
        # Compare against PyTorch chunked version
        lp_match, lp_diff = compare_tensors(logprobs_tr, logprobs_pt)
        ent_match, ent_diff = compare_tensors(entropy_tr, entropy_pt)
    
    results.append(BenchmarkResult(
        name="V-Chunk Triton",
        logprobs_match=lp_match,
        entropy_match=ent_match,
        logprobs_max_diff=lp_diff,
        entropy_max_diff=ent_diff,
        peak_memory_mb=mem_tr,
        avg_time_ms=time_tr,
        std_time_ms=std_tr,
    ))
    
    return results


def print_results(results: list[BenchmarkResult], config: str):
    """Pretty print benchmark results."""
    print(f"\n{'='*80}")
    print(f"Config: {config}")
    print(f"{'='*80}")
    print(f"{'Implementation':<25} {'LogP Match':<12} {'Ent Match':<12} {'LogP Diff':<12} {'Ent Diff':<12} {'Memory (MB)':<12} {'Time (ms)':<15}")
    print(f"{'-'*100}")
    
    for r in results:
        lp_status = "✓" if r.logprobs_match else "✗"
        ent_status = "✓" if r.entropy_match else "✗"
        print(f"{r.name:<25} {lp_status:<12} {ent_status:<12} {r.logprobs_max_diff:<12.2e} {r.entropy_max_diff:<12.2e} {r.peak_memory_mb:<12.1f} {r.avg_time_ms:.2f} ± {r.std_time_ms:.2f}")


def test_correctness_small():
    """Small-scale correctness test with tight tolerances."""
    print("\n" + "="*80)
    print("CORRECTNESS TEST (Small Scale)")
    print("="*80)
    
    torch.manual_seed(42)
    device = "cuda"
    
    N, H, V = 128, 256, 4096
    chunk_size = 512
    
    hidden = torch.randn(N, H, device=device, dtype=torch.float32)
    weight = torch.randn(V, H, device=device, dtype=torch.float32)
    labels = torch.randint(0, V, (N,), device=device, dtype=torch.long)
    
    # Reference
    logprobs_ref, entropy_ref = logprobs_no_chunk(hidden, weight, labels)
    
    # PyTorch chunked
    logprobs_pt, entropy_pt = logprobs_v_chunk_pytorch(hidden, weight, labels, chunk_size=chunk_size)
    
    # Triton chunked
    logprobs_tr, entropy_tr = logprobs_v_chunk_triton(hidden, weight, labels, chunk_size=chunk_size)
    
    # Check
    print(f"\nPyTorch vs Reference:")
    lp_match, lp_diff = compare_tensors(logprobs_pt, logprobs_ref)
    ent_match, ent_diff = compare_tensors(entropy_pt, entropy_ref)
    print(f"  LogProbs: {'PASS' if lp_match else 'FAIL'} (max diff: {lp_diff:.2e})")
    print(f"  Entropy:  {'PASS' if ent_match else 'FAIL'} (max diff: {ent_diff:.2e})")
    
    print(f"\nTriton vs Reference:")
    lp_match, lp_diff = compare_tensors(logprobs_tr, logprobs_ref)
    ent_match, ent_diff = compare_tensors(entropy_tr, entropy_ref)
    print(f"  LogProbs: {'PASS' if lp_match else 'FAIL'} (max diff: {lp_diff:.2e})")
    print(f"  Entropy:  {'PASS' if ent_match else 'FAIL'} (max diff: {ent_diff:.2e})")
    
    print(f"\nTriton vs PyTorch Chunked:")
    lp_match, lp_diff = compare_tensors(logprobs_tr, logprobs_pt)
    ent_match, ent_diff = compare_tensors(entropy_tr, entropy_pt)
    print(f"  LogProbs: {'PASS' if lp_match else 'FAIL'} (max diff: {lp_diff:.2e})")
    print(f"  Entropy:  {'PASS' if ent_match else 'FAIL'} (max diff: {ent_diff:.2e})")


def test_memory_scaling():
    """Test memory usage scaling with vocabulary size."""
    print("\n" + "="*80)
    print("MEMORY SCALING TEST")
    print("="*80)
    
    N, H = 2048, 4096
    chunk_size = 2048
    
    configs = [
        (N, H, 32000, "Small Vocab (32K)"),
        (N, H, 65536, "Medium Vocab (64K)"),
        (N, H, 128000, "Large Vocab (128K)"),
    ]
    
    for n, h, v, name in configs:
        results = run_benchmark(n, h, v, chunk_size=chunk_size)
        print_results(results, f"{name}: N={n}, H={h}, V={v}, chunk={chunk_size}")


def test_performance_scaling():
    """Test performance with different configurations."""
    print("\n" + "="*80)
    print("PERFORMANCE SCALING TEST")
    print("="*80)
    
    configs = [
        # (N, H, V, chunk_size, description)
        (1024, 4096, 128000, 1024, "Batch 1K, Chunk 1K"),
        (1024, 4096, 128000, 2048, "Batch 1K, Chunk 2K"),
        (1024, 4096, 128000, 4096, "Batch 1K, Chunk 4K"),
        (4096, 4096, 128000, 2048, "Batch 4K, Chunk 2K"),
        (8192, 4096, 128000, 2048, "Batch 8K, Chunk 2K"),
    ]
    
    for n, h, v, chunk, desc in configs:
        results = run_benchmark(n, h, v, chunk_size=chunk)
        print_results(results, f"{desc}: N={n}, H={h}, V={v}")


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "="*80)
    print("EDGE CASE TESTS")
    print("="*80)
    
    torch.manual_seed(42)
    device = "cuda"
    
    # Edge case 1: Labels at chunk boundaries
    print("\n1. Labels at chunk boundaries:")
    N, H, V = 64, 128, 1024
    chunk_size = 256
    
    hidden = torch.randn(N, H, device=device, dtype=torch.float32)
    weight = torch.randn(V, H, device=device, dtype=torch.float32)
    
    # Place labels exactly at chunk boundaries
    labels = torch.tensor([0, 255, 256, 511, 512, 767, 768, 1023] * 8, device=device, dtype=torch.long)
    
    logprobs_ref, entropy_ref = logprobs_no_chunk(hidden, weight, labels)
    logprobs_pt, entropy_pt = logprobs_v_chunk_pytorch(hidden, weight, labels, chunk_size=chunk_size)
    logprobs_tr, entropy_tr = logprobs_v_chunk_triton(hidden, weight, labels, chunk_size=chunk_size)
    
    lp_match_pt, lp_diff_pt = compare_tensors(logprobs_pt, logprobs_ref)
    lp_match_tr, lp_diff_tr = compare_tensors(logprobs_tr, logprobs_ref)
    print(f"  PyTorch: {'PASS' if lp_match_pt else 'FAIL'} (diff: {lp_diff_pt:.2e})")
    print(f"  Triton:  {'PASS' if lp_match_tr else 'FAIL'} (diff: {lp_diff_tr:.2e})")
    
    # Edge case 2: Single token
    print("\n2. Single token:")
    N = 1
    hidden = torch.randn(N, H, device=device, dtype=torch.float32)
    labels = torch.randint(0, V, (N,), device=device, dtype=torch.long)
    
    logprobs_ref, entropy_ref = logprobs_no_chunk(hidden, weight, labels)
    logprobs_pt, entropy_pt = logprobs_v_chunk_pytorch(hidden, weight, labels, chunk_size=chunk_size)
    logprobs_tr, entropy_tr = logprobs_v_chunk_triton(hidden, weight, labels, chunk_size=chunk_size)
    
    lp_match_pt, _ = compare_tensors(logprobs_pt, logprobs_ref)
    lp_match_tr, _ = compare_tensors(logprobs_tr, logprobs_ref)
    print(f"  PyTorch: {'PASS' if lp_match_pt else 'FAIL'}")
    print(f"  Triton:  {'PASS' if lp_match_tr else 'FAIL'}")
    
    # Edge case 3: Chunk size larger than vocab
    print("\n3. Chunk size > vocab:")
    chunk_size = 2048  # Larger than V=1024
    
    N = 32
    hidden = torch.randn(N, H, device=device, dtype=torch.float32)
    labels = torch.randint(0, V, (N,), device=device, dtype=torch.long)
    
    logprobs_ref, entropy_ref = logprobs_no_chunk(hidden, weight, labels)
    logprobs_pt, entropy_pt = logprobs_v_chunk_pytorch(hidden, weight, labels, chunk_size=chunk_size)
    logprobs_tr, entropy_tr = logprobs_v_chunk_triton(hidden, weight, labels, chunk_size=chunk_size)
    
    lp_match_pt, _ = compare_tensors(logprobs_pt, logprobs_ref)
    lp_match_tr, _ = compare_tensors(logprobs_tr, logprobs_ref)
    print(f"  PyTorch: {'PASS' if lp_match_pt else 'FAIL'}")
    print(f"  Triton:  {'PASS' if lp_match_tr else 'FAIL'}")
    
    # Edge case 4: Very large logits (numerical stability)
    print("\n4. Large logits (numerical stability):")
    N, H, V = 64, 128, 1024
    chunk_size = 256
    
    hidden = torch.randn(N, H, device=device, dtype=torch.float32) * 10
    weight = torch.randn(V, H, device=device, dtype=torch.float32) * 10
    labels = torch.randint(0, V, (N,), device=device, dtype=torch.long)
    
    logprobs_ref, entropy_ref = logprobs_no_chunk(hidden, weight, labels)
    logprobs_pt, entropy_pt = logprobs_v_chunk_pytorch(hidden, weight, labels, chunk_size=chunk_size)
    logprobs_tr, entropy_tr = logprobs_v_chunk_triton(hidden, weight, labels, chunk_size=chunk_size)
    
    lp_match_pt, lp_diff_pt = compare_tensors(logprobs_pt, logprobs_ref, rtol=1e-3, atol=1e-4)
    lp_match_tr, lp_diff_tr = compare_tensors(logprobs_tr, logprobs_ref, rtol=1e-3, atol=1e-4)
    print(f"  PyTorch: {'PASS' if lp_match_pt else 'FAIL'} (diff: {lp_diff_pt:.2e})")
    print(f"  Triton:  {'PASS' if lp_match_tr else 'FAIL'} (diff: {lp_diff_tr:.2e})")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("ONLINE SOFTMAX LOGPROBS/ENTROPY BENCHMARK")
    print("="*80)
    
    test_correctness_small()
    test_edge_cases()
    test_memory_scaling()
    test_performance_scaling()
    test_fwd_bwd_performance()


if __name__ == "__main__":
    run_all_tests()