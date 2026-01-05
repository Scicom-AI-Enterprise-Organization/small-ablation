import torch
import torch.utils.benchmark as benchmark
from torchao.prototype.moe_training.scaled_grouped_mm import (
    _to_fp8_rowwise_then_scaled_grouped_mm,
)
from torchao.prototype.moe_training.utils import _is_column_major


def test_fp8_backward():
    """Test FP8 grouped_mm backward pass."""
    
    num_experts = 64
    hidden_dim = 4096
    intermediate_dim = 14336
    
    # Test with unaligned expert counts
    experts_count = torch.randint(100, 500, (num_experts,), device='cuda')
    cu_experts = experts_count.cumsum(0).to(torch.int32)
    total_tokens = experts_count.sum().item()
    
    print(f"Config: {num_experts} experts, {total_tokens} tokens")
    print(f"Expert counts (first 8): {experts_count[:8].tolist()}")
    print(f"Any divisible by 16: {(experts_count % 16 == 0).sum().item()}/{num_experts}")
    
    # Inputs with grad
    A = torch.randn(total_tokens, hidden_dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    
    # FP8 weights (column-major via transpose)
    B_storage = torch.randn(num_experts, intermediate_dim, hidden_dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    B_t = B_storage.transpose(-1, -2)
    
    print(f"\nA shape: {A.shape}, requires_grad: {A.requires_grad}")
    print(f"B_t shape: {B_t.shape}, is column-major: {_is_column_major(B_t)}")
    
    # Forward
    print("\n1. Testing forward...")
    try:
        out = _to_fp8_rowwise_then_scaled_grouped_mm(A, B_t, cu_experts)
        print(f"   ✓ Forward passed! Output shape: {out.shape}")
    except Exception as e:
        print(f"   ✗ Forward failed: {e}")
        return False
    
    # Backward
    print("\n2. Testing backward...")
    grad_output = torch.randn_like(out)
    
    try:
        out.backward(grad_output)
        print(f"   ✓ Backward passed!")
        print(f"   A.grad shape: {A.grad.shape if A.grad is not None else None}")
        print(f"   B_storage.grad shape: {B_storage.grad.shape if B_storage.grad is not None else None}")
        return True
    except Exception as e:
        print(f"   ✗ Backward failed/stuck: {e}")
        return False


def test_fp8_backward_aligned():
    """Test FP8 grouped_mm backward with aligned expert counts."""
    
    num_experts = 64
    hidden_dim = 4096
    intermediate_dim = 14336
    
    # Force aligned expert counts (multiples of 16)
    experts_count_raw = torch.randint(100, 500, (num_experts,), device='cuda')
    experts_count = ((experts_count_raw + 15) // 16) * 16  # Align to 16
    
    cu_experts = experts_count.cumsum(0).to(torch.int32)
    total_tokens = experts_count.sum().item()
    
    print(f"\nConfig (ALIGNED): {num_experts} experts, {total_tokens} tokens")
    print(f"Expert counts (first 8): {experts_count[:8].tolist()}")
    print(f"All divisible by 16: {(experts_count % 16 == 0).all().item()}")
    
    A = torch.randn(total_tokens, hidden_dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    B_storage = torch.randn(num_experts, intermediate_dim, hidden_dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    B_t = B_storage.transpose(-1, -2)
    
    # Forward
    print("\n1. Testing forward (aligned)...")
    try:
        out = _to_fp8_rowwise_then_scaled_grouped_mm(A, B_t, cu_experts)
        print(f"   ✓ Forward passed! Output shape: {out.shape}")
    except Exception as e:
        print(f"   ✗ Forward failed: {e}")
        return False
    
    # Backward
    print("\n2. Testing backward (aligned)...")
    grad_output = torch.randn_like(out)
    
    try:
        out.backward(grad_output)
        print(f"   ✓ Backward passed!")
        print(f"   A.grad shape: {A.grad.shape if A.grad is not None else None}")
        print(f"   B_storage.grad shape: {B_storage.grad.shape if B_storage.grad is not None else None}")
        return True
    except Exception as e:
        print(f"   ✗ Backward failed: {e}")
        return False


def test_fp8_backward_with_timeout():
    """Test with timeout to detect hanging."""
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Backward pass is hanging!")
    
    num_experts = 64
    hidden_dim = 4096
    intermediate_dim = 14336
    
    for aligned in [False, True]:
        print(f"\n{'='*60}")
        print(f"Testing {'ALIGNED' if aligned else 'UNALIGNED'} expert counts")
        print('='*60)
        
        experts_count = torch.randint(100, 500, (num_experts,), device='cuda')
        if aligned:
            experts_count = ((experts_count + 15) // 16) * 16
        
        cu_experts = experts_count.cumsum(0).to(torch.int32)
        total_tokens = experts_count.sum().item()
        
        A = torch.randn(total_tokens, hidden_dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        B_storage = torch.randn(num_experts, intermediate_dim, hidden_dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        B_t = B_storage.transpose(-1, -2)
        
        # Forward
        out = _to_fp8_rowwise_then_scaled_grouped_mm(A, B_t, cu_experts)
        grad_output = torch.randn_like(out)
        
        # Set timeout (Unix only)
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10)  # 10 second timeout
            
            torch.cuda.synchronize()
            out.backward(grad_output)
            torch.cuda.synchronize()
            
            signal.alarm(0)  # Cancel timeout
            print(f"✓ Backward completed successfully!")
            
        except TimeoutError:
            print(f"✗ Backward HANGING (timeout after 10s)")
        except Exception as e:
            signal.alarm(0)
            print(f"✗ Backward failed with error: {e}")


def benchmark_with_backward():
    """Benchmark forward + backward for FP8 vs BF16."""
    
    num_experts = 64
    hidden_dim = 4096
    intermediate_dim = 14336
    
    # Use aligned counts if unaligned hangs
    experts_count = torch.randint(100, 500, (num_experts,), device='cuda')
    experts_count_aligned = ((experts_count + 15) // 16) * 16
    
    cu_experts = experts_count.cumsum(0).to(torch.int32)
    cu_experts_aligned = experts_count_aligned.cumsum(0).to(torch.int32)
    
    total_tokens = experts_count.sum().item()
    total_tokens_aligned = experts_count_aligned.sum().item()
    
    print(f"Unaligned tokens: {total_tokens}")
    print(f"Aligned tokens: {total_tokens_aligned} (+{(total_tokens_aligned/total_tokens - 1)*100:.1f}%)")
    
    # BF16 setup
    A_bf16 = torch.randn(total_tokens, hidden_dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    B_bf16 = torch.randn(num_experts, hidden_dim, intermediate_dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    
    # FP8 setup (aligned)
    A_fp8 = torch.randn(total_tokens_aligned, hidden_dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    B_fp8_storage = torch.randn(num_experts, intermediate_dim, hidden_dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    
    def bf16_fwd_bwd():
        A_bf16.grad = None
        B_bf16.grad = None
        out = torch._grouped_mm(A_bf16, B_bf16, cu_experts)
        out.backward(torch.ones_like(out))
        return out
    
    def fp8_fwd_bwd():
        A_fp8.grad = None
        B_fp8_storage.grad = None
        B_t = B_fp8_storage.transpose(-1, -2)
        out = _to_fp8_rowwise_then_scaled_grouped_mm(A_fp8, B_t, cu_experts_aligned)
        out.backward(torch.ones_like(out))
        return out
    
    # Warmup
    print("\nWarming up...")
    for _ in range(5):
        bf16_fwd_bwd()
        fp8_fwd_bwd()
    torch.cuda.synchronize()
    
    # Benchmark
    num_runs = 50
    
    print(f"\nBenchmarking ({num_runs} runs)...")
    
    t_bf16 = benchmark.Timer(stmt='fn()', globals={'fn': bf16_fwd_bwd})
    t_fp8 = benchmark.Timer(stmt='fn()', globals={'fn': fp8_fwd_bwd})
    
    r_bf16 = t_bf16.timeit(num_runs)
    r_fp8 = t_fp8.timeit(num_runs)
    
    print(f"\nSingle matmul forward + backward:")
    print(f"  BF16 (unaligned): {r_bf16.median * 1000:.3f} ms")
    print(f"  FP8 (aligned):    {r_fp8.median * 1000:.3f} ms")
    print(f"  Speedup: {r_bf16.median / r_fp8.median:.2f}x")


def benchmark_full_moe_with_backward():
    """Benchmark full MoE (gate + up + down) with backward."""
    
    num_experts = 64
    hidden_dim = 4096
    intermediate_dim = 14336
    
    experts_count = torch.randint(100, 500, (num_experts,), device='cuda')
    experts_count_aligned = ((experts_count + 15) // 16) * 16
    
    cu_experts = experts_count.cumsum(0).to(torch.int32)
    cu_experts_aligned = experts_count_aligned.cumsum(0).to(torch.int32)
    
    total_tokens = experts_count.sum().item()
    total_tokens_aligned = experts_count_aligned.sum().item()
    
    # BF16 weights
    gate_bf16 = torch.randn(num_experts, hidden_dim, intermediate_dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    up_bf16 = torch.randn(num_experts, hidden_dim, intermediate_dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    down_bf16 = torch.randn(num_experts, intermediate_dim, hidden_dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    
    # FP8 weights
    gate_fp8 = torch.randn(num_experts, intermediate_dim, hidden_dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    up_fp8 = torch.randn(num_experts, intermediate_dim, hidden_dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    down_fp8 = torch.randn(num_experts, hidden_dim, intermediate_dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    
    def moe_bf16_fwd_bwd():
        A = torch.randn(total_tokens, hidden_dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        
        gate_out = torch._grouped_mm(A, gate_bf16, cu_experts)
        up_out = torch._grouped_mm(A, up_bf16, cu_experts)
        intermediate = torch.nn.functional.silu(gate_out) * up_out
        out = torch._grouped_mm(intermediate, down_bf16, cu_experts)
        
        out.backward(torch.ones_like(out))
        return out
    
    def moe_fp8_fwd_bwd():
        A = torch.randn(total_tokens_aligned, hidden_dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        
        gate_out = _to_fp8_rowwise_then_scaled_grouped_mm(A, gate_fp8.transpose(-1, -2), cu_experts_aligned)
        up_out = _to_fp8_rowwise_then_scaled_grouped_mm(A, up_fp8.transpose(-1, -2), cu_experts_aligned)
        intermediate = torch.nn.functional.silu(gate_out) * up_out
        out = _to_fp8_rowwise_then_scaled_grouped_mm(intermediate, down_fp8.transpose(-1, -2), cu_experts_aligned)
        
        out.backward(torch.ones_like(out))
        return out
    
    print(f"\nFull MoE benchmark (forward + backward):")
    print(f"  Experts: {num_experts}")
    print(f"  Unaligned tokens: {total_tokens}")
    print(f"  Aligned tokens: {total_tokens_aligned} (+{(total_tokens_aligned/total_tokens - 1)*100:.1f}% padding)")
    
    # Warmup
    for _ in range(5):
        moe_bf16_fwd_bwd()
        moe_fp8_fwd_bwd()
    torch.cuda.synchronize()
    
    num_runs = 30
    
    t_bf16 = benchmark.Timer(stmt='fn()', globals={'fn': moe_bf16_fwd_bwd})
    t_fp8 = benchmark.Timer(stmt='fn()', globals={'fn': moe_fp8_fwd_bwd})
    
    r_bf16 = t_bf16.timeit(num_runs)
    r_fp8 = t_fp8.timeit(num_runs)
    
    print(f"\nResults ({num_runs} runs):")
    print(f"  BF16: {r_bf16.median * 1000:.3f} ms")
    print(f"  FP8:  {r_fp8.median * 1000:.3f} ms")
    print(f"  Speedup: {r_bf16.median / r_fp8.median:.2f}x")
    
    # Also show tokens/sec
    bf16_tps = total_tokens / r_bf16.median
    fp8_tps = total_tokens_aligned / r_fp8.median
    print(f"\n  BF16 throughput: {bf16_tps/1e6:.2f}M tokens/sec")
    print(f"  FP8 throughput:  {fp8_tps/1e6:.2f}M tokens/sec")


if __name__ == "__main__":
    print("=" * 70)
    print("FP8 GROUPED_MM BACKWARD TEST")
    print("=" * 70)
    
    # Test unaligned first
    print("\n[TEST 1] Unaligned expert counts:")
    unaligned_works = test_fp8_backward()
    
    # Test aligned
    print("\n[TEST 2] Aligned expert counts (multiples of 16):")
    aligned_works = test_fp8_backward_aligned()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Unaligned backward: {'✓ Works' if unaligned_works else '✗ Fails/Hangs'}")
    print(f"Aligned backward:   {'✓ Works' if aligned_works else '✗ Fails/Hangs'}")
    
    if aligned_works:
        print("\n" + "=" * 70)
        print("BENCHMARK (with padding for alignment)")
        print("=" * 70)
        benchmark_with_backward()
        benchmark_full_moe_with_backward()

"""
======================================================================
FP8 GROUPED_MM BACKWARD TEST
======================================================================

[TEST 1] Unaligned expert counts:
Config: 64 experts, 19578 tokens
Expert counts (first 8): [196, 414, 161, 424, 441, 302, 481, 185]
Any divisible by 16: 2/64

A shape: torch.Size([19578, 4096]), requires_grad: True
B_t shape: torch.Size([64, 4096, 14336]), is column-major: True

1. Testing forward...
   ✓ Forward passed! Output shape: torch.Size([19578, 14336])

2. Testing backward...
   ✗ Backward failed/stuck: strides should be multiple of 16 bytes

[TEST 2] Aligned expert counts (multiples of 16):

Config (ALIGNED): 64 experts, 20208 tokens
Expert counts (first 8): [416, 432, 384, 448, 208, 240, 128, 480]
All divisible by 16: True

1. Testing forward (aligned)...
   ✓ Forward passed! Output shape: torch.Size([20208, 14336])

2. Testing backward (aligned)...
   ✓ Backward passed!
   A.grad shape: torch.Size([20208, 4096])
   B_storage.grad shape: torch.Size([64, 14336, 4096])

======================================================================
SUMMARY
======================================================================
Unaligned backward: ✗ Fails/Hangs
Aligned backward:   ✓ Works

======================================================================
BENCHMARK (with padding for alignment)
======================================================================
Unaligned tokens: 19116
Aligned tokens: 19584 (+2.4%)

Warming up...

Benchmarking (50 runs)...

Single matmul forward + backward:
  BF16 (unaligned): 1.142 ms
  FP8 (aligned):    7.379 ms
  Speedup: 0.15x

Full MoE benchmark (forward + backward):
  Experts: 64
  Unaligned tokens: 19382
  Aligned tokens: 19872 (+2.5% padding)

Results (30 runs):
  BF16: 5.711 ms
  FP8:  24.388 ms
  Speedup: 0.23x

  BF16 throughput: 3.39M tokens/sec
  FP8 throughput:  0.81M tokens/sec
"""