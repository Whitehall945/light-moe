"""
Fused MoE Ops Benchmark

Compares:
1. PyTorch separate ops: gate @ weight -> softmax -> topk -> permute
2. Light-MoE fused: single kernel for all operations
"""

import time
import torch
import torch.nn.functional as F


def benchmark_pytorch_separate(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    top_k: int,
    warmup: int = 10,
    iterations: int = 100,
):
    """Benchmark PyTorch separate operations."""
    
    # Warmup
    for _ in range(warmup):
        logits = torch.mm(hidden, gate_weight.t())
        probs = F.softmax(logits, dim=-1)
        weights, indices = torch.topk(probs, top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        logits = torch.mm(hidden, gate_weight.t())
        probs = F.softmax(logits, dim=-1)
        weights, indices = torch.topk(probs, top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    return elapsed / iterations * 1000  # ms


def benchmark_fused(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    top_k: int,
    warmup: int = 10,
    iterations: int = 100,
):
    """Benchmark Light-MoE fused operations."""
    
    from light_moe._C import fused_gate_topk
    
    # Warmup
    for _ in range(warmup):
        indices, weights = fused_gate_topk(hidden, gate_weight, top_k)
        torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        indices, weights = fused_gate_topk(hidden, gate_weight, top_k)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    return elapsed / iterations * 1000  # ms


def run_fused_benchmark():
    """Run comprehensive fused ops benchmark."""
    
    print("\n" + "="*70)
    print("FUSED MoE OPS BENCHMARK")
    print("="*70)
    print("Fusing Gate Projection + Softmax + TopK into efficient CUDA kernel")
    print("="*70)
    
    device = torch.device("cuda")
    
    # Test configurations
    configs = [
        (1024, 4096, 8, 2),
        (2048, 4096, 8, 2),
        (4096, 4096, 8, 2),
        (8192, 4096, 8, 2),
    ]
    
    print("\n" + "-"*70)
    print(f"{'Tokens':<10} {'Hidden':<10} {'Experts':<10} {'PyTorch':<12} {'Fused':<12} {'Speedup':<10}")
    print("-"*70)
    
    for num_tokens, hidden_size, num_experts, top_k in configs:
        hidden = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.float32)
        gate_weight = torch.randn(num_experts, hidden_size, device=device, dtype=torch.float32)
        
        pytorch_time = benchmark_pytorch_separate(hidden, gate_weight, top_k)
        fused_time = benchmark_fused(hidden, gate_weight, top_k)
        
        speedup = pytorch_time / fused_time
        
        print(f"{num_tokens:<10} {hidden_size:<10} {num_experts:<10} {pytorch_time:>8.3f}ms   {fused_time:>8.3f}ms   {speedup:>6.2f}x")
    
    print("-"*70)
    
    # Verify correctness (check if top experts are the same, order may differ)
    print("\nCorrectness Check:")
    hidden = torch.randn(128, 4096, device=device, dtype=torch.float32)
    gate_weight = torch.randn(8, 4096, device=device, dtype=torch.float32)
    
    # PyTorch reference
    logits = torch.mm(hidden, gate_weight.t())
    probs = F.softmax(logits, dim=-1)
    ref_weights, ref_indices = torch.topk(probs, 2, dim=-1)
    ref_weights = ref_weights / ref_weights.sum(dim=-1, keepdim=True)
    
    # Fused
    from light_moe._C import fused_gate_topk
    fused_indices, fused_weights = fused_gate_topk(hidden, gate_weight, 2)
    
    # Check if same experts are selected (order may differ due to ties)
    ref_set = set()
    fused_set = set()
    for i in range(128):
        ref_set.add((i, tuple(sorted(ref_indices[i].tolist()))))
        fused_set.add((i, tuple(sorted(fused_indices[i].tolist()))))
    
    same_experts = len(ref_set & fused_set) / len(ref_set)
    weights_close = torch.allclose(ref_weights.sort(dim=-1)[0], 
                                   fused_weights.sort(dim=-1)[0], 
                                   rtol=1e-3, atol=1e-3)
    
    print(f"  Same expert selection: {same_experts:.1%}")
    print(f"  Weights close (sorted): {weights_close}")
    
    # Max weight difference
    max_diff = (ref_weights.sort(dim=-1)[0] - fused_weights.sort(dim=-1)[0]).abs().max().item()
    print(f"  Max weight difference: {max_diff:.6f}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    run_fused_benchmark()
