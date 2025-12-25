"""
Kernel-level Benchmark for Grouped GEMM

Compares performance of:
- PyTorch baseline (for loop)
- PyTorch optimized (batched matmul)
- Light-MoE CuTe kernel (when available)
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List

import torch
import torch.nn.functional as F


def benchmark_grouped_gemm_pytorch(
    inputs: List[torch.Tensor],
    weights: torch.Tensor,
    warmup: int = 10,
    iterations: int = 100,
) -> Dict[str, float]:
    """
    Benchmark PyTorch grouped GEMM (for loop version).
    """
    num_experts = weights.size(0)
    
    # Warmup
    for _ in range(warmup):
        outputs = []
        for i in range(num_experts):
            if inputs[i].numel() > 0:
                outputs.append(F.linear(inputs[i], weights[i]))
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        outputs = []
        for i in range(num_experts):
            if inputs[i].numel() > 0:
                outputs.append(F.linear(inputs[i], weights[i]))
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    # Compute FLOPS
    total_flops = 0
    for i, inp in enumerate(inputs):
        if inp.numel() > 0:
            m, k = inp.shape
            n = weights.size(1)
            total_flops += 2 * m * n * k
    
    return {
        "avg_time_ms": (elapsed / iterations) * 1000,
        "total_flops": total_flops,
        "tflops": (total_flops * iterations) / elapsed / 1e12,
    }


def benchmark_grouped_gemm_batched(
    input_concat: torch.Tensor,
    weights: torch.Tensor,
    expert_offsets: torch.Tensor,
    warmup: int = 10,
    iterations: int = 100,
) -> Dict[str, float]:
    """
    Benchmark batched approach (concatenate then process).
    """
    num_experts = weights.size(0)
    total_tokens = input_concat.size(0)
    hidden_size = input_concat.size(1)
    out_features = weights.size(1)
    
    output = torch.empty(total_tokens, out_features, device=input_concat.device, dtype=input_concat.dtype)
    
    def run():
        for i in range(num_experts):
            start_idx = expert_offsets[i].item()
            end_idx = expert_offsets[i + 1].item()
            if end_idx > start_idx:
                output[start_idx:end_idx] = F.linear(input_concat[start_idx:end_idx], weights[i])
    
    # Warmup
    for _ in range(warmup):
        run()
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        run()
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    # FLOPS
    total_flops = 2 * total_tokens * hidden_size * out_features
    
    return {
        "avg_time_ms": (elapsed / iterations) * 1000,
        "total_flops": total_flops,
        "tflops": (total_flops * iterations) / elapsed / 1e12,
    }


def run_benchmarks(
    num_experts: int = 8,
    hidden_size: int = 4096,
    intermediate_size: int = 14336,
    total_tokens: int = 2048,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
):
    """Run all grouped GEMM benchmarks."""
    
    print(f"\n{'='*60}")
    print("Grouped GEMM Benchmark")
    print(f"{'='*60}")
    print(f"Experts: {num_experts}")
    print(f"Hidden size: {hidden_size}")
    print(f"Intermediate size: {intermediate_size}")
    print(f"Total tokens: {total_tokens}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"{'='*60}\n")
    
    # Create random token distribution
    torch.manual_seed(42)
    tokens_per_expert = []
    remaining = total_tokens
    
    for i in range(num_experts - 1):
        # Random distribution with some variance
        count = max(1, int(remaining / (num_experts - i) + torch.randn(1).item() * 50))
        count = min(count, remaining - (num_experts - i - 1))
        tokens_per_expert.append(count)
        remaining -= count
    tokens_per_expert.append(remaining)
    
    print("Tokens per expert:", tokens_per_expert)
    
    # Create inputs
    inputs = []
    offsets = [0]
    for i, count in enumerate(tokens_per_expert):
        inputs.append(torch.randn(count, hidden_size, device=device, dtype=dtype))
        offsets.append(offsets[-1] + count)
    
    input_concat = torch.cat(inputs, dim=0)
    expert_offsets = torch.tensor(offsets, device=device, dtype=torch.int64)
    
    # Create weights [num_experts, out_features, in_features]
    weights = torch.randn(num_experts, intermediate_size, hidden_size, device=device, dtype=dtype)
    
    # Run benchmarks
    print("\n1. PyTorch For Loop:")
    result1 = benchmark_grouped_gemm_pytorch(inputs, weights)
    print(f"   Avg time: {result1['avg_time_ms']:.3f} ms")
    print(f"   TFLOPS: {result1['tflops']:.2f}")
    
    print("\n2. PyTorch Batched (Sliced):")
    result2 = benchmark_grouped_gemm_batched(input_concat, weights, expert_offsets)
    print(f"   Avg time: {result2['avg_time_ms']:.3f} ms")
    print(f"   TFLOPS: {result2['tflops']:.2f}")
    
    # Try Light-MoE kernel if available
    try:
        from light_moe._C import grouped_gemm as cute_grouped_gemm
        
        print("\n3. Light-MoE CuTe Kernel:")
        
        # Warmup
        for _ in range(10):
            output = cute_grouped_gemm(input_concat, weights, expert_offsets, num_experts)
            torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(100):
            output = cute_grouped_gemm(input_concat, weights, expert_offsets, num_experts)
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        tflops = (result1['total_flops'] * 100) / elapsed / 1e12
        
        print(f"   Avg time: {(elapsed / 100) * 1000:.3f} ms")
        print(f"   TFLOPS: {tflops:.2f}")
        
        speedup = result2['avg_time_ms'] / ((elapsed / 100) * 1000)
        print(f"   Speedup vs batched: {speedup:.2f}x")
        
    except ImportError:
        print("\n3. Light-MoE CuTe Kernel: Not available (not compiled)")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grouped GEMM Benchmark")
    parser.add_argument("--experts", type=int, default=8, help="Number of experts")
    parser.add_argument("--hidden", type=int, default=4096, help="Hidden size")
    parser.add_argument("--intermediate", type=int, default=14336, help="Intermediate size")
    parser.add_argument("--tokens", type=int, default=2048, help="Total tokens")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    
    args = parser.parse_args()
    
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    
    run_benchmarks(
        num_experts=args.experts,
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        total_tokens=args.tokens,
        dtype=dtype_map[args.dtype],
    )
