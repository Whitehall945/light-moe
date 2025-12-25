"""
Real CUDA Kernel Benchmark

Compares Light-MoE CUDA kernels against PyTorch baseline.
"""

import time
import torch
import torch.nn.functional as F


def benchmark_grouped_gemm():
    """Benchmark Grouped GEMM: CUDA kernel vs PyTorch."""
    
    from light_moe._C import grouped_gemm
    
    print("\n" + "="*70)
    print("GROUPED GEMM BENCHMARK: Light-MoE CUDA vs PyTorch")
    print("="*70)
    
    device = torch.device("cuda")
    dtype = torch.float16
    
    # Configuration (Mixtral-like)
    num_experts = 8
    hidden_size = 4096
    intermediate_size = 14336
    
    for total_tokens in [1024, 2048, 4096]:
        print(f"\n--- Total Tokens: {total_tokens} ---")
        
        # Create token distribution
        tokens_per_expert = [total_tokens // num_experts] * num_experts
        # Add remainder to first experts
        for i in range(total_tokens % num_experts):
            tokens_per_expert[i] += 1
        
        offsets = [0]
        for t in tokens_per_expert:
            offsets.append(offsets[-1] + t)
        
        # Create tensors
        input_tensor = torch.randn(total_tokens, hidden_size, device=device, dtype=dtype)
        weights = torch.randn(num_experts, intermediate_size, hidden_size, device=device, dtype=dtype)
        expert_offsets = torch.tensor(offsets, device=device, dtype=torch.int64)
        
        # Warmup
        for _ in range(10):
            _ = grouped_gemm(input_tensor, weights, expert_offsets, num_experts)
        torch.cuda.synchronize()
        
        # Benchmark CUDA kernel
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            output_cuda = grouped_gemm(input_tensor, weights, expert_offsets, num_experts)
        torch.cuda.synchronize()
        cuda_time = (time.perf_counter() - start) / 100 * 1000
        
        # PyTorch reference
        output_ref = torch.zeros(total_tokens, intermediate_size, device=device, dtype=dtype)
        
        for _ in range(10):  # Warmup
            for e in range(num_experts):
                s, end = offsets[e], offsets[e+1]
                if end > s:
                    output_ref[s:end] = torch.mm(input_tensor[s:end], weights[e].t())
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(100):
            for e in range(num_experts):
                s, end = offsets[e], offsets[e+1]
                if end > s:
                    output_ref[s:end] = torch.mm(input_tensor[s:end], weights[e].t())
        torch.cuda.synchronize()
        pytorch_time = (time.perf_counter() - start) / 100 * 1000
        
        # Verify correctness
        diff = (output_cuda - output_ref).abs().max().item()
        
        # Compute TFLOPS
        flops = 2 * total_tokens * hidden_size * intermediate_size
        cuda_tflops = flops / (cuda_time / 1000) / 1e12
        pytorch_tflops = flops / (pytorch_time / 1000) / 1e12
        
        speedup = pytorch_time / cuda_time
        
        print(f"  Light-MoE CUDA:  {cuda_time:>7.3f}ms  ({cuda_tflops:.1f} TFLOPS)")
        print(f"  PyTorch Loop:    {pytorch_time:>7.3f}ms  ({pytorch_tflops:.1f} TFLOPS)")
        print(f"  Speedup:         {speedup:.2f}x")
        print(f"  Max Error:       {diff:.6f}")
    
    print("="*70)


def benchmark_fused_gate():
    """Benchmark Fused Gate: CUDA kernel vs PyTorch."""
    
    from light_moe._C import fused_gate_topk
    
    print("\n" + "="*70)
    print("FUSED GATE+TOPK BENCHMARK: Light-MoE CUDA vs PyTorch")
    print("="*70)
    
    device = torch.device("cuda")
    
    # Configuration
    num_experts = 8
    hidden_size = 4096
    top_k = 2
    
    for num_tokens in [1024, 2048, 4096]:
        print(f"\n--- Tokens: {num_tokens} ---")
        
        hidden = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.float32)
        gate_weight = torch.randn(num_experts, hidden_size, device=device, dtype=torch.float32)
        
        # Warmup CUDA
        for _ in range(10):
            _, _ = fused_gate_topk(hidden, gate_weight, top_k)
        torch.cuda.synchronize()
        
        # Benchmark CUDA
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            indices, weights = fused_gate_topk(hidden, gate_weight, top_k)
        torch.cuda.synchronize()
        cuda_time = (time.perf_counter() - start) / 100 * 1000
        
        # PyTorch reference
        for _ in range(10):
            logits = torch.mm(hidden, gate_weight.t())
            probs = torch.softmax(logits, dim=-1)
            ref_weights, ref_indices = torch.topk(probs, top_k, dim=-1)
            ref_weights = ref_weights / ref_weights.sum(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(100):
            logits = torch.mm(hidden, gate_weight.t())
            probs = torch.softmax(logits, dim=-1)
            ref_weights, ref_indices = torch.topk(probs, top_k, dim=-1)
            ref_weights = ref_weights / ref_weights.sum(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        pytorch_time = (time.perf_counter() - start) / 100 * 1000
        
        speedup = pytorch_time / cuda_time
        
        print(f"  Light-MoE CUDA:  {cuda_time:>7.3f}ms")
        print(f"  PyTorch:         {pytorch_time:>7.3f}ms")
        print(f"  Speedup:         {speedup:.2f}x")
    
    print("="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("LIGHT-MOE CUDA KERNEL BENCHMARK")
    print("Hardware: " + torch.cuda.get_device_name(0))
    print("="*70)
    
    benchmark_grouped_gemm()
    benchmark_fused_gate()
    
    print("\nDone!")
