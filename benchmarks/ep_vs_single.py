#!/usr/bin/env python
"""
Expert Parallelism Performance Comparison

Compares:
1. Single GPU MoE (all experts on one GPU)
2. Expert Parallel MoE (experts distributed across GPUs)

Launch with:
    torchrun --nproc_per_node=8 benchmarks/ep_vs_single.py
"""

import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional


class SingleGPUMoELayer(nn.Module):
    """MoE layer with all experts on one GPU."""
    
    def __init__(
        self,
        num_experts: int = 8,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        top_k: int = 2,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        self.device = device or torch.device("cuda")
        
        self.gate = nn.Linear(hidden_size, num_experts, bias=False, device=self.device)
        
        # All experts on this GPU
        self.w1 = nn.Parameter(torch.randn(num_experts, intermediate_size, hidden_size, device=self.device) * 0.02)
        self.w2 = nn.Parameter(torch.randn(num_experts, hidden_size, intermediate_size, device=self.device) * 0.02)
        self.w3 = nn.Parameter(torch.randn(num_experts, intermediate_size, hidden_size, device=self.device) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Gate
        logits = self.gate(x)
        probs = torch.softmax(logits, dim=-1)
        weights, indices = torch.topk(probs, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        # Expert computation
        output = torch.zeros_like(x)
        
        for k in range(self.top_k):
            for e in range(self.num_experts):
                mask = (indices[:, k] == e)
                if mask.sum() == 0:
                    continue
                
                tokens = x[mask]
                
                # SwiGLU
                x1 = torch.mm(tokens, self.w1[e].t())
                x3 = torch.mm(tokens, self.w3[e].t())
                hidden = torch.nn.functional.silu(x1) * x3
                expert_out = torch.mm(hidden, self.w2[e].t())
                
                output[mask] += weights[mask, k:k+1] * expert_out
        
        return output


def setup_distributed():
    """Initialize distributed environment."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    torch.cuda.set_device(local_rank)
    
    if not dist.is_initialized() and world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
    
    return rank, world_size, local_rank


def benchmark_single_gpu(
    hidden_size: int = 4096,
    intermediate_size: int = 14336,
    num_experts: int = 8,
    top_k: int = 2,
    batch_sizes: list = [64, 128, 256, 512, 1024],
    warmup: int = 5,
    iterations: int = 20,
):
    """Benchmark single GPU MoE."""
    device = torch.device("cuda:0")
    
    layer = SingleGPUMoELayer(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        top_k=top_k,
        device=device,
    ).eval()
    
    results = {}
    
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, hidden_size, device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = layer(x)
        torch.cuda.synchronize()
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(iterations):
                _ = layer(x)
        
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / iterations
        
        results[batch_size] = {
            "time_ms": elapsed * 1000,
            "throughput": batch_size / elapsed,
        }
    
    return results


def benchmark_expert_parallel(
    hidden_size: int = 4096,
    intermediate_size: int = 14336,
    num_experts: int = 8,
    top_k: int = 2,
    batch_sizes: list = [64, 128, 256, 512, 1024],
    warmup: int = 5,
    iterations: int = 20,
):
    """Benchmark Expert Parallel MoE."""
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    import sys
    sys.path.insert(0, "/home/henglong/light-moe/python")
    from light_moe.distributed.expert_parallel import (
        ExpertParallelConfig,
        ExpertParallelGroup,
        ExpertParallelMoELayer,
    )
    
    config = ExpertParallelConfig(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        ep_size=min(world_size, num_experts),
    )
    
    ep_group = ExpertParallelGroup(ep_size=config.ep_size)
    ep_group.group = dist.group.WORLD
    ep_group.rank = rank
    ep_group.world_size = world_size
    ep_group.initialized = True
    
    layer = ExpertParallelMoELayer(config, ep_group, device=device).eval()
    
    results = {}
    
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, hidden_size, device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = layer(x)
        torch.cuda.synchronize()
        dist.barrier()
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(iterations):
                _ = layer(x)
        
        torch.cuda.synchronize()
        dist.barrier()
        elapsed = (time.perf_counter() - start) / iterations
        
        results[batch_size] = {
            "time_ms": elapsed * 1000,
            "throughput": batch_size / elapsed,
        }
    
    return results, rank


def main():
    """Run comparison benchmark."""
    rank, world_size, local_rank = setup_distributed()
    
    # Use smaller sizes for faster testing
    hidden_size = 1024
    intermediate_size = 2048
    num_experts = 8
    batch_sizes = [64, 128, 256, 512, 1024]
    
    if rank == 0:
        print(f"\n{'#'*70}")
        print("# Single GPU vs Expert Parallel MoE Benchmark")
        print(f"# Hidden: {hidden_size}, Intermediate: {intermediate_size}")
        print(f"# Experts: {num_experts}, GPUs: {world_size}")
        print(f"{'#'*70}\n")
        
        # Single GPU benchmark (only rank 0)
        print("Benchmarking Single GPU MoE...")
        single_results = benchmark_single_gpu(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            batch_sizes=batch_sizes,
        )
    else:
        single_results = None
    
    if world_size > 1:
        dist.barrier()
    
    # Expert Parallel benchmark (all ranks)
    if rank == 0:
        print("Benchmarking Expert Parallel MoE...")
    
    ep_results, _ = benchmark_expert_parallel(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        batch_sizes=batch_sizes,
    )
    
    if world_size > 1:
        dist.barrier()
    
    # Print results (only rank 0)
    if rank == 0:
        print(f"\n{'='*70}")
        print("Results Comparison")
        print(f"{'='*70}")
        print(f"{'Batch':<10} {'Single GPU':<20} {'Expert Parallel':<20} {'Speedup':<10}")
        print(f"{'Size':<10} {'Time (ms)':<10}{'Tok/s':<10} {'Time (ms)':<10}{'Tok/s':<10} {'EP/Single':<10}")
        print("-" * 70)
        
        for bs in batch_sizes:
            s = single_results[bs]
            e = ep_results[bs]
            speedup = s["time_ms"] / e["time_ms"]
            
            print(f"{bs:<10} {s['time_ms']:<10.2f}{s['throughput']:<10.0f} "
                  f"{e['time_ms']:<10.2f}{e['throughput']:<10.0f} {speedup:<10.2f}x")
        
        print("-" * 70)
        print("\nNote: Expert Parallel incurs communication overhead but enables larger models.")
        print("      Real gains come from model parallelism (larger experts per GPU).\n")
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
