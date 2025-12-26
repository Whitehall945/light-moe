#!/usr/bin/env python
"""
Multi-GPU Expert Parallelism Test

Launch with:
    torchrun --nproc_per_node=8 benchmarks/test_expert_parallel.py

Or with mpirun:
    mpirun -np 8 python benchmarks/test_expert_parallel.py
"""

import os
import time
import torch
import torch.distributed as dist


def setup_distributed():
    """Initialize distributed environment."""
    # Get rank from environment (set by torchrun/mpirun)
    rank = int(os.environ.get("RANK", os.environ.get("OMPI_COMM_WORLD_RANK", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", 1)))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0)))
    
    # Set device
    torch.cuda.set_device(local_rank)
    
    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
    
    return rank, world_size, local_rank


def test_all_to_all_basic():
    """Test basic All-to-All communication."""
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print(f"\n{'='*60}")
        print("Test 1: Basic All-to-All Communication")
        print(f"{'='*60}")
    
    dist.barrier()
    
    # Create test tensor: each rank sends different data
    tokens_per_rank = 128
    hidden_size = 256
    
    # Send tensor: [world_size, tokens_per_rank, hidden_size]
    send_tensor = torch.full(
        (world_size, tokens_per_rank, hidden_size),
        fill_value=float(rank),  # Fill with rank ID
        device=device,
    )
    
    recv_tensor = torch.zeros_like(send_tensor)
    
    # All-to-All
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    dist.all_to_all_single(
        recv_tensor.view(-1),
        send_tensor.view(-1),
    )
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    # Verify: recv_tensor[i] should be filled with value i (from rank i)
    correct = True
    for i in range(world_size):
        expected = float(i)
        actual = recv_tensor[i, 0, 0].item()
        if abs(actual - expected) > 1e-5:
            correct = False
            print(f"Rank {rank}: Error - expected {expected}, got {actual}")
    
    if rank == 0:
        data_size_mb = send_tensor.numel() * 4 / 1e6  # float32
        bandwidth = data_size_mb / elapsed / 1000  # GB/s
        print(f"  Data size: {data_size_mb:.2f} MB")
        print(f"  Time: {elapsed*1000:.3f} ms")
        print(f"  Bandwidth: {bandwidth:.2f} GB/s")
        print(f"  Correctness: {'PASS' if correct else 'FAIL'}")
    
    dist.barrier()
    return correct


def test_all_to_all_performance():
    """Benchmark All-to-All performance with different sizes."""
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print(f"\n{'='*60}")
        print("Test 2: All-to-All Performance")
        print(f"{'='*60}")
        print(f"{'Tokens/Rank':<15} {'Hidden':<10} {'Data (MB)':<12} {'Time (ms)':<12} {'BW (GB/s)':<10}")
        print("-" * 60)
    
    dist.barrier()
    
    # Test configurations
    configs = [
        (128, 256),
        (256, 512),
        (512, 1024),
        (1024, 2048),
        (2048, 4096),
    ]
    
    for tokens_per_rank, hidden_size in configs:
        send_tensor = torch.randn(
            world_size, tokens_per_rank, hidden_size,
            device=device,
        )
        recv_tensor = torch.zeros_like(send_tensor)
        
        # Warmup
        for _ in range(5):
            dist.all_to_all_single(recv_tensor.view(-1), send_tensor.view(-1))
        torch.cuda.synchronize()
        
        # Benchmark
        iterations = 20
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for _ in range(iterations):
            dist.all_to_all_single(recv_tensor.view(-1), send_tensor.view(-1))
        
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / iterations
        
        data_size_mb = send_tensor.numel() * 4 / 1e6
        bandwidth = data_size_mb / elapsed / 1000
        
        if rank == 0:
            print(f"{tokens_per_rank:<15} {hidden_size:<10} {data_size_mb:<12.2f} {elapsed*1000:<12.3f} {bandwidth:<10.2f}")
    
    if rank == 0:
        print("-" * 60)
    
    dist.barrier()


def test_expert_parallel_layer():
    """Test the ExpertParallelMoELayer."""
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print(f"\n{'='*60}")
        print("Test 3: Expert Parallel MoE Layer")
        print(f"{'='*60}")
    
    dist.barrier()
    
    # Import after distributed init
    import sys
    sys.path.insert(0, "/home/henglong/light-moe/python")
    from light_moe.distributed.expert_parallel import (
        ExpertParallelConfig,
        ExpertParallelGroup,
        ExpertParallelMoELayer,
    )
    
    # Configuration
    config = ExpertParallelConfig(
        num_experts=8,
        top_k=2,
        hidden_size=256,  # Smaller for testing
        intermediate_size=512,
        ep_size=min(world_size, 8),
    )
    
    # Create EP group (reuse existing dist group)
    ep_group = ExpertParallelGroup(ep_size=config.ep_size)
    ep_group.group = dist.group.WORLD
    ep_group.rank = rank
    ep_group.world_size = world_size
    ep_group.initialized = True
    
    # Create layer
    layer = ExpertParallelMoELayer(config, ep_group, device=device)
    layer.eval()
    
    # Test forward
    batch_size = 64
    hidden_states = torch.randn(batch_size, config.hidden_size, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            output = layer(hidden_states)
    torch.cuda.synchronize()
    
    # Benchmark
    iterations = 10
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(iterations):
            output = layer(hidden_states)
    
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iterations
    
    if rank == 0:
        print(f"  Batch size: {batch_size}")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Experts: {config.num_experts} (distributed across {config.ep_size} GPUs)")
        print(f"  Output shape: {output.shape}")
        print(f"  Time per forward: {elapsed*1000:.3f} ms")
        print(f"  Throughput: {batch_size / elapsed:.0f} tokens/s")
    
    dist.barrier()


def main():
    """Run all tests."""
    rank, world_size, local_rank = setup_distributed()
    
    if rank == 0:
        print(f"\n{'#'*60}")
        print("# Expert Parallelism Multi-GPU Test")
        print(f"# World size: {world_size} GPUs")
        print(f"{'#'*60}")
    
    dist.barrier()
    
    # Run tests
    test_all_to_all_basic()
    test_all_to_all_performance()
    test_expert_parallel_layer()
    
    if rank == 0:
        print(f"\n{'#'*60}")
        print("# All tests completed!")
        print(f"{'#'*60}\n")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
