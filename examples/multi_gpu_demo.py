"""
Multi-GPU inference example with Expert Parallelism.

Run with: torchrun --nproc_per_node=8 examples/multi_gpu_demo.py --model_path /path/to/model
"""

import argparse
import os

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Light-MoE Demo")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model weights",
    )
    parser.add_argument(
        "--expert_parallel_size",
        type=int,
        default=8,
        help="Number of GPUs for expert parallelism",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Write a short story about a robot learning to paint:",
        help="Input prompt",
    )
    args = parser.parse_args()

    # Initialize distributed
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    # Import here to ensure CUDA device is set first
    from light_moe import LightMoEEngine

    if rank == 0:
        print(f"Initializing with {world_size} GPUs...")
        print(f"Expert Parallel Size: {args.expert_parallel_size}")

    engine = LightMoEEngine(
        model_path=args.model_path,
        tensor_parallel_size=1,
        expert_parallel_size=args.expert_parallel_size,
    )

    # Only rank 0 prints output
    if rank == 0:
        print(f"\nPrompt: {args.prompt}\n")
        print("Generating...")

    output = engine.generate(
        prompt=args.prompt,
        max_tokens=512,
        temperature=0.8,
    )

    if rank == 0:
        print(f"\nOutput:\n{output}")

    engine.shutdown()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
