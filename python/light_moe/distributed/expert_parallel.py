"""
Expert Parallelism utilities for MoE models.

Expert Parallelism (EP) distributes experts across GPUs, with each GPU
hosting a subset of experts. Tokens are routed via All-to-All communication.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist

from light_moe.distributed.comm_group import CommGroup


class ExpertParallelGroup:
    """
    Manages expert parallelism across GPUs.
    
    In EP, experts are sharded across GPUs. For example, with 8 experts
    and 4 GPUs, each GPU hosts 2 experts. Tokens are redistributed via
    All-to-All so each GPU processes tokens for its local experts.
    """
    
    def __init__(
        self,
        num_experts: int,
        ep_size: int,
        ranks: Optional[list[int]] = None,
    ):
        """
        Initialize expert parallel group.
        
        Args:
            num_experts: Total number of experts.
            ep_size: Expert parallelism size (number of GPUs).
            ranks: Optional list of ranks in this EP group.
        """
        self.num_experts = num_experts
        self.ep_size = ep_size
        
        if num_experts % ep_size != 0:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size})"
            )
        
        self.experts_per_rank = num_experts // ep_size
        
        # Create communication group
        if ranks is None:
            ranks = list(range(ep_size))
        self.comm_group = CommGroup(ranks=ranks, backend="nccl")
        
        self._rank = self.comm_group.rank
    
    @property
    def rank(self) -> int:
        """Get rank within EP group."""
        return self._rank
    
    def get_local_expert_indices(self) -> list[int]:
        """Get indices of experts hosted on this rank."""
        start = self._rank * self.experts_per_rank
        end = start + self.experts_per_rank
        return list(range(start, end))
    
    def expert_to_rank(self, expert_idx: int) -> int:
        """Get the rank that hosts a given expert."""
        return expert_idx // self.experts_per_rank
    
    def all_to_all(
        self,
        input_tensor: torch.Tensor,
        input_splits: list[int],
        output_splits: list[int],
    ) -> torch.Tensor:
        """
        All-to-All communication for token redistribution.
        
        Args:
            input_tensor: Local tokens to send [total_send, hidden].
            input_splits: Number of tokens to send to each rank.
            output_splits: Number of tokens to receive from each rank.
            
        Returns:
            Received tokens [total_recv, hidden].
        """
        if self.ep_size == 1:
            return input_tensor
        
        # Prepare output tensor
        total_recv = sum(output_splits)
        output_tensor = torch.empty(
            (total_recv, input_tensor.size(1)),
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        
        # Perform All-to-All
        dist.all_to_all_single(
            output_tensor,
            input_tensor,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=self.comm_group.group,
        )
        
        return output_tensor
    
    def permute_tokens(
        self,
        tokens: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
        """
        Permute tokens for All-to-All based on expert assignment.
        
        Args:
            tokens: Input tokens [num_tokens, hidden].
            expert_indices: Expert index for each token [num_tokens].
            
        Returns:
            Tuple of:
            - Permuted tokens
            - Inverse permutation indices
            - Send counts per rank
            - Receive counts per rank
        """
        # Count tokens per expert
        expert_counts = torch.bincount(
            expert_indices, minlength=self.num_experts
        )
        
        # Compute send counts (tokens going to each rank)
        send_counts = []
        for r in range(self.ep_size):
            start_expert = r * self.experts_per_rank
            end_expert = start_expert + self.experts_per_rank
            count = expert_counts[start_expert:end_expert].sum().item()
            send_counts.append(int(count))
        
        # All-to-All to get receive counts
        send_counts_tensor = torch.tensor(
            send_counts, dtype=torch.int64, device=tokens.device
        )
        recv_counts_tensor = torch.empty_like(send_counts_tensor)
        dist.all_to_all_single(
            recv_counts_tensor,
            send_counts_tensor,
            group=self.comm_group.group,
        )
        recv_counts = recv_counts_tensor.tolist()
        
        # Sort tokens by destination rank
        dest_ranks = expert_indices // self.experts_per_rank
        sort_indices = torch.argsort(dest_ranks, stable=True)
        permuted_tokens = tokens[sort_indices]
        
        return permuted_tokens, sort_indices, send_counts, recv_counts
