"""
Communication group management for distributed inference.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist


class CommGroup:
    """
    Wrapper for PyTorch distributed communication group.
    
    Manages NCCL process groups for multi-GPU communication.
    """
    
    def __init__(
        self,
        ranks: list[int],
        backend: str = "nccl",
    ):
        """
        Initialize a communication group.
        
        Args:
            ranks: List of global ranks in this group.
            backend: Communication backend (nccl, gloo).
        """
        self.ranks = ranks
        self.backend = backend
        self._group: Optional[dist.ProcessGroup] = None
        
        if dist.is_initialized():
            self._group = dist.new_group(ranks=ranks, backend=backend)
    
    @property
    def group(self) -> Optional[dist.ProcessGroup]:
        """Get the underlying process group."""
        return self._group
    
    @property
    def size(self) -> int:
        """Get group size."""
        return len(self.ranks)
    
    @property
    def rank(self) -> int:
        """Get rank within this group."""
        if not dist.is_initialized():
            return 0
        global_rank = dist.get_rank()
        if global_rank in self.ranks:
            return self.ranks.index(global_rank)
        return -1
    
    def barrier(self) -> None:
        """Synchronize all processes in this group."""
        if self._group is not None:
            dist.barrier(group=self._group)
    
    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
    ) -> torch.Tensor:
        """
        All-reduce tensor across group.
        
        Args:
            tensor: Input tensor (modified in-place).
            op: Reduction operation.
            
        Returns:
            Reduced tensor.
        """
        if self._group is not None:
            dist.all_reduce(tensor, op=op, group=self._group)
        return tensor
    
    def all_gather(
        self,
        tensor_list: list[torch.Tensor],
        tensor: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        All-gather tensors from all ranks.
        
        Args:
            tensor_list: Output list of tensors.
            tensor: Local tensor to gather.
            
        Returns:
            List of gathered tensors.
        """
        if self._group is not None:
            dist.all_gather(tensor_list, tensor, group=self._group)
        return tensor_list


def init_distributed(
    backend: str = "nccl",
    init_method: Optional[str] = None,
) -> tuple[int, int]:
    """
    Initialize distributed training/inference.
    
    Args:
        backend: Communication backend.
        init_method: Initialization method (env, tcp, file).
        
    Returns:
        Tuple of (world_size, rank).
    """
    if dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    
    # Use environment variables by default
    if init_method is None:
        init_method = "env://"
    
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
    
    return world_size, rank
