"""
Expert Parallel MoE Layer

Implements Expert Parallelism for distributed MoE inference:
- Each GPU holds a subset of experts
- All-to-All communication for token dispatch
- Communication-computation overlap via CUDA streams
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class ExpertParallelConfig:
    """Configuration for Expert Parallelism."""
    num_experts: int = 8
    top_k: int = 2
    hidden_size: int = 4096
    intermediate_size: int = 14336
    ep_size: int = 8  # Number of GPUs for expert parallelism
    
    @property
    def experts_per_rank(self) -> int:
        """Number of experts per GPU."""
        assert self.num_experts % self.ep_size == 0
        return self.num_experts // self.ep_size


class ExpertParallelGroup:
    """Manages the expert parallel process group."""
    
    def __init__(self, ep_size: int = 8):
        self.ep_size = ep_size
        self.group: Optional[dist.ProcessGroup] = None
        self.rank: int = 0
        self.world_size: int = 1
        self.initialized = False
    
    def init_process_group(
        self,
        backend: str = "nccl",
        init_method: str = "env://",
    ):
        """Initialize the distributed process group."""
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
            )
        
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
        # Create expert parallel group
        assert self.world_size >= self.ep_size, \
            f"World size {self.world_size} < EP size {self.ep_size}"
        
        # All ranks participate in EP (for simplicity)
        ranks = list(range(self.ep_size))
        self.group = dist.new_group(ranks)
        
        self.initialized = True
        
        if self.rank == 0:
            print(f"Expert Parallel Group initialized: {self.ep_size} GPUs")
    
    def all_to_all(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """All-to-All communication for token dispatch."""
        dist.all_to_all_single(
            output_tensor,
            input_tensor,
            group=self.group,
        )
        return output_tensor
    
    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-reduce for gradient synchronization."""
        dist.all_reduce(tensor, group=self.group)
        return tensor


class ExpertParallelMoELayer(nn.Module):
    """
    Expert Parallel MoE Layer.
    
    Each GPU holds num_experts/ep_size experts.
    Tokens are dispatched via All-to-All communication.
    """
    
    def __init__(
        self,
        config: ExpertParallelConfig,
        ep_group: ExpertParallelGroup,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        
        self.config = config
        self.ep_group = ep_group
        self.device = device or torch.device(f"cuda:{ep_group.rank}")
        
        # Gate (shared across all ranks)
        self.gate = nn.Linear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            device=self.device,
        )
        
        # Local experts (only this rank's share)
        self.experts_per_rank = config.experts_per_rank
        self.local_expert_start = ep_group.rank * self.experts_per_rank
        self.local_expert_end = self.local_expert_start + self.experts_per_rank
        
        # Expert FFN weights
        self.w1 = nn.Parameter(torch.randn(
            self.experts_per_rank,
            config.intermediate_size,
            config.hidden_size,
            device=self.device,
        ) * 0.02)
        
        self.w2 = nn.Parameter(torch.randn(
            self.experts_per_rank,
            config.hidden_size,
            config.intermediate_size,
            device=self.device,
        ) * 0.02)
        
        self.w3 = nn.Parameter(torch.randn(
            self.experts_per_rank,
            config.intermediate_size,
            config.hidden_size,
            device=self.device,
        ) * 0.02)
        
        # CUDA streams for overlap
        self.compute_stream = torch.cuda.Stream(device=self.device)
        self.comm_stream = torch.cuda.Stream(device=self.device)
    
    def _route_tokens(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.
        
        Returns:
            expert_indices: [num_tokens, top_k]
            expert_weights: [num_tokens, top_k]
            tokens_per_expert: [num_experts]
        """
        # Gate logits
        logits = self.gate(hidden_states)  # [num_tokens, num_experts]
        
        # Softmax + TopK
        probs = torch.softmax(logits, dim=-1)
        weights, indices = torch.topk(probs, self.config.top_k, dim=-1)
        
        # Normalize weights
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        # Count tokens per expert
        tokens_per_expert = torch.zeros(
            self.config.num_experts,
            dtype=torch.int64,
            device=self.device,
        )
        for k in range(self.config.top_k):
            tokens_per_expert.scatter_add_(
                0,
                indices[:, k],
                torch.ones_like(indices[:, k], dtype=torch.int64),
            )
        
        return indices, weights, tokens_per_expert
    
    def _dispatch_tokens(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dispatch tokens to their assigned experts across GPUs.
        
        Uses All-to-All communication.
        """
        num_tokens = hidden_states.size(0)
        hidden_size = hidden_states.size(1)
        top_k = self.config.top_k
        ep_size = self.ep_group.ep_size
        
        # Expand hidden states for top_k
        # [num_tokens, hidden_size] -> [num_tokens * top_k, hidden_size]
        expanded = hidden_states.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, hidden_size)
        flat_indices = expert_indices.flatten()
        
        # Sort by expert for efficient communication
        sorted_idx = torch.argsort(flat_indices)
        sorted_tokens = expanded[sorted_idx]
        sorted_experts = flat_indices[sorted_idx]
        
        # Compute send/recv counts for All-to-All
        # tokens_per_expert: [num_experts]
        # Each rank sends tokens to experts on other ranks
        
        # For simplicity, we reshape to [ep_size, tokens_per_rank, hidden_size]
        # This requires balanced token distribution (pad if needed)
        
        total_tokens = num_tokens * top_k
        tokens_per_rank = total_tokens // ep_size
        
        # Pad to make divisible
        if total_tokens % ep_size != 0:
            pad_tokens = ep_size - (total_tokens % ep_size)
            sorted_tokens = torch.cat([
                sorted_tokens,
                torch.zeros(pad_tokens, hidden_size, device=self.device),
            ])
            tokens_per_rank = (total_tokens + pad_tokens) // ep_size
        
        # Reshape for All-to-All
        send_tensor = sorted_tokens.view(ep_size, tokens_per_rank, hidden_size)
        recv_tensor = torch.empty_like(send_tensor)
        
        # All-to-All
        with torch.cuda.stream(self.comm_stream):
            dist.all_to_all_single(
                recv_tensor.view(-1),
                send_tensor.view(-1),
                group=self.ep_group.group,
            )
        
        # Wait for communication
        torch.cuda.current_stream().wait_stream(self.comm_stream)
        
        # Flatten received tokens
        received_tokens = recv_tensor.view(-1, hidden_size)
        
        return received_tokens, sorted_idx
    
    def _compute_experts(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute local expert FFN.
        
        tokens: [local_tokens, hidden_size]
        """
        # For simplicity, apply all local experts to all tokens
        # (In production, would route to specific experts)
        
        # SwiGLU: output = w2(silu(w1(x)) * w3(x))
        # tokens: [T, H], w1/w3: [E, I, H], w2: [E, H, I]
        
        # Use first local expert for demo
        x1 = torch.mm(tokens, self.w1[0].t())  # [T, I]
        x3 = torch.mm(tokens, self.w3[0].t())  # [T, I]
        
        hidden = torch.nn.functional.silu(x1) * x3
        output = torch.mm(hidden, self.w2[0].t())  # [T, H]
        
        return output
    
    def _combine_tokens(
        self,
        computed_tokens: torch.Tensor,
        sorted_idx: torch.Tensor,
        expert_weights: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """
        Combine expert outputs back to original token order.
        
        Uses All-to-All to send results back.
        """
        hidden_size = computed_tokens.size(-1)
        top_k = self.config.top_k
        ep_size = self.ep_group.ep_size
        
        # All-to-All back
        total_tokens = num_tokens * top_k
        tokens_per_rank = computed_tokens.size(0) // ep_size
        
        send_tensor = computed_tokens.view(ep_size, tokens_per_rank, hidden_size)
        recv_tensor = torch.empty_like(send_tensor)
        
        with torch.cuda.stream(self.comm_stream):
            dist.all_to_all_single(
                recv_tensor.view(-1),
                send_tensor.view(-1),
                group=self.ep_group.group,
            )
        
        torch.cuda.current_stream().wait_stream(self.comm_stream)
        
        # Unsort
        received = recv_tensor.view(-1, hidden_size)[:total_tokens]
        unsorted = torch.empty_like(received)
        unsorted[sorted_idx] = received
        
        # Reshape to [num_tokens, top_k, hidden_size]
        unsorted = unsorted.view(num_tokens, top_k, hidden_size)
        
        # Weighted sum
        weights = expert_weights.unsqueeze(-1)  # [num_tokens, top_k, 1]
        output = (unsorted * weights).sum(dim=1)  # [num_tokens, hidden_size]
        
        return output
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Expert Parallelism.
        
        Args:
            hidden_states: [num_tokens, hidden_size]
            
        Returns:
            output: [num_tokens, hidden_size]
        """
        num_tokens = hidden_states.size(0)
        
        # 1. Route tokens
        expert_indices, expert_weights, tokens_per_expert = self._route_tokens(
            hidden_states
        )
        
        # 2. Dispatch tokens (All-to-All)
        dispatched_tokens, sorted_idx = self._dispatch_tokens(
            hidden_states, expert_indices, tokens_per_expert
        )
        
        # 3. Compute local experts
        with torch.cuda.stream(self.compute_stream):
            computed_tokens = self._compute_experts(dispatched_tokens)
        
        torch.cuda.current_stream().wait_stream(self.compute_stream)
        
        # 4. Combine results (All-to-All back)
        output = self._combine_tokens(
            computed_tokens, sorted_idx, expert_weights, num_tokens
        )
        
        return output


def create_expert_parallel_layer(
    num_experts: int = 8,
    hidden_size: int = 4096,
    intermediate_size: int = 14336,
    top_k: int = 2,
    ep_size: int = 8,
) -> Tuple[ExpertParallelMoELayer, ExpertParallelGroup]:
    """
    Helper to create an Expert Parallel MoE layer.
    
    Must be called after torch.distributed is initialized.
    """
    config = ExpertParallelConfig(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        ep_size=ep_size,
    )
    
    ep_group = ExpertParallelGroup(ep_size=ep_size)
    ep_group.init_process_group()
    
    layer = ExpertParallelMoELayer(config, ep_group)
    
    return layer, ep_group
