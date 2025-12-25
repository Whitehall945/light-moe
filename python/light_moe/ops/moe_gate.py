"""
MoE Gate (Router) Implementation

The gate/router is responsible for computing routing probabilities and
selecting top-K experts for each token. This is a core component of
Mixture-of-Experts models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RoutingOutput:
    """Output from the routing/gate computation."""
    
    # Selected expert indices for each token: [batch_size, seq_len, top_k]
    expert_indices: torch.Tensor
    
    # Routing weights for selected experts: [batch_size, seq_len, top_k]
    routing_weights: torch.Tensor
    
    # Number of tokens assigned to each expert: [num_experts]
    expert_counts: torch.Tensor
    
    # Full routing probabilities (optional, for aux loss): [batch_size, seq_len, num_experts]
    router_probs: Optional[torch.Tensor] = None
    
    # Auxiliary load balancing loss
    aux_loss: Optional[torch.Tensor] = None


def top_k_routing(
    router_logits: torch.Tensor,
    top_k: int,
    normalize: bool = True,
    jitter_noise: float = 0.0,
    training: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute top-K routing from router logits.
    
    Args:
        router_logits: Router output logits [batch_size, seq_len, num_experts]
        top_k: Number of experts to select per token.
        normalize: Whether to normalize routing weights to sum to 1.
        jitter_noise: Amount of noise to add during training for exploration.
        training: Whether in training mode.
        
    Returns:
        Tuple of:
        - expert_indices: Selected expert indices [batch_size, seq_len, top_k]
        - routing_weights: Weights for selected experts [batch_size, seq_len, top_k]
        - router_probs: Full softmax probabilities [batch_size, seq_len, num_experts]
    """
    # Add jitter noise during training for exploration
    if training and jitter_noise > 0:
        noise = torch.rand_like(router_logits) * jitter_noise
        router_logits = router_logits + noise
    
    # Compute softmax probabilities
    router_probs = F.softmax(router_logits, dim=-1)
    
    # Select top-K experts
    routing_weights, expert_indices = torch.topk(router_probs, k=top_k, dim=-1)
    
    # Normalize weights to sum to 1
    if normalize:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    
    return expert_indices, routing_weights, router_probs


def load_balancing_loss(
    router_probs: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """
    Compute auxiliary load balancing loss.
    
    Encourages uniform distribution of tokens across experts.
    Used during training to prevent expert collapse.
    
    Formula: aux_loss = num_experts * sum(f_i * P_i)
    where f_i is fraction of tokens to expert i,
    and P_i is mean routing probability for expert i.
    
    Args:
        router_probs: Full routing probabilities [batch_size, seq_len, num_experts]
        expert_indices: Selected expert indices [batch_size, seq_len, top_k]
        num_experts: Total number of experts.
        
    Returns:
        Scalar loss tensor.
    """
    # Flatten batch and sequence dimensions
    batch_size, seq_len, _ = router_probs.shape
    num_tokens = batch_size * seq_len
    
    router_probs_flat = router_probs.view(-1, num_experts)  # [num_tokens, num_experts]
    expert_indices_flat = expert_indices.view(-1, expert_indices.size(-1))  # [num_tokens, top_k]
    
    # Compute fraction of tokens routed to each expert (f_i)
    expert_mask = F.one_hot(expert_indices_flat, num_experts)  # [num_tokens, top_k, num_experts]
    expert_mask = expert_mask.sum(dim=1)  # [num_tokens, num_experts]
    tokens_per_expert = expert_mask.float().sum(dim=0)  # [num_experts]
    fraction_tokens = tokens_per_expert / num_tokens  # [num_experts]
    
    # Compute mean routing probability per expert (P_i)
    mean_routing_prob = router_probs_flat.mean(dim=0)  # [num_experts]
    
    # Auxiliary loss
    aux_loss = num_experts * (fraction_tokens * mean_routing_prob).sum()
    
    return aux_loss


class MoEGate(nn.Module):
    """
    MoE Gate (Router) module.
    
    Computes routing probabilities and selects top-K experts for each token.
    
    Supports different routing strategies:
    - "softmax": Standard softmax routing
    - "sigmoid": Expert-choice routing (experimental)
    
    Args:
        hidden_size: Input hidden dimension.
        num_experts: Number of experts.
        top_k: Number of experts to select per token.
        routing_type: Type of routing ("softmax" or "sigmoid").
        jitter_noise: Noise to add during training.
        aux_loss_coef: Coefficient for auxiliary load balancing loss.
    
    Example:
        >>> gate = MoEGate(4096, num_experts=8, top_k=2)
        >>> hidden_states = torch.randn(2, 128, 4096)
        >>> output = gate(hidden_states)
        >>> print(output.expert_indices.shape)  # (2, 128, 2)
        >>> print(output.routing_weights.shape)  # (2, 128, 2)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        routing_type: str = "softmax",
        jitter_noise: float = 0.0,
        aux_loss_coef: float = 0.01,
        bias: bool = False,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.routing_type = routing_type
        self.jitter_noise = jitter_noise
        self.aux_loss_coef = aux_loss_coef
        
        # Router projection: hidden_size -> num_experts
        self.gate = nn.Linear(hidden_size, num_experts, bias=bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        return_aux_loss: bool = True,
    ) -> RoutingOutput:
        """
        Compute routing for input hidden states.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            return_aux_loss: Whether to compute auxiliary load balancing loss.
            
        Returns:
            RoutingOutput containing indices, weights, and optional aux loss.
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute router logits
        router_logits = self.gate(hidden_states)  # [batch, seq, num_experts]
        
        # Get top-K routing
        expert_indices, routing_weights, router_probs = top_k_routing(
            router_logits,
            top_k=self.top_k,
            normalize=True,
            jitter_noise=self.jitter_noise if self.training else 0.0,
            training=self.training,
        )
        
        # Count tokens per expert
        expert_counts = torch.zeros(
            self.num_experts, 
            dtype=torch.int64, 
            device=hidden_states.device
        )
        for k in range(self.top_k):
            counts = torch.bincount(
                expert_indices[:, :, k].flatten(),
                minlength=self.num_experts,
            )
            expert_counts += counts
        
        # Compute auxiliary loss if requested (training only)
        aux_loss = None
        if return_aux_loss and self.training:
            aux_loss = load_balancing_loss(
                router_probs, expert_indices, self.num_experts
            ) * self.aux_loss_coef
        
        return RoutingOutput(
            expert_indices=expert_indices,
            routing_weights=routing_weights,
            expert_counts=expert_counts,
            router_probs=router_probs,
            aux_loss=aux_loss,
        )
    
    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_experts={self.num_experts}, "
            f"top_k={self.top_k}"
        )
