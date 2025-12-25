"""
MoE Block Implementation

Combines the MoE Gate (router) with expert FFN networks.
This is the core building block of Mixture-of-Experts models.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from light_moe.ops.moe_gate import MoEGate, RoutingOutput
from light_moe.ops.grouped_gemm import (
    GroupedLinear,
    permute_tokens,
    unpermute_tokens,
)
from light_moe.ops.activation import swiglu


class MoEExpert(nn.Module):
    """
    Single MoE Expert (FFN with SwiGLU activation).
    
    Each expert is a standard FFN: gate_proj, up_proj, down_proj with SwiGLU.
    
    Args:
        hidden_size: Input and output dimension.
        intermediate_size: FFN intermediate dimension.
        bias: Whether to use bias in linear layers.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through expert FFN.
        
        Args:
            x: Input tensor [..., hidden_size]
            
        Returns:
            Output tensor [..., hidden_size]
        """
        return self.down_proj(swiglu(self.up_proj(x), self.gate_proj(x)))


class SparseMoEBlock(nn.Module):
    """
    Sparse Mixture-of-Experts Block.
    
    Routes each token to top-K experts and combines their outputs.
    
    Architecture:
    1. Gate computes routing probabilities and selects top-K experts
    2. Tokens are permuted so same-expert tokens are contiguous
    3. Each expert processes its assigned tokens
    4. Outputs are unpermuted and combined with routing weights
    
    Args:
        hidden_size: Input/output dimension.
        intermediate_size: Expert FFN intermediate dimension.
        num_experts: Number of experts.
        top_k: Number of experts per token.
        aux_loss_coef: Coefficient for load balancing loss.
    
    Example:
        >>> moe = SparseMoEBlock(4096, 14336, num_experts=8, top_k=2)
        >>> x = torch.randn(2, 128, 4096)
        >>> output, aux_loss = moe(x)
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        top_k: int = 2,
        aux_loss_coef: float = 0.01,
        bias: bool = False,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Gate/Router
        self.gate = MoEGate(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            aux_loss_coef=aux_loss_coef,
        )
        
        # Experts - using separate modules for clarity
        # In production, would use GroupedLinear for efficiency
        self.experts = nn.ModuleList([
            MoEExpert(hidden_size, intermediate_size, bias=bias)
            for _ in range(num_experts)
        ])
    
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through MoE block.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            
        Returns:
            Tuple of:
            - Output tensor [batch, seq_len, hidden_size]
            - Auxiliary load balancing loss (if training)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Compute routing
        routing_output = self.gate(hidden_states, return_aux_loss=self.training)
        
        # Flatten batch and sequence dimensions
        flat_hidden = hidden_states.view(-1, hidden_size)  # [batch*seq, hidden]
        
        # Process tokens through experts
        # This is the simple version - permute_tokens version is more efficient
        final_output = torch.zeros_like(flat_hidden)
        
        for k in range(self.top_k):
            # Get expert indices and weights for this k
            expert_indices = routing_output.expert_indices[:, :, k].flatten()  # [batch*seq]
            routing_weights = routing_output.routing_weights[:, :, k].flatten()  # [batch*seq]
            
            # Process each expert
            for expert_idx in range(self.num_experts):
                # Find tokens for this expert
                mask = (expert_indices == expert_idx)
                if mask.sum() == 0:
                    continue
                
                # Get tokens and process
                expert_input = flat_hidden[mask]  # [num_tokens, hidden]
                expert_output = self.experts[expert_idx](expert_input)
                
                # Weight by routing weight and add to output
                expert_weights = routing_weights[mask].unsqueeze(-1)  # [num_tokens, 1]
                final_output[mask] += expert_output * expert_weights
        
        # Reshape output
        output = final_output.view(batch_size, seq_len, hidden_size)
        
        return output, routing_output.aux_loss
    
    def extra_repr(self) -> str:
        return (
            f"num_experts={self.num_experts}, "
            f"top_k={self.top_k}, "
            f"hidden_size={self.hidden_size}"
        )


class SparseMoEBlockFast(nn.Module):
    """
    Optimized Sparse MoE Block using grouped operations.
    
    Uses token permutation and grouped GEMM for better efficiency.
    This is the production version that should be used in practice.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        top_k: int = 2,
        aux_loss_coef: float = 0.01,
        bias: bool = False,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Gate/Router
        self.gate = MoEGate(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            aux_loss_coef=aux_loss_coef,
        )
        
        # Grouped projections for all experts
        self.gate_proj = GroupedLinear(num_experts, hidden_size, intermediate_size, bias=bias)
        self.up_proj = GroupedLinear(num_experts, hidden_size, intermediate_size, bias=bias)
        self.down_proj = GroupedLinear(num_experts, intermediate_size, hidden_size, bias=bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optimized grouped operations.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            
        Returns:
            Tuple of (output, aux_loss)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Compute routing
        routing_output = self.gate(hidden_states, return_aux_loss=self.training)
        
        # Permute tokens by expert assignment
        permuted_tokens, expert_offsets, reverse_indices = permute_tokens(
            hidden_states,
            routing_output.expert_indices,
            self.num_experts,
        )
        
        # Split by expert
        expert_inputs = []
        for i in range(self.num_experts):
            start = expert_offsets[i].item()
            end = expert_offsets[i + 1].item()
            expert_inputs.append(permuted_tokens[start:end])
        
        # Apply grouped projections
        gate_outputs = self.gate_proj(expert_inputs)
        up_outputs = self.up_proj(expert_inputs)
        
        # Apply SwiGLU activation
        intermediate = [swiglu(up, gate) for up, gate in zip(up_outputs, gate_outputs)]
        
        # Apply down projection
        expert_outputs = self.down_proj(intermediate)
        
        # Concatenate expert outputs
        if any(out.numel() > 0 for out in expert_outputs):
            permuted_output = torch.cat([out for out in expert_outputs if out.numel() > 0], dim=0)
        else:
            permuted_output = torch.zeros(
                batch_size * seq_len * self.top_k, hidden_size,
                device=hidden_states.device, dtype=hidden_states.dtype
            )
        
        # Unpermute and combine with routing weights
        output = unpermute_tokens(
            permuted_output,
            reverse_indices,
            routing_output.routing_weights,
        )
        
        return output, routing_output.aux_loss
