"""
Grouped GEMM (General Matrix Multiplication) for MoE

In MoE models, different experts process different numbers of tokens.
Grouped GEMM efficiently handles these variable-sized batches by
batching multiple small GEMMs together.

This PyTorch implementation serves as a reference and baseline.
The CuTe/CUDA implementation will be much faster.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def grouped_gemm(
    inputs: List[torch.Tensor],
    weights: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    """
    Perform grouped GEMM for MoE expert computation.
    
    Each group (expert) processes a variable number of tokens.
    This function computes: output[i] = inputs[i] @ weights[i].T + bias[i]
    
    Args:
        inputs: List of input tensors, one per expert.
                Each tensor has shape [num_tokens_i, hidden_size]
        weights: Expert weights tensor [num_experts, out_features, in_features]
        bias: Optional bias tensor [num_experts, out_features]
        
    Returns:
        List of output tensors, one per expert.
        Each tensor has shape [num_tokens_i, out_features]
    
    Example:
        >>> # 8 experts, each processing different number of tokens
        >>> inputs = [torch.randn(32, 4096) for _ in range(8)]  # Variable tokens
        >>> weights = torch.randn(8, 14336, 4096)  # [experts, out, in]
        >>> outputs = grouped_gemm(inputs, weights)
        >>> print([o.shape for o in outputs])
    """
    num_experts = weights.size(0)
    outputs = []
    
    for i in range(num_experts):
        if inputs[i].numel() == 0:
            # Empty input for this expert
            out_features = weights.size(1)
            outputs.append(torch.empty(0, out_features, device=weights.device, dtype=weights.dtype))
        else:
            # Compute: output = input @ weight.T
            output = F.linear(inputs[i], weights[i], bias[i] if bias is not None else None)
            outputs.append(output)
    
    return outputs


def grouped_gemm_single_tensor(
    input: torch.Tensor,
    weights: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Grouped GEMM with single concatenated input/output tensors.
    
    More efficient than list-based version as it avoids Python overhead.
    
    Args:
        input: Concatenated input tensor [total_tokens, hidden_size]
               Tokens are ordered by expert assignment.
        weights: Expert weights [num_experts, out_features, in_features]
        expert_indices: Expert index for each token [total_tokens]
        expert_offsets: Start offset for each expert [num_experts + 1]
        bias: Optional bias [num_experts, out_features]
        
    Returns:
        Concatenated output tensor [total_tokens, out_features]
    """
    num_experts = weights.size(0)
    out_features = weights.size(1)
    total_tokens = input.size(0)
    
    output = torch.empty(
        total_tokens, out_features,
        device=input.device, dtype=input.dtype
    )
    
    for i in range(num_experts):
        start = expert_offsets[i].item()
        end = expert_offsets[i + 1].item()
        
        if start < end:
            expert_input = input[start:end]
            expert_bias = bias[i] if bias is not None else None
            output[start:end] = F.linear(expert_input, weights[i], expert_bias)
    
    return output


class GroupedLinear(nn.Module):
    """
    Grouped Linear layer for MoE experts.
    
    Contains weights for all experts and performs grouped GEMM.
    
    Args:
        num_experts: Number of experts.
        in_features: Input dimension.
        out_features: Output dimension.
        bias: Whether to use bias.
    
    Example:
        >>> layer = GroupedLinear(8, 4096, 14336)
        >>> inputs = [torch.randn(32, 4096) for _ in range(8)]
        >>> outputs = layer(inputs)
    """
    
    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        
        # Expert weights: [num_experts, out_features, in_features]
        self.weight = nn.Parameter(
            torch.empty(num_experts, out_features, in_features)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.empty(num_experts, out_features))
        else:
            self.register_parameter("bias", None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Kaiming uniform."""
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.weight[i])
            if self.bias is not None:
                nn.init.zeros_(self.bias[i])
    
    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass with list of inputs.
        
        Args:
            inputs: List of tensors, one per expert [num_tokens_i, in_features]
            
        Returns:
            List of output tensors [num_tokens_i, out_features]
        """
        return grouped_gemm(inputs, self.weight, self.bias)
    
    def forward_single(
        self,
        input: torch.Tensor,
        expert_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with single concatenated tensor.
        
        Args:
            input: Concatenated input [total_tokens, in_features]
            expert_offsets: Start offset per expert [num_experts + 1]
            
        Returns:
            Concatenated output [total_tokens, out_features]
        """
        return grouped_gemm_single_tensor(
            input, self.weight, 
            expert_indices=None,  # Not needed with offsets
            expert_offsets=expert_offsets,
            bias=self.bias,
        )
    
    def extra_repr(self) -> str:
        return (
            f"num_experts={self.num_experts}, "
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


def permute_tokens(
    hidden_states: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Permute tokens so that tokens for the same expert are contiguous.
    
    This is necessary before calling grouped_gemm_single_tensor.
    
    Args:
        hidden_states: Input tensor [batch_size, seq_len, hidden_size]
        expert_indices: Expert index for each token [batch_size, seq_len, top_k]
        num_experts: Total number of experts.
        
    Returns:
        Tuple of:
        - permuted_tokens: Tokens reordered by expert [total_tokens * top_k, hidden_size]
        - expert_offsets: Start offset for each expert [num_experts + 1]
        - reverse_indices: Indices to restore original order
    """
    batch_size, seq_len, top_k = expert_indices.shape
    hidden_size = hidden_states.size(-1)
    
    # Flatten batch and sequence dimensions
    flat_hidden = hidden_states.view(-1, hidden_size)  # [batch * seq, hidden]
    flat_indices = expert_indices.view(-1)  # [batch * seq * top_k]
    
    # Expand hidden states for each top-k selection
    # Each token appears top_k times
    expanded_hidden = flat_hidden.unsqueeze(1).expand(-1, top_k, -1)  # [batch*seq, top_k, hidden]
    expanded_hidden = expanded_hidden.reshape(-1, hidden_size)  # [batch*seq*top_k, hidden]
    
    # Sort by expert index
    sorted_indices = torch.argsort(flat_indices)
    permuted_tokens = expanded_hidden[sorted_indices]
    
    # Compute expert offsets
    expert_counts = torch.bincount(flat_indices, minlength=num_experts)
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.long, device=hidden_states.device)
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)
    
    # Compute reverse indices for unpermutation
    reverse_indices = torch.argsort(sorted_indices)
    
    return permuted_tokens, expert_offsets, reverse_indices


def unpermute_tokens(
    permuted_output: torch.Tensor,
    reverse_indices: torch.Tensor,
    routing_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Unpermute tokens back to original order and combine expert outputs.
    
    Args:
        permuted_output: Expert outputs [total_tokens * top_k, hidden_size]
        reverse_indices: Indices from permute_tokens
        routing_weights: Routing weights [batch_size, seq_len, top_k]
        
    Returns:
        Combined output [batch_size, seq_len, hidden_size]
    """
    batch_size, seq_len, top_k = routing_weights.shape
    hidden_size = permuted_output.size(-1)
    
    # Unpermute to original order
    unpermuted = permuted_output[reverse_indices]  # [batch*seq*top_k, hidden]
    
    # Reshape to [batch*seq, top_k, hidden]
    unpermuted = unpermuted.view(batch_size * seq_len, top_k, hidden_size)
    
    # Weight by routing weights and sum
    weights = routing_weights.view(-1, top_k, 1)  # [batch*seq, top_k, 1]
    combined = (unpermuted * weights).sum(dim=1)  # [batch*seq, hidden]
    
    return combined.view(batch_size, seq_len, hidden_size)
