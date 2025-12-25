"""
Activation Functions for MoE Models

Includes SiLU (Swish) and SwiGLU (Swish-Gated Linear Unit) used in
modern LLMs like LLaMA, Mixtral, and DeepSeek.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def silu(x: torch.Tensor) -> torch.Tensor:
    """
    SiLU (Sigmoid Linear Unit) activation, also known as Swish.
    
    Formula: silu(x) = x * sigmoid(x)
    
    Args:
        x: Input tensor of any shape.
        
    Returns:
        Activated tensor of same shape.
    """
    return F.silu(x)


def swiglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """
    SwiGLU (Swish-Gated Linear Unit) activation.
    
    Used in the FFN of LLaMA, Mixtral, and similar models.
    
    Formula: swiglu(x, gate) = silu(gate) * x
    
    Args:
        x: Input tensor (the "up" projection).
        gate: Gate tensor (the "gate" projection).
        
    Returns:
        Activated tensor of same shape as x.
    
    Example:
        >>> x = torch.randn(2, 128, 14336)
        >>> gate = torch.randn(2, 128, 14336)
        >>> output = swiglu(x, gate)  # Shape: (2, 128, 14336)
    """
    return F.silu(gate) * x


class SwiGLU(nn.Module):
    """
    SwiGLU activation module with optional linear projections.
    
    Implements the FFN structure used in LLaMA/Mixtral:
    - gate_proj: hidden_size -> intermediate_size
    - up_proj: hidden_size -> intermediate_size  
    - down_proj: intermediate_size -> hidden_size
    
    Forward: down_proj(silu(gate_proj(x)) * up_proj(x))
    
    Args:
        hidden_size: Input and output dimension.
        intermediate_size: FFN intermediate dimension.
        bias: Whether to use bias in linear layers.
    
    Example:
        >>> ffn = SwiGLU(4096, 14336)
        >>> x = torch.randn(2, 128, 4096)
        >>> output = ffn(x)  # Shape: (2, 128, 4096)
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
        Forward pass of SwiGLU FFN.
        
        Args:
            x: Input tensor of shape (..., hidden_size)
            
        Returns:
            Output tensor of shape (..., hidden_size)
        """
        return self.down_proj(swiglu(self.up_proj(x), self.gate_proj(x)))
    
    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size}"


class FusedSwiGLU(nn.Module):
    """
    Fused SwiGLU that combines gate and up projections into one matrix multiply.
    
    More memory efficient as it computes gate and up in a single GEMM.
    
    Args:
        hidden_size: Input and output dimension.
        intermediate_size: FFN intermediate dimension.
        bias: Whether to use bias.
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
        
        # Fused gate+up projection: hidden -> 2 * intermediate
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fused gate+up projection.
        
        Args:
            x: Input tensor of shape (..., hidden_size)
            
        Returns:
            Output tensor of shape (..., hidden_size)
        """
        # Single GEMM for gate and up
        gate_up = self.gate_up_proj(x)
        
        # Split and apply SwiGLU
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = swiglu(up, gate)
        
        return self.down_proj(hidden)
