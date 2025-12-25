"""
RMSNorm (Root Mean Square Layer Normalization)

Used in LLaMA, Mixtral, and other modern LLMs.
More efficient than LayerNorm as it doesn't require mean computation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Normalizes the input by the root mean square, then scales by a learnable parameter.
    
    Formula: output = (x / sqrt(mean(x^2) + eps)) * weight
    
    Args:
        hidden_size: The size of the last dimension to normalize over.
        eps: A small constant for numerical stability.
    
    Example:
        >>> norm = RMSNorm(4096)
        >>> x = torch.randn(2, 128, 4096)
        >>> output = norm(x)  # Shape: (2, 128, 4096)
    """
    
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to input tensor.
        
        Args:
            x: Input tensor of shape (..., hidden_size)
            
        Returns:
            Normalized tensor of same shape as input.
        """
        # Compute RMS
        # x.pow(2).mean(-1, keepdim=True) computes mean of squares
        input_dtype = x.dtype
        x = x.to(torch.float32)
        
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        
        # Cast back and apply learnable scale
        return (self.weight * x).to(input_dtype)
    
    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.eps}"


class FastRMSNorm(nn.Module):
    """
    Fused RMSNorm using torch.compile for better performance.
    
    Falls back to regular RMSNorm if torch.compile is not available.
    """
    
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        
        # Try to compile the forward function
        self._compiled_forward = None
        try:
            self._compiled_forward = torch.compile(self._forward_impl)
        except Exception:
            pass
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Implementation that can be compiled."""
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._compiled_forward is not None:
            return self._compiled_forward(x)
        return self._forward_impl(x)
