"""
Attention Module with Rotary Position Embeddings (RoPE)

Implements Multi-Head Attention with Grouped Query Attention (GQA)
and Rotary Position Embeddings used in modern LLMs.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    Encodes position information by rotating query and key vectors.
    This allows the model to learn relative positions effectively.
    
    Args:
        dim: Dimension of the embedding (head_dim).
        max_position_embeddings: Maximum sequence length.
        base: Base for the frequency computation.
    
    Reference:
        RoFormer: Enhanced Transformer with Rotary Position Embedding
        https://arxiv.org/abs/2104.09864
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Build cache for cos/sin
        self._set_cos_sin_cache(max_position_embeddings)
    
    def _set_cos_sin_cache(self, seq_len: int):
        """Pre-compute cos and sin for positions up to seq_len."""
        self.max_seq_len_cached = seq_len
        
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim/2]
        
        # Concatenate to get full dimension
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
        
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos and sin for the given positions.
        
        Args:
            x: Input tensor [batch, seq_len, ...] (only used for seq_len and device)
            position_ids: Optional position indices [batch, seq_len]
            
        Returns:
            Tuple of (cos, sin) tensors for RoPE.
        """
        seq_len = x.shape[1]
        
        # Extend cache if needed
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        
        if position_ids is None:
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]
        else:
            cos = self.cos_cached[position_ids]
            sin = self.sin_cached[position_ids]
        
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.
    
    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_kv_heads, seq_len, head_dim]
        cos: Cosine tensor for positions
        sin: Sine tensor for positions
        position_ids: Optional position indices
        
    Returns:
        Tuple of rotated (query, key) tensors.
    """
    # Reshape cos/sin for broadcasting
    if position_ids is not None:
        cos = cos.unsqueeze(1)  # [batch, 1, seq, dim]
        sin = sin.unsqueeze(1)
    else:
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim]
        sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


class Attention(nn.Module):
    """
    Multi-Head Attention with Grouped Query Attention (GQA) support.
    
    Supports:
    - Standard Multi-Head Attention (num_heads == num_kv_heads)
    - Grouped Query Attention (num_heads > num_kv_heads)
    - Multi-Query Attention (num_kv_heads == 1)
    
    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of attention heads.
        num_kv_heads: Number of key-value heads (for GQA).
        head_dim: Dimension per head (default: hidden_size // num_heads).
        max_position_embeddings: Maximum sequence length.
        rope_theta: Base for RoPE frequency computation.
        attention_dropout: Dropout probability for attention weights.
    
    Example:
        >>> attn = Attention(4096, num_heads=32, num_kv_heads=8)
        >>> x = torch.randn(2, 128, 4096)
        >>> output = attn(x)  # Shape: (2, 128, 4096)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        max_position_embeddings: int = 32768,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.attention_dropout = attention_dropout
        
        # Validate dimensions
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
            )
        
        # Projection layers
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=bias)
        
        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of attention.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Optional mask [batch, 1, seq_len, seq_len]
            position_ids: Optional position indices [batch, seq_len]
            past_key_value: Optional cached (key, value) from previous step
            use_cache: Whether to return updated cache
            
        Returns:
            Tuple of:
            - Output tensor [batch, seq_len, hidden_size]
            - Optional updated (key, value) cache
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape to [batch, num_heads, seq_len, head_dim]
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        
        # Handle KV cache
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        # Expand KV for GQA
        if self.num_kv_groups > 1:
            key_states = key_states.repeat_interleave(self.num_kv_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_kv_groups, dim=1)
        
        # Compute attention
        attn_weights = torch.matmul(
            query_states, key_states.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)
        
        # Apply mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_weights = F.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, past_key_value
    
    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_heads={self.num_heads}, "
            f"num_kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}"
        )
