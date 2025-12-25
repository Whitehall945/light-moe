"""
KV Cache Management

Efficient KV cache implementation for autoregressive inference.
Supports both contiguous and paged memory layouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class CacheConfig:
    """Configuration for KV cache."""
    
    num_layers: int
    """Number of transformer layers."""
    
    num_kv_heads: int
    """Number of key-value heads."""
    
    head_dim: int
    """Dimension per head."""
    
    max_batch_size: int = 32
    """Maximum batch size."""
    
    max_seq_len: int = 4096
    """Maximum sequence length."""
    
    dtype: torch.dtype = torch.float16
    """Data type for cache tensors."""


class KVCache:
    """
    Key-Value cache for efficient autoregressive generation.
    
    Pre-allocates memory for the full sequence length to avoid
    dynamic memory allocation during generation.
    
    Args:
        config: Cache configuration.
        device: Target device.
    
    Example:
        >>> config = CacheConfig(num_layers=32, num_kv_heads=8, head_dim=128)
        >>> cache = KVCache(config, device="cuda")
        >>> 
        >>> # During prefill
        >>> cache.update(layer_idx=0, key=k, value=v, positions=pos)
        >>> 
        >>> # During decode
        >>> k_cache, v_cache = cache.get(layer_idx=0, seq_len=current_len)
    """
    
    def __init__(
        self,
        config: CacheConfig,
        device: torch.device = torch.device("cuda"),
    ):
        self.config = config
        self.device = device
        
        # Pre-allocate cache tensors
        # Shape: [batch_size, num_kv_heads, max_seq_len, head_dim]
        cache_shape = (
            config.max_batch_size,
            config.num_kv_heads,
            config.max_seq_len,
            config.head_dim,
        )
        
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        
        for _ in range(config.num_layers):
            self.key_cache.append(
                torch.zeros(cache_shape, dtype=config.dtype, device=device)
            )
            self.value_cache.append(
                torch.zeros(cache_shape, dtype=config.dtype, device=device)
            )
        
        # Track current sequence length per batch item
        self.seq_lengths = torch.zeros(
            config.max_batch_size, dtype=torch.long, device=device
        )
    
    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key-value pairs.
        
        Args:
            layer_idx: Layer index.
            key: New key tensor [batch, num_kv_heads, seq_len, head_dim]
            value: New value tensor [batch, num_kv_heads, seq_len, head_dim]
            positions: Optional position indices [batch, seq_len]
            
        Returns:
            Tuple of (full_key_cache, full_value_cache) up to current position.
        """
        batch_size, num_heads, seq_len, head_dim = key.shape
        
        # Ensure dtype matches cache
        key = key.to(dtype=self.config.dtype)
        value = value.to(dtype=self.config.dtype)
        
        if positions is None:
            # Append sequentially
            start_pos = self.seq_lengths[0].item()
            positions = torch.arange(
                start_pos, start_pos + seq_len,
                device=self.device
            ).unsqueeze(0).expand(batch_size, -1)
        
        # Scatter into cache
        # positions: [batch, seq_len] -> indices for dim 2
        for b in range(batch_size):
            pos = positions[b]  # [seq_len]
            self.key_cache[layer_idx][b, :, pos, :] = key[b]
            self.value_cache[layer_idx][b, :, pos, :] = value[b]
        
        # Update sequence lengths
        new_len = positions.max().item() + 1
        self.seq_lengths[:batch_size] = new_len
        
        # Return cached values up to current position
        return (
            self.key_cache[layer_idx][:batch_size, :, :new_len, :],
            self.value_cache[layer_idx][:batch_size, :, :new_len, :],
        )
    
    def get(
        self,
        layer_idx: int,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cached key-value pairs.
        
        Args:
            layer_idx: Layer index.
            batch_size: Number of batch items to return.
            seq_len: Sequence length to return (default: current length).
            
        Returns:
            Tuple of (key_cache, value_cache).
        """
        if batch_size is None:
            batch_size = (self.seq_lengths > 0).sum().item()
        if seq_len is None:
            seq_len = self.seq_lengths[0].item()
        
        return (
            self.key_cache[layer_idx][:batch_size, :, :seq_len, :],
            self.value_cache[layer_idx][:batch_size, :, :seq_len, :],
        )
    
    def get_seq_length(self, batch_idx: int = 0) -> int:
        """Get current sequence length for a batch item."""
        return self.seq_lengths[batch_idx].item()
    
    def reset(self, batch_indices: Optional[torch.Tensor] = None):
        """
        Reset cache for specified batch indices.
        
        Args:
            batch_indices: Indices to reset. If None, reset all.
        """
        if batch_indices is None:
            self.seq_lengths.zero_()
            for layer_idx in range(self.config.num_layers):
                self.key_cache[layer_idx].zero_()
                self.value_cache[layer_idx].zero_()
        else:
            self.seq_lengths[batch_indices] = 0
            for layer_idx in range(self.config.num_layers):
                self.key_cache[layer_idx][batch_indices].zero_()
                self.value_cache[layer_idx][batch_indices].zero_()
    
    def to_list(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Convert to list format for model forward pass.
        
        Returns:
            List of (key, value) tuples per layer.
        """
        batch_size = (self.seq_lengths > 0).sum().item()
        seq_len = self.seq_lengths[0].item() if batch_size > 0 else 0
        
        if seq_len == 0:
            return [None] * self.config.num_layers
        
        return [
            (
                self.key_cache[i][:batch_size, :, :seq_len, :],
                self.value_cache[i][:batch_size, :, :seq_len, :],
            )
            for i in range(self.config.num_layers)
        ]
    
    @classmethod
    def from_model_config(
        cls,
        model_config,
        max_batch_size: int = 32,
        max_seq_len: int = 4096,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> "KVCache":
        """
        Create KVCache from model configuration.
        
        Args:
            model_config: MoEConfig instance.
            max_batch_size: Maximum batch size.
            max_seq_len: Maximum sequence length.
            device: Target device.
            dtype: Data type.
            
        Returns:
            KVCache instance.
        """
        head_dim = model_config.hidden_size // model_config.num_attention_heads
        
        cache_config = CacheConfig(
            num_layers=model_config.num_hidden_layers,
            num_kv_heads=model_config.num_key_value_heads,
            head_dim=head_dim,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            dtype=dtype,
        )
        
        return cls(cache_config, device=torch.device(device))
