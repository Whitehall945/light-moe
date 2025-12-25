"""
Transformer Model Implementation

Complete MoE Transformer model combining:
- Attention with RoPE
- MoE FFN blocks
- RMSNorm normalization
"""

from __future__ import annotations

from typing import Optional, Tuple, List

import torch
import torch.nn as nn

from light_moe.ops.rmsnorm import RMSNorm
from light_moe.model.attention import Attention
from light_moe.model.moe_block import SparseMoEBlock
from light_moe.model.config import MoEConfig


class TransformerBlock(nn.Module):
    """
    Single Transformer block with MoE FFN.
    
    Architecture:
    - Pre-norm with RMSNorm
    - Attention (with GQA support)
    - Residual connection
    - Pre-norm with RMSNorm
    - MoE FFN
    - Residual connection
    
    Args:
        config: Model configuration.
        layer_idx: Index of this layer (for KV cache).
    """
    
    def __init__(self, config: MoEConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        # Attention
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
        )
        
        # MoE FFN
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.moe = SparseMoEBlock(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_token,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """
        Forward pass through transformer block.
        
        Args:
            hidden_states: Input [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            past_key_value: Optional KV cache
            use_cache: Whether to return updated cache
            
        Returns:
            Tuple of:
            - Output hidden states
            - Updated KV cache (if use_cache)
            - Auxiliary MoE loss (if training)
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        
        # MoE FFN with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, aux_loss = self.moe(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_value, aux_loss


class MoETransformer(nn.Module):
    """
    Complete MoE Transformer model.
    
    Consists of:
    - Token embedding
    - Stack of TransformerBlocks
    - Final RMSNorm
    - LM head for next-token prediction
    
    Args:
        config: Model configuration.
    
    Example:
        >>> config = MoEConfig(num_experts=8, num_hidden_layers=32)
        >>> model = MoETransformer(config)
        >>> input_ids = torch.randint(0, 32000, (2, 128))
        >>> logits = model(input_ids)
    """
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        
        # Final norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # LM head (tied to embeddings by default)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights with normal distribution."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _create_causal_mask(
        self,
        batch_size: int,
        seq_len: int,
        past_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create causal attention mask."""
        # Create causal mask
        total_len = past_len + seq_len
        mask = torch.full(
            (seq_len, total_len),
            float("-inf"),
            device=device,
            dtype=dtype,
        )
        mask = torch.triu(mask, diagonal=past_len + 1)
        
        # Expand for batch and heads
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, total]
        
        return mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        return_aux_loss: bool = True,
    ) -> Tuple[torch.Tensor, Optional[List], Optional[torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Optional padding mask
            position_ids: Optional position indices
            past_key_values: Optional KV cache from previous steps
            use_cache: Whether to return updated cache
            return_aux_loss: Whether to accumulate MoE aux losses
            
        Returns:
            Tuple of:
            - Logits [batch, seq_len, vocab_size]
            - Updated past_key_values (if use_cache)
            - Accumulated aux loss (if training and return_aux_loss)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Determine past length for position IDs
        past_len = 0
        if past_key_values is not None and past_key_values[0] is not None:
            past_len = past_key_values[0][0].size(2)
        
        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(
                past_len, past_len + seq_len,
                device=device
            ).unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Create causal mask
        causal_mask = self._create_causal_mask(
            batch_size, seq_len, past_len,
            device=device, dtype=hidden_states.dtype
        )
        
        # Combine with attention mask if provided
        if attention_mask is not None:
            # Expand attention mask
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(hidden_states.dtype)
            attention_mask = (1.0 - attention_mask) * float("-inf")
            causal_mask = causal_mask + attention_mask
        
        # Process through layers
        new_past_key_values = [] if use_cache else None
        total_aux_loss = 0.0
        
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            
            hidden_states, present_kv, aux_loss = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                use_cache=use_cache,
            )
            
            if use_cache:
                new_past_key_values.append(present_kv)
            
            if aux_loss is not None and return_aux_loss:
                total_aux_loss = total_aux_loss + aux_loss
        
        # Final norm and LM head
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        # Prepare aux loss
        aux_loss_out = total_aux_loss if return_aux_loss and self.training else None
        
        return logits, new_past_key_values, aux_loss_out
    
    @torch.no_grad()
    def generate_next_token(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> Tuple[torch.Tensor, List]:
        """
        Generate the next token.
        
        Args:
            input_ids: Current token IDs
            past_key_values: KV cache
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            
        Returns:
            Tuple of (next_token, updated_cache)
        """
        logits, new_cache, _ = self.forward(
            input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_aux_loss=False,
        )
        
        # Get last token logits
        next_token_logits = logits[:, -1, :]
        
        # Apply temperature
        if temperature > 0:
            next_token_logits = next_token_logits / temperature
        
        # Apply top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            next_token_logits[indices_to_remove] = float("-inf")
        
        # Sample
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token, new_cache
