"""
Sampling Strategies for Text Generation

Implements various sampling methods:
- Greedy decoding
- Temperature sampling
- Top-K sampling
- Top-P (nucleus) sampling
- Beam search (planned)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class SamplingParams:
    """Parameters for text generation sampling."""
    
    temperature: float = 1.0
    """Sampling temperature. 0 = greedy, higher = more random."""
    
    top_k: int = -1
    """Top-K sampling. -1 = disabled."""
    
    top_p: float = 1.0
    """Top-P (nucleus) sampling. 1.0 = disabled."""
    
    min_p: float = 0.0
    """Minimum probability threshold."""
    
    repetition_penalty: float = 1.0
    """Penalty for repeated tokens. 1.0 = no penalty."""
    
    max_tokens: int = 128
    """Maximum tokens to generate."""
    
    stop_token_ids: Optional[list] = None
    """Token IDs that trigger stop."""
    
    seed: Optional[int] = None
    """Random seed for reproducibility."""


class Sampler:
    """
    Token sampler for autoregressive generation.
    
    Applies temperature, top-k, top-p filtering and samples from
    the resulting distribution.
    
    Example:
        >>> sampler = Sampler()
        >>> logits = model(input_ids)[..., -1, :]  # [batch, vocab]
        >>> next_token = sampler(logits, temperature=0.7, top_p=0.9)
    """
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
    
    def __call__(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = 1.0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        past_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample next tokens from logits.
        
        Args:
            logits: Logits tensor [batch, vocab_size]
            temperature: Sampling temperature.
            top_k: Top-K filtering.
            top_p: Top-P (nucleus) filtering.
            min_p: Minimum probability filtering.
            repetition_penalty: Penalty for repeated tokens.
            past_tokens: Previously generated tokens for repetition penalty.
            
        Returns:
            Sampled token IDs [batch, 1].
        """
        # Apply repetition penalty
        if repetition_penalty != 1.0 and past_tokens is not None:
            logits = self._apply_repetition_penalty(
                logits, past_tokens, repetition_penalty
            )
        
        # Greedy decoding
        if temperature == 0:
            return logits.argmax(dim=-1, keepdim=True)
        
        # Apply temperature
        logits = logits / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            logits = self._top_k_filter(logits, top_k)
        
        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            logits = self._top_p_filter(logits, top_p)
        
        # Apply min-p filtering
        if min_p > 0.0:
            logits = self._min_p_filter(logits, min_p)
        
        # Sample from distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token
    
    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        past_tokens: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        """Apply repetition penalty to logits."""
        batch_size = logits.size(0)
        
        for b in range(batch_size):
            unique_tokens = past_tokens[b].unique()
            for token in unique_tokens:
                if logits[b, token] > 0:
                    logits[b, token] /= penalty
                else:
                    logits[b, token] *= penalty
        
        return logits
    
    def _top_k_filter(
        self,
        logits: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Apply top-K filtering."""
        top_k = min(k, logits.size(-1))
        
        # Get top-K values
        values, _ = torch.topk(logits, top_k, dim=-1)
        min_values = values[:, -1].unsqueeze(-1)
        
        # Mask out tokens below threshold
        logits = torch.where(
            logits < min_values,
            torch.full_like(logits, float("-inf")),
            logits,
        )
        
        return logits
    
    def _top_p_filter(
        self,
        logits: torch.Tensor,
        p: float,
    ) -> torch.Tensor:
        """Apply top-P (nucleus) filtering."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > p
        
        # Keep at least one token
        sorted_indices_to_remove[:, 0] = False
        
        # Scatter back to original indices
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        
        logits = logits.masked_fill(indices_to_remove, float("-inf"))
        
        return logits
    
    def _min_p_filter(
        self,
        logits: torch.Tensor,
        min_p: float,
    ) -> torch.Tensor:
        """Apply minimum probability filtering."""
        probs = F.softmax(logits, dim=-1)
        max_prob = probs.max(dim=-1, keepdim=True).values
        threshold = max_prob * min_p
        
        logits = torch.where(
            probs < threshold,
            torch.full_like(logits, float("-inf")),
            logits,
        )
        
        return logits


def sample_top_p(
    logits: torch.Tensor,
    top_p: float,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Convenience function for top-P sampling.
    
    Args:
        logits: Logits [batch, vocab]
        top_p: Nucleus probability threshold.
        temperature: Sampling temperature.
        
    Returns:
        Sampled tokens [batch, 1].
    """
    sampler = Sampler(logits.size(-1))
    return sampler(logits, temperature=temperature, top_p=top_p)


def sample_greedy(logits: torch.Tensor) -> torch.Tensor:
    """
    Greedy decoding (argmax).
    
    Args:
        logits: Logits [batch, vocab]
        
    Returns:
        Token with highest probability [batch, 1].
    """
    return logits.argmax(dim=-1, keepdim=True)
