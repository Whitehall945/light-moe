"""
MoE Model Configuration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MoEConfig:
    """
    Configuration for MoE (Mixture of Experts) models.
    
    Supports common MoE architectures like Mixtral and DeepSeek.
    """
    
    # Model architecture
    vocab_size: int = 32000
    """Vocabulary size."""
    
    hidden_size: int = 4096
    """Hidden dimension."""
    
    intermediate_size: int = 14336
    """FFN intermediate dimension."""
    
    num_hidden_layers: int = 32
    """Number of transformer layers."""
    
    num_attention_heads: int = 32
    """Number of attention heads."""
    
    num_key_value_heads: int = 8
    """Number of key-value heads (for GQA)."""
    
    # MoE specific
    num_experts: int = 8
    """Total number of experts."""
    
    num_experts_per_token: int = 2
    """Number of experts activated per token (top-k)."""
    
    moe_layer_frequency: int = 1
    """Every N-th layer is an MoE layer (1 = all layers)."""
    
    # RoPE
    rope_theta: float = 10000.0
    """RoPE base frequency."""
    
    max_position_embeddings: int = 32768
    """Maximum sequence length."""
    
    # Normalization
    rms_norm_eps: float = 1e-5
    """RMSNorm epsilon."""
    
    # Activation
    hidden_act: str = "silu"
    """Activation function (silu, gelu, etc.)."""
    
    # Parallelism
    tensor_parallel_size: int = 1
    """Tensor parallelism degree."""
    
    expert_parallel_size: int = 1
    """Expert parallelism degree."""
    
    @classmethod
    def from_pretrained(cls, model_path: str) -> MoEConfig:
        """
        Load configuration from a pretrained model.
        
        Args:
            model_path: Path to model directory or HuggingFace model ID.
            
        Returns:
            MoEConfig instance.
        """
        import json
        from pathlib import Path
        
        config_path = Path(model_path) / "config.json"
        
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
            
            # Map HuggingFace config to our config
            return cls(
                vocab_size=config_dict.get("vocab_size", 32000),
                hidden_size=config_dict.get("hidden_size", 4096),
                intermediate_size=config_dict.get("intermediate_size", 14336),
                num_hidden_layers=config_dict.get("num_hidden_layers", 32),
                num_attention_heads=config_dict.get("num_attention_heads", 32),
                num_key_value_heads=config_dict.get("num_key_value_heads", 8),
                num_experts=config_dict.get("num_local_experts", 8),
                num_experts_per_token=config_dict.get("num_experts_per_tok", 2),
                max_position_embeddings=config_dict.get("max_position_embeddings", 32768),
                rope_theta=config_dict.get("rope_theta", 10000.0),
                rms_norm_eps=config_dict.get("rms_norm_eps", 1e-5),
            )
        else:
            raise FileNotFoundError(f"Config file not found at {config_path}")
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "num_experts": self.num_experts,
            "num_experts_per_token": self.num_experts_per_token,
            "moe_layer_frequency": self.moe_layer_frequency,
            "rope_theta": self.rope_theta,
            "max_position_embeddings": self.max_position_embeddings,
            "rms_norm_eps": self.rms_norm_eps,
            "hidden_act": self.hidden_act,
            "tensor_parallel_size": self.tensor_parallel_size,
            "expert_parallel_size": self.expert_parallel_size,
        }
