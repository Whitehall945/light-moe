"""
Model Weight Loader

Utilities for loading model weights from HuggingFace format (Mixtral, DeepSeek).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from light_moe.model.config import MoEConfig
from light_moe.model.transformer import MoETransformer

logger = logging.getLogger(__name__)


def load_config_from_hf(model_path: str) -> MoEConfig:
    """
    Load model configuration from HuggingFace format.
    
    Args:
        model_path: Path to model directory.
        
    Returns:
        MoEConfig instance.
    """
    return MoEConfig.from_pretrained(model_path)


def load_safetensors(file_path: str) -> Dict[str, torch.Tensor]:
    """
    Load weights from safetensors file.
    
    Args:
        file_path: Path to .safetensors file.
        
    Returns:
        Dictionary of weight tensors.
    """
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("Please install safetensors: pip install safetensors")
    
    tensors = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    
    return tensors


def load_pytorch_bin(file_path: str) -> Dict[str, torch.Tensor]:
    """
    Load weights from PyTorch .bin file.
    
    Args:
        file_path: Path to .bin file.
        
    Returns:
        Dictionary of weight tensors.
    """
    return torch.load(file_path, map_location="cpu", weights_only=True)


def convert_hf_to_lightmoe(
    hf_state_dict: Dict[str, torch.Tensor],
    config: MoEConfig,
) -> Dict[str, torch.Tensor]:
    """
    Convert HuggingFace state dict to Light-MoE format.
    
    Handles name mapping between Mixtral/LLaMA format and Light-MoE format.
    
    Args:
        hf_state_dict: HuggingFace model state dict.
        config: Model configuration.
        
    Returns:
        Light-MoE format state dict.
    """
    new_state_dict = {}
    
    # Name mapping from HuggingFace to Light-MoE
    name_map = {
        "model.embed_tokens.weight": "embed_tokens.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    
    for hf_name, tensor in hf_state_dict.items():
        # Direct mappings
        if hf_name in name_map:
            new_state_dict[name_map[hf_name]] = tensor
            continue
        
        # Layer mappings
        # Format: model.layers.{layer_idx}.{component}.weight
        if hf_name.startswith("model.layers."):
            # Remove 'model.' prefix
            new_name = hf_name.replace("model.layers.", "layers.")
            
            # Map attention components
            new_name = new_name.replace("self_attn.q_proj", "self_attn.q_proj")
            new_name = new_name.replace("self_attn.k_proj", "self_attn.k_proj")
            new_name = new_name.replace("self_attn.v_proj", "self_attn.v_proj")
            new_name = new_name.replace("self_attn.o_proj", "self_attn.o_proj")
            
            # Map norm components
            new_name = new_name.replace("input_layernorm", "input_layernorm")
            new_name = new_name.replace("post_attention_layernorm", "post_attention_layernorm")
            
            # Map MoE components (Mixtral format)
            # block_sparse_moe.gate.weight -> moe.gate.gate.weight
            new_name = new_name.replace("block_sparse_moe.gate.weight", "moe.gate.gate.weight")
            
            # block_sparse_moe.experts.{i}.w1 -> moe.experts.{i}.gate_proj (gate)
            # block_sparse_moe.experts.{i}.w2 -> moe.experts.{i}.down_proj
            # block_sparse_moe.experts.{i}.w3 -> moe.experts.{i}.up_proj
            new_name = new_name.replace(".w1.", ".gate_proj.")
            new_name = new_name.replace(".w2.", ".down_proj.")
            new_name = new_name.replace(".w3.", ".up_proj.")
            new_name = new_name.replace("block_sparse_moe.experts", "moe.experts")
            
            new_state_dict[new_name] = tensor
            continue
        
        # Fallback: keep original name
        logger.warning(f"Unknown weight: {hf_name}, keeping original name")
        new_state_dict[hf_name] = tensor
    
    return new_state_dict


def load_model_from_hf(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> MoETransformer:
    """
    Load complete model from HuggingFace format.
    
    Args:
        model_path: Path to model directory.
        device: Target device.
        dtype: Target dtype.
        
    Returns:
        Loaded MoETransformer model.
    """
    model_path = Path(model_path)
    
    # Load config
    logger.info(f"Loading config from {model_path}")
    config = load_config_from_hf(str(model_path))
    
    # Create model
    logger.info(f"Creating model with config: {config.to_dict()}")
    model = MoETransformer(config)
    
    # Find weight files
    safetensor_files = list(model_path.glob("*.safetensors"))
    pytorch_files = list(model_path.glob("*.bin"))
    
    if safetensor_files:
        logger.info(f"Loading from {len(safetensor_files)} safetensor files")
        state_dict = {}
        for file in safetensor_files:
            state_dict.update(load_safetensors(str(file)))
    elif pytorch_files:
        logger.info(f"Loading from {len(pytorch_files)} PyTorch files")
        state_dict = {}
        for file in pytorch_files:
            state_dict.update(load_pytorch_bin(str(file)))
    else:
        raise FileNotFoundError(f"No weight files found in {model_path}")
    
    # Convert state dict
    logger.info("Converting HuggingFace weights to Light-MoE format")
    converted_state_dict = convert_hf_to_lightmoe(state_dict, config)
    
    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(converted_state_dict, strict=False)
    
    if missing_keys:
        logger.warning(f"Missing keys: {missing_keys[:10]}...")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {unexpected_keys[:10]}...")
    
    # Move to device and dtype
    model = model.to(device=device, dtype=dtype)
    model.eval()
    
    logger.info(f"Model loaded successfully to {device} with dtype {dtype}")
    
    return model


class ModelLoader:
    """
    High-level model loader with caching.
    
    Example:
        >>> loader = ModelLoader()
        >>> model = loader.load("path/to/mixtral-8x7b")
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._loaded_models: Dict[str, MoETransformer] = {}
    
    def load(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        use_cache: bool = True,
    ) -> MoETransformer:
        """
        Load model with optional caching.
        
        Args:
            model_path: Path to model.
            device: Target device.
            dtype: Target dtype.
            use_cache: Whether to cache loaded model.
            
        Returns:
            Loaded model.
        """
        cache_key = f"{model_path}_{device}_{dtype}"
        
        if use_cache and cache_key in self._loaded_models:
            return self._loaded_models[cache_key]
        
        model = load_model_from_hf(model_path, device, dtype)
        
        if use_cache:
            self._loaded_models[cache_key] = model
        
        return model
    
    def clear_cache(self):
        """Clear model cache."""
        self._loaded_models.clear()
