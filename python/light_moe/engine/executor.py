"""
Light-MoE Inference Engine Executor

Main entry point for running MoE model inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Union, List

import torch

from light_moe.model import MoEConfig, MoETransformer, load_model_from_hf
from light_moe.engine.kv_cache import KVCache, CacheConfig
from light_moe.engine.sampler import Sampler, SamplingParams

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """Configuration for the inference engine."""
    
    model_path: str
    """Path to the model weights."""
    
    tensor_parallel_size: int = 1
    """Number of GPUs for tensor parallelism."""
    
    expert_parallel_size: int = 1
    """Number of GPUs for expert parallelism."""
    
    max_batch_size: int = 32
    """Maximum batch size."""
    
    max_seq_len: int = 4096
    """Maximum sequence length."""
    
    dtype: str = "float16"
    """Data type for computation (float16, bfloat16, float32)."""
    
    use_int4_quantization: bool = False
    """Enable INT4 weight-only quantization."""
    
    quantization_group_size: int = 128
    """Group size for quantization."""
    
    device: str = "cuda"
    """Target device."""


class LightMoEEngine:
    """
    High-performance MoE inference engine.
    
    This engine implements:
    - Expert Parallelism for distributed MoE inference
    - Efficient KV caching for autoregressive generation
    - Various sampling strategies
    
    Example:
        >>> engine = LightMoEEngine(
        ...     model_path="path/to/mixtral-8x7b",
        ...     expert_parallel_size=8,
        ... )
        >>> output = engine.generate("Hello, world!", max_tokens=100)
    """
    
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        expert_parallel_size: int = 1,
        max_batch_size: int = 32,
        max_seq_len: int = 4096,
        dtype: str = "float16",
        device: str = "cuda",
        **kwargs,
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to model weights.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            expert_parallel_size: Number of GPUs for expert parallelism.
            max_batch_size: Maximum batch size.
            max_seq_len: Maximum sequence length.
            dtype: Data type (float16, bfloat16, float32).
            device: Target device.
            **kwargs: Additional configuration options.
        """
        self.config = EngineConfig(
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            expert_parallel_size=expert_parallel_size,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            dtype=dtype,
            device=device,
            **kwargs,
        )
        
        self._initialized = False
        self._model: Optional[MoETransformer] = None
        self._model_config: Optional[MoEConfig] = None
        self._tokenizer = None
        self._kv_cache: Optional[KVCache] = None
        self._sampler: Optional[Sampler] = None
        
        # Convert dtype string to torch dtype
        self._dtype = getattr(torch, dtype)
        self._device = torch.device(device)
        
        logger.info(f"Initializing LightMoEEngine with config: {self.config}")
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize the engine components."""
        try:
            # Load model
            logger.info(f"Loading model from {self.config.model_path}")
            self._model = load_model_from_hf(
                self.config.model_path,
                device=self.config.device,
                dtype=self._dtype,
            )
            self._model_config = self._model.config
            
            # Initialize KV cache
            self._kv_cache = KVCache.from_model_config(
                self._model_config,
                max_batch_size=self.config.max_batch_size,
                max_seq_len=self.config.max_seq_len,
                device=self.config.device,
                dtype=self._dtype,
            )
            
            # Initialize sampler
            self._sampler = Sampler(self._model_config.vocab_size)
            
            # Try to load tokenizer
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
                logger.info("Tokenizer loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load tokenizer: {e}")
                self._tokenizer = None
            
            self._initialized = True
            logger.info("LightMoEEngine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize engine: {e}")
            raise
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        repetition_penalty: float = 1.0,
        stop_token_ids: Optional[List[int]] = None,
        stream: bool = False,
    ) -> Union[str, List[str]]:
        """
        Generate text from prompts.
        
        Args:
            prompt: Input prompt(s).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling parameter.
            repetition_penalty: Penalty for repeated tokens.
            stop_token_ids: Token IDs that trigger stop.
            stream: Whether to stream output (yield tokens).
            
        Returns:
            Generated text(s).
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized")
        
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not available. Cannot generate from string prompts.")
        
        # Handle single string input
        is_single = isinstance(prompt, str)
        if is_single:
            prompt = [prompt]
        
        # Tokenize
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_len - max_tokens,
        )
        input_ids = inputs["input_ids"].to(self._device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)
        
        # Generate
        output_ids = self._generate_tokens(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop_token_ids=stop_token_ids,
        )
        
        # Decode
        outputs = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        
        return outputs[0] if is_single else outputs
    
    @torch.inference_mode()
    def _generate_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        repetition_penalty: float = 1.0,
        stop_token_ids: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Generate tokens from input IDs.
        
        Args:
            input_ids: Input token IDs [batch, seq_len].
            attention_mask: Optional attention mask.
            max_new_tokens: Maximum new tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling.
            repetition_penalty: Repetition penalty.
            stop_token_ids: Stop token IDs.
            
        Returns:
            Generated token IDs [batch, total_seq_len].
        """
        batch_size, seq_len = input_ids.shape
        
        # Reset KV cache
        self._kv_cache.reset()
        
        # Prefill phase
        logits, past_key_values, _ = self._model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_aux_loss=False,
        )
        
        # Get next token from last position
        next_token_logits = logits[:, -1, :]
        next_token = self._sampler(
            next_token_logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            past_tokens=input_ids,
        )
        
        # Collect generated tokens
        generated_tokens = [next_token]
        all_tokens = torch.cat([input_ids, next_token], dim=1)
        
        # Decode phase
        for _ in range(max_new_tokens - 1):
            # Check stop condition
            if stop_token_ids is not None:
                if any(next_token.item() == stop_id for stop_id in stop_token_ids):
                    break
            
            # Forward with cache
            logits, past_key_values, _ = self._model(
                next_token,
                past_key_values=past_key_values,
                use_cache=True,
                return_aux_loss=False,
            )
            
            # Sample next token
            next_token_logits = logits[:, -1, :]
            next_token = self._sampler(
                next_token_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                past_tokens=all_tokens,
            )
            
            generated_tokens.append(next_token)
            all_tokens = torch.cat([all_tokens, next_token], dim=1)
        
        return all_tokens
    
    def generate_from_ids(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate from token IDs directly (no tokenization).
        
        Args:
            input_ids: Input token IDs [batch, seq_len].
            max_new_tokens: Maximum new tokens.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            
        Returns:
            Generated token IDs.
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized")
        
        return self._generate_tokens(
            input_ids.to(self._device),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    
    def get_model(self) -> MoETransformer:
        """Get the underlying model."""
        if not self._initialized:
            raise RuntimeError("Engine not initialized")
        return self._model
    
    def shutdown(self) -> None:
        """Shutdown the engine and release resources."""
        if self._model is not None:
            del self._model
            self._model = None
        
        if self._kv_cache is not None:
            del self._kv_cache
            self._kv_cache = None
        
        self._initialized = False
        
        # Free CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("LightMoEEngine shutdown complete")


# Convenience function
def create_engine(
    model_path: str,
    device: str = "cuda",
    dtype: str = "float16",
    **kwargs,
) -> LightMoEEngine:
    """
    Create a LightMoEEngine instance.
    
    Args:
        model_path: Path to model.
        device: Target device.
        dtype: Data type.
        **kwargs: Additional engine config.
        
    Returns:
        LightMoEEngine instance.
    """
    return LightMoEEngine(
        model_path=model_path,
        device=device,
        dtype=dtype,
        **kwargs,
    )
