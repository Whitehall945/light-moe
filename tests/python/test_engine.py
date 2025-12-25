"""
Tests for inference engine (Phase 3).
"""

import pytest
import torch


class TestKVCache:
    """Test KV Cache implementation."""

    def test_cache_creation(self):
        """Test cache creation."""
        from light_moe.engine import KVCache, CacheConfig

        config = CacheConfig(
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
            max_batch_size=2,
            max_seq_len=128,
        )
        
        cache = KVCache(config, device=torch.device("cpu"))
        
        assert len(cache.key_cache) == 2
        assert len(cache.value_cache) == 2
        assert cache.key_cache[0].shape == (2, 4, 128, 64)

    def test_cache_update(self):
        """Test cache update operation."""
        from light_moe.engine import KVCache, CacheConfig

        config = CacheConfig(
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
            max_batch_size=2,
            max_seq_len=128,
        )
        
        cache = KVCache(config, device=torch.device("cpu"))
        
        # First update
        key = torch.randn(2, 4, 8, 64)
        value = torch.randn(2, 4, 8, 64)
        
        k_out, v_out = cache.update(0, key, value)
        
        assert k_out.shape == (2, 4, 8, 64)
        assert cache.get_seq_length() == 8

    def test_cache_reset(self):
        """Test cache reset."""
        from light_moe.engine import KVCache, CacheConfig

        config = CacheConfig(
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
            max_batch_size=2,
            max_seq_len=128,
        )
        
        cache = KVCache(config, device=torch.device("cpu"))
        
        # Add some data
        key = torch.randn(2, 4, 8, 64)
        value = torch.randn(2, 4, 8, 64)
        cache.update(0, key, value)
        
        assert cache.get_seq_length() == 8
        
        # Reset
        cache.reset()
        assert cache.get_seq_length() == 0


class TestSampler:
    """Test Sampler implementation."""

    def test_greedy_sampling(self):
        """Test greedy (temperature=0) sampling."""
        from light_moe.engine import Sampler

        sampler = Sampler(vocab_size=100)
        
        logits = torch.randn(2, 100)
        next_token = sampler(logits, temperature=0)
        
        # Greedy should select argmax
        expected = logits.argmax(dim=-1, keepdim=True)
        assert torch.equal(next_token, expected)

    def test_temperature_sampling(self):
        """Test temperature sampling."""
        from light_moe.engine import Sampler

        sampler = Sampler(vocab_size=100)
        
        logits = torch.randn(2, 100)
        
        # Sample with temperature
        next_token = sampler(logits, temperature=1.0)
        
        assert next_token.shape == (2, 1)
        assert (next_token >= 0).all()
        assert (next_token < 100).all()

    def test_top_k_sampling(self):
        """Test top-K sampling."""
        from light_moe.engine import Sampler

        sampler = Sampler(vocab_size=100)
        
        logits = torch.randn(2, 100)
        next_token = sampler(logits, temperature=1.0, top_k=10)
        
        assert next_token.shape == (2, 1)

    def test_top_p_sampling(self):
        """Test top-P (nucleus) sampling."""
        from light_moe.engine import Sampler

        sampler = Sampler(vocab_size=100)
        
        logits = torch.randn(2, 100)
        next_token = sampler(logits, temperature=1.0, top_p=0.9)
        
        assert next_token.shape == (2, 1)


class TestLightMoEEngine:
    """Test LightMoEEngine (basic structure only, full test requires model)."""

    def test_engine_config(self):
        """Test EngineConfig creation."""
        from light_moe.engine import EngineConfig

        config = EngineConfig(
            model_path="/tmp/test",
            tensor_parallel_size=1,
            expert_parallel_size=4,
        )
        
        assert config.model_path == "/tmp/test"
        assert config.expert_parallel_size == 4

    @pytest.mark.skip(reason="Requires model files")
    def test_engine_initialization(self):
        """Test engine initialization with model."""
        from light_moe.engine import LightMoEEngine

        engine = LightMoEEngine(
            model_path="/path/to/model",
            device="cpu",
        )
        
        assert engine._initialized

    def test_sampling_params(self):
        """Test SamplingParams dataclass."""
        from light_moe.engine import SamplingParams

        params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=256,
        )
        
        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.max_tokens == 256


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
