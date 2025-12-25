"""
Tests for the inference engine.
"""

import pytest


class TestEngineConfig:
    """Test EngineConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from light_moe.engine.executor import EngineConfig

        config = EngineConfig(model_path="/tmp/test_model")
        
        assert config.model_path == "/tmp/test_model"
        assert config.tensor_parallel_size == 1
        assert config.expert_parallel_size == 1
        assert config.max_batch_size == 256
        assert config.dtype == "float16"

    def test_custom_config(self):
        """Test custom configuration."""
        from light_moe.engine.executor import EngineConfig

        config = EngineConfig(
            model_path="/tmp/model",
            tensor_parallel_size=2,
            expert_parallel_size=4,
            use_int4_quantization=True,
        )
        
        assert config.tensor_parallel_size == 2
        assert config.expert_parallel_size == 4
        assert config.use_int4_quantization is True


class TestLightMoEEngine:
    """Test LightMoEEngine class."""

    @pytest.mark.skip(reason="Engine initialization requires model files")
    def test_engine_init(self):
        """Test engine initialization."""
        from light_moe import LightMoEEngine

        engine = LightMoEEngine(
            model_path="/tmp/test_model",
            expert_parallel_size=1,
        )
        
        assert engine._initialized is True
        engine.shutdown()

    def test_engine_not_initialized_error(self):
        """Test error when generating without initialization."""
        # This test would require mocking the initialization
        pass
