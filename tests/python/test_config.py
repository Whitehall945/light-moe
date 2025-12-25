"""
Tests for model configuration.
"""

import json
import tempfile
from pathlib import Path

import pytest


class TestMoEConfig:
    """Test MoEConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        from light_moe.model import MoEConfig

        config = MoEConfig()
        
        assert config.num_experts == 8
        assert config.num_experts_per_token == 2
        assert config.hidden_size == 4096
        assert config.num_attention_heads == 32

    def test_custom_config(self):
        """Test custom configuration."""
        from light_moe.model import MoEConfig

        config = MoEConfig(
            num_experts=16,
            num_experts_per_token=4,
            hidden_size=8192,
        )
        
        assert config.num_experts == 16
        assert config.num_experts_per_token == 4
        assert config.hidden_size == 8192

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from light_moe.model import MoEConfig

        config = MoEConfig(num_experts=16)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["num_experts"] == 16
        assert "hidden_size" in config_dict

    def test_from_pretrained(self):
        """Test loading from pretrained config file."""
        from light_moe.model import MoEConfig

        # Create temporary config file
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_data = {
                "vocab_size": 32000,
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "num_local_experts": 8,
                "num_experts_per_tok": 2,
            }
            
            with open(config_path, "w") as f:
                json.dump(config_data, f)
            
            config = MoEConfig.from_pretrained(tmpdir)
            
            assert config.num_experts == 8
            assert config.num_experts_per_token == 2
            assert config.hidden_size == 4096

    def test_from_pretrained_not_found(self):
        """Test error when config file not found."""
        from light_moe.model import MoEConfig

        with pytest.raises(FileNotFoundError):
            MoEConfig.from_pretrained("/nonexistent/path")
