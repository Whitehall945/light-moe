"""
Tests for model layer (Phase 2).
"""

import pytest
import torch


class TestAttention:
    """Test Attention implementation."""

    def test_attention_forward(self):
        """Test basic attention forward pass."""
        from light_moe.model import Attention

        hidden_size = 256
        num_heads = 8
        
        attn = Attention(hidden_size, num_heads)
        x = torch.randn(2, 32, hidden_size)
        
        output, _ = attn(x)
        assert output.shape == x.shape

    def test_gqa_attention(self):
        """Test Grouped Query Attention."""
        from light_moe.model import Attention

        hidden_size = 256
        num_heads = 8
        num_kv_heads = 2  # GQA
        
        attn = Attention(hidden_size, num_heads, num_kv_heads=num_kv_heads)
        x = torch.randn(2, 32, hidden_size)
        
        output, _ = attn(x)
        assert output.shape == x.shape

    def test_attention_with_cache(self):
        """Test attention with KV cache."""
        from light_moe.model import Attention

        hidden_size = 256
        num_heads = 8
        
        attn = Attention(hidden_size, num_heads)
        
        # First forward (prefill)
        x1 = torch.randn(2, 32, hidden_size)
        output1, cache = attn(x1, use_cache=True)
        
        # Second forward (decode with cache)
        x2 = torch.randn(2, 1, hidden_size)  # Single token
        output2, cache2 = attn(x2, past_key_value=cache, use_cache=True)
        
        assert output1.shape == x1.shape
        assert output2.shape == x2.shape
        # Cache should have grown
        assert cache2[0].size(2) == 33  # 32 + 1


class TestRoPE:
    """Test Rotary Position Embeddings."""

    def test_rope_shape(self):
        """Test RoPE output shape."""
        from light_moe.model import RotaryEmbedding

        dim = 64
        rope = RotaryEmbedding(dim, max_position_embeddings=128)
        
        x = torch.randn(2, 32, dim)
        cos, sin = rope(x)
        
        assert cos.shape[-1] == dim
        assert sin.shape[-1] == dim


class TestMoEBlock:
    """Test MoE Block implementation."""

    def test_moe_block_forward(self):
        """Test MoE block forward pass."""
        from light_moe.model import SparseMoEBlock

        hidden_size = 256
        intermediate_size = 512
        num_experts = 4
        top_k = 2
        
        moe = SparseMoEBlock(hidden_size, intermediate_size, num_experts, top_k)
        moe.eval()
        
        x = torch.randn(2, 16, hidden_size)
        output, aux_loss = moe(x)
        
        assert output.shape == x.shape
        assert aux_loss is None  # No aux loss in eval mode

    def test_moe_block_aux_loss(self):
        """Test MoE block with auxiliary loss."""
        from light_moe.model import SparseMoEBlock

        moe = SparseMoEBlock(256, 512, num_experts=4, top_k=2)
        moe.train()
        
        x = torch.randn(2, 16, 256)
        output, aux_loss = moe(x)
        
        assert output.shape == x.shape
        assert aux_loss is not None


class TestTransformer:
    """Test Transformer implementation."""

    def test_transformer_block(self):
        """Test single transformer block."""
        from light_moe.model import TransformerBlock, MoEConfig

        config = MoEConfig(
            hidden_size=256,
            intermediate_size=512,
            num_attention_heads=8,
            num_key_value_heads=2,
            num_experts=4,
            num_experts_per_token=2,
        )
        
        block = TransformerBlock(config)
        block.eval()
        
        x = torch.randn(2, 16, config.hidden_size)
        output, cache, aux_loss = block(x)
        
        assert output.shape == x.shape

    def test_moe_transformer(self):
        """Test complete MoE Transformer."""
        from light_moe.model import MoETransformer, MoEConfig

        config = MoEConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_experts=4,
            num_experts_per_token=2,
        )
        
        model = MoETransformer(config)
        model.eval()
        
        input_ids = torch.randint(0, 1000, (2, 8))
        logits, cache, aux_loss = model(input_ids)
        
        assert logits.shape == (2, 8, 1000)

    def test_transformer_generate(self):
        """Test transformer generation."""
        from light_moe.model import MoETransformer, MoEConfig

        config = MoEConfig(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_experts=2,
            num_experts_per_token=1,
        )
        
        model = MoETransformer(config)
        model.eval()
        
        # Initial input
        input_ids = torch.randint(0, 100, (1, 4))
        
        # Generate one token
        next_token, cache = model.generate_next_token(input_ids)
        
        assert next_token.shape == (1, 1)
        assert cache is not None
        assert len(cache) == config.num_hidden_layers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
