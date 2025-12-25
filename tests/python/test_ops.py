"""
Tests for basic operators (Phase 1).
"""

import pytest
import torch


class TestRMSNorm:
    """Test RMSNorm implementation."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        from light_moe.ops import RMSNorm

        hidden_size = 4096
        norm = RMSNorm(hidden_size)
        
        x = torch.randn(2, 128, hidden_size)
        output = norm(x)
        
        assert output.shape == x.shape

    def test_normalized_output(self):
        """Test that output has approximately unit variance."""
        from light_moe.ops import RMSNorm

        hidden_size = 256
        norm = RMSNorm(hidden_size)
        
        x = torch.randn(4, 32, hidden_size)
        output = norm(x)
        
        # RMS of output should be close to 1 (before weight scaling)
        rms = output.pow(2).mean(-1).sqrt()
        # With learnable weight=1, RMS should be approximately 1
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    def test_different_dtypes(self):
        """Test with different input dtypes."""
        from light_moe.ops import RMSNorm

        hidden_size = 256
        norm = RMSNorm(hidden_size)
        
        for dtype in [torch.float32, torch.float16]:
            x = torch.randn(2, 32, hidden_size, dtype=dtype)
            norm_typed = norm.to(dtype)
            output = norm_typed(x)
            assert output.dtype == dtype


class TestActivations:
    """Test activation functions."""

    def test_silu(self):
        """Test SiLU activation."""
        from light_moe.ops import silu

        x = torch.randn(2, 32, 256)
        output = silu(x)
        
        # SiLU should have same shape
        assert output.shape == x.shape
        
        # SiLU(0) should be 0
        assert torch.allclose(silu(torch.zeros(1)), torch.zeros(1))

    def test_swiglu(self):
        """Test SwiGLU activation."""
        from light_moe.ops import swiglu

        x = torch.randn(2, 32, 256)
        gate = torch.randn(2, 32, 256)
        output = swiglu(x, gate)
        
        assert output.shape == x.shape

    def test_swiglu_module(self):
        """Test SwiGLU module."""
        from light_moe.ops import SwiGLU

        hidden_size = 256
        intermediate_size = 512
        
        ffn = SwiGLU(hidden_size, intermediate_size)
        x = torch.randn(2, 32, hidden_size)
        output = ffn(x)
        
        assert output.shape == x.shape


class TestMoEGate:
    """Test MoE Gate implementation."""

    def test_routing_output_shape(self):
        """Test routing output shapes."""
        from light_moe.ops import MoEGate

        hidden_size = 256
        num_experts = 8
        top_k = 2
        batch_size = 2
        seq_len = 32
        
        gate = MoEGate(hidden_size, num_experts, top_k)
        gate.eval()  # Disable aux loss computation
        
        x = torch.randn(batch_size, seq_len, hidden_size)
        output = gate(x, return_aux_loss=False)
        
        assert output.expert_indices.shape == (batch_size, seq_len, top_k)
        assert output.routing_weights.shape == (batch_size, seq_len, top_k)
        assert output.expert_counts.shape == (num_experts,)

    def test_routing_weights_sum_to_one(self):
        """Test that routing weights sum to 1 per token."""
        from light_moe.ops import MoEGate

        gate = MoEGate(256, num_experts=8, top_k=2)
        gate.eval()
        
        x = torch.randn(2, 32, 256)
        output = gate(x, return_aux_loss=False)
        
        weight_sums = output.routing_weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    def test_expert_indices_valid(self):
        """Test that expert indices are in valid range."""
        from light_moe.ops import MoEGate

        num_experts = 8
        gate = MoEGate(256, num_experts=num_experts, top_k=2)
        gate.eval()
        
        x = torch.randn(2, 32, 256)
        output = gate(x, return_aux_loss=False)
        
        assert (output.expert_indices >= 0).all()
        assert (output.expert_indices < num_experts).all()

    def test_aux_loss_computation(self):
        """Test auxiliary loss computation during training."""
        from light_moe.ops import MoEGate

        gate = MoEGate(256, num_experts=8, top_k=2, aux_loss_coef=0.01)
        gate.train()
        
        x = torch.randn(2, 32, 256)
        output = gate(x, return_aux_loss=True)
        
        assert output.aux_loss is not None
        assert output.aux_loss.numel() == 1
        assert output.aux_loss >= 0


class TestGroupedGemm:
    """Test Grouped GEMM implementation."""

    def test_grouped_gemm_list(self):
        """Test grouped GEMM with list of inputs."""
        from light_moe.ops import grouped_gemm

        num_experts = 4
        in_features = 256
        out_features = 512
        
        # Variable number of tokens per expert
        inputs = [
            torch.randn(32, in_features),
            torch.randn(16, in_features),
            torch.randn(48, in_features),
            torch.randn(24, in_features),
        ]
        weights = torch.randn(num_experts, out_features, in_features)
        
        outputs = grouped_gemm(inputs, weights)
        
        assert len(outputs) == num_experts
        for i, output in enumerate(outputs):
            assert output.shape == (inputs[i].size(0), out_features)

    def test_grouped_linear(self):
        """Test GroupedLinear module."""
        from light_moe.ops import GroupedLinear

        num_experts = 4
        in_features = 256
        out_features = 512
        
        layer = GroupedLinear(num_experts, in_features, out_features)
        
        inputs = [torch.randn(32, in_features) for _ in range(num_experts)]
        outputs = layer(inputs)
        
        assert len(outputs) == num_experts
        for output in outputs:
            assert output.shape[1] == out_features

    def test_permute_tokens(self):
        """Test token permutation for grouped GEMM."""
        from light_moe.ops.grouped_gemm import permute_tokens

        batch_size = 2
        seq_len = 8
        hidden_size = 64
        num_experts = 4
        top_k = 2
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        expert_indices = torch.randint(0, num_experts, (batch_size, seq_len, top_k))
        
        permuted, offsets, reverse_idx = permute_tokens(
            hidden_states, expert_indices, num_experts
        )
        
        expected_total = batch_size * seq_len * top_k
        assert permuted.shape == (expected_total, hidden_size)
        assert offsets.shape == (num_experts + 1,)
        assert reverse_idx.shape == (expected_total,)

    def test_unpermute_tokens(self):
        """Test token unpermutation after grouped GEMM."""
        from light_moe.ops.grouped_gemm import permute_tokens, unpermute_tokens

        batch_size = 2
        seq_len = 8
        hidden_size = 64
        num_experts = 4
        top_k = 2
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        expert_indices = torch.randint(0, num_experts, (batch_size, seq_len, top_k))
        routing_weights = torch.softmax(torch.randn(batch_size, seq_len, top_k), dim=-1)
        
        # Permute
        permuted, offsets, reverse_idx = permute_tokens(
            hidden_states, expert_indices, num_experts
        )
        
        # Simulate expert processing (identity for test)
        processed = permuted.clone()
        
        # Unpermute
        output = unpermute_tokens(processed, reverse_idx, routing_weights)
        
        assert output.shape == (batch_size, seq_len, hidden_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
