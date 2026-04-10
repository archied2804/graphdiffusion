"""
Tests for graph_diffusion.building_blocks.mlp
===============================================

Unit tests for MLP and SinusoidalTimeEmbedding.
"""

import torch
from context import graph_diffusion  # noqa: F401

from graph_diffusion.building_blocks.mlp import MLP, SinusoidalTimeEmbedding

# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class TestMLP:
    def test_output_shape(self):
        """Output shape must match (*, out_dim)."""
        torch.manual_seed(0)
        mlp = MLP(in_dim=16, hidden_dims=[32, 32], out_dim=8)
        x = torch.randn(10, 16)
        out = mlp(x)
        assert out.shape == (10, 8)

    def test_batch_dims_preserved(self):
        """Arbitrary leading batch dimensions should be preserved."""
        torch.manual_seed(0)
        mlp = MLP(in_dim=4, hidden_dims=[8], out_dim=6)
        x = torch.randn(3, 5, 4)
        out = mlp(x)
        assert out.shape == (3, 5, 6)

    def test_activation_relu(self):
        """MLP should accept relu activation without error."""
        torch.manual_seed(0)
        mlp = MLP(in_dim=4, hidden_dims=[8], out_dim=4, activation="relu")
        out = mlp(torch.randn(5, 4))
        assert out.shape == (5, 4)

    def test_activation_silu(self):
        """MLP should accept silu activation without error."""
        torch.manual_seed(0)
        mlp = MLP(in_dim=4, hidden_dims=[8], out_dim=4, activation="silu")
        out = mlp(torch.randn(5, 4))
        assert out.shape == (5, 4)

    def test_activation_gelu(self):
        """MLP should accept gelu activation without error."""
        torch.manual_seed(0)
        mlp = MLP(in_dim=4, hidden_dims=[8], out_dim=4, activation="gelu")
        out = mlp(torch.randn(5, 4))
        assert out.shape == (5, 4)

    def test_invalid_activation(self):
        """Unknown activation should raise ValueError."""
        try:
            MLP(in_dim=4, hidden_dims=[8], out_dim=4, activation="tanh")
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_residual_same_dim(self):
        """Residual connection should be active when in_dim == out_dim."""
        torch.manual_seed(0)
        mlp = MLP(in_dim=8, hidden_dims=[16], out_dim=8, residual=True)
        assert mlp.residual is True
        x = torch.randn(5, 8)
        out = mlp(x)
        assert out.shape == (5, 8)
        # Output should differ from non-residual version
        mlp_no_res = MLP(in_dim=8, hidden_dims=[16], out_dim=8, residual=False)
        mlp_no_res.load_state_dict(mlp.state_dict())
        out_no_res = mlp_no_res(x)
        # residual adds x, so out should equal out_no_res + x
        assert torch.allclose(out, out_no_res + x, atol=1e-6)

    def test_residual_diff_dim(self):
        """Residual should be inactive when in_dim != out_dim."""
        torch.manual_seed(0)
        mlp = MLP(in_dim=8, hidden_dims=[16], out_dim=4, residual=True)
        assert mlp.residual is False

    def test_layer_norm(self):
        """LayerNorm should be present when layer_norm=True."""
        torch.manual_seed(0)
        mlp = MLP(in_dim=4, hidden_dims=[8, 8], out_dim=4, layer_norm=True)
        # Count LayerNorm modules
        ln_count = sum(
            1 for m in mlp.net.modules() if isinstance(m, torch.nn.LayerNorm)
        )
        assert ln_count == 2  # one per hidden layer

    def test_no_layer_norm(self):
        """No LayerNorm should be present when layer_norm=False."""
        torch.manual_seed(0)
        mlp = MLP(in_dim=4, hidden_dims=[8, 8], out_dim=4, layer_norm=False)
        ln_count = sum(
            1 for m in mlp.net.modules() if isinstance(m, torch.nn.LayerNorm)
        )
        assert ln_count == 0

    def test_empty_hidden_dims(self):
        """MLP with no hidden layers should just be a single Linear."""
        torch.manual_seed(0)
        mlp = MLP(in_dim=4, hidden_dims=[], out_dim=8)
        x = torch.randn(5, 4)
        out = mlp(x)
        assert out.shape == (5, 8)


# ---------------------------------------------------------------------------
# SinusoidalTimeEmbedding
# ---------------------------------------------------------------------------


class TestSinusoidalTimeEmbedding:
    def test_output_shape_batch(self):
        """Output shape should be (B, embed_dim) for batched input."""
        torch.manual_seed(0)
        emb = SinusoidalTimeEmbedding(embed_dim=64)
        t = torch.tensor([0, 50, 100, 999])
        out = emb(t)
        assert out.shape == (4, 64)

    def test_output_shape_scalar(self):
        """Scalar input should produce (1, embed_dim)."""
        torch.manual_seed(0)
        emb = SinusoidalTimeEmbedding(embed_dim=32)
        t = torch.tensor(42)
        out = emb(t)
        assert out.shape == (1, 32)

    def test_different_timesteps_differ(self):
        """Different timesteps should produce different embeddings."""
        torch.manual_seed(0)
        emb = SinusoidalTimeEmbedding(embed_dim=64)
        t = torch.tensor([0, 500])
        out = emb(t)
        assert not torch.allclose(out[0], out[1])

    def test_deterministic(self):
        """Same input should produce same output (deterministic after init)."""
        torch.manual_seed(0)
        emb = SinusoidalTimeEmbedding(embed_dim=64)
        t = torch.tensor([10, 20, 30])
        out1 = emb(t)
        out2 = emb(t)
        assert torch.allclose(out1, out2)

    def test_embed_dim_128(self):
        """Should work with various embed_dim values."""
        torch.manual_seed(0)
        emb = SinusoidalTimeEmbedding(embed_dim=128)
        t = torch.tensor([0, 1, 2])
        out = emb(t)
        assert out.shape == (3, 128)
