"""Tests for graph_diffusion.building_blocks.feature_transforms."""

import pytest
import torch

from graph_diffusion.building_blocks.feature_transforms import (
    FeatureTransform,
    LogitNormTransform,
)


def test_logit_norm_transform_init_valid() -> None:
    t = LogitNormTransform(r_min=0.5, r_max=1.5)
    assert t.r_min == 0.5
    assert t.r_max == 1.5


def test_logit_norm_transform_init_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="r_min must be < r_max"):
        LogitNormTransform(r_min=1.5, r_max=0.5)


def test_logit_norm_transform_init_invalid_eps() -> None:
    with pytest.raises(ValueError, match="eps must be in"):
        LogitNormTransform(r_min=0.5, r_max=1.5, eps=0.0)


def test_logit_norm_is_abstract_subclass() -> None:
    assert issubclass(LogitNormTransform, FeatureTransform)


def test_logit_norm_forward_shape() -> None:
    torch.manual_seed(0)
    t = LogitNormTransform(r_min=0.5, r_max=1.5)
    x = torch.rand(64, 1) * (1.5 - 0.5) + 0.5  # uniform in [0.5, 1.5]
    z = t.forward(x)
    assert z.shape == x.shape


def test_logit_norm_inverse_shape() -> None:
    torch.manual_seed(0)
    t = LogitNormTransform(r_min=0.5, r_max=1.5)
    z = torch.randn(64, 1)
    r = t.inverse(z)
    assert r.shape == z.shape


def test_logit_norm_roundtrip() -> None:
    """forward followed by inverse should recover the original values."""
    torch.manual_seed(0)
    t = LogitNormTransform(r_min=0.5, r_max=1.5)
    x = torch.rand(100, 1) * (1.5 - 0.5) + 0.5
    recovered = t.inverse(t.forward(x))
    assert torch.allclose(
        recovered, x, atol=1e-5
    ), f"Max roundtrip error: {(recovered - x).abs().max().item():.2e}"


def test_logit_norm_inverse_bounds() -> None:
    """inverse should produce values strictly within [r_min, r_max]."""
    torch.manual_seed(0)
    t = LogitNormTransform(r_min=0.5, r_max=1.5)
    z = torch.randn(1000, 1) * 5  # wide range in logit space
    r = t.inverse(z)
    assert (r >= 0.5).all(), f"Min generated radius: {r.min().item():.4f}"
    assert (r <= 1.5).all(), f"Max generated radius: {r.max().item():.4f}"


def test_logit_norm_unit_circle_maps_to_zero() -> None:
    """Unit circle (r=1.0) should map to logit(0.5) = 0."""
    t = LogitNormTransform(r_min=0.5, r_max=1.5)
    x = torch.tensor([[1.0]])
    z = t.forward(x)
    assert abs(z.item()) < 1e-5, f"logit(0.5) should be 0, got {z.item()}"


def test_logit_norm_forward_unbounded() -> None:
    """forward should produce finite values even at near-boundary inputs."""
    t = LogitNormTransform(r_min=0.5, r_max=1.5, eps=1e-6)
    x = torch.tensor([[0.5001], [1.4999]])
    z = t.forward(x)
    assert torch.isfinite(z).all()


def test_logit_norm_multi_feature() -> None:
    """Transform should broadcast correctly over multi-feature tensors."""
    torch.manual_seed(0)
    t = LogitNormTransform(r_min=0.5, r_max=1.5)
    x = torch.rand(64, 3) * (1.5 - 0.5) + 0.5
    z = t.forward(x)
    r = t.inverse(z)
    assert torch.allclose(r, x, atol=1e-5)
