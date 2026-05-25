"""
Tests for graph_diffusion.model.pressure_head
==============================================

Unit tests for PressurePredictionHead: shape, gradient flow, and
permutation invariance.
"""

import pytest
import torch
from context import graph_diffusion  # noqa: F401

from graph_diffusion.model.pressure_head import PressurePredictionHead


def _make_head(in_dim: int = 3, out_dim: int = 8) -> PressurePredictionHead:
    return PressurePredictionHead(
        in_dim=in_dim,
        out_dim=out_dim,
        node_hidden=[16, 16],
        global_hidden=[16, 16],
        node_embed_dim=16,
    )


def test_invalid_in_dim() -> None:
    with pytest.raises(ValueError, match="in_dim"):
        PressurePredictionHead(
            in_dim=0, out_dim=8, node_hidden=[16], global_hidden=[16]
        )


def test_invalid_out_dim() -> None:
    with pytest.raises(ValueError, match="out_dim"):
        PressurePredictionHead(
            in_dim=3, out_dim=0, node_hidden=[16], global_hidden=[16]
        )


def test_output_shape_single_graph() -> None:
    torch.manual_seed(0)
    head = _make_head()
    n = 16
    x0 = torch.randn(n, 1)
    pos = torch.randn(n, 2)
    batch = torch.zeros(n, dtype=torch.long)
    out = head(x0, pos, batch)
    assert out.shape == (1, 8)


def test_output_shape_batched() -> None:
    torch.manual_seed(0)
    head = _make_head()
    n_per_graph = [12, 16, 8]
    x0 = torch.cat([torch.randn(n, 1) for n in n_per_graph], dim=0)
    pos = torch.cat([torch.randn(n, 2) for n in n_per_graph], dim=0)
    batch = torch.cat(
        [torch.full((n,), i, dtype=torch.long) for i, n in enumerate(n_per_graph)]
    )
    out = head(x0, pos, batch)
    assert out.shape == (3, 8)


def test_gradient_flows_to_x0() -> None:
    torch.manual_seed(0)
    head = _make_head()
    n = 16
    x0 = torch.randn(n, 1, requires_grad=True)
    pos = torch.randn(n, 2)
    batch = torch.zeros(n, dtype=torch.long)
    target = torch.zeros(1, 8)
    pred = head(x0, pos, batch)
    loss = (pred - target).pow(2).mean()
    loss.backward()
    assert x0.grad is not None
    assert x0.grad.abs().sum().item() > 0.0


def test_permutation_invariance() -> None:
    torch.manual_seed(0)
    head = _make_head()
    head.eval()
    n = 12
    x0 = torch.randn(n, 1)
    pos = torch.randn(n, 2)
    batch = torch.zeros(n, dtype=torch.long)
    out_a = head(x0, pos, batch)

    perm = torch.randperm(n)
    out_b = head(x0[perm], pos[perm], batch[perm])
    assert torch.allclose(out_a, out_b, atol=1e-5)
