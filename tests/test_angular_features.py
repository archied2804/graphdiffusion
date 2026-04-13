"""
Tests for ComputeAngularEdgeFeatures transform
================================================

Unit tests for the angular edge feature computation transform.
"""

import math

import torch
from context import graph_diffusion  # noqa: F401
from torch_geometric.data import Data

from graph_diffusion.data.transforms import ComputeAngularEdgeFeatures


class TestComputeAngularEdgeFeatures:
    def test_output_shape(self):
        """edge_attr should have shape (E, 2)."""
        torch.manual_seed(0)
        n_nodes = 8
        theta = torch.linspace(0, 2 * math.pi, n_nodes + 1)[:-1]
        pos = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
        # Simple ring: node i -> node (i+1)%n
        src = torch.arange(n_nodes)
        dst = (src + 1) % n_nodes
        edge_index = torch.stack([src, dst])

        data = Data(pos=pos, edge_index=edge_index)
        transform = ComputeAngularEdgeFeatures()
        result = transform(data)

        assert result.edge_attr is not None
        assert result.edge_attr.shape == (n_nodes, 2)

    def test_values_in_range(self):
        """sin and cos values should be in [-1, 1]."""
        torch.manual_seed(0)
        n_nodes = 16
        theta = torch.linspace(0, 2 * math.pi, n_nodes + 1)[:-1]
        pos = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
        src = torch.arange(n_nodes)
        dst = (src + 1) % n_nodes
        edge_index = torch.stack([src, dst])

        data = Data(pos=pos, edge_index=edge_index)
        transform = ComputeAngularEdgeFeatures()
        result = transform(data)

        assert result.edge_attr[:, 0].abs().max() <= 1.0 + 1e-6
        assert result.edge_attr[:, 1].abs().max() <= 1.0 + 1e-6

    def test_uniform_spacing_equal_delta_theta(self):
        """For uniformly spaced nodes, all angular differences should be equal."""
        torch.manual_seed(0)
        n_nodes = 32
        theta = torch.linspace(0, 2 * math.pi, n_nodes + 1)[:-1]
        pos = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
        src = torch.arange(n_nodes)
        dst = (src + 1) % n_nodes
        edge_index = torch.stack([src, dst])

        data = Data(pos=pos, edge_index=edge_index)
        transform = ComputeAngularEdgeFeatures()
        result = transform(data)

        # All edges should have same sin(Δθ) and cos(Δθ)
        sin_vals = result.edge_attr[:, 0]
        cos_vals = result.edge_attr[:, 1]
        assert torch.allclose(sin_vals, sin_vals[0].expand_as(sin_vals), atol=1e-5)
        assert torch.allclose(cos_vals, cos_vals[0].expand_as(cos_vals), atol=1e-5)

    def test_expected_delta_theta_value(self):
        """For 4 nodes on a circle, forward step should be Δθ = π/2."""
        torch.manual_seed(0)
        n_nodes = 4
        theta = torch.linspace(0, 2 * math.pi, n_nodes + 1)[:-1]
        pos = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
        src = torch.tensor([0])
        dst = torch.tensor([1])
        edge_index = torch.stack([src, dst])

        data = Data(pos=pos, edge_index=edge_index)
        transform = ComputeAngularEdgeFeatures()
        result = transform(data)

        # Δθ = π/2 → sin(π/2) = 1, cos(π/2) = 0
        assert torch.allclose(
            result.edge_attr[0],
            torch.tensor([1.0, 0.0]),
            atol=1e-5,
        )

    def test_raises_on_missing_pos(self):
        """Should raise ValueError when data.pos is None."""
        data = Data(
            edge_index=torch.tensor([[0], [1]]),
        )
        transform = ComputeAngularEdgeFeatures()
        try:
            transform(data)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_raises_on_missing_edge_index(self):
        """Should raise ValueError when data.edge_index is None."""
        data = Data(
            pos=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        )
        transform = ComputeAngularEdgeFeatures()
        try:
            transform(data)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_sin_cos_identity(self):
        """sin²(Δθ) + cos²(Δθ) should equal 1 for all edges."""
        torch.manual_seed(0)
        n_nodes = 16
        theta = torch.linspace(0, 2 * math.pi, n_nodes + 1)[:-1]
        pos = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
        src = torch.arange(n_nodes)
        dst = (src + 1) % n_nodes
        edge_index = torch.stack([src, dst])

        data = Data(pos=pos, edge_index=edge_index)
        transform = ComputeAngularEdgeFeatures()
        result = transform(data)

        norm_sq = result.edge_attr[:, 0] ** 2 + result.edge_attr[:, 1] ** 2
        assert torch.allclose(norm_sq, torch.ones_like(norm_sq), atol=1e-5)
