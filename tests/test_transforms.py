"""
Tests for graph_diffusion.data.transforms
==========================================

Unit tests for BaseTransform, NormalizeNodeFeatures, AddSelfLoops,
KNNGraph, and Compose.
"""

import pytest
import torch
from context import graph_diffusion  # noqa: F401
from torch_geometric.data import Data

from graph_diffusion.data.transforms import (
    AddSelfLoops,
    BaseTransform,
    Compose,
    KNNGraph,
    NormalizeNodeFeatures,
)

_has_torch_cluster = pytest.importorskip is not None
try:
    import torch_cluster  # noqa: F401

    _has_torch_cluster = True
except ImportError:
    _has_torch_cluster = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_graph(
    n_nodes: int = 10,
    node_dim: int = 4,
    seed: int = 0,
) -> Data:
    """Create a small test graph with features, positions, and edges."""
    torch.manual_seed(seed)
    x = torch.randn(n_nodes, node_dim)
    pos = torch.rand(n_nodes, 2)
    # Simple chain graph: 0→1→2→…
    src = torch.arange(0, n_nodes - 1, dtype=torch.long)
    dst = torch.arange(1, n_nodes, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)
    return Data(x=x, pos=pos, edge_index=edge_index)


# ---------------------------------------------------------------------------
# BaseTransform — abstract, cannot be instantiated
# ---------------------------------------------------------------------------


class TestBaseTransform:
    def test_cannot_instantiate(self):
        """BaseTransform is abstract and cannot be directly instantiated."""
        try:
            BaseTransform()  # type: ignore[abstract]
            raise AssertionError("Should have raised TypeError")
        except TypeError:
            pass


# ---------------------------------------------------------------------------
# NormalizeNodeFeatures
# ---------------------------------------------------------------------------


class TestNormalizeNodeFeatures:
    def test_output_shape(self):
        """Output shape must match input shape."""
        torch.manual_seed(0)
        data = _make_simple_graph()
        original_shape = data.x.shape
        result = NormalizeNodeFeatures()(data)
        assert result.x.shape == original_shape

    def test_zero_mean(self):
        """After normalisation, each feature column should have ~zero mean."""
        torch.manual_seed(0)
        data = _make_simple_graph(n_nodes=100, node_dim=8)
        result = NormalizeNodeFeatures()(data)
        col_means = result.x.mean(dim=0)
        assert torch.allclose(col_means, torch.zeros_like(col_means), atol=1e-5)

    def test_unit_variance(self):
        """After normalisation, each feature column should have ~unit std."""
        torch.manual_seed(0)
        data = _make_simple_graph(n_nodes=100, node_dim=8)
        result = NormalizeNodeFeatures()(data)
        col_stds = result.x.std(dim=0)
        assert torch.allclose(col_stds, torch.ones_like(col_stds), atol=1e-2)

    def test_raises_on_none_x(self):
        """Should raise ValueError when data.x is None."""
        data = Data(edge_index=torch.zeros(2, 0, dtype=torch.long))
        try:
            NormalizeNodeFeatures()(data)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_constant_features(self):
        """Constant features should not produce NaN (eps guards division)."""
        torch.manual_seed(0)
        data = Data(
            x=torch.ones(5, 3),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
        )
        result = NormalizeNodeFeatures()(data)
        assert not torch.isnan(result.x).any()

    def test_preserves_other_attrs(self):
        """Transform should not remove other Data attributes."""
        torch.manual_seed(0)
        data = _make_simple_graph()
        original_pos = data.pos.clone()
        original_edge_index = data.edge_index.clone()
        result = NormalizeNodeFeatures()(data)
        assert torch.equal(result.pos, original_pos)
        assert torch.equal(result.edge_index, original_edge_index)


# ---------------------------------------------------------------------------
# AddSelfLoops
# ---------------------------------------------------------------------------


class TestAddSelfLoops:
    def test_adds_correct_number_of_edges(self):
        """Self-loops should add exactly n_nodes edges."""
        torch.manual_seed(0)
        data = _make_simple_graph(n_nodes=5)
        n_edges_before = data.edge_index.size(1)
        n_nodes = data.x.size(0)
        result = AddSelfLoops()(data)
        assert result.edge_index.size(1) == n_edges_before + n_nodes

    def test_output_shape(self):
        """edge_index should still have 2 rows."""
        torch.manual_seed(0)
        data = _make_simple_graph()
        result = AddSelfLoops()(data)
        assert result.edge_index.size(0) == 2

    def test_self_loops_present(self):
        """Each node should appear as a self-loop (i, i) in edge_index."""
        torch.manual_seed(0)
        n_nodes = 5
        data = _make_simple_graph(n_nodes=n_nodes)
        result = AddSelfLoops()(data)
        src, dst = result.edge_index
        self_loop_mask = src == dst
        self_loop_nodes = src[self_loop_mask].unique()
        assert self_loop_nodes.numel() == n_nodes

    def test_with_edge_attr(self):
        """edge_attr should be extended with fill_value for self-loop edges."""
        torch.manual_seed(0)
        data = _make_simple_graph(n_nodes=5)
        n_edges = data.edge_index.size(1)
        data.edge_attr = torch.randn(n_edges, 3)
        result = AddSelfLoops(fill_value=0.0)(data)
        assert result.edge_attr.shape == (n_edges + 5, 3)

    def test_raises_on_none_edge_index(self):
        """Should raise ValueError when data.edge_index is None."""
        data = Data(x=torch.randn(5, 3))
        try:
            AddSelfLoops()(data)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# KNNGraph
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_torch_cluster, reason="torch-cluster not installed")
class TestKNNGraph:
    def test_output_edge_index_shape(self):
        """edge_index should have 2 rows."""
        torch.manual_seed(0)
        data = Data(
            x=torch.randn(20, 4),
            pos=torch.rand(20, 2),
        )
        result = KNNGraph(k=3)(data)
        assert result.edge_index.size(0) == 2

    def test_correct_number_of_edges(self):
        """k-NN with k neighbours should produce n_nodes * k edges."""
        torch.manual_seed(0)
        n_nodes = 20
        k = 3
        data = Data(
            x=torch.randn(n_nodes, 4),
            pos=torch.rand(n_nodes, 2),
        )
        result = KNNGraph(k=k)(data)
        assert result.edge_index.size(1) == n_nodes * k

    def test_with_loop(self):
        """When loop=True, self-loops should be included in edges."""
        torch.manual_seed(0)
        data = Data(
            x=torch.randn(10, 4),
            pos=torch.rand(10, 2),
        )
        result = KNNGraph(k=3, loop=True)(data)
        src, dst = result.edge_index
        self_loops = (src == dst).sum().item()
        assert self_loops > 0

    def test_raises_on_none_pos(self):
        """Should raise ValueError when data.pos is None."""
        data = Data(x=torch.randn(5, 3))
        try:
            KNNGraph(k=3)(data)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_preserves_node_features(self):
        """KNNGraph should not modify data.x."""
        torch.manual_seed(0)
        data = Data(
            x=torch.randn(10, 4),
            pos=torch.rand(10, 2),
        )
        original_x = data.x.clone()
        result = KNNGraph(k=3)(data)
        assert torch.equal(result.x, original_x)


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


class TestCompose:
    def test_applies_transforms_in_order(self):
        """Compose should apply transforms sequentially."""
        torch.manual_seed(0)
        n_nodes = 10
        data = _make_simple_graph(n_nodes=n_nodes, node_dim=4)
        n_edges_before = data.edge_index.size(1)
        transform = Compose(
            [
                AddSelfLoops(),
                NormalizeNodeFeatures(),
            ]
        )
        result = transform(data)
        # After AddSelfLoops: original edges + n_nodes self-loops
        assert result.edge_index.size(1) == n_edges_before + n_nodes
        # After NormalizeNodeFeatures: mean ~ 0
        col_means = result.x.mean(dim=0)
        assert torch.allclose(col_means, torch.zeros_like(col_means), atol=1e-5)

    @pytest.mark.skipif(not _has_torch_cluster, reason="torch-cluster not installed")
    def test_applies_transforms_with_knn(self):
        """Compose with KNNGraph should apply all transforms sequentially."""
        torch.manual_seed(0)
        data = Data(
            x=torch.randn(20, 4),
            pos=torch.rand(20, 2),
        )
        transform = Compose(
            [
                KNNGraph(k=3),
                AddSelfLoops(),
                NormalizeNodeFeatures(),
            ]
        )
        result = transform(data)
        # After KNNGraph: 20*3 = 60 edges
        # After AddSelfLoops: 60 + 20 = 80 edges
        assert result.edge_index.size(1) == 80
        col_means = result.x.mean(dim=0)
        assert torch.allclose(col_means, torch.zeros_like(col_means), atol=1e-5)

    def test_empty_compose(self):
        """Compose with empty list should return data unchanged."""
        torch.manual_seed(0)
        data = _make_simple_graph()
        original_x = data.x.clone()
        result = Compose([])(data)
        assert torch.equal(result.x, original_x)

    def test_single_transform(self):
        """Compose with one transform should behave like that transform alone."""
        torch.manual_seed(0)
        data = _make_simple_graph(n_nodes=50, node_dim=4)
        data_direct = _make_simple_graph(n_nodes=50, node_dim=4)

        result_composed = Compose([NormalizeNodeFeatures()])(data)
        result_direct = NormalizeNodeFeatures()(data_direct)
        assert torch.allclose(result_composed.x, result_direct.x)
