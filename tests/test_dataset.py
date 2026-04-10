"""
Tests for graph_diffusion.data.dataset
=======================================

Unit tests for BaseGraphDataset and SyntheticGraphDataset.
"""

import torch
from context import graph_diffusion  # noqa: F401

from graph_diffusion.data.dataset import BaseGraphDataset, SyntheticGraphDataset

# ---------------------------------------------------------------------------
# BaseGraphDataset — abstract, cannot be instantiated
# ---------------------------------------------------------------------------


class TestBaseGraphDataset:
    def test_cannot_instantiate(self):
        """BaseGraphDataset is abstract and cannot be directly instantiated."""
        try:
            BaseGraphDataset(root="/tmp/test_abstract")  # type: ignore[abstract]
            raise AssertionError("Should have raised TypeError")
        except TypeError:
            pass


# ---------------------------------------------------------------------------
# SyntheticGraphDataset
# ---------------------------------------------------------------------------


class TestSyntheticGraphDataset:
    def test_length(self, tmp_path):
        """Dataset should contain the requested number of graphs."""
        torch.manual_seed(0)
        n_graphs = 10
        ds = SyntheticGraphDataset(
            root=str(tmp_path / "syn"),
            n_graphs=n_graphs,
            n_nodes_range=(5, 10),
            node_feature_dim=4,
            k=3,
        )
        assert len(ds) == n_graphs

    def test_data_attributes(self, tmp_path):
        """Each Data object should have x, edge_index, pos, and u."""
        torch.manual_seed(0)
        ds = SyntheticGraphDataset(
            root=str(tmp_path / "syn"),
            n_graphs=5,
            n_nodes_range=(10, 15),
            node_feature_dim=4,
            k=3,
        )
        data = ds[0]
        assert data.x is not None
        assert data.edge_index is not None
        assert data.pos is not None
        assert data.u is not None

    def test_node_feature_dim(self, tmp_path):
        """Node features should have the requested dimensionality."""
        torch.manual_seed(0)
        node_dim = 6
        ds = SyntheticGraphDataset(
            root=str(tmp_path / "syn"),
            n_graphs=5,
            n_nodes_range=(10, 15),
            node_feature_dim=node_dim,
            k=3,
        )
        data = ds[0]
        assert data.x.size(1) == node_dim

    def test_global_dim(self, tmp_path):
        """Global attribute u should have shape (1, global_dim)."""
        torch.manual_seed(0)
        global_dim = 8
        ds = SyntheticGraphDataset(
            root=str(tmp_path / "syn"),
            n_graphs=5,
            n_nodes_range=(10, 15),
            node_feature_dim=4,
            k=3,
            global_dim=global_dim,
        )
        data = ds[0]
        assert data.u.shape == (1, global_dim)

    def test_pos_shape(self, tmp_path):
        """Positions should be 2D coordinates."""
        torch.manual_seed(0)
        ds = SyntheticGraphDataset(
            root=str(tmp_path / "syn"),
            n_graphs=5,
            n_nodes_range=(10, 15),
            node_feature_dim=4,
            k=3,
        )
        data = ds[0]
        assert data.pos.size(1) == 2

    def test_edge_index_shape(self, tmp_path):
        """edge_index should have 2 rows (COO format)."""
        torch.manual_seed(0)
        ds = SyntheticGraphDataset(
            root=str(tmp_path / "syn"),
            n_graphs=5,
            n_nodes_range=(10, 15),
            node_feature_dim=4,
            k=3,
        )
        data = ds[0]
        assert data.edge_index.size(0) == 2

    def test_node_count_in_range(self, tmp_path):
        """Number of nodes should fall within the specified range."""
        torch.manual_seed(0)
        lo, hi = 10, 20
        ds = SyntheticGraphDataset(
            root=str(tmp_path / "syn"),
            n_graphs=20,
            n_nodes_range=(lo, hi),
            node_feature_dim=4,
            k=3,
        )
        for i in range(len(ds)):
            n_nodes = ds[i].x.size(0)
            assert lo <= n_nodes <= hi, f"Graph {i}: n_nodes={n_nodes}"

    def test_reproducibility(self, tmp_path):
        """Two datasets with the same seed should produce identical data."""
        torch.manual_seed(0)
        kwargs = dict(
            n_graphs=5,
            n_nodes_range=(10, 15),
            node_feature_dim=4,
            k=3,
            seed=123,
        )
        ds1 = SyntheticGraphDataset(root=str(tmp_path / "syn1"), **kwargs)
        ds2 = SyntheticGraphDataset(root=str(tmp_path / "syn2"), **kwargs)
        for i in range(len(ds1)):
            assert torch.allclose(ds1[i].x, ds2[i].x)
            assert torch.equal(ds1[i].edge_index, ds2[i].edge_index)

    def test_invalid_n_graphs(self, tmp_path):
        """Should raise ValueError for n_graphs < 1."""
        try:
            SyntheticGraphDataset(root=str(tmp_path / "syn"), n_graphs=0)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_invalid_n_nodes_range(self, tmp_path):
        """Should raise ValueError for invalid n_nodes_range."""
        try:
            SyntheticGraphDataset(root=str(tmp_path / "syn"), n_nodes_range=(10, 5))
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_invalid_node_feature_dim(self, tmp_path):
        """Should raise ValueError for node_feature_dim < 1."""
        try:
            SyntheticGraphDataset(root=str(tmp_path / "syn"), node_feature_dim=0)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_invalid_k(self, tmp_path):
        """Should raise ValueError for k < 1."""
        try:
            SyntheticGraphDataset(root=str(tmp_path / "syn"), k=0)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_with_pre_transform(self, tmp_path):
        """pre_transform should be applied during processing."""
        torch.manual_seed(0)
        from graph_diffusion.data.transforms import NormalizeNodeFeatures

        ds = SyntheticGraphDataset(
            root=str(tmp_path / "syn"),
            n_graphs=5,
            n_nodes_range=(20, 30),
            node_feature_dim=4,
            k=3,
            pre_transform=NormalizeNodeFeatures(),
        )
        data = ds[0]
        # After normalisation, mean should be approximately zero
        col_means = data.x.mean(dim=0)
        assert torch.allclose(col_means, torch.zeros_like(col_means), atol=0.15)
