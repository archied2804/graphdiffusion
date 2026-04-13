"""
Tests for graph_diffusion.data.circledataset
=============================================

Unit tests for UnitCircleDataset.
"""

import torch
from context import graph_diffusion  # noqa: F401

from graph_diffusion.data.circledataset import UnitCircleDataset


class TestUnitCircleDataset:
    def test_dataset_length(self, tmp_path):
        """Dataset should contain exactly n_graphs graphs."""
        torch.manual_seed(0)
        ds = UnitCircleDataset(
            root=str(tmp_path / "circle"),
            n_graphs=10,
            n_nodes=16,
            seed=42,
        )
        assert len(ds) == 10

    def test_node_feature_shape(self, tmp_path):
        """Node features x should have shape (n_nodes, 1)."""
        torch.manual_seed(0)
        n_nodes = 32
        ds = UnitCircleDataset(
            root=str(tmp_path / "circle"),
            n_graphs=5,
            n_nodes=n_nodes,
            seed=42,
        )
        for data in ds:
            assert data.x.shape == (n_nodes, 1)

    def test_pos_shape(self, tmp_path):
        """Reference positions should have shape (n_nodes, 2)."""
        torch.manual_seed(0)
        n_nodes = 32
        ds = UnitCircleDataset(
            root=str(tmp_path / "circle"),
            n_graphs=5,
            n_nodes=n_nodes,
            seed=42,
        )
        for data in ds:
            assert data.pos.shape == (n_nodes, 2)

    def test_pos_on_unit_circle(self, tmp_path):
        """Reference positions should lie on the unit circle."""
        torch.manual_seed(0)
        ds = UnitCircleDataset(
            root=str(tmp_path / "circle"),
            n_graphs=3,
            n_nodes=16,
            seed=42,
        )
        for data in ds:
            radii = torch.norm(data.pos, dim=1)
            assert torch.allclose(radii, torch.ones_like(radii), atol=1e-6)

    def test_global_shape(self, tmp_path):
        """Global attribute u should have shape (1, global_dim)."""
        torch.manual_seed(0)
        global_dim = 12
        ds = UnitCircleDataset(
            root=str(tmp_path / "circle"),
            n_graphs=3,
            n_nodes=16,
            global_dim=global_dim,
            seed=42,
        )
        for data in ds:
            assert data.u.shape == (1, global_dim)

    def test_radii_within_bounds(self, tmp_path):
        """All radial values should be within [r_min, r_max]."""
        torch.manual_seed(0)
        r_min, r_max = 0.5, 1.5
        ds = UnitCircleDataset(
            root=str(tmp_path / "circle"),
            n_graphs=20,
            n_nodes=64,
            r_min=r_min,
            r_max=r_max,
            amplitude_scale=0.3,
            seed=42,
        )
        for data in ds:
            assert data.x.min().item() >= r_min - 1e-6
            assert data.x.max().item() <= r_max + 1e-6

    def test_ring_connectivity(self, tmp_path):
        """Each node should have exactly 2 * k_neighbors edges."""
        torch.manual_seed(0)
        n_nodes = 16
        k_neighbors = 2
        ds = UnitCircleDataset(
            root=str(tmp_path / "circle"),
            n_graphs=3,
            n_nodes=n_nodes,
            k_neighbors=k_neighbors,
            seed=42,
        )
        expected_edges = 2 * n_nodes * k_neighbors
        for data in ds:
            assert data.edge_index.shape == (2, expected_edges)

    def test_ring_edge_index_valid(self, tmp_path):
        """All edge indices should be within [0, n_nodes)."""
        torch.manual_seed(0)
        n_nodes = 20
        ds = UnitCircleDataset(
            root=str(tmp_path / "circle"),
            n_graphs=3,
            n_nodes=n_nodes,
            seed=42,
        )
        for data in ds:
            assert data.edge_index.min().item() >= 0
            assert data.edge_index.max().item() < n_nodes

    def test_seed_reproducibility(self, tmp_path):
        """Same seed should produce identical datasets."""
        torch.manual_seed(0)
        ds1 = UnitCircleDataset(
            root=str(tmp_path / "circle1"),
            n_graphs=5,
            n_nodes=16,
            seed=123,
        )
        ds2 = UnitCircleDataset(
            root=str(tmp_path / "circle2"),
            n_graphs=5,
            n_nodes=16,
            seed=123,
        )
        for g1, g2 in zip(ds1, ds2, strict=True):
            assert torch.allclose(g1.x, g2.x)
            assert torch.equal(g1.edge_index, g2.edge_index)

    def test_different_seeds_produce_different_data(self, tmp_path):
        """Different seeds should produce different radial values."""
        torch.manual_seed(0)
        ds1 = UnitCircleDataset(
            root=str(tmp_path / "circle1"),
            n_graphs=5,
            n_nodes=16,
            seed=42,
        )
        ds2 = UnitCircleDataset(
            root=str(tmp_path / "circle2"),
            n_graphs=5,
            n_nodes=16,
            seed=99,
        )
        assert not torch.allclose(ds1[0].x, ds2[0].x)

    def test_invalid_n_graphs(self, tmp_path):
        """n_graphs < 1 should raise ValueError."""
        try:
            UnitCircleDataset(
                root=str(tmp_path / "circle"),
                n_graphs=0,
            )
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_invalid_n_nodes(self, tmp_path):
        """n_nodes < 3 should raise ValueError."""
        try:
            UnitCircleDataset(
                root=str(tmp_path / "circle"),
                n_nodes=2,
            )
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_invalid_r_range(self, tmp_path):
        """r_min >= r_max should raise ValueError."""
        try:
            UnitCircleDataset(
                root=str(tmp_path / "circle"),
                r_min=1.5,
                r_max=0.5,
            )
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_invalid_k_neighbors(self, tmp_path):
        """k_neighbors >= n_nodes should raise ValueError."""
        try:
            UnitCircleDataset(
                root=str(tmp_path / "circle"),
                n_nodes=8,
                k_neighbors=8,
            )
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_k_neighbors_connectivity_pattern(self, tmp_path):
        """Node 0 should be connected to nodes 1, n-1 for k=1."""
        torch.manual_seed(0)
        n_nodes = 8
        ds = UnitCircleDataset(
            root=str(tmp_path / "circle"),
            n_graphs=1,
            n_nodes=n_nodes,
            k_neighbors=1,
            seed=42,
        )
        data = ds[0]
        src, dst = data.edge_index

        # Find edges from node 0
        mask = src == 0
        neighbours = set(dst[mask].tolist())
        assert neighbours == {1, n_nodes - 1}
