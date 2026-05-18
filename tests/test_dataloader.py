"""
Tests for graph_diffusion.data.dataloader
==========================================

Unit tests for GraphDataLoader.
"""

import torch
from context import graph_diffusion  # noqa: F401
from torch_geometric.data import Data

from graph_diffusion.data.base_dataset import BaseGraphDataset
from graph_diffusion.data.dataloader import GraphDataLoader


class _FixedGraphDataset(BaseGraphDataset):
    """Minimal dataset of identical graphs for loader testing."""

    def __init__(self, root: str, n_graphs: int = 50) -> None:
        self.n_graphs = n_graphs
        super().__init__(root)

    def _build_graphs(self) -> list[Data]:
        return [
            Data(
                x=torch.randn(8, 4),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                u=torch.zeros(1, 8),
            )
            for _ in range(self.n_graphs)
        ]


# ---------------------------------------------------------------------------
# Helper: small dataset fixture
# ---------------------------------------------------------------------------


def _make_dataset(tmp_path, n_graphs=50):
    """Create a small dataset for testing."""
    return _FixedGraphDataset(root=str(tmp_path / "fixed"), n_graphs=n_graphs)


# ---------------------------------------------------------------------------
# GraphDataLoader
# ---------------------------------------------------------------------------


class TestGraphDataLoader:
    def test_split_sizes(self, tmp_path):
        """Train + val should cover the entire dataset."""
        torch.manual_seed(0)
        n_graphs = 50
        val_split = 0.2
        ds = _make_dataset(tmp_path, n_graphs=n_graphs)
        loader = GraphDataLoader(ds, batch_size=8, val_split=val_split)

        n_val = max(1, int(n_graphs * val_split))
        n_train = n_graphs - n_val

        train_count = sum(1 for _ in loader.train_loader().dataset)
        val_count = sum(1 for _ in loader.val_loader().dataset)
        assert train_count == n_train
        assert val_count == n_val

    def test_train_loader_returns_batches(self, tmp_path):
        """train_loader should yield batched Data objects."""
        torch.manual_seed(0)
        ds = _make_dataset(tmp_path, n_graphs=20)
        loader = GraphDataLoader(ds, batch_size=4, val_split=0.2)
        batch = next(iter(loader.train_loader()))
        assert hasattr(batch, "x")
        assert hasattr(batch, "batch")
        assert batch.x is not None

    def test_val_loader_returns_batches(self, tmp_path):
        """val_loader should yield batched Data objects."""
        torch.manual_seed(0)
        ds = _make_dataset(tmp_path, n_graphs=20)
        loader = GraphDataLoader(ds, batch_size=4, val_split=0.2)
        batch = next(iter(loader.val_loader()))
        assert hasattr(batch, "x")
        assert batch.x is not None

    def test_batch_has_batch_vector(self, tmp_path):
        """Batched Data should include a batch assignment vector."""
        torch.manual_seed(0)
        ds = _make_dataset(tmp_path, n_graphs=20)
        loader = GraphDataLoader(ds, batch_size=4, val_split=0.2)
        batch = next(iter(loader.train_loader()))
        assert batch.batch is not None
        # batch vector should assign nodes to graphs 0..batch_size-1
        unique_graphs = batch.batch.unique()
        assert len(unique_graphs) <= 4

    def test_reproducible_split(self, tmp_path):
        """Two loaders with the same seed should have identical splits."""
        torch.manual_seed(0)
        ds = _make_dataset(tmp_path, n_graphs=20)
        loader1 = GraphDataLoader(ds, batch_size=4, val_split=0.2, seed=99)
        loader2 = GraphDataLoader(ds, batch_size=4, val_split=0.2, seed=99)

        for d1, d2 in zip(
            loader1.val_loader().dataset,
            loader2.val_loader().dataset,
            strict=True,
        ):
            assert torch.equal(d1.x, d2.x)

    def test_invalid_batch_size(self, tmp_path):
        """Should raise ValueError for batch_size < 1."""
        torch.manual_seed(0)
        ds = _make_dataset(tmp_path, n_graphs=10)
        try:
            GraphDataLoader(ds, batch_size=0)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_invalid_val_split_zero(self, tmp_path):
        """Should raise ValueError for val_split == 0."""
        torch.manual_seed(0)
        ds = _make_dataset(tmp_path, n_graphs=10)
        try:
            GraphDataLoader(ds, val_split=0.0)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_invalid_val_split_one(self, tmp_path):
        """Should raise ValueError for val_split == 1."""
        torch.manual_seed(0)
        ds = _make_dataset(tmp_path, n_graphs=10)
        try:
            GraphDataLoader(ds, val_split=1.0)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_invalid_num_workers(self, tmp_path):
        """Should raise ValueError for num_workers < 0."""
        torch.manual_seed(0)
        ds = _make_dataset(tmp_path, n_graphs=10)
        try:
            GraphDataLoader(ds, num_workers=-1)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass
