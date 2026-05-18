"""
Tests for graph_diffusion.data.base_dataset
=============================================

Unit tests for BaseGraphDataset. Uses a minimal in-process concrete
subclass to verify the abstract contract without real data.
"""

import pytest
import torch
from context import graph_diffusion  # noqa: F401
from torch_geometric.data import Data

from graph_diffusion.data.base_dataset import BaseGraphDataset


class _MinimalDataset(BaseGraphDataset):
    """Minimal concrete subclass: returns a single fixed graph."""

    def _build_graphs(self) -> list[Data]:
        return [
            Data(
                x=torch.zeros(4, 1),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
            )
        ]


class TestBaseGraphDataset:
    def test_cannot_instantiate_abstract(self) -> None:
        """BaseGraphDataset is abstract — direct instantiation raises TypeError."""
        with pytest.raises(TypeError):
            BaseGraphDataset(root="/tmp/test_abstract")  # type: ignore[abstract]

    def test_concrete_subclass_length(self, tmp_path: pytest.TempPathFactory) -> None:
        """A concrete subclass with one graph has length 1."""
        ds = _MinimalDataset(root=str(tmp_path / "minimal"))
        assert len(ds) == 1

    def test_concrete_subclass_graph_attributes(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """The graph returned by _build_graphs is accessible via indexing."""
        ds = _MinimalDataset(root=str(tmp_path / "minimal"))
        graph = ds[0]
        assert graph.x.shape == (4, 1)

    def test_pre_transform_applied(self, tmp_path: pytest.TempPathFactory) -> None:
        """pre_transform is applied to each graph during processing."""
        from graph_diffusion.data.transforms import NormalizeNodeFeatures

        class _NonZeroDataset(BaseGraphDataset):
            def _build_graphs(self) -> list[Data]:
                return [
                    Data(
                        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
                        edge_index=torch.zeros(2, 0, dtype=torch.long),
                    )
                ]

        ds = _NonZeroDataset(
            root=str(tmp_path / "nonzero"),
            pre_transform=NormalizeNodeFeatures(),
        )
        graph = ds[0]
        mean = graph.x.mean(dim=0)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=0.15)
