"""
graph_diffusion.data.dataset
=============================

Abstract base dataset and a concrete synthetic graph dataset built on
``torch_geometric.data.InMemoryDataset``.
"""

import abc
from collections.abc import Callable

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import knn_graph

__all__ = [
    "BaseGraphDataset",
    "SyntheticGraphDataset",
]


class BaseGraphDataset(InMemoryDataset, abc.ABC):  # type: ignore[misc]
    """Abstract base class for all graph datasets in this library.

    Subclasses must implement ``_build_graphs`` to define the concrete
    list of ``Data`` objects that constitute the dataset.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (Callable | None): A transform applied to each ``Data``
            object at access time. Defaults to ``None``.
        pre_transform (Callable | None): A transform applied once during
            ``process()`` before saving. Defaults to ``None``.

    # TODO(extensibility): subclass ``BaseGraphDataset`` and implement
    # ``_build_graphs`` to add a new dataset without modifying existing code.
    """

    def __init__(
        self,
        root: str,
        transform: Callable[[Data], Data] | None = None,
        pre_transform: Callable[[Data], Data] | None = None,
    ) -> None:
        super().__init__(root, transform=transform, pre_transform=pre_transform)
        self.load(self.processed_paths[0])

    @abc.abstractmethod
    def _build_graphs(self) -> list[Data]:
        """Construct and return the list of graph Data objects.

        Subclasses implement this to define the actual dataset.
        The parent class calls this during ``process()`` and stores results
        via ``self.save(graph_list, ...)``.

        Returns:
            list[Data]: The raw graph data objects.
        """

    @property
    def raw_file_names(self) -> list[str]:
        """No raw files needed for programmatically generated datasets."""
        return []

    @property
    def processed_file_names(self) -> list[str]:
        """Single processed file containing the collated dataset."""
        return ["data.pt"]

    def download(self) -> None:
        """No-op — synthetic datasets do not require downloads."""

    def process(self) -> None:
        """Build graphs, apply pre_transform, collate and save."""
        graph_list = self._build_graphs()

        if self.pre_transform is not None:
            graph_list = [self.pre_transform(g) for g in graph_list]

        self.save(graph_list, self.processed_paths[0])


class SyntheticGraphDataset(BaseGraphDataset):
    """Concrete baseline dataset of random geometric graphs.

    Generates random k-NN graphs with Gaussian node features. Provides a
    runnable demo without any external data files.

    Args:
        root (str): Root directory where the dataset should be saved.
        n_graphs (int): Number of graphs to generate. Defaults to ``1000``.
        n_nodes_range (tuple[int, int]): Range ``(lo, hi)`` for sampling the
            number of nodes per graph from ``Uniform(lo, hi)``.
            Defaults to ``(20, 50)``.
        node_feature_dim (int): Dimensionality of random node features.
            Defaults to ``8``.
        k (int): Number of nearest neighbours for k-NN connectivity.
            Defaults to ``6``.
        global_dim (int): Dimensionality of the global graph attribute ``u``.
            Defaults to ``16``.
        transform (Callable | None): Runtime transform. Defaults to ``None``.
        pre_transform (Callable | None): Processing-time transform.
            Defaults to ``None``.
        seed (int): Random seed for reproducibility. Defaults to ``42``.

    Raises:
        ValueError: If ``n_graphs < 1``.
        ValueError: If ``n_nodes_range[0] < 1`` or
            ``n_nodes_range[1] < n_nodes_range[0]``.
        ValueError: If ``node_feature_dim < 1``.
        ValueError: If ``k < 1``.
        ValueError: If ``global_dim < 1``.
    """

    def __init__(
        self,
        root: str,
        n_graphs: int = 1000,
        n_nodes_range: tuple[int, int] = (20, 50),
        node_feature_dim: int = 8,
        k: int = 6,
        global_dim: int = 16,
        transform: Callable[[Data], Data] | None = None,
        pre_transform: Callable[[Data], Data] | None = None,
        seed: int = 42,
    ) -> None:
        if n_graphs < 1:
            raise ValueError(f"n_graphs must be >= 1, got {n_graphs}")
        if n_nodes_range[0] < 1 or n_nodes_range[1] < n_nodes_range[0]:
            raise ValueError(
                f"n_nodes_range must satisfy 1 <= lo <= hi, got {n_nodes_range}"
            )
        if node_feature_dim < 1:
            raise ValueError(f"node_feature_dim must be >= 1, got {node_feature_dim}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if global_dim < 1:
            raise ValueError(f"global_dim must be >= 1, got {global_dim}")

        self.n_graphs = n_graphs
        self.n_nodes_range = n_nodes_range
        self.node_feature_dim = node_feature_dim
        self.k = k
        self.global_dim = global_dim
        self.seed = seed

        super().__init__(root, transform=transform, pre_transform=pre_transform)

    def _build_graphs(self) -> list[Data]:
        """Generate random geometric graphs with Gaussian node features.

        Returns:
            list[Data]: A list of ``n_graphs`` random graph ``Data`` objects.
        """
        rng = np.random.default_rng(self.seed)
        graphs: list[Data] = []

        for _ in range(self.n_graphs):
            n_nodes = int(
                rng.integers(self.n_nodes_range[0], self.n_nodes_range[1] + 1)
            )

            # Spatial positions ~ Uniform([0, 1]^2)
            pos = torch.tensor(rng.uniform(size=(n_nodes, 2)), dtype=torch.float32)

            # k-NN edges from pos
            edge_index = knn_graph(pos, k=self.k, loop=False)

            # Node features ~ N(0, 1)
            x = torch.tensor(
                rng.standard_normal(size=(n_nodes, self.node_feature_dim)),
                dtype=torch.float32,
            )

            # Global attribute initialised to zeros
            u = torch.zeros(1, self.global_dim)

            graphs.append(Data(x=x, edge_index=edge_index, pos=pos, u=u))

        return graphs
