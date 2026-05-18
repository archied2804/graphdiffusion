"""
graph_diffusion.data.base_dataset
===================================

Abstract base dataset for all boundary mesh datasets in this library.
Subclass this and implement ``_build_graphs`` to define a new dataset.
"""

import abc
from collections.abc import Callable

from torch_geometric.data import Data, InMemoryDataset

__all__ = [
    "BaseGraphDataset",
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

        Returns:
            list[Data]: The raw graph data objects.
        """

    @property
    def raw_file_names(self) -> list[str]:
        """No raw files required by default."""
        return []

    @property
    def processed_file_names(self) -> list[str]:
        """Single processed file containing the collated dataset."""
        return ["data.pt"]

    def download(self) -> None:
        """No-op — override in subclasses that download from remote sources."""

    def process(self) -> None:
        """Build graphs, apply pre_transform, collate and save."""
        graph_list = self._build_graphs()
        if self.pre_transform is not None:
            graph_list = [self.pre_transform(g) for g in graph_list]
        self.save(graph_list, self.processed_paths[0])
