"""
graph_diffusion.data.transforms
================================

Graph transforms for pre-processing ``torch_geometric.data.Data`` objects.

Provides a thin abstraction layer (``BaseTransform``) over PyG's
``BaseTransform`` plus four concrete transforms used by the data pipeline.
"""

import abc

import torch
import torch_geometric.transforms
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.utils import add_self_loops as _add_self_loops

__all__ = [
    "BaseTransform",
    "NormalizeNodeFeatures",
    "AddSelfLoops",
    "KNNGraph",
    "Compose",
    "ComputeAngularEdgeFeatures",
]


class BaseTransform(torch_geometric.transforms.BaseTransform, abc.ABC):  # type: ignore[misc]
    """Abstract base class for all graph transforms in this library.

    Subclasses must implement ``__call__`` to transform a single
    ``torch_geometric.data.Data`` object.

    # TODO(extensibility): subclass this ABC and implement ``forward``
    # to add a new transform without modifying existing code.
    """

    @abc.abstractmethod
    def forward(self, data: Data) -> Data:
        """Transform a single graph Data object in-place or return a new one.

        Args:
            data (Data): A PyG ``Data`` object.

        Returns:
            Data: The transformed ``Data`` object.
        """


class NormalizeNodeFeatures(BaseTransform):
    """Standardise node features to zero mean and unit variance per feature dim.

    For each feature column, the transform subtracts the mean and divides by
    the standard deviation (plus a small epsilon for numerical stability).

    Args:
        eps (float): Small constant added to the standard deviation to avoid
            division by zero. Defaults to ``1e-8``.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

    def forward(self, data: Data) -> Data:
        """Normalise ``data.x`` to zero mean and unit variance per feature.

        Args:
            data (Data): A PyG ``Data`` object with ``data.x`` set.

        Returns:
            Data: The same ``Data`` object with ``data.x`` standardised.

        Raises:
            ValueError: If ``data.x`` is ``None``.
        """
        if data.x is None:
            raise ValueError("data.x must not be None for NormalizeNodeFeatures")
        mean = data.x.mean(dim=0, keepdim=True)
        std = data.x.std(dim=0, keepdim=True)
        data.x = (data.x - mean) / (std + self.eps)
        return data


class AddSelfLoops(BaseTransform):
    """Add self-loop edges to ``data.edge_index``.

    Delegates to ``torch_geometric.utils.add_self_loops``. If ``data.edge_attr``
    is present, self-loop edge attributes are filled with the specified value.

    Args:
        fill_value (float): Value used to fill self-loop edge attributes when
            ``data.edge_attr`` is not ``None``. Defaults to ``0.0``.
    """

    def __init__(self, fill_value: float = 0.0) -> None:
        self.fill_value = fill_value

    def forward(self, data: Data) -> Data:
        """Add self-loop edges to ``data.edge_index``.

        Args:
            data (Data): A PyG ``Data`` object with ``data.edge_index`` set.

        Returns:
            Data: The same ``Data`` object with self-loops added.

        Raises:
            ValueError: If ``data.edge_index`` is ``None``.
        """
        if data.edge_index is None:
            raise ValueError("data.edge_index must not be None for AddSelfLoops")
        num_nodes: int | None = None
        if data.x is not None:
            num_nodes = data.x.size(0)

        if data.edge_attr is not None:
            edge_index, edge_attr = _add_self_loops(
                data.edge_index,
                data.edge_attr,
                fill_value=self.fill_value,
                num_nodes=num_nodes,
            )
            data.edge_index = edge_index
            data.edge_attr = edge_attr
        else:
            edge_index, _ = _add_self_loops(data.edge_index, num_nodes=num_nodes)
            data.edge_index = edge_index

        return data


class KNNGraph(BaseTransform):
    """Build k-NN edge index from ``data.pos``.

    Delegates to ``torch_geometric.nn.knn_graph``.

    Args:
        k (int): Number of nearest neighbours. Defaults to ``6``.
        loop (bool): If ``True``, include self-loops in the k-NN graph.
            Defaults to ``False``.
    """

    def __init__(self, k: int = 6, loop: bool = False) -> None:
        self.k = k
        self.loop = loop

    def forward(self, data: Data) -> Data:
        """Compute k-NN edges from spatial positions and set ``data.edge_index``.

        Args:
            data (Data): A PyG ``Data`` object with ``data.pos`` set.

        Returns:
            Data: The same ``Data`` object with ``data.edge_index`` built
                from k-NN connectivity.

        Raises:
            ValueError: If ``data.pos`` is ``None``.
        """
        if data.pos is None:
            raise ValueError("data.pos must not be None for KNNGraph")
        data.edge_index = knn_graph(
            data.pos,
            k=self.k,
            batch=data.batch if hasattr(data, "batch") else None,
            loop=self.loop,
        )
        return data


class Compose(BaseTransform):
    """Apply a sequence of transforms in order.

    Args:
        transforms (list[BaseTransform]): Ordered list of transforms to apply.
    """

    def __init__(self, transforms: list[BaseTransform]) -> None:
        self.transforms = transforms

    def forward(self, data: Data) -> Data:
        """Apply each transform sequentially.

        Args:
            data (Data): A PyG ``Data`` object.

        Returns:
            Data: The ``Data`` object after all transforms have been applied.
        """
        for t in self.transforms:
            data = t(data)
        return data


class ComputeAngularEdgeFeatures(BaseTransform):
    """Compute angular edge features from node positions on a circle.

    For each edge ``(i, j)``, computes the angular difference
    ``Δθ = θ_j − θ_i`` (where ``θ = atan2(pos_y, pos_x)``) and sets
    ``data.edge_attr = [sin(Δθ), cos(Δθ)]``.

    This encodes relative angular separation in a rotation-invariant
    manner suitable for ring graphs on a unit circle.
    """

    def forward(self, data: Data) -> Data:
        """Compute angular edge features from ``data.pos``.

        Args:
            data (Data): A PyG ``Data`` object with ``data.pos`` (Cartesian
                positions) and ``data.edge_index`` set.

        Returns:
            Data: The same ``Data`` object with ``data.edge_attr`` set to
                shape ``(E, 2)`` containing ``[sin(Δθ), cos(Δθ)]``.

        Raises:
            ValueError: If ``data.pos`` is ``None``.
            ValueError: If ``data.edge_index`` is ``None``.
        """
        if data.pos is None:
            raise ValueError("data.pos must not be None for ComputeAngularEdgeFeatures")
        if data.edge_index is None:
            raise ValueError(
                "data.edge_index must not be None for " "ComputeAngularEdgeFeatures"
            )

        # Compute per-node angle from Cartesian positions
        theta = torch.atan2(data.pos[:, 1], data.pos[:, 0])  # (N,)

        # Compute angular difference per edge
        src, dst = data.edge_index
        delta_theta = theta[dst] - theta[src]  # (E,)

        data.edge_attr = torch.stack(
            [torch.sin(delta_theta), torch.cos(delta_theta)], dim=-1
        )  # (E, 2)

        return data
