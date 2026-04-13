"""
graph_diffusion.data.circledataset
===================================

Unit circle dataset for radial diffusion experiments.  Generates ring
graphs whose nodes lie on a perturbed unit circle, with radial
coordinates as the sole diffused node feature.
"""

from collections.abc import Callable

import numpy as np
import torch
from torch_geometric.data import Data

from graph_diffusion.data.dataset import BaseGraphDataset

__all__ = [
    "UnitCircleDataset",
]


class UnitCircleDataset(BaseGraphDataset):
    """Ring graphs on a Fourier-perturbed unit circle.

    Each graph consists of ``n_nodes`` nodes uniformly spaced in
    ``θ ∈ [0, 2π)`` with radial coordinates drawn from a random Fourier
    series:

    ``r(θ) = 1 + Σ_{n=1}^{M} (a_n sin(nθ) + b_n cos(nθ))``

    where ``a_n, b_n ~ N(0, amplitude_scale / n)``.

    Nodes are connected in a bidirectional ring to their ``k_neighbors``
    nearest neighbours in the θ direction on each side.

    **Data attributes per graph:**

    - ``x``: radial coordinate ``[r_i]``, shape ``(n_nodes, 1)``
    - ``pos``: reference unit-circle Cartesian positions
      ``[cos θ_i, sin θ_i]``, shape ``(n_nodes, 2)``
    - ``edge_index``: ring connectivity, shape ``(2, E)``
    - ``u``: zero-initialised global attribute, shape ``(1, global_dim)``

    Args:
        root (str): Root directory where the dataset should be saved.
        n_graphs (int): Number of graphs to generate. Defaults to ``2000``.
        n_nodes (int): Number of nodes per graph. Defaults to ``64``.
        n_fourier_modes (int): Number of Fourier modes ``M`` used for
            radial perturbation. Defaults to ``5``.
        amplitude_scale (float): Controls perturbation magnitude ``σ``.
            Defaults to ``0.15``.
        r_min (float): Minimum allowed radius (clamped). Defaults to ``0.5``.
        r_max (float): Maximum allowed radius (clamped). Defaults to ``1.5``.
        k_neighbors (int): Number of ring neighbours on each side of a node.
            Defaults to ``2``.
        global_dim (int): Dimensionality of the global attribute ``u``.
            Defaults to ``8``.
        transform (Callable | None): Runtime transform. Defaults to ``None``.
        pre_transform (Callable | None): Processing-time transform.
            Defaults to ``None``.
        seed (int): Random seed for reproducibility. Defaults to ``42``.

    Raises:
        ValueError: If ``n_graphs < 1``.
        ValueError: If ``n_nodes < 3``.
        ValueError: If ``n_fourier_modes < 1``.
        ValueError: If ``amplitude_scale <= 0``.
        ValueError: If ``r_min >= r_max``.
        ValueError: If ``k_neighbors < 1`` or ``k_neighbors >= n_nodes``.
        ValueError: If ``global_dim < 1``.
    """

    def __init__(
        self,
        root: str,
        n_graphs: int = 2000,
        n_nodes: int = 64,
        n_fourier_modes: int = 5,
        amplitude_scale: float = 0.15,
        r_min: float = 0.5,
        r_max: float = 1.5,
        k_neighbors: int = 2,
        global_dim: int = 8,
        transform: Callable[[Data], Data] | None = None,
        pre_transform: Callable[[Data], Data] | None = None,
        seed: int = 42,
    ) -> None:
        if n_graphs < 1:
            raise ValueError(f"n_graphs must be >= 1, got {n_graphs}")
        if n_nodes < 3:
            raise ValueError(f"n_nodes must be >= 3, got {n_nodes}")
        if n_fourier_modes < 1:
            raise ValueError(f"n_fourier_modes must be >= 1, got {n_fourier_modes}")
        if amplitude_scale <= 0:
            raise ValueError(f"amplitude_scale must be > 0, got {amplitude_scale}")
        if r_min >= r_max:
            raise ValueError(f"r_min must be < r_max, got r_min={r_min}, r_max={r_max}")
        if k_neighbors < 1 or k_neighbors >= n_nodes:
            raise ValueError(
                f"k_neighbors must satisfy 1 <= k < n_nodes, "
                f"got k_neighbors={k_neighbors}, n_nodes={n_nodes}"
            )
        if global_dim < 1:
            raise ValueError(f"global_dim must be >= 1, got {global_dim}")

        self.n_graphs = n_graphs
        self.n_nodes = n_nodes
        self.n_fourier_modes = n_fourier_modes
        self.amplitude_scale = amplitude_scale
        self.r_min = r_min
        self.r_max = r_max
        self.k_neighbors = k_neighbors
        self.global_dim = global_dim
        self.seed = seed

        super().__init__(root, transform=transform, pre_transform=pre_transform)

    def _build_graphs(self) -> list[Data]:
        """Generate Fourier-perturbed unit circle ring graphs.

        Returns:
            list[Data]: A list of ``n_graphs`` ring graph ``Data`` objects.
        """
        rng = np.random.default_rng(self.seed)
        graphs: list[Data] = []

        # Uniform angular spacing shared by all graphs
        theta = np.linspace(0.0, 2.0 * np.pi, self.n_nodes, endpoint=False)

        # Build ring edge_index once (shared topology)
        edge_index = self._build_ring_edge_index()

        for _ in range(self.n_graphs):
            # Sample Fourier coefficients with 1/n amplitude decay
            r = np.ones(self.n_nodes, dtype=np.float64)
            for mode in range(1, self.n_fourier_modes + 1):
                scale = self.amplitude_scale / mode
                a_n = rng.normal(0.0, scale)
                b_n = rng.normal(0.0, scale)
                r += a_n * np.sin(mode * theta) + b_n * np.cos(mode * theta)

            # Clamp to [r_min, r_max]
            r = np.clip(r, self.r_min, self.r_max)

            # Node features: radial coordinate only
            x = torch.tensor(r, dtype=torch.float32).unsqueeze(1)  # (N, 1)

            # Reference positions on the unit circle
            pos = torch.tensor(
                np.stack([np.cos(theta), np.sin(theta)], axis=1),
                dtype=torch.float32,
            )  # (N, 2)

            # Global attribute
            u = torch.zeros(1, self.global_dim)

            graphs.append(Data(x=x, edge_index=edge_index.clone(), pos=pos, u=u))

        return graphs

    def _build_ring_edge_index(self) -> torch.Tensor:
        """Build bidirectional ring edge index.

        Each node ``i`` is connected to its ``k_neighbors`` nearest
        neighbours on each side in the θ direction.

        Returns:
            torch.Tensor: Edge index of shape ``(2, 2 * n_nodes * k_neighbors)``.
        """
        sources: list[int] = []
        targets: list[int] = []

        for i in range(self.n_nodes):
            for k in range(1, self.k_neighbors + 1):
                # Forward neighbour (clockwise)
                j_fwd = (i + k) % self.n_nodes
                sources.append(i)
                targets.append(j_fwd)

                # Backward neighbour (counter-clockwise)
                j_bwd = (i - k) % self.n_nodes
                sources.append(i)
                targets.append(j_bwd)

        return torch.tensor([sources, targets], dtype=torch.long)
