"""
graph_diffusion.data.circledataset
===================================

Unit circle dataset for radial diffusion experiments.  Generates ring
graphs whose nodes lie on a perturbed unit circle, with radial
coordinates as the sole diffused node feature.
"""

from collections.abc import Callable
from typing import cast

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

    When ``include_curvature`` or ``include_arc_length`` is ``True``, the
    node feature vector ``x`` is extended:

    - ``include_curvature=True``:  appends signed curvature ``κ_i`` computed
      via 2-D finite differences on the Cartesian arc ``(x_i, y_i)``.
    - ``include_arc_length=True``: appends the normalised cumulative
      arc-length fraction ``s_i / L ∈ [0, 1]``.

    The resulting ``x`` shape is:

    +---------------------------+------------+
    | flags                     | x shape    |
    +===========================+============+
    | neither                   | (N, 1)     |
    +---------------------------+------------+
    | curvature only            | (N, 2)     |
    +---------------------------+------------+
    | arc-length only           | (N, 2)     |
    +---------------------------+------------+
    | both                      | (N, 3)     |
    +---------------------------+------------+

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
        include_curvature (bool): If ``True``, append per-node curvature
            to ``x``. Defaults to ``False``.
        include_arc_length (bool): If ``True``, append normalised cumulative
            arc-length fraction to ``x``. Defaults to ``False``.
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
        include_curvature: bool = False,
        include_arc_length: bool = False,
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
        self.include_curvature = include_curvature
        self.include_arc_length = include_arc_length
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

            # Cartesian positions of perturbed curve for geometric features
            xc = r * np.cos(theta)
            yc = r * np.sin(theta)

            # Node features: start with radial coordinate
            features = [r]

            if self.include_curvature:
                features.append(self._compute_curvature(xc, yc))

            if self.include_arc_length:
                features.append(self._compute_arc_length_fraction(xc, yc))

            x = torch.tensor(
                np.stack(features, axis=1), dtype=torch.float32
            )  # (N, F)

            # Reference positions on the unit circle
            pos = torch.tensor(
                np.stack([np.cos(theta), np.sin(theta)], axis=1),
                dtype=torch.float32,
            )  # (N, 2)

            # Global attribute
            u = torch.zeros(1, self.global_dim)

            graphs.append(Data(x=x, edge_index=edge_index.clone(), pos=pos, u=u))

        return graphs

    @staticmethod
    def _compute_curvature(xc: np.ndarray, yc: np.ndarray) -> np.ndarray:
        """Compute 2-D curvature via periodic central finite differences.

        Uses the standard signed curvature formula for a parametric planar
        curve:  κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)

        First and second derivatives are estimated with periodic central
        differences so the ring boundary is handled correctly.

        Args:
            xc: x-coordinates of the perturbed curve, shape ``(N,)``.
            yc: y-coordinates, shape ``(N,)``.

        Returns:
            np.ndarray: Curvature values, shape ``(N,)``.
        """
        # Periodic first differences: proportional to x', y'
        dx = np.roll(xc, -1) - np.roll(xc, 1)
        dy = np.roll(yc, -1) - np.roll(yc, 1)

        # Periodic second differences: proportional to x'', y''
        d2x = np.roll(xc, -1) - 2.0 * xc + np.roll(xc, 1)
        d2y = np.roll(yc, -1) - 2.0 * yc + np.roll(yc, 1)

        # Cross product of first and second derivatives (scaled numerator)
        # and squared speed (scaled denominator) — Δθ factors cancel exactly
        numerator = np.abs(dx * d2y - dy * d2x)
        denominator = (dx**2 + dy**2 + 1e-10) ** 1.5

        return cast(np.ndarray, (4.0 * numerator / denominator).astype(np.float64))

    @staticmethod
    def _compute_arc_length_fraction(
        xc: np.ndarray, yc: np.ndarray
    ) -> np.ndarray:
        """Compute normalised cumulative arc-length fraction s_i / L.

        Arc-length elements are estimated from chord lengths between
        consecutive Cartesian positions with periodic wrap-around.

        Args:
            xc: x-coordinates of the perturbed curve, shape ``(N,)``.
            yc: y-coordinates, shape ``(N,)``.

        Returns:
            np.ndarray: Normalised arc-length fractions in ``[0, 1]``,
                shape ``(N,)``.
        """
        # Chord length to the next node (periodic)
        ds = np.sqrt(
            (np.roll(xc, -1) - xc) ** 2 + (np.roll(yc, -1) - yc) ** 2
        )
        s = np.cumsum(ds)
        total = s[-1]
        # Shift so node 0 starts at 0 (cumsum of ds gives s_1…s_N; s_0 = 0)
        s_shifted = np.concatenate([[0.0], s[:-1]])
        return cast(np.ndarray, (s_shifted / (total + 1e-10)).astype(np.float64))

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
