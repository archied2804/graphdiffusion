"""
graph_diffusion.data.pOnEllipseConditional
============================================

Pressure-conditioned variant of :class:`pOnEllipseDataset` for inverse
shape design (EXP-020).

Each graph carries the same boundary topology as
:class:`pOnEllipseDataset`, plus a target pressure descriptor used as
diffusion conditioning:

- ``Data.cond`` — shape ``(1, K)``. The first ``K`` real DCT-II
  coefficients of the time-averaged surface pressure curve
  :math:`\\langle C_p\\rangle(x/c)` sorted by chordwise coordinate.
  Fixed-size and source-agnostic, so an aerofoil :math:`C_p` curve
  encoded the same way can drive sampling at inference time.
- ``Data.cp_nodal`` — shape ``(N, 1)`` when ``cond_mode="nodal"``.
  Per-node time-averaged Cp, for the nodal-conditioning extensibility
  path. Otherwise unset.

The base class' ``Data.x`` / ``Data.pos`` / ``Data.u`` / ``Data.edge_index``
(and ``Data.p_cond`` if the parent feature mode sets it) are preserved
unchanged — this subclass only attaches additional conditioning
attributes.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from graph_diffusion.data.pOnEllipse import (
    _H5_COL_P_START,
    _H5_COL_X,
    _H5_COL_Y,
    _H5_N_TIMESTEPS,
    _load_h5_raw,
    _node_count,
    pOnEllipseDataset,
)

__all__ = [
    "pOnEllipseConditionalDataset",
    "dct_ii",
    "dct_ii_inverse",
    "N_CP_GRID_DEFAULT",
]


_VALID_COND_MODES = ("fourier", "nodal", "fourier_dual")
N_CP_GRID_DEFAULT = 64  # interpolation grid for fourier_dual upper/lower halves


def dct_ii(x: np.ndarray, k_modes: int) -> np.ndarray:
    """Real DCT-II of a 1-D signal, truncated to ``k_modes`` coefficients.

    Uses the canonical type-II definition

        X[k] = sum_{n=0..N-1} x[n] * cos(pi * (2n+1) * k / (2N))

    Orthonormalised: multiplied by ``sqrt(2/N)`` for ``k>0`` and
    ``sqrt(1/N)`` for ``k=0``, so the encoding is scale-invariant under
    a change in sample count ``N``. This is what makes the resulting
    descriptor source-agnostic across different node counts.

    Args:
        x (np.ndarray): Real input signal, shape ``(N,)``.
        k_modes (int): Number of DCT coefficients to retain.

    Returns:
        np.ndarray: First ``k_modes`` DCT-II coefficients, shape
            ``(k_modes,)``, dtype ``float32``.
    """
    n_samples = x.shape[0]
    n_idx = np.arange(n_samples, dtype=np.float32)
    k_idx = np.arange(k_modes, dtype=np.float32)[:, None]
    basis = np.cos(np.pi * (2.0 * n_idx + 1.0) * k_idx / (2.0 * n_samples))
    coeffs = basis @ x.astype(np.float32)
    norm = np.full(k_modes, np.sqrt(2.0 / n_samples), dtype=np.float32)
    norm[0] = np.sqrt(1.0 / n_samples)
    return (coeffs * norm).astype(np.float32)  # type: ignore[no-any-return]


def dct_ii_inverse(coeffs: np.ndarray, n_samples: int) -> np.ndarray:
    """Inverse of :func:`dct_ii`: reconstruct a signal from K DCT-II modes.

    Uses the same orthonormalised type-II basis as :func:`dct_ii`, so that
    for any K-band-limited signal ``s`` of length ``N``::

        dct_ii(dct_ii_inverse(c, N), K) ≈ c     (round-trip up to float)

    Args:
        coeffs (np.ndarray): The first ``K`` DCT-II coefficients, shape
            ``(K,)``.
        n_samples (int): Length ``N`` of the reconstructed signal.

    Returns:
        np.ndarray: Reconstructed signal of length ``n_samples``, dtype
            ``float32``.
    """
    k_modes = coeffs.shape[0]
    n_idx = np.arange(n_samples, dtype=np.float32)
    k_idx = np.arange(k_modes, dtype=np.float32)[:, None]
    basis = np.cos(np.pi * (2.0 * n_idx + 1.0) * k_idx / (2.0 * n_samples))
    norm = np.full(k_modes, np.sqrt(2.0 / n_samples), dtype=np.float32)
    norm[0] = np.sqrt(1.0 / n_samples)
    return ((coeffs.astype(np.float32) * norm) @ basis).astype(np.float32)  # type: ignore[no-any-return]


class pOnEllipseConditionalDataset(pOnEllipseDataset):
    """Pressure-conditioned ellipse dataset for inverse design.

    Reads the same HDF5 file as :class:`pOnEllipseDataset` but additionally
    derives a steady-state pressure descriptor from the 101 unsteady
    timesteps (columns 4–104) and attaches it as conditioning.

    Args:
        root (str): Root directory for downloading and caching.
        cond_mode (str): One of ``"fourier"`` (default) — store
            ``Data.cond`` as ``K`` DCT modes of ``⟨Cp⟩(x/c)``; or
            ``"nodal"`` — additionally store per-node ``Data.cp_nodal``
            (mean Cp per node) for node-level conditioning experiments.
        k_modes (int): Number of DCT coefficients to retain in
            ``Data.cond``. Defaults to ``8``.
        feature_mode (str): Coordinate representation for the diffused
            node feature. Same options as the parent class. Defaults to
            ``"radial_norm"``.
        split (str): ``"train"`` or ``"test"``.
        n_samples (int | None): Limit the number of simulations loaded.
        k_neighbors (int): Ring connectivity. Defaults to ``2``.
        global_dim (int): Global attribute width.
        coord_scale (float | None): Global scale for ``"cartesian"`` mode.
        transform (Callable | None): Runtime transform.
        pre_transform (Callable | None): Processing-time transform.

    Raises:
        ValueError: If ``cond_mode`` is not one of the supported strings.
        ValueError: If ``k_modes < 1``.
    """

    def __init__(
        self,
        root: str,
        cond_mode: str = "fourier",
        k_modes: int = 8,
        feature_mode: str = "radial_norm",
        split: str = "train",
        n_samples: int | None = None,
        k_neighbors: int = 2,
        global_dim: int = 8,
        coord_scale: float | None = None,
        variant: str = "default",
        transform: Callable[[Data], Data] | None = None,
        pre_transform: Callable[[Data], Data] | None = None,
    ) -> None:
        if cond_mode not in _VALID_COND_MODES:
            raise ValueError(
                f"cond_mode must be one of {_VALID_COND_MODES}, got '{cond_mode}'"
            )
        if k_modes < 1:
            raise ValueError(f"k_modes must be >= 1, got {k_modes}")

        self.cond_mode = cond_mode
        self.k_modes = k_modes

        super().__init__(
            root=root,
            feature_mode=feature_mode,
            split=split,
            n_samples=n_samples,
            k_neighbors=k_neighbors,
            global_dim=global_dim,
            coord_scale=coord_scale,
            variant=variant,
            transform=transform,
            pre_transform=pre_transform,
        )

    @property
    def processed_file_names(self) -> list[str]:
        # Include `variant` so the default (AoA=0) and aoa10 caches never
        # collide — they share every other field but hold different data.
        suffix = f"_{self.variant}" if self.variant != "default" else ""
        return [
            f"data_cond_{self.cond_mode}_K{self.k_modes}"
            f"_{self.feature_mode}_{self.split}_k{self.k_neighbors}{suffix}.pt"
        ]

    def _build_graphs(self) -> list[Data]:
        """Build graphs and attach pressure conditioning.

        Returns:
            list[Data]: One graph per CFD simulation with ``Data.cond``
                (and optionally ``Data.p_cond``) set.
        """
        graphs = super()._build_graphs()

        h5_path = Path(self.raw_dir) / self.raw_file_names[0]
        raw = _load_h5_raw(h5_path)

        for i, graph in enumerate(graphs):
            sample = raw[i]
            n_nodes = _node_count(sample)
            data = sample[:n_nodes]

            # Steady-state mean over the 101 unsteady timesteps
            cp = data[
                :, _H5_COL_P_START : _H5_COL_P_START + _H5_N_TIMESTEPS
            ]  # (N, 101)
            cp_mean = cp.mean(axis=-1).astype(np.float32)  # (N,)

            # Order by chordwise coordinate to form Cp(x/c)
            xc = data[:, _H5_COL_X].astype(np.float32)
            xc = xc - xc.mean()
            x_min = float(xc.min())
            x_max = float(xc.max())
            chord = max(x_max - x_min, 1e-10)
            x_over_c = (xc - x_min) / chord

            if self.cond_mode == "fourier_dual":
                yc = (
                    data[:, _H5_COL_Y].astype(np.float32)
                    - data[:, _H5_COL_Y].astype(np.float32).mean()
                )
                modes = self._fourier_dual_modes(x_over_c, yc, cp_mean)
            else:
                order = np.argsort(x_over_c)
                cp_ordered = cp_mean[order]
                modes = dct_ii(cp_ordered, self.k_modes)  # (K,)
            graph.cond = torch.tensor(modes[None, :], dtype=torch.float32)

            if self.cond_mode == "nodal":
                graph.cp_nodal = torch.tensor(cp_mean[:, None], dtype=torch.float32)

        return graphs

    def _fourier_dual_modes(
        self,
        x_over_c: np.ndarray,
        y_centred: np.ndarray,
        cp_per_node: np.ndarray,
    ) -> np.ndarray:
        """Interpolate upper/lower halves onto a shared grid then DCT each.

        Args:
            x_over_c: Per-node chord-fraction ``∈ [0, 1]``, shape ``(N,)``.
            y_centred: Per-node y - y.mean(), shape ``(N,)``. Sign determines
                upper (≥ 0) vs lower (< 0) surface.
            cp_per_node: Per-node steady-state Cp, shape ``(N,)``.

        Returns:
            Concatenated ``[upper_modes (K), lower_modes (K)]`` array of
            length ``2 * self.k_modes``, dtype float32.
        """
        upper_mask = y_centred >= 0.0
        lower_mask = ~upper_mask
        grid = np.linspace(0.0, 1.0, N_CP_GRID_DEFAULT, dtype=np.float32)

        def _half_modes(mask: np.ndarray) -> np.ndarray:
            if mask.sum() < 2:
                # Degenerate: not enough nodes on this side. Return zeros so
                # downstream code still produces a 2K vector.
                return np.zeros(self.k_modes, dtype=np.float32)
            xc_half = x_over_c[mask]
            cp_half = cp_per_node[mask]
            order = np.argsort(xc_half)
            xc_sorted = xc_half[order]
            cp_sorted = cp_half[order]
            cp_interp = np.interp(grid, xc_sorted, cp_sorted).astype(np.float32)
            return dct_ii(cp_interp, self.k_modes)

        upper = _half_modes(upper_mask)
        lower = _half_modes(lower_mask)
        return np.concatenate([upper, lower], axis=0).astype(np.float32)
