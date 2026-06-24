"""
graph_diffusion.data.cp_manifold
=================================

The dataset surface-pressure (Cp) manifold: resample every simulation's
mean Cp(x/c) onto a common grid and fit a small PCA basis. Used to build
principled OOD Cp targets (perturbing the dataset mean along its own
principal modes) and to project samples into the dataset Cp plane for
context plots. Pure NumPy; depends only on the pOnEllipse column layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from graph_diffusion.data.pOnEllipse import (
    _H5_COL_P_START,
    _H5_COL_X,
    _H5_N_TIMESTEPS,
    _node_count,
)

__all__ = [
    "CpPcaBasis",
    "cp_pca_basis",
    "dataset_cp_curves",
    "dataset_cp_pca",
    "pc_ladder_targets",
]


@dataclass
class CpPcaBasis:
    """Mean curve + leading principal components of the dataset Cp manifold.

    Attributes:
        mean (np.ndarray): Mean Cp curve, shape ``(n_grid,)``.
        components (np.ndarray): Unit principal components, shape
            ``(n_components, n_grid)``.
        proj_std (np.ndarray): Std of the dataset projections along each
            component, shape ``(n_components,)`` — the natural amplitude
            unit for an OOD ladder.
    """

    mean: np.ndarray
    components: np.ndarray
    proj_std: np.ndarray

    def project(self, curve: np.ndarray) -> np.ndarray:
        """Project a Cp curve (or array of curves) onto the components.

        Args:
            curve (np.ndarray): Shape ``(n_grid,)`` or ``(M, n_grid)``.

        Returns:
            np.ndarray: Coordinates, shape ``(n_components,)`` or
                ``(M, n_components)``.
        """
        centred = np.asarray(curve, dtype=np.float64) - self.mean
        return cast(np.ndarray, centred @ self.components.T)


def dataset_cp_curves(raw: np.ndarray, n_grid: int = 128) -> np.ndarray:
    """Every sim's mean Cp(x/c) resampled onto a shared ``n_grid`` grid.

    Args:
        raw (np.ndarray): Full HDF5 array ``(N_sims, MAX_NODES, 105)``.
        n_grid (int): Number of x/c grid points.

    Returns:
        np.ndarray: Shape ``(N_sims, n_grid)``, dtype float64.
    """
    grid = np.linspace(0.0, 1.0, n_grid)
    out = np.empty((raw.shape[0], n_grid), dtype=np.float64)
    for i in range(raw.shape[0]):
        n = _node_count(raw[i])
        data = raw[i, :n]
        xc = data[:, _H5_COL_X].astype(np.float64)
        chord = max(xc.max() - xc.min(), 1e-12)
        x_over_c = (xc - xc.min()) / chord
        cp_mean = data[:, _H5_COL_P_START : _H5_COL_P_START + _H5_N_TIMESTEPS].mean(
            axis=1
        )
        order = np.argsort(x_over_c)
        out[i] = np.interp(grid, x_over_c[order], cp_mean[order])
    return out


def cp_pca_basis(curves: np.ndarray, n_components: int = 2) -> CpPcaBasis:
    """Fit a PCA basis to a set of Cp curves via thin SVD.

    Args:
        curves (np.ndarray): Shape ``(N, n_grid)``.
        n_components (int): Number of principal components to retain.

    Returns:
        CpPcaBasis: Mean, unit components, and per-component projection std.

    Raises:
        ValueError: If ``n_components`` exceeds the available rank.
    """
    curves = np.asarray(curves, dtype=np.float64)
    if n_components < 1 or n_components > min(curves.shape):
        raise ValueError(
            f"n_components must be in [1, {min(curves.shape)}], got {n_components}"
        )
    mean = curves.mean(axis=0)
    centred = curves - mean
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    components = vt[:n_components]
    proj_std = (centred @ components.T).std(axis=0)
    return CpPcaBasis(mean=mean, components=components, proj_std=proj_std)


def dataset_cp_pca(
    raw: np.ndarray, n_grid: int = 128, n_components: int = 2
) -> tuple[np.ndarray, CpPcaBasis]:
    """Convenience: resample dataset Cp curves and fit a PCA basis.

    Args:
        raw (np.ndarray): Full HDF5 array.
        n_grid (int): x/c grid resolution.
        n_components (int): Components to retain.

    Returns:
        tuple[np.ndarray, CpPcaBasis]: ``(grid, basis)`` where ``grid`` is
            ``np.linspace(0, 1, n_grid)``.
    """
    grid = np.linspace(0.0, 1.0, n_grid)
    curves = dataset_cp_curves(raw, n_grid=n_grid)
    return grid, cp_pca_basis(curves, n_components=n_components)


def pc_ladder_targets(
    basis: CpPcaBasis,
    amplitudes: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0),
    n_modes: int = 2,
) -> list[dict[str, Any]]:
    """Systematic OOD Cp targets along the dataset's principal Cp modes.

    For each mode ``m`` in ``1..n_modes``, signed amplitude ``s in {+, -}``
    and ``alpha`` in ``amplitudes``::

        cp_target = mean + s * alpha * proj_std[m-1] * components[m-1]

    The amplitude is in units of the dataset projection std, so ``alpha=1``
    is interior, ``~2`` is the border of the dataset spread, and ``>=3`` is
    out of distribution.

    Args:
        basis (CpPcaBasis): Dataset Cp-PCA basis.
        amplitudes (tuple[float, ...]): Sigma-unit amplitudes to sweep.
        n_modes (int): Number of leading modes to perturb along.

    Returns:
        list[dict[str, Any]]: One dict per target with keys ``name``
            (e.g. ``"pc1_p2sig"``), ``mode`` (int, 1-based), ``sign``
            (``+1``/``-1``), ``amp_sigma`` (float), and ``cp_target``
            (``np.ndarray (n_grid,)``).

    Raises:
        ValueError: If ``n_modes`` exceeds the available components.
    """
    if n_modes < 1 or n_modes > basis.components.shape[0]:
        raise ValueError(
            f"n_modes must be in [1, {basis.components.shape[0]}], got {n_modes}"
        )
    targets: list[dict[str, Any]] = []
    for m in range(1, n_modes + 1):
        comp = basis.components[m - 1]
        sigma = float(basis.proj_std[m - 1])
        for alpha in amplitudes:
            for sign, tag in ((1.0, "p"), (-1.0, "m")):
                cp_target = basis.mean + sign * alpha * sigma * comp
                targets.append(
                    {
                        "name": f"pc{m}_{tag}{int(round(alpha))}sig",
                        "mode": m,
                        "sign": int(sign),
                        "amp_sigma": float(alpha),
                        "cp_target": cp_target.astype(np.float64),
                    }
                )
    return targets
