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
from typing import cast

import numpy as np

from graph_diffusion.data.pOnEllipse import (
    _H5_COL_P_START,
    _H5_COL_X,
    _H5_N_TIMESTEPS,
    _node_count,
)

__all__ = ["CpPcaBasis", "cp_pca_basis", "dataset_cp_curves", "dataset_cp_pca"]


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
