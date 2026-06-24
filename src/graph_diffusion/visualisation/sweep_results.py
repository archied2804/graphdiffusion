"""
graph_diffusion.visualisation.sweep_results
============================================

Figures for assessing OpenLB validation sweeps (in-distribution and OOD
inverse-design samples) against the dataset distribution. Filesystem-free:
functions take loaded arrays / ``SampleRecord`` lists and return Figures.
The per-set on-disk loaders live in ``scripts/plot_sweep_results.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from graph_diffusion.data.cp_manifold import dataset_cp_curves, dataset_cp_pca
from graph_diffusion.data.pOnEllipse import _H5_COL_H, _H5_COL_RE


def cp_match_metrics(
    target_x: np.ndarray,
    target_cp: np.ndarray,
    meas_x: np.ndarray,
    meas_cp: np.ndarray,
) -> dict[str, float]:
    """Match between a target Cp curve and an OpenLB-measured Cp.

    The OpenLB surface Cp is non-monotonic in x/c, so the monotonic target
    is interpolated onto the measured node positions (not the reverse), then
    compared. Reports Pearson correlation, the affine fit
    ``meas ≈ a*target_on_meas + b`` (``a`` is the magnitude metric), and RMS.

    Args:
        target_x (np.ndarray): Target x/c, strictly increasing, shape ``(K,)``.
        target_cp (np.ndarray): Target Cp, shape ``(K,)``.
        meas_x (np.ndarray): Measured x/c (may be non-monotonic), shape ``(N,)``.
        meas_cp (np.ndarray): Measured Cp, shape ``(N,)``.

    Returns:
        dict[str, float]: Keys ``corr``, ``affine_a``, ``affine_b``, ``rms``.
    """
    target_on_meas = np.interp(meas_x, target_x, target_cp)
    corr = float(np.corrcoef(meas_cp, target_on_meas)[0, 1])
    design = np.vstack([target_on_meas, np.ones_like(target_on_meas)]).T
    lstsq_result = np.linalg.lstsq(design, meas_cp, rcond=None)
    coeffs: np.ndarray = lstsq_result[0]
    a = float(coeffs[0])
    b = float(coeffs[1])
    rms = float(np.sqrt(np.mean((meas_cp - target_on_meas) ** 2)))
    return {"corr": corr, "affine_a": a, "affine_b": b, "rms": rms}


@dataclass
class SampleRecord:
    """One validated sample: generated/dataset shape + target & measured Cp.

    Attributes:
        name (str): Case name (matches the run dir).
        re (float): Reynolds number.
        h (float): Channel height.
        ood_dist (float | None): OOD distance (min-L2 in cond space) if known.
        polygon (np.ndarray): Body outline ``(N, 2)``.
        target_x (np.ndarray): Target Cp x/c, increasing, shape ``(K,)``.
        target_cp (np.ndarray): Target/reference Cp, shape ``(K,)``.
        meas_x (np.ndarray): OpenLB-measured x/c, shape ``(M,)``.
        meas_cp (np.ndarray): OpenLB-measured Cp, shape ``(M,)``.
        cd (float | None): Time-averaged drag coefficient (Cd_mean), if known.
    """

    name: str
    re: float
    h: float
    ood_dist: float | None
    polygon: np.ndarray
    target_x: np.ndarray
    target_cp: np.ndarray
    meas_x: np.ndarray
    meas_cp: np.ndarray
    cd: float | None = None


def plot_sample_gallery(
    samples: list[SampleRecord],
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
) -> matplotlib.figure.Figure:
    """Grid of panels: each sample's shape + measured vs target Cp + metrics.

    Args:
        samples (list[SampleRecord]): Samples to draw.
        ncols (int): Panels per row.
        figsize (tuple[float, float] | None): Figure size; auto if None.

    Returns:
        matplotlib.figure.Figure: The rendered gallery.
    """
    n = len(samples)
    nrows = int(np.ceil(n / ncols))
    figsize = figsize or (4.0 * ncols, 3.2 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for k, s in enumerate(samples):
        ax = axes[k // ncols][k % ncols]
        ax.axis("on")
        m = cp_match_metrics(s.target_x, s.target_cp, s.meas_x, s.meas_cp)
        order = np.argsort(s.meas_x)
        ax.plot(s.target_x, s.target_cp, "k-", lw=2, label="target")
        ax.plot(s.meas_x[order], s.meas_cp[order], "C1.-", ms=3, label="OpenLB")
        ax.invert_yaxis()
        ax.set_title(
            f"{s.name}\n" r"$\rho$=" f"{m['corr']:.2f}  a={m['affine_a']:.2f}",
            fontsize=8,
        )
        inset = ax.inset_axes((0.62, 0.62, 0.36, 0.36))
        poly = np.vstack([s.polygon, s.polygon[:1]])  # close the loop
        inset.plot(poly[:, 0], poly[:, 1], color="navy", lw=0.8)
        inset.set_aspect("equal")
        inset.set_xticks([])
        inset.set_yticks([])
        ax.set_xlabel("x/c", fontsize=7)
        ax.set_ylabel("Cp", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    return fig


def plot_design_space_context(
    raw: np.ndarray,
    samples: list[SampleRecord],
    figsize: tuple[float, float] = (7.0, 5.0),
) -> matplotlib.figure.Figure:
    """Sample (Re, H) points over the full dataset cloud.

    Args:
        raw (np.ndarray): Full HDF5 array.
        samples (list[SampleRecord]): Samples to overlay.
        figsize (tuple[float, float]): Figure size.

    Returns:
        matplotlib.figure.Figure: The rendered figure.
    """
    re = raw[:, 0, _H5_COL_RE]
    h = raw[:, 0, _H5_COL_H]
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(re, h, s=8, c="0.7", alpha=0.5, label="dataset")
    ax.scatter(
        [s.re for s in samples],
        [s.h for s in samples],
        s=70,
        facecolors="none",
        edgecolors="red",
        linewidths=1.5,
        label="samples",
    )
    ax.set_xlabel("Reynolds number $Re$")
    ax.set_ylabel("channel height $H$")
    ax.set_title("Samples in dataset design space")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_cp_manifold_context(
    raw: np.ndarray,
    samples: list[SampleRecord],
    n_grid: int = 128,
    figsize: tuple[float, float] = (7.0, 5.0),
) -> matplotlib.figure.Figure:
    """Project sample target & measured Cp into the dataset Cp-PCA plane.

    The dataset cloud is the backdrop; an OOD sample sits outside it. Each
    sample's target (square) and measured (filled circle) are joined by a
    thin line so the generation+solver drift is visible.

    Args:
        raw (np.ndarray): Full HDF5 array.
        samples (list[SampleRecord]): Samples to project.
        n_grid (int): Cp grid for the basis.
        figsize (tuple[float, float]): Figure size.

    Returns:
        matplotlib.figure.Figure: The rendered figure.
    """
    grid, basis = dataset_cp_pca(raw, n_grid=n_grid, n_components=2)
    cloud = basis.project(dataset_cp_curves(raw, n_grid=n_grid))
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(cloud[:, 0], cloud[:, 1], s=8, c="0.75", alpha=0.5, label="dataset")
    for s in samples:
        tgt = basis.project(np.interp(grid, s.target_x, s.target_cp))
        order = np.argsort(s.meas_x)
        mea = basis.project(np.interp(grid, s.meas_x[order], s.meas_cp[order]))
        ax.plot([tgt[0], mea[0]], [tgt[1], mea[1]], "0.5", lw=0.6, zorder=1)
        ax.scatter(tgt[0], tgt[1], marker="s", s=45, c="C0", zorder=2)
        ax.scatter(mea[0], mea[1], marker="o", s=35, c="C3", zorder=2)
    ax.scatter([], [], marker="s", c="C0", label="target")
    ax.scatter([], [], marker="o", c="C3", label="measured")
    ax.set_xlabel("Cp PC 1")
    ax.set_ylabel("Cp PC 2")
    ax.set_title("Samples on the dataset Cp manifold")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_cp_envelope_overlay(
    raw: np.ndarray,
    samples: list[SampleRecord],
    n_grid: int = 128,
    figsize: tuple[float, float] = (7.0, 5.0),
) -> matplotlib.figure.Figure:
    """Dataset Cp mean±std band with sample measured & target curves on top.

    Args:
        raw (np.ndarray): Full HDF5 array.
        samples (list[SampleRecord]): Samples to overlay.
        n_grid (int): Cp grid.
        figsize (tuple[float, float]): Figure size.

    Returns:
        matplotlib.figure.Figure: The rendered figure.
    """
    grid = np.linspace(0.0, 1.0, n_grid)
    curves = dataset_cp_curves(raw, n_grid=n_grid)
    mean = curves.mean(axis=0)
    std = curves.std(axis=0)
    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(
        grid, mean - std, mean + std, color="0.8", label=r"dataset mean$\pm\sigma$"
    )
    ax.plot(grid, mean, "k-", lw=1.5)
    for s in samples:
        order = np.argsort(s.meas_x)
        ax.plot(s.target_x, s.target_cp, "C0--", lw=0.8, alpha=0.7)
        ax.plot(s.meas_x[order], s.meas_cp[order], "C3-", lw=0.8, alpha=0.7)
    ax.plot([], [], "C0--", label="sample target")
    ax.plot([], [], "C3-", label="sample measured")
    ax.invert_yaxis()
    ax.set_xlabel("x/c")
    ax.set_ylabel("Cp")
    ax.set_title("Samples vs dataset Cp envelope")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_results_summary(
    samples: list[SampleRecord],
    figsize: tuple[float, float] = (11.0, 4.0),
) -> matplotlib.figure.Figure:
    """Correlation, affine slope ``a``, and drag Cd vs OOD distance across samples.

    Args:
        samples (list[SampleRecord]): Samples (need ``ood_dist`` set).
        figsize (tuple[float, float]): Figure size.

    Returns:
        matplotlib.figure.Figure: Three-panel figure (corr, a, and Cd vs ood_dist).
    """
    metrics = [
        cp_match_metrics(s.target_x, s.target_cp, s.meas_x, s.meas_cp) for s in samples
    ]
    ood = np.array([s.ood_dist if s.ood_dist is not None else np.nan for s in samples])
    corr = np.array([m["corr"] for m in metrics])
    aff = np.array([m["affine_a"] for m in metrics])
    cd = np.array([s.cd if s.cd is not None else np.nan for s in samples])
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    ax1.scatter(ood, corr, c="C0")
    ax1.set_xlabel("OOD distance")
    ax1.set_ylabel(r"correlation $\rho$")
    ax1.set_title("Fidelity vs OOD distance")
    ax2.scatter(ood, aff, c="C3")
    ax2.axhline(1.0, color="0.6", ls="--", lw=1)
    ax2.set_xlabel("OOD distance")
    ax2.set_ylabel("affine slope $a$ (magnitude)")
    ax2.set_title("Magnitude creep vs OOD distance")
    ax3.scatter(ood, cd, c="C2")
    ax3.set_xlabel("OOD distance")
    ax3.set_ylabel("drag $C_d$")
    ax3.set_title("Drag vs OOD distance")
    fig.tight_layout()
    return fig
