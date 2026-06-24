"""
graph_diffusion.visualisation.dataset_overview
===============================================

Read-only Matplotlib figure builders that characterise the whole
pOnEllipse dataset: a design-space (Re, b) -> pressure-drag contour, the
distribution of surface Cp curves, design-space coverage, and supporting
extras. Figure functions take the raw HDF5 array (or derived arrays) and
return a Figure; they perform no file I/O.
"""

from __future__ import annotations

from typing import Any

import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from graph_diffusion.data.cp_manifold import cp_pca_basis
from graph_diffusion.data.pOnEllipse import (
    _H5_COL_H,
    _H5_COL_P_START,
    _H5_COL_RE,
    _H5_COL_X,
    _H5_COL_Y,
    _H5_N_TIMESTEPS,
    _node_count,
)

__all__ = [
    "pressure_drag_coefficient",
    "sim_summary",
    "plot_design_space_drag",
    "plot_cp_envelope",
    "plot_design_space_coverage",
    "plot_shape_gallery",
    "plot_drag_vs_re",
    "plot_cp_pca",
]


def pressure_drag_coefficient(polygon_xy: np.ndarray, cp: np.ndarray) -> float:
    """Pressure-drag coefficient from a closed boundary polygon and nodal Cp.

    Computes the discrete line integral ``Cd_p = oint Cp (n . x_hat) ds``
    around the body. For a counter-clockwise polygon the outward-normal
    times segment-length has x-component ``dy``, so the integral reduces to
    ``sum_i cp_mid_i * (y_{i+1} - y_i)`` with ``cp_mid`` the edge-midpoint
    Cp. The body is pressure-only, so this is pressure drag (no skin
    friction).

    Args:
        polygon_xy (np.ndarray): Boundary points ordered around the body,
            shape ``(N, 2)``. The contour is closed implicitly (wraps).
        cp (np.ndarray): Per-node pressure coefficient, shape ``(N,)``,
            aligned with ``polygon_xy``.

    Returns:
        float: The pressure-drag coefficient.
    """
    if polygon_xy.ndim != 2 or polygon_xy.shape[1] != 2:
        raise ValueError(f"polygon_xy must be (N, 2), got {polygon_xy.shape}")
    if cp.shape[0] != polygon_xy.shape[0]:
        raise ValueError("cp and polygon_xy must have matching length")
    y = polygon_xy[:, 1].astype(np.float64)
    cp = cp.astype(np.float64)
    y_next = np.roll(y, -1)
    cp_next = np.roll(cp, -1)
    cp_mid = 0.5 * (cp + cp_next)
    return float(np.sum(cp_mid * (y_next - y)))


def sim_summary(raw: np.ndarray, idx: int) -> dict[str, Any]:
    """Derive per-figure quantities for one simulation.

    Args:
        raw (np.ndarray): Full HDF5 array, shape ``(N_sims, MAX_NODES, 105)``.
        idx (int): Simulation index.

    Returns:
        dict[str, Any]: Keys ``re``, ``h``, ``a``, ``b`` (floats),
            ``polygon`` ``(N, 2)`` CCW, ``cp_mean`` ``(N,)`` aligned with
            ``polygon``, ``x_over_c`` ``(N,)`` ascending, ``cp_curve``
            ``(N,)`` Cp sorted by x/c, and ``cd_p`` (float).
    """
    sample = raw[idx]
    n = _node_count(sample)
    data = sample[:n]
    re = float(data[0, _H5_COL_RE])
    h = float(data[0, _H5_COL_H])
    xy = np.stack([data[:, _H5_COL_X], data[:, _H5_COL_Y]], axis=1).astype(np.float64)
    a = float((xy[:, 0].max() - xy[:, 0].min()) / 2.0)
    b = float((xy[:, 1].max() - xy[:, 1].min()) / 2.0)
    cp_mean = data[:, _H5_COL_P_START : _H5_COL_P_START + _H5_N_TIMESTEPS].mean(axis=1)
    cp_mean = cp_mean.astype(np.float64)

    # Ordered polygon + Cp aligned to the same ordering for Cd_p.
    centroid = xy.mean(axis=0)
    order_poly = np.argsort(np.arctan2(xy[:, 1] - centroid[1], xy[:, 0] - centroid[0]))
    polygon = xy[order_poly]
    cp_poly = cp_mean[order_poly]
    cd_p = pressure_drag_coefficient(polygon, cp_poly)

    # Cp(x/c) sorted by chordwise coordinate.
    xc = xy[:, 0]
    chord = max(xc.max() - xc.min(), 1e-12)
    x_over_c = (xc - xc.min()) / chord
    order_xc = np.argsort(x_over_c)

    return {
        "re": re,
        "h": h,
        "a": a,
        "b": b,
        "polygon": polygon,
        "cp_mean": cp_poly,
        "x_over_c": x_over_c[order_xc],
        "cp_curve": cp_mean[order_xc],
        "cd_p": cd_p,
    }


def _summaries(raw: np.ndarray, indices: list[int]) -> list[dict[str, Any]]:
    return [sim_summary(raw, i) for i in indices]


def plot_design_space_drag(
    raw: np.ndarray,
    indices: list[int],
    figsize: tuple[float, float] = (7.0, 5.0),
) -> matplotlib.figure.Figure:
    """Hexbin of pressure-drag coefficient over the (Re, b) design space.

    Args:
        raw (np.ndarray): Full HDF5 array.
        indices (list[int]): Sim indices to include.
        figsize (tuple[float, float]): Figure size in inches.

    Returns:
        matplotlib.figure.Figure: The rendered figure. Colour encodes
            pressure drag (Cd_p); the label states it is pressure-only.
    """
    s = _summaries(raw, indices)
    re = np.array([d["re"] for d in s])
    b = np.array([d["b"] for d in s])
    cd = np.array([d["cd_p"] for d in s])
    fig, ax = plt.subplots(figsize=figsize)
    hb = ax.hexbin(re, b, C=cd, gridsize=25, cmap="viridis")
    fig.colorbar(hb, ax=ax, label=r"pressure drag $C_{d,p}$ (no skin friction)")
    ax.set_xlabel("Reynolds number $Re$")
    ax.set_ylabel("semi-minor axis $b$")
    ax.set_title("Design space coloured by pressure drag")
    fig.tight_layout()
    return fig


def plot_cp_envelope(
    raw: np.ndarray,
    indices: list[int],
    colour_by: str = "re",
    figsize: tuple[float, float] = (7.0, 5.0),
) -> matplotlib.figure.Figure:
    """Overlay all surface Cp(x/c) curves coloured by Re or b, with a band.

    Args:
        raw (np.ndarray): Full HDF5 array.
        indices (list[int]): Sim indices to include.
        colour_by (str): ``"re"`` or ``"b"`` — the per-curve colour key.
        figsize (tuple[float, float]): Figure size in inches.

    Returns:
        matplotlib.figure.Figure: The rendered figure with a mean +/- std
            band over a common x/c grid.

    Raises:
        ValueError: If ``colour_by`` is not ``"re"`` or ``"b"``.
    """
    if colour_by not in ("re", "b"):
        raise ValueError(f"colour_by must be 're' or 'b', got '{colour_by}'")
    s = _summaries(raw, indices)
    key = np.array([d[colour_by] for d in s])
    norm = matplotlib.colors.Normalize(vmin=float(key.min()), vmax=float(key.max()))
    cmap = matplotlib.colormaps["plasma"]
    grid = np.linspace(0.0, 1.0, 100)
    fig, ax = plt.subplots(figsize=figsize)
    resampled = []
    for d in s:
        cp_g = np.interp(grid, d["x_over_c"], d["cp_curve"])
        resampled.append(cp_g)
        ax.plot(grid, cp_g, color=cmap(norm(d[colour_by])), alpha=0.3, linewidth=0.7)
    stack = np.stack(resampled, axis=0)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0)
    ax.plot(grid, mean, color="black", linewidth=2.0, label="mean")
    ax.fill_between(
        grid, mean - std, mean + std, color="grey", alpha=0.3, label=r"$\pm\sigma$"
    )
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb_label: dict[str, str] = {"re": "Reynolds $Re$", "b": "semi-minor $b$"}
    fig.colorbar(sm, ax=ax, label=cb_label[colour_by])
    ax.set_xlabel("$x/c$")
    ax.set_ylabel("$C_p$")
    ax.set_title(f"Surface $C_p$ distribution (coloured by {colour_by})")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_design_space_coverage(
    raw: np.ndarray,
    indices: list[int],
    eval_indices: list[int] | None = None,
    figsize: tuple[float, float] = (7.0, 5.0),
) -> matplotlib.figure.Figure:
    """Scatter of (Re, b) coloured by Cd_p, marking evaluation samples.

    Args:
        raw (np.ndarray): Full HDF5 array.
        indices (list[int]): All sim indices (the background cloud).
        eval_indices (list[int] | None): Indices to highlight as the
            chosen evaluation samples.
        figsize (tuple[float, float]): Figure size in inches.

    Returns:
        matplotlib.figure.Figure: The rendered figure.
    """
    s = _summaries(raw, indices)
    re = np.array([d["re"] for d in s])
    b = np.array([d["b"] for d in s])
    cd = np.array([d["cd_p"] for d in s])
    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(re, b, c=cd, cmap="viridis", s=12, alpha=0.7)
    fig.colorbar(sc, ax=ax, label=r"pressure drag $C_{d,p}$")
    if eval_indices:
        es = _summaries(raw, eval_indices)
        ax.scatter(
            [d["re"] for d in es],
            [d["b"] for d in es],
            facecolors="none",
            edgecolors="red",
            s=90,
            linewidths=1.5,
            label="evaluation samples",
        )
        ax.legend()
    ax.set_xlabel("Reynolds number $Re$")
    ax.set_ylabel("semi-minor axis $b$")
    ax.set_title("Design-space coverage")
    fig.tight_layout()
    return fig


def plot_shape_gallery(
    raw: np.ndarray,
    indices: list[int],
    figsize: tuple[float, float] = (9.0, 6.0),
) -> matplotlib.figure.Figure:
    """Grid of body outlines, one per selected sim, labelled by Re and b.

    Args:
        raw (np.ndarray): Full HDF5 array.
        indices (list[int]): Sim indices to draw.
        figsize (tuple[float, float]): Figure size in inches.

    Returns:
        matplotlib.figure.Figure: The rendered gallery.
    """
    n = len(indices)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for k, idx in enumerate(indices):
        d = sim_summary(raw, idx)
        ax = axes[k // ncols][k % ncols]
        ax.axis("on")
        poly = np.vstack([d["polygon"], d["polygon"][:1]])
        ax.plot(poly[:, 0], poly[:, 1], color="navy")
        ax.set_aspect("equal")
        ax.set_title(f"Re={d['re']:.0f}, b={d['b']:.3f}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return fig


def plot_drag_vs_re(
    raw: np.ndarray,
    indices: list[int],
    n_bins: int = 4,
    figsize: tuple[float, float] = (7.0, 5.0),
) -> matplotlib.figure.Figure:
    """Pressure drag vs Re, with sims grouped into ``n_bins`` thickness bands.

    Args:
        raw (np.ndarray): Full HDF5 array.
        indices (list[int]): Sim indices to include.
        n_bins (int): Number of b (thickness) bands.
        figsize (tuple[float, float]): Figure size in inches.

    Returns:
        matplotlib.figure.Figure: The rendered figure.
    """
    s = _summaries(raw, indices)
    re = np.array([d["re"] for d in s])
    b = np.array([d["b"] for d in s])
    cd = np.array([d["cd_p"] for d in s])
    edges = np.linspace(b.min(), b.max() + 1e-9, n_bins + 1)
    fig, ax = plt.subplots(figsize=figsize)
    for j in range(n_bins):
        mask = (b >= edges[j]) & (b < edges[j + 1])
        if not mask.any():
            continue
        order = np.argsort(re[mask])
        ax.plot(
            re[mask][order],
            cd[mask][order],
            marker="o",
            markersize=3,
            label=f"b in [{edges[j]:.3f}, {edges[j + 1]:.3f})",
        )
    ax.set_xlabel("Reynolds number $Re$")
    ax.set_ylabel(r"pressure drag $C_{d,p}$")
    ax.set_title("Pressure drag vs Re by thickness band")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_cp_pca(
    raw: np.ndarray,
    indices: list[int],
    colour_by: str = "re",
    figsize: tuple[float, float] = (7.0, 5.0),
) -> matplotlib.figure.Figure:
    """2-D PCA projection of the resampled Cp(x/c) curves.

    A learned-latent view: each sim's Cp curve is resampled onto a common
    grid, then projected onto its first two principal components.

    Args:
        raw (np.ndarray): Full HDF5 array.
        indices (list[int]): Sim indices to include.
        colour_by (str): ``"re"`` or ``"b"``.
        figsize (tuple[float, float]): Figure size in inches.

    Returns:
        matplotlib.figure.Figure: The rendered figure.

    Raises:
        ValueError: If ``colour_by`` is not ``"re"`` or ``"b"``.
    """
    if colour_by not in ("re", "b"):
        raise ValueError(f"colour_by must be 're' or 'b', got '{colour_by}'")
    s = _summaries(raw, indices)
    grid = np.linspace(0.0, 1.0, 100)
    curves = np.stack(
        [np.interp(grid, d["x_over_c"], d["cp_curve"]) for d in s], axis=0
    )
    basis = cp_pca_basis(curves, n_components=2)
    coords = basis.project(curves)
    key = np.array([d[colour_by] for d in s])
    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=key, cmap="plasma", s=14)
    cb_labels: dict[str, str] = {"re": "Reynolds $Re$", "b": "semi-minor $b$"}
    fig.colorbar(sc, ax=ax, label=cb_labels[colour_by])
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(f"Cp-curve PCA (coloured by {colour_by})")
    fig.tight_layout()
    return fig
