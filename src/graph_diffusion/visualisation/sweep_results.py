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
        ax.set_xlabel("x/c", fontsize=7)
        ax.set_ylabel("Cp", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    return fig
