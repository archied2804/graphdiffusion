"""
graph_diffusion.visualisation.sweep_results
============================================

Figures for assessing OpenLB validation sweeps (in-distribution and OOD
inverse-design samples) against the dataset distribution. Filesystem-free:
functions take loaded arrays / ``SampleRecord`` lists and return Figures.
The per-set on-disk loaders live in ``scripts/plot_sweep_results.py``.
"""

from __future__ import annotations

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
