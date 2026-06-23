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

import numpy as np

from graph_diffusion.data.pOnEllipse import (
    _H5_COL_H,
    _H5_COL_P_START,
    _H5_COL_RE,
    _H5_COL_X,
    _H5_COL_Y,
    _H5_N_TIMESTEPS,
    _node_count,
)

__all__ = ["pressure_drag_coefficient", "sim_summary"]


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
