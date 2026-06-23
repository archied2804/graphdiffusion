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

import numpy as np


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
