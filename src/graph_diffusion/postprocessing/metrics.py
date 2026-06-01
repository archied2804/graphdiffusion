"""
graph_diffusion.postprocessing.metrics
========================================

Comparative metrics for evaluating EXP-021..025 against the EXP-020
baseline. Currently exposes :func:`compute_boundary_roughness`, the
second-difference-of-r(theta) metric used as the surface-quality
yardstick for the training-improvement series.
"""

from __future__ import annotations

import numpy as np


def _roughness_one(xy: np.ndarray) -> float:
    """Roughness for a single closed boundary, ``(N, 2)``."""
    r = np.linalg.norm(xy, axis=1)
    theta = np.arctan2(xy[:, 1], xy[:, 0])
    order = np.argsort(theta)
    r_sorted = r[order]
    # Wrap-around second difference: |r[i+1] - 2 r[i] + r[i-1]|, indices mod N.
    second_diff = np.roll(r_sorted, -1) - 2.0 * r_sorted + np.roll(r_sorted, 1)
    return float(np.mean(np.abs(second_diff)))


def compute_boundary_roughness(shapes_xy: np.ndarray) -> float:
    """Mean second-difference of ``r(θ)`` across one or many boundaries.

    The metric quantifies high-frequency wobble on a closed boundary:
    a perfect circle has roughness ``0`` (radius is constant in θ); a
    boundary with per-node jitter scores positive in proportion to the
    jitter amplitude.

    The boundary is first re-expressed in polar coordinates
    ``r(θ) = ‖xy‖`` and sorted by θ, then we take the mean absolute
    second difference of ``r`` with wrap-around indices.

    Args:
        shapes_xy: Either a single ``(N, 2)`` boundary or a batched
            ``(B, N, 2)`` stack of boundaries. The metric is averaged
            across ``B`` in the batched case.

    Returns:
        The (mean) roughness score, ``>= 0``. Lower = smoother.
    """
    arr = np.asarray(shapes_xy)
    if arr.ndim == 2:
        return _roughness_one(arr)
    if arr.ndim == 3:
        return float(np.mean([_roughness_one(arr[i]) for i in range(arr.shape[0])]))
    raise ValueError(f"shapes_xy must be (N, 2) or (B, N, 2); got shape {arr.shape}")
