"""
metrics — Experiment-specific post-processing metrics
======================================================

Reusable metric functions for evaluating diffusion model outputs.
Each experiment type gets its own section of metrics.

Circle experiment metrics
-------------------------
- ``compute_radii_stats``: Summary statistics for radii distributions.
- ``compute_smoothness``: Shape smoothness via 2nd-order finite differences.
- ``compute_closure_error``: Gap between first/last node on closed curves.
- ``compute_boundary_violations``: Fraction of nodes outside clamp bounds.
- ``compute_circularity``: Coefficient of variation (0 = perfect circle).
- ``ks_statistic``: Two-sample KS test without scipy.
- ``extract_sorted_radii``: Extract angularly-sorted radii from a sample.
"""

from __future__ import annotations

import numpy as np
from torch_geometric.data import Data

# ---------------------------------------------------------------------------
# Circle experiment metrics
# ---------------------------------------------------------------------------


def compute_radii_stats(radii: np.ndarray) -> dict[str, float]:
    """Compute summary statistics for a set of radii.

    Args:
        radii: 1-D array of radial values.

    Returns:
        Dictionary with mean, std, min, max, median.
    """
    return {
        "mean": float(np.mean(radii)),
        "std": float(np.std(radii)),
        "min": float(np.min(radii)),
        "max": float(np.max(radii)),
        "median": float(np.median(radii)),
    }


def compute_smoothness(radii: np.ndarray) -> float:
    """Measure shape smoothness as mean absolute 2nd-order finite difference.

    Lower values indicate smoother shapes. A perfect circle returns 0.

    Args:
        radii: 1-D array of angularly-sorted radii.

    Returns:
        Scalar smoothness value.
    """
    d2r = np.diff(radii, n=2, prepend=radii[-1], append=radii[0])
    return float(np.mean(np.abs(d2r)))


def compute_closure_error(radii: np.ndarray) -> float:
    """Absolute difference between first and last radius.

    Args:
        radii: 1-D array of angularly-sorted radii.

    Returns:
        Closure gap magnitude.
    """
    return float(np.abs(radii[0] - radii[-1]))


def compute_boundary_violations(
    radii: np.ndarray,
    r_min: float,
    r_max: float,
) -> dict[str, float]:
    """Fraction of nodes outside the clamp range.

    Args:
        radii: 1-D array of radial values.
        r_min: Lower clamp bound.
        r_max: Upper clamp bound.

    Returns:
        Dictionary with below_min, above_max, total fractions.
    """
    below = float(np.mean(radii < r_min))
    above = float(np.mean(radii > r_max))
    return {"below_min": below, "above_max": above, "total": below + above}


def compute_circularity(radii: np.ndarray) -> float:
    """Coefficient of variation of radii — 0 = perfect circle.

    Args:
        radii: 1-D array of radial values.

    Returns:
        CV value (std / mean).
    """
    mean = np.mean(radii)
    return float(np.std(radii) / mean) if mean > 0 else float("inf")


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (no scipy dependency).

    Args:
        a: First sample array.
        b: Second sample array.

    Returns:
        Maximum absolute CDF difference.
    """
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    all_vals = np.sort(np.concatenate([a_sorted, b_sorted]))
    cdf_a = np.searchsorted(a_sorted, all_vals, side="right") / len(a_sorted)
    cdf_b = np.searchsorted(b_sorted, all_vals, side="right") / len(b_sorted)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def extract_sorted_radii(
    result: Data,
    template: Data,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract radii from a sample, sorted by angle.

    Args:
        result: Data object with generated ``x[:, 0]`` = radii.
        template: Data object with ``pos`` = Cartesian positions.

    Returns:
        Tuple of (sorted_radii, sorted_theta).
    """
    r = result.x[:, 0].cpu().numpy()
    theta = np.arctan2(
        template.pos[:, 1].cpu().numpy(),
        template.pos[:, 0].cpu().numpy(),
    )
    order = np.argsort(theta)
    return r[order], theta[order]
