"""
graph_diffusion.data.design_space
===================================

Systematic sampling of the pOnEllipse design space for model evaluation.
Sampling axes are the controllable physical inputs Reynolds number (Re)
and channel height (H); thickness b is emergent from the generated shape
and is therefore recorded but not sampled. Provides a trackable, sortable
case-naming scheme so evaluation and CFD cases share one namespace.
"""

from __future__ import annotations

import numpy as np

from graph_diffusion.data.pOnEllipse import _H5_COL_H, _H5_COL_RE

_RE_RANGE = (500.0, 1000.0)
_H_RANGE = (5.0, 6.0)


def make_case_name(re: float, h: float, idx: int) -> str:
    """Build a sortable, unambiguous case name from (Re, H, source index).

    Format ``re{RRRR}_h{HH}_idx{NNNN}`` — Re zero-padded to 4 digits, H
    multiplied by 10 to 2 digits, source HDF5 index to 4 digits.

    Args:
        re (float): Reynolds number (500-1000).
        h (float): Channel height (5-6).
        idx (int): Source HDF5 simulation index.

    Returns:
        str: e.g. ``"re0500_h53_idx0000"``.
    """
    return f"re{int(round(re)):04d}_h{int(round(h * 10)):02d}_idx{int(idx):04d}"


def sample_design_space(
    raw: np.ndarray,
    n_re: int = 5,
    n_h: int = 3,
) -> list[int]:
    """Select sims spanning an (Re, H) grid, snapped to nearest real sims.

    A regular ``n_re`` x ``n_h`` grid is laid over the design space; each
    grid node is matched to the dataset sim nearest in normalised (Re, H)
    space. The four corners are included by construction. Duplicate matches
    are removed, preserving first-seen order.

    Args:
        raw (np.ndarray): Full HDF5 array, shape ``(N_sims, MAX_NODES, 105)``.
        n_re (int): Number of Reynolds grid points (>= 2 to hit corners).
        n_h (int): Number of channel-height grid points (>= 2).

    Returns:
        list[int]: Selected simulation indices.

    Raises:
        ValueError: If ``n_re < 2`` or ``n_h < 2``.
    """
    if n_re < 2 or n_h < 2:
        raise ValueError(f"n_re and n_h must be >= 2, got {n_re}, {n_h}")
    re = raw[:, 0, _H5_COL_RE].astype(np.float64)
    h = raw[:, 0, _H5_COL_H].astype(np.float64)
    re_n = (re - _RE_RANGE[0]) / (_RE_RANGE[1] - _RE_RANGE[0])
    h_n = (h - _H_RANGE[0]) / (_H_RANGE[1] - _H_RANGE[0])

    re_targets = np.linspace(0.0, 1.0, n_re)
    h_targets = np.linspace(0.0, 1.0, n_h)
    selected: list[int] = []
    for rt in re_targets:
        for ht in h_targets:
            d2 = (re_n - rt) ** 2 + (h_n - ht) ** 2
            idx = int(np.argmin(d2))
            if idx not in selected:
                selected.append(idx)
    return selected
