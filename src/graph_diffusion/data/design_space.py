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
