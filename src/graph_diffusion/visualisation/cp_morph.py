"""
graph_diffusion.visualisation.cp_morph
=======================================

Scripted (non-interactive) Cp-morph animation. Reuses the interactive
editor's spline/DCT maths (:func:`cp_spline_and_dct`) and renders a
clip in which the Cp spline morphs through a sequence of target curves
while a pre-sampled shape is shown for the most-recently-reached target.

Filesystem-free and model-free: the writer takes *pre-sampled* shapes
as arrays, so the heavy reverse-diffusion sampling happens in the
calling script, not here.
"""

from __future__ import annotations

import numpy as np


def build_handle_keyframes(
    targets: np.ndarray,
    handle_x: np.ndarray,
    x_grid: np.ndarray,
) -> np.ndarray:
    """Sample each dense target Cp curve at the editor handle positions.

    Args:
        targets: Dense target Cp curves, shape ``(n_targets, n_grid)``.
        handle_x: Handle x-positions, shape ``(n_handles,)``.
        x_grid: Dense ``x/c`` grid the targets live on, shape ``(n_grid,)``.

    Returns:
        Handle-y keyframes, shape ``(n_targets, n_handles)``, ``float32``.
    """
    keyframes = np.stack(
        [np.interp(handle_x, x_grid, targets[t]) for t in range(targets.shape[0])],
        axis=0,
    )
    return keyframes.astype(np.float32)
