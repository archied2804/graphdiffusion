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

from dataclasses import dataclass

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


@dataclass
class FrameSpec:
    """One animation frame: the handle y-values and the active target.

    Attributes:
        handle_y (np.ndarray): Handle y-values for this frame, ``(n_handles,)``.
        target_idx (int): Index of the most-recently-reached target. Drives
            which pre-sampled shape and OOD flag the frame displays.
    """

    handle_y: np.ndarray
    target_idx: int


def _smoothstep(s: float) -> float:
    """Ease-in-out (cubic Hermite) blend factor for ``s`` in ``[0, 1]``."""
    return 3.0 * s * s - 2.0 * s * s * s


def tween_handles(
    keyframes: np.ndarray,
    n_tween: int,
    n_hold: int,
) -> list[FrameSpec]:
    """Build per-frame handle states: holds at each target + eased tweens.

    Sequence: hold target 0 for ``n_hold`` frames, ease-in-out tween to
    target 1 over ``n_tween`` frames, hold target 1, ... The tween frames
    keep ``target_idx`` at the *previous* target (the shown shape swaps only
    once the next target is reached), so a tween block from target ``t`` to
    ``t+1`` is tagged ``target_idx = t``.

    Args:
        keyframes: Handle-y keyframes, shape ``(n_targets, n_handles)``.
        n_tween: Frames per tween between consecutive targets (``>= 1``).
        n_hold: Frames each target is held (``>= 1``).

    Returns:
        Frames, length ``n_targets * n_hold + (n_targets - 1) * n_tween``.

    Raises:
        ValueError: If ``n_tween < 1`` or ``n_hold < 1``.
    """
    if n_tween < 1:
        raise ValueError(f"n_tween must be >= 1, got {n_tween}")
    if n_hold < 1:
        raise ValueError(f"n_hold must be >= 1, got {n_hold}")
    n_targets = int(keyframes.shape[0])
    frames: list[FrameSpec] = []
    for t in range(n_targets):
        for _ in range(n_hold):
            frames.append(FrameSpec(handle_y=keyframes[t].copy(), target_idx=t))
        if t < n_targets - 1:
            start = keyframes[t]
            end = keyframes[t + 1]
            for j in range(n_tween):
                s = (j + 1) / n_tween  # (0, 1]; final tween frame lands on `end`
                blend = _smoothstep(s)
                handle_y = (start + blend * (end - start)).astype(np.float32)
                frames.append(FrameSpec(handle_y=handle_y, target_idx=t))
    return frames
