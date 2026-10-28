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

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.animation as manim
import matplotlib.artist
import matplotlib.pyplot as plt
import numpy as np

from graph_diffusion.visualisation.cp_editor import cp_spline_and_dct


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


def write_cp_morph_animation(
    targets: np.ndarray,
    shapes: list[np.ndarray],
    labels: list[str],
    ood_flags: list[bool],
    handle_x: np.ndarray,
    x_grid: np.ndarray,
    k_modes: int,
    out_path_mp4: Path | None,
    out_path_gif: Path | None = None,
    n_tween: int = 25,
    n_hold: int = 20,
    fps: int = 25,
) -> None:
    """Render the scripted Cp-morph clip to MP4 and/or GIF.

    Left panel: the editor view (grey spline, blue DCT-k reconstruction,
    red handles) morphing through the targets. Right panel: the pre-sampled
    boundary shape for the most-recently-reached target, its title flagged
    red + "(OOD)" when that target is out-of-distribution. Axis bounds are
    fixed across the clip so nothing jitters.

    Args:
        targets: Dense target Cp curves, shape ``(n_targets, n_grid)``.
        shapes: Pre-sampled shapes, ``n_targets`` arrays of ``(N_nodes, 2)``.
        labels: Per-target captions, length ``n_targets``.
        ood_flags: Per-target OOD markers, length ``n_targets``.
        handle_x: Handle x-positions, shape ``(n_handles,)``.
        x_grid: Dense ``x/c`` grid, shape ``(n_grid,)``.
        k_modes: DCT modes kept for the blue reconstruction.
        out_path_mp4: MP4 destination; skipped if ``None`` or ffmpeg absent.
        out_path_gif: GIF destination; skipped if ``None``.
        n_tween: Frames per tween between targets.
        n_hold: Frames each target is held.
        fps: Frames per second.

    Returns:
        None. Files are written to disk.
    """
    keyframes = build_handle_keyframes(targets, handle_x, x_grid)
    frames = tween_handles(keyframes, n_tween=n_tween, n_hold=n_hold)

    fig, (ax_cp, ax_shape) = plt.subplots(
        1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1.2, 1]}
    )

    # Left (Cp editor) axis: fixed limits from the union of all targets.
    cp_lo = float(targets.min()) - 0.15
    cp_hi = float(targets.max()) + 0.15
    ax_cp.set_xlim(-0.02, 1.02)
    ax_cp.set_ylim(cp_lo, cp_hi)
    ax_cp.set_xlabel("x/c")
    ax_cp.set_ylabel("Cp")
    ax_cp.set_title("Cp spline (drag handles)")
    (spline_line,) = ax_cp.plot([], [], color="0.7", lw=1.0, label="spline")
    (dct_line,) = ax_cp.plot([], [], color="C0", lw=1.8, label=f"DCT-{k_modes}")
    (handles_line,) = ax_cp.plot([], [], "o", color="C3", ms=8, label="handles")
    ax_cp.legend(loc="best", fontsize=8)

    # Right (shape) axis: fixed limits from the union of all shapes.
    all_xy = np.concatenate(shapes, axis=0)
    xy_min = all_xy.min(axis=0) - 0.1
    xy_max = all_xy.max(axis=0) + 0.1
    ax_shape.set_aspect("equal")
    ax_shape.set_xlim(xy_min[0], xy_max[0])
    ax_shape.set_ylim(xy_min[1], xy_max[1])
    ax_shape.set_xticks([])
    ax_shape.set_yticks([])
    (shape_line,) = ax_shape.plot([], [], color="C2", lw=1.5)
    shape_title = ax_shape.set_title("", fontsize=11)

    def update(frame_idx: int) -> Iterable[matplotlib.artist.Artist]:
        fs = frames[frame_idx]
        spline, dct_curve = cp_spline_and_dct(handle_x, fs.handle_y, x_grid, k_modes)
        spline_line.set_data(x_grid, spline)
        dct_line.set_data(x_grid, dct_curve)
        handles_line.set_data(handle_x, fs.handle_y)

        xy = shapes[fs.target_idx]
        closed = np.vstack([xy, xy[:1]])
        shape_line.set_data(closed[:, 0], closed[:, 1])

        is_ood = ood_flags[fs.target_idx]
        label = labels[fs.target_idx]
        shape_title.set_text(f"{label}{'  (OOD)' if is_ood else ''}")
        shape_title.set_color("C3" if is_ood else "black")
        return (spline_line, dct_line, handles_line, shape_line, shape_title)

    anim = manim.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000.0 / fps,
        blit=False,
    )

    if out_path_mp4 is not None:
        if manim.writers.is_available("ffmpeg"):
            anim.save(str(out_path_mp4), writer="ffmpeg", fps=fps)
        else:
            print(
                f"[visualisation] ffmpeg not available; skipping MP4 "
                f"output at {out_path_mp4}"
            )

    if out_path_gif is not None:
        anim.save(str(out_path_gif), writer="pillow", fps=fps)

    plt.close(fig)
