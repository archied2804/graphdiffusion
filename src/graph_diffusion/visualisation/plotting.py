"""
graph_diffusion.visualisation.plotting
========================================

Matplotlib figure builders for diffusion experiment writeups: a
conditioning grid, a forward+reverse trajectory filmstrip, and an
animation writer for the reverse pass.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.animation as manim
import matplotlib.artist
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "plot_conditioning_grid",
    "plot_trajectory_filmstrip",
    "write_trajectory_animation",
]


def plot_conditioning_grid(
    target_cps: list[np.ndarray],
    sample_shapes: list[list[np.ndarray]],
    head_pred_cps: list[np.ndarray],
    head_pred_stds: list[np.ndarray],
    row_labels: list[str],
    figsize: tuple[float, float] = (16, 10),
) -> matplotlib.figure.Figure:
    """Render the Figure A 4×(2+S) conditioning grid.

    Args:
        target_cps: One Cp curve per row, shape ``(N_cp_grid,)``.
        sample_shapes: Per-row list of generated boundary shapes, each
            shape ``(N_nodes, 2)`` in (x, y).
        head_pred_cps: Per-row mean head-predicted Cp, shape ``(N_cp_grid,)``.
        head_pred_stds: Per-row std across samples, shape ``(N_cp_grid,)``.
        row_labels: Per-row labels (left ylabel of column 0).
        figsize: Figure size in inches.

    Returns:
        The created figure. Caller is responsible for ``fig.savefig(...)``.
    """
    n_targets = len(target_cps)
    n_samples = len(sample_shapes[0])
    ncols = 2 + n_samples
    fig, axes = plt.subplots(n_targets, ncols, figsize=figsize, squeeze=False)
    x_grid = np.linspace(0.0, 1.0, target_cps[0].shape[0])

    for r in range(n_targets):
        ax = axes[r, 0]
        ax.plot(x_grid, target_cps[r], color="C0", lw=2)
        ax.set_ylabel(row_labels[r], fontsize=10)
        if r == 0:
            ax.set_title("target Cp(x/c)", fontsize=10)
        ax.grid(alpha=0.3)

        ax = axes[r, 1]
        mean = head_pred_cps[r]
        std = head_pred_stds[r]
        ax.plot(x_grid, mean, color="C1", lw=2, label="head pred")
        ax.fill_between(x_grid, mean - std, mean + std, alpha=0.3, color="C1")
        ax.plot(x_grid, target_cps[r], color="C0", lw=1, ls="--", label="target")
        if r == 0:
            ax.set_title("predicted Cp ±σ", fontsize=10)
            ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)

        for s in range(n_samples):
            ax = axes[r, 2 + s]
            xy = sample_shapes[r][s]
            closed = np.vstack([xy, xy[:1]])
            ax.plot(closed[:, 0], closed[:, 1], color="C2", lw=1.5)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"sample {s + 1}", fontsize=10)

    fig.tight_layout()
    return fig


def plot_trajectory_filmstrip(
    forward_snapshots: list[np.ndarray],
    reverse_snapshots: list[np.ndarray],
    timesteps: list[int],
    target_cp: np.ndarray,
    figsize: tuple[float, float] = (14, 5),
) -> matplotlib.figure.Figure:
    """Two-row diffusion-trajectory filmstrip + side Cp panel.

    Top row plots forward noising at the given timesteps; bottom row
    plots reverse denoising at the same timesteps. A side panel shows
    the target Cp curve.

    Args:
        forward_snapshots: F shapes captured during forward diffusion,
            each ``(N_nodes, 2)``.
        reverse_snapshots: F shapes captured during reverse diffusion,
            each ``(N_nodes, 2)``.
        timesteps: F timesteps the snapshots correspond to (used for
            column titles). Length must match the snapshot lists.
        target_cp: Target pressure curve, shape ``(N_cp_grid,)``.
        figsize: Figure size in inches.

    Returns:
        The created figure.
    """
    n_frames = len(timesteps)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, n_frames + 1, width_ratios=[1] * n_frames + [1.2])

    for col in range(n_frames):
        ax_fwd = fig.add_subplot(gs[0, col])
        xy = forward_snapshots[col]
        closed = np.vstack([xy, xy[:1]])
        ax_fwd.plot(closed[:, 0], closed[:, 1], color="C3", lw=1.2)
        ax_fwd.set_aspect("equal")
        ax_fwd.set_xticks([])
        ax_fwd.set_yticks([])
        ax_fwd.set_title(f"t={timesteps[col]}", fontsize=9)
        if col == 0:
            ax_fwd.set_ylabel("forward", fontsize=10)

        ax_rev = fig.add_subplot(gs[1, col])
        xy = reverse_snapshots[col]
        closed = np.vstack([xy, xy[:1]])
        ax_rev.plot(closed[:, 0], closed[:, 1], color="C2", lw=1.2)
        ax_rev.set_aspect("equal")
        ax_rev.set_xticks([])
        ax_rev.set_yticks([])
        if col == 0:
            ax_rev.set_ylabel("reverse", fontsize=10)

    ax_cp = fig.add_subplot(gs[:, n_frames])
    x_grid = np.linspace(0.0, 1.0, target_cp.shape[0])
    ax_cp.plot(x_grid, target_cp, color="C0", lw=2)
    ax_cp.set_title("target Cp(x/c)", fontsize=10)
    ax_cp.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def write_trajectory_animation(
    reverse_snapshots: list[np.ndarray],
    target_cp: np.ndarray,
    out_path_mp4: Path | None,
    out_path_gif: Path | None = None,
    fps: int = 25,
) -> None:
    """Render the reverse-diffusion trajectory as MP4 and/or GIF.

    Each frame is a side-by-side figure: the current shape on the left,
    the target Cp curve on the right. The shape line is replotted each
    frame; axis bounds are fixed to the global min/max across all frames
    so the animation doesn't jitter.

    Args:
        reverse_snapshots: T+1 shapes ``(N_nodes, 2)``, ordered from
            noise (frame 0) to clean (final frame).
        target_cp: Static target Cp curve shown alongside.
        out_path_mp4: Where to write the MP4. Skip MP4 if ``None`` or
            if ``ffmpeg`` is unavailable (logs a warning).
        out_path_gif: Where to write the GIF. Skip GIF if ``None``.
        fps: Frames per second.

    Returns:
        None. Both files are written to disk.
    """
    all_xy = np.concatenate(reverse_snapshots, axis=0)
    xy_min = all_xy.min(axis=0) - 0.1
    xy_max = all_xy.max(axis=0) + 0.1

    fig, (ax_shape, ax_cp) = plt.subplots(
        1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [1, 1.2]}
    )
    ax_shape.set_aspect("equal")
    ax_shape.set_xlim(xy_min[0], xy_max[0])
    ax_shape.set_ylim(xy_min[1], xy_max[1])
    ax_shape.set_xticks([])
    ax_shape.set_yticks([])
    (line,) = ax_shape.plot([], [], color="C2", lw=1.5)

    x_grid = np.linspace(0.0, 1.0, target_cp.shape[0])
    ax_cp.plot(x_grid, target_cp, color="C0", lw=2)
    ax_cp.set_title("target Cp(x/c)", fontsize=10)
    ax_cp.grid(alpha=0.3)

    title = ax_shape.set_title("", fontsize=10)

    def update(frame_idx: int) -> Iterable[matplotlib.artist.Artist]:
        xy = reverse_snapshots[frame_idx]
        closed = np.vstack([xy, xy[:1]])
        line.set_data(closed[:, 0], closed[:, 1])
        title.set_text(f"step {frame_idx + 1}/{len(reverse_snapshots)}")
        return (line, title)

    anim = manim.FuncAnimation(
        fig,
        update,
        frames=len(reverse_snapshots),
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
