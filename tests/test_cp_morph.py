"""
Tests for graph_diffusion.visualisation.cp_morph
=================================================

Unit tests for the scripted Cp-morph animation helpers. The writer is
exercised with stub pre-sampled shapes so no diffusion model is needed.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend for CI

import numpy as np  # noqa: E402
from context import graph_diffusion  # noqa: F401, E402

from graph_diffusion.visualisation.cp_morph import (  # noqa: E402
    build_handle_keyframes,
)


def test_build_handle_keyframes_matches_interp() -> None:
    rng = np.random.default_rng(0)
    targets = rng.standard_normal((3, 32)).astype(np.float32)
    handle_x = np.linspace(0.0, 1.0, 7, dtype=np.float32)
    x_grid = np.linspace(0.0, 1.0, 32, dtype=np.float32)

    keyframes = build_handle_keyframes(targets, handle_x, x_grid)

    assert keyframes.shape == (3, 7)
    for t in range(3):
        expected = np.interp(handle_x, x_grid, targets[t])
        np.testing.assert_allclose(keyframes[t], expected, atol=1e-6)


from graph_diffusion.visualisation.cp_morph import tween_handles  # noqa: E402


def test_tween_handles_frame_count_and_target_progression() -> None:
    rng = np.random.default_rng(1)
    keyframes = rng.standard_normal((3, 7)).astype(np.float32)
    n_tween, n_hold = 4, 2

    frames = tween_handles(keyframes, n_tween=n_tween, n_hold=n_hold)

    # 3 holds + 2 tweens between them.
    assert len(frames) == 3 * n_hold + 2 * n_tween

    target_idx = [f.target_idx for f in frames]
    # Non-decreasing, starts at 0, ends at the last target.
    assert target_idx == sorted(target_idx)
    assert target_idx[0] == 0
    assert target_idx[-1] == 2

    # First hold frame sits exactly on keyframe 0.
    np.testing.assert_allclose(frames[0].handle_y, keyframes[0], atol=1e-6)
    # The first tween block ends on keyframe 1.
    first_tween_last = frames[n_hold + n_tween - 1]
    np.testing.assert_allclose(first_tween_last.handle_y, keyframes[1], atol=1e-6)


from graph_diffusion.visualisation.cp_morph import (  # noqa: E402
    write_cp_morph_animation,
)


def test_write_cp_morph_animation_writes_gif(tmp_path) -> None:
    n_grid, n_handles = 16, 7
    x_grid = np.linspace(0.0, 1.0, n_grid, dtype=np.float32)
    handle_x = np.linspace(0.0, 1.0, n_handles, dtype=np.float32)
    targets = np.stack(
        [np.linspace(0.6, -0.6, n_grid), np.linspace(-0.4, 0.4, n_grid)],
        axis=0,
    ).astype(np.float32)
    # Two stub shapes (closed-ish unit polygons), shape (N_nodes, 2).
    theta = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    shape_a = np.stack([np.cos(theta), 0.5 * np.sin(theta)], axis=1).astype(np.float32)
    shape_b = np.stack([0.7 * np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)

    out_gif = tmp_path / "cp_morph.gif"
    write_cp_morph_animation(
        targets=targets,
        shapes=[shape_a, shape_b],
        labels=["in-dist", "OOD 3sigma"],
        ood_flags=[False, True],
        handle_x=handle_x,
        x_grid=x_grid,
        k_modes=4,
        out_path_mp4=None,
        out_path_gif=out_gif,
        n_tween=2,
        n_hold=1,
        fps=5,
    )

    assert out_gif.exists()
    assert out_gif.stat().st_size > 0
