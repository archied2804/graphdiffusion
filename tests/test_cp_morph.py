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
