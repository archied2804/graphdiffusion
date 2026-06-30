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
