"""
Tests for graph_diffusion.postprocessing.metrics
==================================================

Boundary-roughness metric for comparing generated shapes across the
EXP-021..025 training-improvement series.
"""

from __future__ import annotations

import numpy as np
from context import graph_diffusion  # noqa: F401

from graph_diffusion.postprocessing.metrics import compute_boundary_roughness


def _circle_xy(n: int, r: float = 1.0) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1).astype(np.float32)


def test_perfect_circle_has_zero_roughness() -> None:
    xy = _circle_xy(64, r=1.0)
    roughness = compute_boundary_roughness(xy)
    # A clean N-point circle has uniform radius; second-difference of r(θ) ≡ 0
    # up to float32 noise in np.linalg.norm.
    assert roughness < 1e-6


def test_noisy_circle_has_positive_roughness() -> None:
    rng = np.random.default_rng(0)
    n = 64
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r_noisy = 1.0 + 0.1 * rng.standard_normal(n)
    xy = np.stack([r_noisy * np.cos(theta), r_noisy * np.sin(theta)], axis=1).astype(
        np.float32
    )
    roughness = compute_boundary_roughness(xy)
    assert roughness > 0.0


def test_rougher_shape_scores_higher() -> None:
    rng = np.random.default_rng(1)
    n = 64
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r_smooth = 1.0 + 0.01 * rng.standard_normal(n)
    r_rough = 1.0 + 0.20 * rng.standard_normal(n)
    smooth_xy = np.stack(
        [r_smooth * np.cos(theta), r_smooth * np.sin(theta)], axis=1
    ).astype(np.float32)
    rough_xy = np.stack(
        [r_rough * np.cos(theta), r_rough * np.sin(theta)], axis=1
    ).astype(np.float32)
    assert compute_boundary_roughness(rough_xy) > compute_boundary_roughness(smooth_xy)


def test_batch_input_returns_mean_over_samples() -> None:
    clean = _circle_xy(64, r=1.0)
    rng = np.random.default_rng(2)
    n = 64
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r_noisy = 1.0 + 0.1 * rng.standard_normal(n)
    noisy = np.stack([r_noisy * np.cos(theta), r_noisy * np.sin(theta)], axis=1).astype(
        np.float32
    )

    batch = np.stack([clean, noisy], axis=0)  # (2, N, 2)
    mean_roughness = compute_boundary_roughness(batch)
    clean_only = compute_boundary_roughness(clean)
    noisy_only = compute_boundary_roughness(noisy)
    assert mean_roughness == 0.5 * (clean_only + noisy_only)


def test_roughness_is_rotation_invariant() -> None:
    rng = np.random.default_rng(3)
    n = 64
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r = 1.0 + 0.1 * rng.standard_normal(n)
    xy = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1).astype(np.float32)
    base = compute_boundary_roughness(xy)

    # Rotate the same boundary by 45°: roughness depends on r(θ) shape, not
    # absolute orientation, so the metric must be invariant.
    angle = np.pi / 4.0
    rot = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float32,
    )
    rotated = xy @ rot.T
    rotated_score = compute_boundary_roughness(rotated)
    assert abs(rotated_score - base) < 1e-5
