"""Tests for graph_diffusion.visualisation.dataset_overview."""

import numpy as np
import pytest
from context import graph_diffusion  # noqa: F401

from graph_diffusion.visualisation.dataset_overview import (
    pressure_drag_coefficient,
    sim_summary,
)


def _unit_circle(n: int) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float64)


def test_uniform_cp_gives_zero_drag():
    xy = _unit_circle(64)
    cp = np.ones(64)
    assert pressure_drag_coefficient(xy, cp) == pytest.approx(0.0, abs=1e-9)


def test_drag_sign_flips_with_cp_sign():
    xy = _unit_circle(128)
    # Front-loaded pressure: high Cp where x < 0 (upstream face).
    cp = -xy[:, 0]
    cd_pos = pressure_drag_coefficient(xy, cp)
    cd_neg = pressure_drag_coefficient(xy, -cp)
    assert cd_pos == pytest.approx(-cd_neg, rel=1e-9)
    assert abs(cd_pos) > 1e-3  # non-degenerate


def test_drag_matches_analytic_line_integral():
    # For cp = -x on the unit circle, Cd_p = oint (-x) dy = -area = -pi.
    xy = _unit_circle(512)
    cp = -xy[:, 0]
    assert pressure_drag_coefficient(xy, cp) == pytest.approx(-np.pi, rel=1e-2)


def _fake_raw(n_sims: int = 4, max_nodes: int = 16) -> np.ndarray:
    rng = np.random.default_rng(0)
    raw = np.full((n_sims, max_nodes, 105), np.nan, dtype=np.float32)
    for s in range(n_sims):
        n = 12
        theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        a, b = 0.5, 0.25 + 0.1 * s / n_sims
        raw[s, :n, 0] = (a * np.cos(theta)).astype(np.float32)
        raw[s, :n, 1] = (b * np.sin(theta)).astype(np.float32)
        raw[s, :n, 2] = 500.0 + 100.0 * s  # Re
        raw[s, :n, 3] = 5.0 + 0.2 * s  # H
        raw[s, :n, 4:105] = rng.standard_normal((n, 101)).astype(np.float32)
    return raw


def test_sim_summary_fields():
    raw = _fake_raw()
    s = sim_summary(raw, 1)
    assert s["re"] == pytest.approx(600.0)
    assert s["h"] == pytest.approx(5.2, abs=1e-5)
    assert s["polygon"].shape[1] == 2
    assert s["x_over_c"].min() >= 0.0 and s["x_over_c"].max() <= 1.0
    assert np.isfinite(s["cd_p"])
    assert s["cp_mean"].shape[0] == s["polygon"].shape[0]
