"""Tests for graph_diffusion.visualisation.dataset_overview."""

import numpy as np
import pytest
from context import graph_diffusion  # noqa: F401

from graph_diffusion.visualisation.dataset_overview import (
    pressure_drag_coefficient,
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
