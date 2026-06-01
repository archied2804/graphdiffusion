"""
Tests for graph_diffusion.visualisation.cp_editor
===================================================

Unit tests for the :class:`CpCurveEditor` widget. Mouse interaction is
not simulated end-to-end (matplotlib event handling is awkward to
fake); instead we exercise the deterministic computation paths that
back the widget: initial state, programmatic ``set_cp``, programmatic
handle moves, and callback dispatch.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend for CI

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from context import graph_diffusion  # noqa: F401, E402

from graph_diffusion.data.pOnEllipseConditional import (  # noqa: E402
    dct_ii,
    dct_ii_inverse,
)
from graph_diffusion.visualisation.cp_editor import CpCurveEditor  # noqa: E402


def _make_axes() -> plt.Axes:
    fig, ax = plt.subplots()
    return ax


def test_editor_initial_cond_vector_shape() -> None:
    ax = _make_axes()
    initial = np.linspace(0.5, -1.5, 64, dtype=np.float32)
    editor = CpCurveEditor(ax, initial, n_control_points=7, k_modes=8)
    cond = editor.get_cond_vector()
    assert cond.shape == (8,)
    assert cond.dtype == np.float32
    plt.close(ax.figure)


def test_editor_initial_dct_curve_matches_inverse_dct_of_initial() -> None:
    # The blue (DCT-8) curve drawn by the editor should equal
    # dct_ii_inverse(dct_ii(initial, K), N) once the editor is built.
    ax = _make_axes()
    n_grid = 64
    initial = np.sin(np.linspace(0.0, 2.0 * np.pi, n_grid)).astype(np.float32)
    editor = CpCurveEditor(ax, initial, n_control_points=7, k_modes=8)

    cond = editor.get_cond_vector()
    dct_curve = editor.get_dct_curve()
    expected_curve = dct_ii_inverse(cond, n_samples=n_grid)
    np.testing.assert_allclose(dct_curve, expected_curve, atol=1e-5)
    plt.close(ax.figure)


def test_editor_set_cp_updates_cond_vector() -> None:
    ax = _make_axes()
    initial = np.zeros(64, dtype=np.float32)
    editor = CpCurveEditor(ax, initial, n_control_points=7, k_modes=8)
    cond_initial = editor.get_cond_vector().copy()

    new_curve = np.cos(np.linspace(0.0, 2.0 * np.pi, 64)).astype(np.float32)
    editor.set_cp(new_curve)
    cond_after = editor.get_cond_vector()
    assert not np.allclose(cond_initial, cond_after)
    # Self-consistency: the cond vector and the displayed DCT curve must
    # both come from the same spline.
    np.testing.assert_allclose(
        editor.get_dct_curve(), dct_ii_inverse(cond_after, 64), atol=1e-5
    )
    np.testing.assert_allclose(
        cond_after, dct_ii(editor.get_spline_curve(), 8), atol=1e-5
    )
    plt.close(ax.figure)


def test_editor_set_handle_changes_cond_without_firing_callback() -> None:
    fired: list[np.ndarray] = []

    def on_release(c: np.ndarray) -> None:
        fired.append(c)

    ax = _make_axes()
    initial = np.zeros(64, dtype=np.float32)
    editor = CpCurveEditor(
        ax,
        initial,
        n_control_points=7,
        k_modes=8,
        on_release=on_release,
    )
    cond_before = editor.get_cond_vector().copy()
    editor.set_handle(3, -1.0)
    cond_after = editor.get_cond_vector()
    assert not np.allclose(cond_before, cond_after)
    # Programmatic move must NOT fire the user-release callback.
    assert fired == []
    plt.close(ax.figure)


def test_editor_fire_release_invokes_callback_with_cond() -> None:
    fired: list[np.ndarray] = []

    def on_release(c: np.ndarray) -> None:
        fired.append(c.copy())

    ax = _make_axes()
    initial = np.zeros(64, dtype=np.float32)
    editor = CpCurveEditor(
        ax,
        initial,
        n_control_points=7,
        k_modes=8,
        on_release=on_release,
    )
    editor.set_handle(2, 0.5)
    editor.fire_release()  # simulates a mouse-release end-of-drag
    assert len(fired) == 1
    np.testing.assert_allclose(fired[0], editor.get_cond_vector())
    plt.close(ax.figure)


def test_editor_raises_on_too_few_control_points() -> None:
    ax = _make_axes()
    initial = np.zeros(64, dtype=np.float32)
    try:
        CpCurveEditor(ax, initial, n_control_points=1, k_modes=8)
    except ValueError as e:
        assert "n_control_points" in str(e)
    else:
        raise AssertionError("expected ValueError")
    plt.close(ax.figure)
