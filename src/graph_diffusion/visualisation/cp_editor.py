"""
graph_diffusion.visualisation.cp_editor
========================================

A matplotlib widget for editing a Cp(x/c) curve via a handful of
draggable control points. Drives the interactive pressure-conditioning
notebook used to probe the EXP-020 model.

The editor renders three layers on a single :class:`~matplotlib.axes.Axes`:

* the cubic spline through the user-controlled handles (light grey) —
  this is "what the user is drawing";
* the first ``k_modes`` DCT-II reconstruction of that spline (solid
  blue) — this is "what the model actually sees";
* the handles themselves, draggable on left-click.

On mouse release the editor fires an ``on_release`` callback with the
current K-mode cond vector, so the notebook can re-sample shapes.

The class avoids any Jupyter-specific coupling — it works inside any
matplotlib figure with an interactive backend (``%matplotlib widget``
in JupyterLab, ``TkAgg`` from a script, etc.).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event, MouseEvent
from scipy.interpolate import CubicSpline

from graph_diffusion.data.pOnEllipseConditional import dct_ii, dct_ii_inverse

_DEFAULT_HANDLE_X = np.array([0.0, 0.15, 0.30, 0.50, 0.70, 0.85, 1.0], dtype=np.float32)


class CpCurveEditor:
    """Draggable Cp(x/c) curve editor.

    Args:
        ax: The :class:`matplotlib.axes.Axes` to draw into. The editor
            takes over its limits, title, and labels.
        initial_cp: Initial dense Cp curve, shape ``(N_grid,)``. The
            grid is assumed to span ``x/c ∈ [0, 1]`` uniformly.
        n_control_points: Number of draggable handles. Must be ``>= 2``.
            Defaults to ``7`` (positions at ``0, 0.15, 0.30, 0.50,
            0.70, 0.85, 1.0``).
        k_modes: Number of DCT-II coefficients kept when computing the
            cond vector. Defaults to ``8``.
        on_release: Callback fired with the current cond vector
            ``(K,)`` on mouse release after a drag. Pass ``None`` to
            disable.
        handle_pick_radius: Pixel tolerance for picking a handle on
            mouse press. Defaults to ``10``.

    Raises:
        ValueError: If ``n_control_points < 2``.
    """

    def __init__(
        self,
        ax: Axes,
        initial_cp: np.ndarray,
        n_control_points: int = 7,
        k_modes: int = 8,
        on_release: Callable[[np.ndarray], None] | None = None,
        handle_pick_radius: float = 10.0,
    ) -> None:
        if n_control_points < 2:
            raise ValueError(f"n_control_points must be >= 2, got {n_control_points}")
        self.ax = ax
        self.k_modes = int(k_modes)
        self.on_release = on_release
        self.handle_pick_radius = float(handle_pick_radius)

        self._n_grid = int(initial_cp.shape[0])
        self._x_grid = np.linspace(0.0, 1.0, self._n_grid, dtype=np.float32)
        if n_control_points == _DEFAULT_HANDLE_X.shape[0]:
            self._control_x = _DEFAULT_HANDLE_X.copy()
        else:
            self._control_x = np.linspace(0.0, 1.0, n_control_points, dtype=np.float32)
        self._control_y = self._sample_at_handles(initial_cp.astype(np.float32))

        (self._spline_line,) = ax.plot(
            self._x_grid,
            self._spline_curve(),
            color="0.7",
            lw=1.0,
            label="spline",
        )
        (self._dct_line,) = ax.plot(
            self._x_grid,
            self._compute_dct_curve(),
            color="C0",
            lw=1.8,
            label="DCT-8",
        )
        (self._handles_line,) = ax.plot(
            self._control_x,
            self._control_y,
            "o",
            color="C3",
            ms=8,
            label="handles",
        )

        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("x/c")
        ax.set_ylabel("Cp")
        ax.set_title("Cp editor (drag handles)")
        ax.legend(loc="best", fontsize=8)

        self._dragging_idx: int | None = None
        self._cid_press = ax.figure.canvas.mpl_connect(
            "button_press_event", self._on_press
        )
        self._cid_motion = ax.figure.canvas.mpl_connect(
            "motion_notify_event", self._on_motion
        )
        self._cid_release = ax.figure.canvas.mpl_connect(
            "button_release_event", self._on_release
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_cond_vector(self) -> np.ndarray:
        """Return the current cond vector ``(K,)`` (DCT-II of the spline)."""
        return dct_ii(self._spline_curve(), self.k_modes)

    def get_dct_curve(self) -> np.ndarray:
        """Return the dense DCT-K reconstruction ``(N_grid,)`` of the spline."""
        return self._compute_dct_curve()

    def get_spline_curve(self) -> np.ndarray:
        """Return the dense spline ``(N_grid,)`` (what the user is drawing)."""
        return self._spline_curve()

    def set_cp(self, new_cp: np.ndarray) -> None:
        """Replace the current curve programmatically (e.g. reset button).

        Args:
            new_cp: New dense Cp curve, shape ``(N_grid,)``.
        """
        self._control_y = self._sample_at_handles(new_cp.astype(np.float32))
        self._redraw()

    def set_handle(self, index: int, y: float) -> None:
        """Move a single handle programmatically without firing on_release.

        Useful for tests and "reset to handle defaults" flows.
        """
        self._control_y[index] = float(y)
        self._redraw()

    def fire_release(self) -> None:
        """Manually invoke the ``on_release`` callback (used by tests)."""
        if self.on_release is not None:
            self.on_release(self.get_cond_vector())

    # ------------------------------------------------------------------
    # Internal: curve maths
    # ------------------------------------------------------------------

    def _sample_at_handles(self, dense: np.ndarray) -> np.ndarray:
        out: np.ndarray = np.interp(self._control_x, self._x_grid, dense).astype(
            np.float32
        )
        return out

    def _spline_curve(self) -> np.ndarray:
        spline = CubicSpline(self._control_x, self._control_y, bc_type="natural")
        out: np.ndarray = spline(self._x_grid).astype(np.float32)
        return out

    def _compute_dct_curve(self) -> np.ndarray:
        coeffs = dct_ii(self._spline_curve(), self.k_modes)
        return dct_ii_inverse(coeffs, self._n_grid)

    def _redraw(self) -> None:
        self._spline_line.set_ydata(self._spline_curve())
        self._dct_line.set_ydata(self._compute_dct_curve())
        self._handles_line.set_data(self._control_x, self._control_y)
        if self.ax.figure.canvas is not None:
            self.ax.figure.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Internal: mouse handling
    # ------------------------------------------------------------------

    def _pick_handle(self, event: MouseEvent) -> int | None:
        if event.inaxes is not self.ax:
            return None
        if event.x is None or event.y is None:
            return None
        # Convert handle positions to display coords for pixel-distance picking.
        handle_disp = self.ax.transData.transform(
            np.column_stack([self._control_x, self._control_y])
        )
        d2 = (handle_disp[:, 0] - event.x) ** 2 + (handle_disp[:, 1] - event.y) ** 2
        idx = int(np.argmin(d2))
        if d2[idx] <= self.handle_pick_radius**2:
            return idx
        return None

    def _on_press(self, event: Event) -> None:
        if not isinstance(event, MouseEvent) or event.button != 1:
            return
        self._dragging_idx = self._pick_handle(event)

    def _on_motion(self, event: Event) -> None:
        if self._dragging_idx is None:
            return
        if not isinstance(event, MouseEvent) or event.inaxes is not self.ax:
            return
        if event.ydata is None:
            return
        # First and last handles are fixed in x; only move in y.
        self._control_y[self._dragging_idx] = float(event.ydata)
        self._redraw()

    def _on_release(self, event: Event) -> None:
        del event  # button identity is not needed; we end any active drag.
        if self._dragging_idx is None:
            return
        self._dragging_idx = None
        if self.on_release is not None:
            self.on_release(self.get_cond_vector())
