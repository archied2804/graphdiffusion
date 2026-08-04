# Cp-Morph Generation Animation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a self-running gif/mp4 in which a Cp spline programmatically morphs through three in-distribution Cp targets and one 3σ OOD target, with the EXP-021 conditional diffusion model regenerating one boundary shape at each target.

**Architecture:** A new filesystem-free visualisation module (`cp_morph.py`) provides pure curve/keyframe/tween helpers plus a `FuncAnimation` writer that takes *pre-sampled* shapes (so it is testable without the model). A thin script (`make_cp_morph_animation.py`) loads the EXP-021 model via `load_exp020`, reads the target Cp CSVs from the EXP-043 sweep dirs, samples one shape per target, and drives the writer. The spline/DCT maths shared with the existing `CpCurveEditor` is extracted into a single helper so both render identically.

**Tech Stack:** Python 3.9+, NumPy, SciPy (`CubicSpline`), matplotlib (`animation.FuncAnimation`, Agg/pillow/ffmpeg), PyTorch + PyTorch Geometric (model sampling, in the script only).

---

## File Structure

- **Create** `src/graph_diffusion/visualisation/cp_morph.py` — pure helpers (`build_handle_keyframes`, `tween_handles`, `FrameSpec`) + the `write_cp_morph_animation` writer. No imports from `model/`.
- **Modify** `src/graph_diffusion/visualisation/cp_editor.py` — extract `cp_spline_and_dct(...)` module function; refactor `CpCurveEditor._spline_curve` / `_compute_dct_curve` to call it.
- **Create** `scripts/make_cp_morph_animation.py` — model/data wiring + CSV loaders + CLI.
- **Create** `tests/test_cp_morph.py` — unit + smoke tests for the new module.
- **Modify** `tests/test_cp_editor.py` — add one test for the extracted helper.

Conventions to follow: tests start with `from __future__ import annotations`, set `matplotlib.use("Agg")` before importing pyplot, and import the package via `from context import graph_diffusion  # noqa: F401`. Lowercase generics (`list[int]`), `torch.Tensor`, Google docstrings, module-level docstring in every file.

---

### Task 1: Extract shared spline/DCT helper into `cp_editor.py`

**Files:**
- Modify: `src/graph_diffusion/visualisation/cp_editor.py`
- Test: `tests/test_cp_editor.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cp_editor.py` (after the existing imports — `dct_ii`, `dct_ii_inverse`, and `numpy as np` are already imported there):

```python
from graph_diffusion.visualisation.cp_editor import (  # noqa: E402
    CpCurveEditor,
    cp_spline_and_dct,
)


def test_cp_spline_and_dct_shapes_and_dct_identity() -> None:
    handle_x = np.linspace(0.0, 1.0, 7, dtype=np.float32)
    handle_y = np.array([0.6, 0.2, -0.3, -0.8, -0.4, 0.1, 0.5], dtype=np.float32)
    x_grid = np.linspace(0.0, 1.0, 64, dtype=np.float32)

    spline, dct_curve = cp_spline_and_dct(handle_x, handle_y, x_grid, k_modes=8)

    assert spline.shape == (64,)
    assert dct_curve.shape == (64,)
    expected = dct_ii_inverse(dct_ii(spline, 8), n_samples=64)
    np.testing.assert_allclose(dct_curve, expected, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cp_editor.py::test_cp_spline_and_dct_shapes_and_dct_identity -v`
Expected: FAIL with `ImportError: cannot import name 'cp_spline_and_dct'`.

- [ ] **Step 3: Add the helper and refactor the editor to use it**

In `src/graph_diffusion/visualisation/cp_editor.py`, add this module-level function after the imports and `_DEFAULT_HANDLE_X` constant (before `class CpCurveEditor`):

```python
def cp_spline_and_dct(
    handle_x: np.ndarray,
    handle_y: np.ndarray,
    x_grid: np.ndarray,
    k_modes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Natural cubic spline through the handles and its DCT-k reconstruction.

    Args:
        handle_x: Handle x-positions, shape ``(n_handles,)``.
        handle_y: Handle y-values, shape ``(n_handles,)``.
        x_grid: Dense ``x/c`` grid, shape ``(n_grid,)``.
        k_modes: Number of DCT-II coefficients kept.

    Returns:
        Tuple ``(spline, dct_curve)``, each shape ``(n_grid,)`` and
        ``float32``. ``spline`` is the cubic-spline curve the user
        "draws"; ``dct_curve`` is the DCT-k reconstruction the model sees.
    """
    spline_fn = CubicSpline(handle_x, handle_y, bc_type="natural")
    spline: np.ndarray = spline_fn(x_grid).astype(np.float32)
    coeffs = dct_ii(spline, int(k_modes))
    dct_curve: np.ndarray = dct_ii_inverse(coeffs, int(x_grid.shape[0])).astype(
        np.float32
    )
    return spline, dct_curve
```

Add `dct_ii` to the existing import from `pOnEllipseConditional` (it currently imports only `dct_ii` and `dct_ii_inverse` — confirm both are present; if `dct_ii` is missing, add it):

```python
from graph_diffusion.data.pOnEllipseConditional import dct_ii, dct_ii_inverse
```

Then replace the two private methods so they delegate to the helper:

```python
    def _spline_curve(self) -> np.ndarray:
        spline, _ = cp_spline_and_dct(
            self._control_x, self._control_y, self._x_grid, self.k_modes
        )
        return spline

    def _compute_dct_curve(self) -> np.ndarray:
        _, dct_curve = cp_spline_and_dct(
            self._control_x, self._control_y, self._x_grid, self.k_modes
        )
        return dct_curve
```

- [ ] **Step 4: Run the editor tests to verify the refactor is green**

Run: `pytest tests/test_cp_editor.py -v`
Expected: PASS (the new helper test plus all pre-existing editor tests — they pin the spline/DCT behaviour, so equal output proves the refactor is behaviour-preserving).

- [ ] **Step 5: Commit**

```bash
git add src/graph_diffusion/visualisation/cp_editor.py tests/test_cp_editor.py
git commit -m "refactor(integration): extract cp_spline_and_dct helper from CpCurveEditor"
```

---

### Task 2: `build_handle_keyframes` in new `cp_morph.py`

**Files:**
- Create: `src/graph_diffusion/visualisation/cp_morph.py`
- Test: `tests/test_cp_morph.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cp_morph.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cp_morph.py::test_build_handle_keyframes_matches_interp -v`
Expected: FAIL with `ModuleNotFoundError`/`ImportError` for `cp_morph`.

- [ ] **Step 3: Create the module with the helper**

Create `src/graph_diffusion/visualisation/cp_morph.py`:

```python
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

import numpy as np


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cp_morph.py::test_build_handle_keyframes_matches_interp -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/graph_diffusion/visualisation/cp_morph.py tests/test_cp_morph.py
git commit -m "feat(integration): add build_handle_keyframes for Cp-morph animation"
```

---

### Task 3: `tween_handles` + `FrameSpec`

**Files:**
- Modify: `src/graph_diffusion/visualisation/cp_morph.py`
- Test: `tests/test_cp_morph.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cp_morph.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cp_morph.py::test_tween_handles_frame_count_and_target_progression -v`
Expected: FAIL with `ImportError: cannot import name 'tween_handles'`.

- [ ] **Step 3: Implement `FrameSpec` and `tween_handles`**

Add to `src/graph_diffusion/visualisation/cp_morph.py` — first extend the imports at the top:

```python
from dataclasses import dataclass
```

Then add after `build_handle_keyframes`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cp_morph.py -v`
Expected: PASS (both `cp_morph` tests).

- [ ] **Step 5: Commit**

```bash
git add src/graph_diffusion/visualisation/cp_morph.py tests/test_cp_morph.py
git commit -m "feat(integration): add tween_handles + FrameSpec for Cp-morph animation"
```

---

### Task 4: `write_cp_morph_animation` writer

**Files:**
- Modify: `src/graph_diffusion/visualisation/cp_morph.py`
- Test: `tests/test_cp_morph.py`

- [ ] **Step 1: Write the failing smoke test**

Append to `tests/test_cp_morph.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cp_morph.py::test_write_cp_morph_animation_writes_gif -v`
Expected: FAIL with `ImportError: cannot import name 'write_cp_morph_animation'`.

- [ ] **Step 3: Implement the writer**

Add to the top imports of `src/graph_diffusion/visualisation/cp_morph.py`:

```python
from collections.abc import Iterable
from pathlib import Path

import matplotlib
import matplotlib.animation as manim
import matplotlib.artist
import matplotlib.pyplot as plt

from graph_diffusion.visualisation.cp_editor import cp_spline_and_dct
```

Then append the writer:

```python
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
```

Note: `matplotlib` is imported for the `matplotlib.artist` type annotation used in `update`'s return type; keep the import even though only the submodule is referenced.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cp_morph.py -v`
Expected: PASS (all three `cp_morph` tests; the GIF smoke test writes a non-empty file via the pillow writer).

- [ ] **Step 5: Run the focused quality gate on the new module**

Run: `uv run ruff check src/graph_diffusion/visualisation/cp_morph.py tests/test_cp_morph.py && uv run black --check src/graph_diffusion/visualisation/cp_morph.py tests/test_cp_morph.py && uv run mypy src/graph_diffusion/visualisation/cp_morph.py`
Expected: all pass. Fix any reported issues (e.g. run `black` to format) and re-run.

- [ ] **Step 6: Commit**

```bash
git add src/graph_diffusion/visualisation/cp_morph.py tests/test_cp_morph.py
git commit -m "feat(integration): add write_cp_morph_animation gif/mp4 writer"
```

---

### Task 5: Driver script `make_cp_morph_animation.py`

**Files:**
- Create: `scripts/make_cp_morph_animation.py`

This task has no unit test (it needs the trained checkpoint and the dataset download); it is verified by a real run in Task 6. Keep the logic thin — all reusable maths already lives in tested helpers.

- [ ] **Step 1: Create the script**

Create `scripts/make_cp_morph_animation.py`:

```python
"""
make_cp_morph_animation
=======================

Render the scripted Cp-morph generation clip: the Cp spline morphs through
three in-distribution target Cp curves (auto-selected from the EXP-043
``eval_grid`` sweep to span the Reynolds range) and one 3-sigma OOD target
(``pc1_p3sig`` from the ``ood_systematic`` sweep). One shape is sampled per
target from the EXP-021 conditional diffusion model; the morph itself is
cheap (no per-frame sampling).

Example:
    python scripts/make_cp_morph_animation.py \
        --experiment-dir outputs/EXP-021_long_train \
        --config configs/EXP-021_long_train.yaml --device cpu
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from graph_diffusion.data.pOnEllipseConditional import dct_ii
from graph_diffusion.postprocessing.inference import (
    radial_to_xy,
    sample_shapes_from_cond,
    template_thetas,
)
from graph_diffusion.postprocessing.loaders import load_exp020
from graph_diffusion.visualisation.cp_morph import write_cp_morph_animation

N_CP_GRID = 128
EVAL_GRID_DIR = Path("cfd/ellipse2d/shapes/eval_grid")
OOD_DIR = Path("cfd/ellipse2d/shapes/ood_systematic")


def _read_cp_csv(path: Path, value_col: str, x_grid: np.ndarray) -> np.ndarray:
    """Read a 2-column Cp CSV and resample its curve onto ``x_grid``.

    Args:
        path: CSV with header ``x_over_c,<value_col>``.
        value_col: Name of the Cp column (``cp_ref`` or ``cp_target``).
        x_grid: Dense grid to resample onto, shape ``(N_CP_GRID,)``.

    Returns:
        Dense Cp curve on ``x_grid``, shape ``(N_CP_GRID,)`` ``float32``.
    """
    xs: list[float] = []
    ys: list[float] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            xs.append(float(row["x_over_c"]))
            ys.append(float(row[value_col]))
    order = np.argsort(xs)
    x_sorted = np.asarray(xs)[order]
    y_sorted = np.asarray(ys)[order]
    return np.interp(x_grid, x_sorted, y_sorted).astype(np.float32)


def _select_indist_cases(n_indist: int) -> list[tuple[str, Path]]:
    """Pick ``n_indist`` Re-spanning eval_grid cases as ``(label, csv_path)``.

    Reads ``eval_grid/manifest.csv``, sorts by Reynolds number, and takes
    evenly-spaced cases (first .. last) so the clip traverses the dataset.
    """
    rows: list[dict[str, str]] = []
    with (EVAL_GRID_DIR / "manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda r: float(r["Re"]))
    picks = np.linspace(0, len(rows) - 1, n_indist).round().astype(int)
    selected: list[tuple[str, Path]] = []
    for i in picks:
        row = rows[int(i)]
        label = f"Re={float(row['Re']):.0f}, H={float(row['H']):.1f}"
        csv_path = EVAL_GRID_DIR / f"sim_{row['idx']}_ref_cp.csv"
        selected.append((label, csv_path))
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=Path("outputs/EXP-021_long_train"))
    parser.add_argument("--config", type=str, default="configs/EXP-021_long_train.yaml")
    parser.add_argument("--checkpoint-name", type=str, default="checkpoint_best.pt")
    parser.add_argument("--ood-case", type=str, default="pc1_p3sig")
    parser.add_argument("--n-indist", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x_grid = np.linspace(0.0, 1.0, N_CP_GRID, dtype=np.float32)

    model, dataset, config = load_exp020(
        experiment_dir=str(args.experiment_dir),
        config_path=args.config,
        device=args.device,
        checkpoint_name=args.checkpoint_name,
    )
    k_modes = int(config["ellipse_dataset"].get("k_modes", 8))
    guidance_scale = float(config.get("sampling", {}).get("guidance_scale", 1.0))

    import copy

    template = copy.copy(dataset[0]).to(args.device)
    theta = template_thetas(template)

    # Build the ordered target list: in-distribution cases, then the OOD climax.
    targets_meta: list[tuple[str, np.ndarray, bool]] = []
    for label, csv_path in _select_indist_cases(args.n_indist):
        targets_meta.append((label, _read_cp_csv(csv_path, "cp_ref", x_grid), False))
    ood_csv = OOD_DIR / f"{args.ood_case}_target_cp.csv"
    targets_meta.append(
        (f"{args.ood_case}", _read_cp_csv(ood_csv, "cp_target", x_grid), True)
    )

    targets = np.stack([cp for _, cp, _ in targets_meta], axis=0)
    labels = [lab for lab, _, _ in targets_meta]
    ood_flags = [flag for _, _, flag in targets_meta]

    # Sample one shape per target (single sample = fast, reproducible).
    shapes: list[np.ndarray] = []
    for _, cp_dense, _ in targets_meta:
        cond_vec = torch.from_numpy(dct_ii(cp_dense, k_modes)).float()
        radii, _ = sample_shapes_from_cond(
            model=model,
            template=template,
            cond_vec=cond_vec,
            n_samples=1,
            guidance_scale=guidance_scale,
            device=args.device,
            seed=args.seed,
        )
        shapes.append(radial_to_xy(radii[0], theta))

    out_dir = args.out_dir if args.out_dir is not None else args.experiment_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_cp_morph_animation(
        targets=targets,
        shapes=shapes,
        labels=labels,
        ood_flags=ood_flags,
        handle_x=np.linspace(0.0, 1.0, 7, dtype=np.float32),
        x_grid=x_grid,
        k_modes=k_modes,
        out_path_mp4=out_dir / "cp_morph.mp4",
        out_path_gif=out_dir / "cp_morph.gif",
        fps=args.fps,
    )
    print(f"[cp_morph] wrote {out_dir / 'cp_morph.gif'} (+ mp4 if ffmpeg available)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Lint/format the script**

Run: `uv run ruff check scripts/make_cp_morph_animation.py && uv run black --check scripts/make_cp_morph_animation.py`
Expected: pass. If black reports formatting (e.g. the long argparse lines), run `uv run black scripts/make_cp_morph_animation.py` and re-run ruff.

- [ ] **Step 3: Commit**

```bash
git add scripts/make_cp_morph_animation.py
git commit -m "feat(integration): add make_cp_morph_animation driver script"
```

---

### Task 6: End-to-end verification run + full quality gate

**Files:** none (verification only).

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest tests/test_cp_morph.py tests/test_cp_editor.py -q`
Expected: PASS (no regressions in the editor; all morph tests green).

- [ ] **Step 2: Run the full project quality gate**

Run: `uv run ruff check src/ tests/ && uv run black --check src/ tests/ && uv run mypy src/`
Expected: all pass. Fix anything reported.

- [ ] **Step 3: Generate the real clip on CPU**

Run:
```bash
python scripts/make_cp_morph_animation.py \
    --experiment-dir outputs/EXP-021_long_train \
    --config configs/EXP-021_long_train.yaml --device cpu
```
Expected: prints `[cp_morph] wrote outputs/EXP-021_long_train/cp_morph.gif ...` and the file exists. If the EXP-021 dataset is not yet downloaded, this triggers the HuggingFace download once (expected on first run). If `checkpoint_best.pt` is missing, pass `--checkpoint-name checkpoint.pt`.

- [ ] **Step 4: Eyeball the output**

Open `outputs/EXP-021_long_train/cp_morph.gif`. Confirm: the spline (grey) + DCT-8 (blue) curve morphs smoothly through 4 Cp shapes; the red handles slide; the right-panel shape swaps at each target; the final `pc1_p3sig` frame title is red with "(OOD)".

- [ ] **Step 5: Commit the generated artifact (optional)**

The repo already tracks per-experiment `figure_b_reverse.gif/mp4` under `outputs/`, so committing the clip matches convention:
```bash
git add outputs/EXP-021_long_train/cp_morph.gif outputs/EXP-021_long_train/cp_morph.mp4
git commit -m "docs(integration): add EXP-021 Cp-morph generation clip"
```

---

## Self-Review

**Spec coverage:**
- Reuse editor + `sample_shapes_from_cond` framework → Task 1 (helper extraction), Task 5 (sampling). ✓
- Smooth morph, resample at targets (pre-sample 4 shapes, swap on arrival) → Task 3 (`tween_handles` target tagging), Task 4 (writer uses `shapes[target_idx]`), Task 5 (one sample/target). ✓
- 3 Re-spanning in-dist + 1 OOD `pc1_p3sig` → Task 5 (`_select_indist_cases`, `--ood-case`). ✓
- Two-panel layout, OOD flagged, fixed axis limits → Task 4. ✓
- Outputs to `outputs/EXP-021_long_train/cp_morph.{mp4,gif}` → Task 5. ✓
- Tests (pure helpers + gif smoke + editor parity) → Tasks 1-4. ✓
- Quality gate → Task 6. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; exact commands with expected output. ✓

**Type consistency:** `cp_spline_and_dct(handle_x, handle_y, x_grid, k_modes)` defined in Task 1, used identically in Task 4. `build_handle_keyframes(targets, handle_x, x_grid)` and `tween_handles(keyframes, n_tween, n_hold)` and `FrameSpec(handle_y, target_idx)` defined in Tasks 2-3, used consistently in Task 4. `write_cp_morph_animation(...)` signature in Task 4 matches the call in Task 5. `load_exp020`, `sample_shapes_from_cond`, `radial_to_xy`, `template_thetas`, `dct_ii` used with their real signatures verified against the codebase. ✓
