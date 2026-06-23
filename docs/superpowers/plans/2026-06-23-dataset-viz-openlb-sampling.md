# Dataset Visualisation, OpenLB Domain Correctness & Systematic Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dataset-overview figure suite, make the OpenLB validation domain match the dataset per-sample, and replace the Re-only evaluation sampler with a systematic (Re, H) sampler plus a trackable naming scheme.

**Architecture:** Three independent workstreams. Part A adds a read-only plotting module (`dataset_overview.py`) + a CLI that renders figures from the raw HDF5. Part B hardens the OpenLB C++ case against a wrong channel-height default and adds a manifest-driven run driver. Part C adds an importable, tested (Re, H) sampler + naming helper in the data package and wires it into the export CLI. Part D (model Re-conditioning) is deferred per the spec.

**Tech Stack:** Python 3.13, NumPy, Matplotlib, h5py, PyTorch/PyG (existing), pytest; OpenLB 1.9.0 C++ (`cfd/ellipse2d/`), bash run drivers.

**Spec:** `docs/superpowers/specs/2026-06-23-dataset-viz-physical-conditioning-design.md`

**Conventions to follow (verified in the codebase):**
- HDF5 columns: `_H5_COL_X=0, _H5_COL_Y=1, _H5_COL_RE=2, _H5_COL_H=3, _H5_COL_P_START=4, _H5_N_TIMESTEPS=101` (in `graph_diffusion.data.pOnEllipse`). Re ∈ [500,1000], H ∈ [5,6] (verified empirically).
- Reuse `_load_h5_raw`, `_node_count` from `graph_diffusion.data.pOnEllipse`.
- Tests build synthetic in-memory HDF5-shaped arrays (see `tests/test_pOnEllipse.py::_fake_ellipse_nodes`); do NOT depend on the real `.h5` file in unit tests. Tests import via `from context import graph_diffusion`.
- Plotting functions return a `matplotlib.figure.Figure` and do NO file I/O (see `visualisation/plotting.py`). Module docstring + Google-style docstrings required. `from __future__ import annotations`, lowercase generics, fully-qualified imports.
- Quality gate before each commit: `uv run pytest tests/ -q && uv run ruff check src/ tests/ && uv run black --check src/ tests/ && uv run mypy src/`.

---

## Part A — Dataset visualisation suite

### Task A1: Pressure-drag coefficient integrator

**Files:**
- Create: `src/graph_diffusion/visualisation/dataset_overview.py`
- Test: `tests/test_dataset_overview.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dataset_overview.py -q`
Expected: FAIL — `ImportError`/`ModuleNotFoundError` (module/function not defined).

- [ ] **Step 3: Write minimal implementation**

```python
"""
graph_diffusion.visualisation.dataset_overview
===============================================

Read-only Matplotlib figure builders that characterise the whole
pOnEllipse dataset: a design-space (Re, b) -> pressure-drag contour, the
distribution of surface Cp curves, design-space coverage, and supporting
extras. Figure functions take the raw HDF5 array (or derived arrays) and
return a Figure; they perform no file I/O.
"""

from __future__ import annotations

import numpy as np


def pressure_drag_coefficient(polygon_xy: np.ndarray, cp: np.ndarray) -> float:
    """Pressure-drag coefficient from a closed boundary polygon and nodal Cp.

    Computes the discrete line integral ``Cd_p = oint Cp (n . x_hat) ds``
    around the body. For a counter-clockwise polygon the outward-normal
    times segment-length has x-component ``dy``, so the integral reduces to
    ``sum_i cp_mid_i * (y_{i+1} - y_i)`` with ``cp_mid`` the edge-midpoint
    Cp. The body is pressure-only, so this is pressure drag (no skin
    friction).

    Args:
        polygon_xy (np.ndarray): Boundary points ordered around the body,
            shape ``(N, 2)``. The contour is closed implicitly (wraps).
        cp (np.ndarray): Per-node pressure coefficient, shape ``(N,)``,
            aligned with ``polygon_xy``.

    Returns:
        float: The pressure-drag coefficient.
    """
    if polygon_xy.ndim != 2 or polygon_xy.shape[1] != 2:
        raise ValueError(f"polygon_xy must be (N, 2), got {polygon_xy.shape}")
    if cp.shape[0] != polygon_xy.shape[0]:
        raise ValueError("cp and polygon_xy must have matching length")
    y = polygon_xy[:, 1].astype(np.float64)
    cp = cp.astype(np.float64)
    y_next = np.roll(y, -1)
    cp_next = np.roll(cp, -1)
    cp_mid = 0.5 * (cp + cp_next)
    return float(np.sum(cp_mid * (y_next - y)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dataset_overview.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/graph_diffusion/visualisation/dataset_overview.py tests/test_dataset_overview.py
git commit -m "feat(integration): add pressure-drag coefficient integrator"
```

---

### Task A2: Per-sim accessor helper

**Files:**
- Modify: `src/graph_diffusion/visualisation/dataset_overview.py`
- Test: `tests/test_dataset_overview.py`

A small helper that turns one raw HDF5 sim slice into the quantities every figure needs: ordered polygon, x/c, mean Cp, Re, H, semi-axes a/b, and Cd_p. Keeps the figure functions DRY.

- [ ] **Step 1: Write the failing test** (append to `tests/test_dataset_overview.py`)

```python
from graph_diffusion.visualisation.dataset_overview import sim_summary


def _fake_raw(n_sims: int = 4, max_nodes: int = 16) -> np.ndarray:
    rng = np.random.default_rng(0)
    raw = np.full((n_sims, max_nodes, 105), np.nan, dtype=np.float32)
    for s in range(n_sims):
        n = 12
        theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        a, b = 0.5, 0.25 + 0.1 * s / n_sims
        raw[s, :n, 0] = (a * np.cos(theta)).astype(np.float32)
        raw[s, :n, 1] = (b * np.sin(theta)).astype(np.float32)
        raw[s, :n, 2] = 500.0 + 100.0 * s          # Re
        raw[s, :n, 3] = 5.0 + 0.2 * s              # H
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dataset_overview.py::test_sim_summary_fields -q`
Expected: FAIL — `ImportError` (`sim_summary` not defined).

- [ ] **Step 3: Write minimal implementation** (append to `dataset_overview.py`; add imports at top)

```python
# add to the imports block at the top of the file:
from typing import Any

from graph_diffusion.data.pOnEllipse import (
    _H5_COL_H,
    _H5_COL_P_START,
    _H5_COL_RE,
    _H5_COL_X,
    _H5_COL_Y,
    _H5_N_TIMESTEPS,
    _node_count,
)
```

```python
def _ordered_polygon(xy: np.ndarray) -> np.ndarray:
    """Angular-sort convex boundary points into a CCW simple polygon."""
    centroid = xy.mean(axis=0)
    rel = xy - centroid
    theta = np.arctan2(rel[:, 1], rel[:, 0])
    return xy[np.argsort(theta)]


def sim_summary(raw: np.ndarray, idx: int) -> dict[str, Any]:
    """Derive per-figure quantities for one simulation.

    Args:
        raw (np.ndarray): Full HDF5 array, shape ``(N_sims, MAX_NODES, 105)``.
        idx (int): Simulation index.

    Returns:
        dict[str, Any]: Keys ``re``, ``h``, ``a``, ``b`` (floats),
            ``polygon`` ``(N, 2)`` CCW, ``cp_mean`` ``(N,)`` aligned with
            ``polygon``, ``x_over_c`` ``(N,)`` ascending, ``cp_curve``
            ``(N,)`` Cp sorted by x/c, and ``cd_p`` (float).
    """
    sample = raw[idx]
    n = _node_count(sample)
    data = sample[:n]
    re = float(data[0, _H5_COL_RE])
    h = float(data[0, _H5_COL_H])
    xy = np.stack([data[:, _H5_COL_X], data[:, _H5_COL_Y]], axis=1).astype(np.float64)
    a = float((xy[:, 0].max() - xy[:, 0].min()) / 2.0)
    b = float((xy[:, 1].max() - xy[:, 1].min()) / 2.0)
    cp_mean = data[:, _H5_COL_P_START : _H5_COL_P_START + _H5_N_TIMESTEPS].mean(axis=1)
    cp_mean = cp_mean.astype(np.float64)

    # Ordered polygon + Cp aligned to the same ordering for Cd_p.
    centroid = xy.mean(axis=0)
    order_poly = np.argsort(np.arctan2(xy[:, 1] - centroid[1], xy[:, 0] - centroid[0]))
    polygon = xy[order_poly]
    cp_poly = cp_mean[order_poly]
    cd_p = pressure_drag_coefficient(polygon, cp_poly)

    # Cp(x/c) sorted by chordwise coordinate.
    xc = xy[:, 0]
    chord = max(xc.max() - xc.min(), 1e-12)
    x_over_c = (xc - xc.min()) / chord
    order_xc = np.argsort(x_over_c)

    return {
        "re": re,
        "h": h,
        "a": a,
        "b": b,
        "polygon": polygon,
        "cp_mean": cp_poly,
        "x_over_c": x_over_c[order_xc],
        "cp_curve": cp_mean[order_xc],
        "cd_p": cd_p,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dataset_overview.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/graph_diffusion/visualisation/dataset_overview.py tests/test_dataset_overview.py
git commit -m "feat(integration): add per-sim summary accessor for dataset figures"
```

---

### Task A3: Core figures — drag contour, Cp envelope, coverage scatter

**Files:**
- Modify: `src/graph_diffusion/visualisation/dataset_overview.py`
- Test: `tests/test_dataset_overview.py`

Three figure builders. Correctness of a plot is not unit-testable, so the tests are smoke tests: each returns a `Figure` with at least one axes and no exception on a small synthetic `raw`.

- [ ] **Step 1: Write the failing test** (append)

```python
import matplotlib

matplotlib.use("Agg")
import matplotlib.figure  # noqa: E402

from graph_diffusion.visualisation.dataset_overview import (  # noqa: E402
    plot_design_space_drag,
    plot_cp_envelope,
    plot_design_space_coverage,
)


def test_core_figures_smoke():
    raw = _fake_raw(n_sims=8)
    idxs = list(range(8))
    f1 = plot_design_space_drag(raw, idxs)
    f2 = plot_cp_envelope(raw, idxs, colour_by="re")
    f3 = plot_cp_envelope(raw, idxs, colour_by="b")
    f4 = plot_design_space_coverage(raw, idxs, eval_indices=[0, 7])
    for fig in (f1, f2, f3, f4):
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) >= 1
    matplotlib.pyplot.close("all")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dataset_overview.py::test_core_figures_smoke -q`
Expected: FAIL — `ImportError` (functions not defined).

- [ ] **Step 3: Write minimal implementation** (append; add these to the top imports block — submodules must be imported explicitly)

```python
import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.figure
import matplotlib.pyplot as plt
```

```python
def _summaries(raw: np.ndarray, indices: list[int]) -> list[dict[str, Any]]:
    return [sim_summary(raw, i) for i in indices]


def plot_design_space_drag(
    raw: np.ndarray,
    indices: list[int],
    figsize: tuple[float, float] = (7.0, 5.0),
) -> matplotlib.figure.Figure:
    """Hexbin of pressure-drag coefficient over the (Re, b) design space.

    Args:
        raw (np.ndarray): Full HDF5 array.
        indices (list[int]): Sim indices to include.
        figsize (tuple[float, float]): Figure size in inches.

    Returns:
        matplotlib.figure.Figure: The rendered figure. Colour encodes
            pressure drag (Cd_p); the label states it is pressure-only.
    """
    s = _summaries(raw, indices)
    re = np.array([d["re"] for d in s])
    b = np.array([d["b"] for d in s])
    cd = np.array([d["cd_p"] for d in s])
    fig, ax = plt.subplots(figsize=figsize)
    hb = ax.hexbin(re, b, C=cd, gridsize=25, cmap="viridis")
    fig.colorbar(hb, ax=ax, label=r"pressure drag $C_{d,p}$ (no skin friction)")
    ax.set_xlabel("Reynolds number $Re$")
    ax.set_ylabel("semi-minor axis $b$")
    ax.set_title("Design space coloured by pressure drag")
    fig.tight_layout()
    return fig


def plot_cp_envelope(
    raw: np.ndarray,
    indices: list[int],
    colour_by: str = "re",
    figsize: tuple[float, float] = (7.0, 5.0),
) -> matplotlib.figure.Figure:
    """Overlay all surface Cp(x/c) curves coloured by Re or b, with a band.

    Args:
        raw (np.ndarray): Full HDF5 array.
        indices (list[int]): Sim indices to include.
        colour_by (str): ``"re"`` or ``"b"`` — the per-curve colour key.
        figsize (tuple[float, float]): Figure size in inches.

    Returns:
        matplotlib.figure.Figure: The rendered figure with a mean +/- std
            band over a common x/c grid.

    Raises:
        ValueError: If ``colour_by`` is not ``"re"`` or ``"b"``.
    """
    if colour_by not in ("re", "b"):
        raise ValueError(f"colour_by must be 're' or 'b', got '{colour_by}'")
    s = _summaries(raw, indices)
    key = np.array([d[colour_by] for d in s])
    norm = matplotlib.colors.Normalize(vmin=float(key.min()), vmax=float(key.max()))
    cmap = matplotlib.colormaps["plasma"]
    grid = np.linspace(0.0, 1.0, 100)
    fig, ax = plt.subplots(figsize=figsize)
    resampled = []
    for d in s:
        cp_g = np.interp(grid, d["x_over_c"], d["cp_curve"])
        resampled.append(cp_g)
        ax.plot(grid, cp_g, color=cmap(norm(d[colour_by])), alpha=0.3, linewidth=0.7)
    stack = np.stack(resampled, axis=0)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0)
    ax.plot(grid, mean, color="black", linewidth=2.0, label="mean")
    ax.fill_between(grid, mean - std, mean + std, color="grey", alpha=0.3, label=r"$\pm\sigma$")
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, label={"re": "Reynolds $Re$", "b": "semi-minor $b$"}[colour_by])
    ax.set_xlabel("$x/c$")
    ax.set_ylabel("$C_p$")
    ax.set_title(f"Surface $C_p$ distribution (coloured by {colour_by})")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_design_space_coverage(
    raw: np.ndarray,
    indices: list[int],
    eval_indices: list[int] | None = None,
    figsize: tuple[float, float] = (7.0, 5.0),
) -> matplotlib.figure.Figure:
    """Scatter of (Re, b) coloured by Cd_p, marking evaluation samples.

    Args:
        raw (np.ndarray): Full HDF5 array.
        indices (list[int]): All sim indices (the background cloud).
        eval_indices (list[int] | None): Indices to highlight as the
            chosen evaluation samples.
        figsize (tuple[float, float]): Figure size in inches.

    Returns:
        matplotlib.figure.Figure: The rendered figure.
    """
    s = _summaries(raw, indices)
    re = np.array([d["re"] for d in s])
    b = np.array([d["b"] for d in s])
    cd = np.array([d["cd_p"] for d in s])
    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(re, b, c=cd, cmap="viridis", s=12, alpha=0.7)
    fig.colorbar(sc, ax=ax, label=r"pressure drag $C_{d,p}$")
    if eval_indices:
        es = _summaries(raw, eval_indices)
        ax.scatter(
            [d["re"] for d in es],
            [d["b"] for d in es],
            facecolors="none",
            edgecolors="red",
            s=90,
            linewidths=1.5,
            label="evaluation samples",
        )
        ax.legend()
    ax.set_xlabel("Reynolds number $Re$")
    ax.set_ylabel("semi-minor axis $b$")
    ax.set_title("Design-space coverage")
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dataset_overview.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/graph_diffusion/visualisation/dataset_overview.py tests/test_dataset_overview.py
git commit -m "feat(integration): add drag-contour, Cp-envelope, coverage figures"
```

---

### Task A4: Extra figures — shape gallery, Cd_p-vs-Re binned, Cp PCA projection

**Files:**
- Modify: `src/graph_diffusion/visualisation/dataset_overview.py`
- Test: `tests/test_dataset_overview.py`

- [ ] **Step 1: Write the failing test** (append)

```python
from graph_diffusion.visualisation.dataset_overview import (  # noqa: E402
    plot_shape_gallery,
    plot_drag_vs_re,
    plot_cp_pca,
)


def test_extra_figures_smoke():
    raw = _fake_raw(n_sims=8)
    idxs = list(range(8))
    figs = [
        plot_shape_gallery(raw, idxs[:6]),
        plot_drag_vs_re(raw, idxs, n_bins=3),
        plot_cp_pca(raw, idxs, colour_by="re"),
    ]
    for fig in figs:
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) >= 1
    matplotlib.pyplot.close("all")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dataset_overview.py::test_extra_figures_smoke -q`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation** (append)

```python
def plot_shape_gallery(
    raw: np.ndarray,
    indices: list[int],
    figsize: tuple[float, float] = (9.0, 6.0),
) -> matplotlib.figure.Figure:
    """Grid of body outlines, one per selected sim, labelled by Re and b.

    Args:
        raw (np.ndarray): Full HDF5 array.
        indices (list[int]): Sim indices to draw.
        figsize (tuple[float, float]): Figure size in inches.

    Returns:
        matplotlib.figure.Figure: The rendered gallery.
    """
    n = len(indices)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for k, idx in enumerate(indices):
        d = sim_summary(raw, idx)
        ax = axes[k // ncols][k % ncols]
        ax.axis("on")
        poly = np.vstack([d["polygon"], d["polygon"][:1]])
        ax.plot(poly[:, 0], poly[:, 1], color="navy")
        ax.set_aspect("equal")
        ax.set_title(f"Re={d['re']:.0f}, b={d['b']:.3f}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return fig


def plot_drag_vs_re(
    raw: np.ndarray,
    indices: list[int],
    n_bins: int = 4,
    figsize: tuple[float, float] = (7.0, 5.0),
) -> matplotlib.figure.Figure:
    """Pressure drag vs Re, with sims grouped into ``n_bins`` thickness bands.

    Args:
        raw (np.ndarray): Full HDF5 array.
        indices (list[int]): Sim indices to include.
        n_bins (int): Number of b (thickness) bands.
        figsize (tuple[float, float]): Figure size in inches.

    Returns:
        matplotlib.figure.Figure: The rendered figure.
    """
    s = _summaries(raw, indices)
    re = np.array([d["re"] for d in s])
    b = np.array([d["b"] for d in s])
    cd = np.array([d["cd_p"] for d in s])
    edges = np.linspace(b.min(), b.max() + 1e-9, n_bins + 1)
    fig, ax = plt.subplots(figsize=figsize)
    for j in range(n_bins):
        mask = (b >= edges[j]) & (b < edges[j + 1])
        if not mask.any():
            continue
        order = np.argsort(re[mask])
        ax.plot(
            re[mask][order],
            cd[mask][order],
            marker="o",
            markersize=3,
            label=f"b in [{edges[j]:.3f}, {edges[j + 1]:.3f})",
        )
    ax.set_xlabel("Reynolds number $Re$")
    ax.set_ylabel(r"pressure drag $C_{d,p}$")
    ax.set_title("Pressure drag vs Re by thickness band")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_cp_pca(
    raw: np.ndarray,
    indices: list[int],
    colour_by: str = "re",
    figsize: tuple[float, float] = (7.0, 5.0),
) -> matplotlib.figure.Figure:
    """2-D PCA projection of the resampled Cp(x/c) curves.

    A learned-latent view: each sim's Cp curve is resampled onto a common
    grid, then projected onto its first two principal components.

    Args:
        raw (np.ndarray): Full HDF5 array.
        indices (list[int]): Sim indices to include.
        colour_by (str): ``"re"`` or ``"b"``.
        figsize (tuple[float, float]): Figure size in inches.

    Returns:
        matplotlib.figure.Figure: The rendered figure.

    Raises:
        ValueError: If ``colour_by`` is not ``"re"`` or ``"b"``.
    """
    if colour_by not in ("re", "b"):
        raise ValueError(f"colour_by must be 're' or 'b', got '{colour_by}'")
    s = _summaries(raw, indices)
    grid = np.linspace(0.0, 1.0, 100)
    curves = np.stack(
        [np.interp(grid, d["x_over_c"], d["cp_curve"]) for d in s], axis=0
    )
    centred = curves - curves.mean(axis=0, keepdims=True)
    # SVD-based PCA; first two right-singular vectors are the components.
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    coords = centred @ vt[:2].T
    key = np.array([d[colour_by] for d in s])
    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=key, cmap="plasma", s=14)
    fig.colorbar(sc, ax=ax, label={"re": "Reynolds $Re$", "b": "semi-minor $b$"}[colour_by])
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(f"Cp-curve PCA (coloured by {colour_by})")
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dataset_overview.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/graph_diffusion/visualisation/dataset_overview.py tests/test_dataset_overview.py
git commit -m "feat(integration): add shape gallery, drag-vs-Re, Cp PCA figures"
```

---

### Task A5: CLI to render all figures + package exports

**Files:**
- Create: `scripts/plot_dataset_overview.py`
- Modify: `src/graph_diffusion/visualisation/__init__.py`

- [ ] **Step 1: Export the new functions** from `visualisation/__init__.py`

Add to the imports and `__all__`:

```python
from graph_diffusion.visualisation.dataset_overview import (
    plot_cp_envelope,
    plot_cp_pca,
    plot_design_space_coverage,
    plot_design_space_drag,
    plot_drag_vs_re,
    plot_shape_gallery,
    pressure_drag_coefficient,
    sim_summary,
)
```

Append the same eight names to the existing `__all__` list.

- [ ] **Step 2: Write the CLI script**

```python
"""
Render the full dataset-overview figure suite from the pOnEllipse HDF5.

Writes one PNG + one PDF per figure into ``--out-dir``. Read-only on the
dataset; no model is loaded. Final figure selection for the report is made
by hand from the rendered set.

Usage:
    python scripts/plot_dataset_overview.py \
        --h5 data/ellipse/raw/pOnEllipseTrain.h5 \
        --out-dir outputs/dataset_overview \
        --max-sims 800
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from graph_diffusion.data.pOnEllipse import _load_h5_raw
from graph_diffusion.visualisation.dataset_overview import (
    plot_cp_envelope,
    plot_cp_pca,
    plot_design_space_coverage,
    plot_design_space_drag,
    plot_drag_vs_re,
    plot_shape_gallery,
)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=Path("data/ellipse/raw/pOnEllipseTrain.h5"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/dataset_overview"))
    parser.add_argument(
        "--max-sims",
        type=int,
        default=800,
        help="Cap the number of sims plotted (subsampled evenly) for speed.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw = _load_h5_raw(args.h5)
    n = raw.shape[0]
    step = max(1, n // args.max_sims)
    indices = list(range(0, n, step))
    # Evaluation-sample overlay: reuse the same even subsample's corners.
    eval_indices = [indices[0], indices[len(indices) // 2], indices[-1]]

    figures = {
        "design_space_drag": plot_design_space_drag(raw, indices),
        "cp_envelope_by_re": plot_cp_envelope(raw, indices, colour_by="re"),
        "cp_envelope_by_b": plot_cp_envelope(raw, indices, colour_by="b"),
        "design_space_coverage": plot_design_space_coverage(raw, indices, eval_indices),
        "shape_gallery": plot_shape_gallery(raw, indices[:9]),
        "drag_vs_re": plot_drag_vs_re(raw, indices),
        "cp_pca_by_re": plot_cp_pca(raw, indices, colour_by="re"),
    }
    for name, fig in figures.items():
        for ext in ("png", "pdf"):
            fig.savefig(args.out_dir / f"{name}.{ext}", dpi=200, bbox_inches="tight")
        print(f"wrote {name}.png / .pdf")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the quality gate**

Run: `uv run pytest tests/ -q && uv run ruff check src/ tests/ && uv run black --check src/ tests/ scripts/plot_dataset_overview.py && uv run mypy src/`
Expected: all pass. (If black reports formatting, run `uv run black src/ tests/ scripts/plot_dataset_overview.py` and re-run.)

- [ ] **Step 4: Smoke-run the CLI on the real dataset**

Run: `python scripts/plot_dataset_overview.py --max-sims 300`
Expected: prints `wrote …` for all 7 figures; PNG+PDF files exist under `outputs/dataset_overview/`.

- [ ] **Step 5: Commit**

```bash
git add scripts/plot_dataset_overview.py src/graph_diffusion/visualisation/__init__.py
git commit -m "feat(integration): add dataset-overview CLI and package exports"
```

---

## Part B — OpenLB domain correctness

### Task B1: Remove the silent wrong CHANNEL_HEIGHT default

**Files:**
- Modify: `cfd/ellipse2d/ellipse2d.cpp:120` and just after the CLI parse at `cfd/ellipse2d/ellipse2d.cpp:149`

Make the channel height fail loudly when unset, instead of silently using 8.0. Set a sentinel default, then abort after `fromCLI` if it was not overridden to a valid (5–6-ish, but accept any positive) value.

- [ ] **Step 1: Change the default to a sentinel** at `ellipse2d.cpp:120`

Replace:

```cpp
    myCaseParameters.set<CHANNEL_HEIGHT>(8.0);
```

with:

```cpp
    // Sentinel: a valid channel height MUST be supplied via --CHANNEL_HEIGHT
    // (dataset range 5-6). A negative sentinel makes an omitted flag fail
    // loudly below rather than silently simulating the wrong domain.
    myCaseParameters.set<CHANNEL_HEIGHT>(-1.0);
```

- [ ] **Step 2: Add the post-parse guard** immediately after `myCaseParameters.fromCLI(argc, argv);` (line 149)

```cpp
  myCaseParameters.fromCLI(argc, argv);

  // Guard: refuse to run with an unset/invalid channel height. The dataset
  // domain height is 5-6; an omitted --CHANNEL_HEIGHT must not silently fall
  // back to a wrong domain.
  if (!(myCaseParameters.get<CHANNEL_HEIGHT>() > T{0})) {
    std::cerr << "FATAL: --CHANNEL_HEIGHT must be supplied (dataset range 5-6). "
                 "Refusing to run with an unset channel height.\n";
    return 1;
  }
```

(If `<iostream>` is not already included, add `#include <iostream>` near the top includes.)

- [ ] **Step 3: Rebuild the case** (GPU build is preferred per project memory)

Run:
```bash
cd cfd/ellipse2d && make OLB_ROOT=$HOME/Software/OpenLB-1.9-0-cuda/olb-release
```
Expected: compiles to `./ellipse2d` with no errors.
(If the build environment is unavailable in this session, dispatch the `openlb-build-run` agent to compile; do not skip the build.)

- [ ] **Step 4: Verify the guard fires when the flag is omitted**

Run:
```bash
cd cfd/ellipse2d && ./ellipse2d shapes/baseline/sim_0.dat --REYNOLDS 500 --RESOLUTION 100 --MAX_PHYS_T 1 ; echo "exit=$?"
```
Expected: prints the `FATAL: --CHANNEL_HEIGHT must be supplied …` message and `exit=1`.

- [ ] **Step 5: Verify a valid run still starts**

Run:
```bash
cd cfd/ellipse2d && ./ellipse2d shapes/baseline/sim_0.dat --REYNOLDS 500 --CHANNEL_HEIGHT 5.3 --RESOLUTION 100 --MAX_PHYS_T 1 2>&1 | head -20
```
Expected: the UnitConverter prints; no FATAL message.

- [ ] **Step 6: Commit**

```bash
git add cfd/ellipse2d/ellipse2d.cpp
git commit -m "fix(integration): require --CHANNEL_HEIGHT, drop silent 8.0 default"
```

---

### Task B2: Manifest-driven run driver (no SEMI_MINOR_B)

**Files:**
- Create: `cfd/ellipse2d/runs/run_from_manifest.sh`

A driver that reads a case `manifest.csv` (columns include `name` or `idx`, `Re`, `H`) and launches each row with its own `--REYNOLDS`/`--CHANNEL_HEIGHT`. `--SEMI_MINOR_B` is intentionally NOT passed — the body geometry (and hence b) is fully determined by the shape `.dat` file (`SEMI_MINOR_B` is declared but unused in the C++ case).

- [ ] **Step 1: Write the driver script**

```bash
#!/bin/bash
# Launch one OpenLB run per manifest row, using that row's own Re and H so
# the simulated domain always matches the dataset sample. The body polygon
# (.dat) determines the shape; --SEMI_MINOR_B is deliberately omitted.
#
# Usage:
#   run_from_manifest.sh <shapes_dir> [RESOLUTION] [MAX_PHYS_T]
# Expects <shapes_dir>/manifest.csv with header containing idx,Re,H and a
# sibling <shapes_dir>/sim_<idx>.dat (or <name>.dat) per row.
set -euo pipefail

BIN="$(cd "$(dirname "$0")/.." && pwd)/ellipse2d"
SHAPES_DIR="${1:?usage: run_from_manifest.sh <shapes_dir> [RES] [MAX_T]}"
RES="${2:-100}"
MAX_T="${3:-50}"
MANIFEST="$SHAPES_DIR/manifest.csv"

# Resolve column indices from the header (robust to column order).
header=$(head -1 "$MANIFEST")
col() { echo "$header" | tr ',' '\n' | grep -n "^$1$" | cut -d: -f1; }
c_idx=$(col idx); c_re=$(col Re); c_h=$(col H); c_name=$(col name || true)

tail -n +2 "$MANIFEST" | while IFS=',' read -r -a f; do
  idx="${f[$((c_idx-1))]}"; re="${f[$((c_re-1))]}"; h="${f[$((c_h-1))]}"
  if [ -n "${c_name:-}" ]; then
    name="${f[$((c_name-1))]}"; dat="$SHAPES_DIR/${name}.dat"
  else
    name="sim_${idx}"; dat="$SHAPES_DIR/sim_${idx}.dat"
  fi
  [ -f "$dat" ] || dat="$SHAPES_DIR/sim_${idx}.dat"
  d="runs/$name"; mkdir -p "$d"; rm -rf "$d/tmp" "$d/run.log"
  echo "launching $name: Re=$re H=$h RES=$RES"
  ( cd "$d" && "$BIN" "$dat" --REYNOLDS "$re" --CHANNEL_HEIGHT "$h" \
      --RESOLUTION "$RES" --MAX_PHYS_T "$MAX_T" --AVG_WINDOW_CHORDS 25 \
      > run.log 2>&1 ) || echo "  WARN: $name exited non-zero"
  echo "  $name done: $(tail -1 "$d/tmp/ellipse2d_force.csv" 2>/dev/null || echo '(no force.csv)')"
done
echo ALL_DONE
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x cfd/ellipse2d/runs/run_from_manifest.sh`

- [ ] **Step 3: Dry-verify column parsing on the existing baseline manifest**

Run:
```bash
cd cfd/ellipse2d && bash -n runs/run_from_manifest.sh && \
  head -1 shapes/baseline/manifest.csv
```
Expected: `bash -n` reports no syntax errors; header shows `idx,Re,H,a,b,n_nodes`.

- [ ] **Step 4: Commit**

```bash
git add cfd/ellipse2d/runs/run_from_manifest.sh
git commit -m "feat(integration): add manifest-driven per-sample OpenLB run driver"
```

---

### Task B3: Exporter asserts manifest H/Re match the HDF5

**Files:**
- Modify: `scripts/export_ellipse_for_openlb.py` (`export_sim`)
- Test: `tests/test_export_openlb.py` (create)

Guard against future drift: the H/Re written into the manifest must equal the HDF5 column for that sim. Factor the check into a tiny verifiable function.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for scripts/export_ellipse_for_openlb helpers."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from context import graph_diffusion  # noqa: F401

_spec = importlib.util.spec_from_file_location(
    "export_openlb",
    Path(__file__).resolve().parent.parent / "scripts" / "export_ellipse_for_openlb.py",
)
export_openlb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_openlb)


def test_verify_manifest_row_matches_raw():
    raw = np.full((2, 8, 105), np.nan, dtype=np.float32)
    raw[0, :6, 0] = np.linspace(-0.5, 0.5, 6)
    raw[0, :6, 1] = np.linspace(-0.2, 0.2, 6)
    raw[0, :6, 2] = 754.0  # Re
    raw[0, :6, 3] = 5.5    # H
    row = {"idx": 0, "Re": 754.0, "H": 5.5}
    export_openlb.verify_manifest_row(raw, row)  # should not raise


def test_verify_manifest_row_detects_mismatch():
    raw = np.full((1, 8, 105), np.nan, dtype=np.float32)
    raw[0, :6, 0] = np.linspace(-0.5, 0.5, 6)
    raw[0, :6, 1] = np.linspace(-0.2, 0.2, 6)
    raw[0, :6, 2] = 754.0
    raw[0, :6, 3] = 5.5
    bad = {"idx": 0, "Re": 754.0, "H": 8.0}  # wrong H
    with pytest.raises(ValueError, match="H mismatch"):
        export_openlb.verify_manifest_row(raw, bad)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_export_openlb.py -q`
Expected: FAIL — `AttributeError` (`verify_manifest_row` not defined).

- [ ] **Step 3: Add the helper** to `scripts/export_ellipse_for_openlb.py` (after `export_sim`)

```python
def verify_manifest_row(raw: np.ndarray, row: dict[str, float]) -> None:
    """Assert a manifest row's Re and H equal the HDF5 columns for that sim.

    Args:
        raw (np.ndarray): Full HDF5 array.
        row (dict[str, float]): A manifest row with ``idx``, ``Re``, ``H``.

    Raises:
        ValueError: If the manifest Re or H differs from the HDF5 value.
    """
    idx = int(row["idx"])
    re_h5 = float(raw[idx, 0, _H5_COL_RE])
    h_h5 = float(raw[idx, 0, _H5_COL_H])
    if abs(re_h5 - float(row["Re"])) > 1e-3:
        raise ValueError(f"Re mismatch for sim {idx}: manifest {row['Re']} vs h5 {re_h5}")
    if abs(h_h5 - float(row["H"])) > 1e-3:
        raise ValueError(f"H mismatch for sim {idx}: manifest {row['H']} vs h5 {h_h5}")
```

Ensure `_H5_COL_RE` and `_H5_COL_H` are imported at the top of the script (it already imports from `graph_diffusion.data.pOnEllipse`; add `_H5_COL_H` if missing). Then, in `main()`, after building `rows`, add:

```python
    for row in rows:
        verify_manifest_row(raw, row)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_export_openlb.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/export_ellipse_for_openlb.py tests/test_export_openlb.py
git commit -m "test(integration): assert exported manifest Re/H match the HDF5"
```

---

## Part C — Systematic design-space sampling + naming

### Task C1: Case-name helper

**Files:**
- Create: `src/graph_diffusion/data/design_space.py`
- Test: `tests/test_design_space.py`

Naming: `re{RRRR}_h{HH}_idx{NNNN}` — Re zero-padded to 4 digits, H × 10 as 2 digits, source HDF5 index 4 digits. Sortable and unambiguous. b is NOT in the name (emergent, not a sampling axis).

- [ ] **Step 1: Write the failing test**

```python
"""Tests for graph_diffusion.data.design_space."""

import numpy as np
import pytest
from context import graph_diffusion  # noqa: F401

from graph_diffusion.data.design_space import make_case_name


def test_make_case_name_format():
    assert make_case_name(500.0, 5.3, 0) == "re0500_h53_idx0000"
    assert make_case_name(1000.0, 6.0, 4849) == "re1000_h60_idx4849"


def test_make_case_name_sorts_by_re_then_h():
    names = [
        make_case_name(1000.0, 5.0, 1),
        make_case_name(500.0, 6.0, 2),
        make_case_name(500.0, 5.0, 3),
    ]
    assert sorted(names)[0].startswith("re0500_h50")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_design_space.py -q`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
"""
graph_diffusion.data.design_space
===================================

Systematic sampling of the pOnEllipse design space for model evaluation.
Sampling axes are the controllable physical inputs Reynolds number (Re)
and channel height (H); thickness b is emergent from the generated shape
and is therefore recorded but not sampled. Provides a trackable, sortable
case-naming scheme so evaluation and CFD cases share one namespace.
"""

from __future__ import annotations

import numpy as np


def make_case_name(re: float, h: float, idx: int) -> str:
    """Build a sortable, unambiguous case name from (Re, H, source index).

    Format ``re{RRRR}_h{HH}_idx{NNNN}`` — Re zero-padded to 4 digits, H
    multiplied by 10 to 2 digits, source HDF5 index to 4 digits.

    Args:
        re (float): Reynolds number (500-1000).
        h (float): Channel height (5-6).
        idx (int): Source HDF5 simulation index.

    Returns:
        str: e.g. ``"re0500_h53_idx0000"``.
    """
    return f"re{int(round(re)):04d}_h{int(round(h * 10)):02d}_idx{int(idx):04d}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_design_space.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/graph_diffusion/data/design_space.py tests/test_design_space.py
git commit -m "feat(data): add design-space case-naming helper"
```

---

### Task C2: (Re, H) grid sampler snapping to real sims

**Files:**
- Modify: `src/graph_diffusion/data/design_space.py`
- Test: `tests/test_design_space.py`

Build a grid over (Re, H), and for each grid point select the nearest real dataset sim in normalised (Re, H) space (Re normalised over [500,1000], H over [5,6]). Deduplicate while preserving order. Guarantees the four design-space corners are represented.

- [ ] **Step 1: Write the failing test** (append)

```python
from graph_diffusion.data.design_space import sample_design_space


def _fake_raw_grid() -> np.ndarray:
    # 36 sims spanning Re in [500,1000], H in [5,6].
    res = np.repeat(np.linspace(500, 1000, 6), 6)
    hs = np.tile(np.linspace(5.0, 6.0, 6), 6)
    raw = np.full((36, 8, 105), np.nan, dtype=np.float32)
    for s in range(36):
        raw[s, :6, 0] = np.linspace(-0.5, 0.5, 6)
        raw[s, :6, 1] = np.linspace(-0.2, 0.2, 6)
        raw[s, :6, 2] = res[s]
        raw[s, :6, 3] = hs[s]
    return raw


def test_sample_returns_unique_in_range_indices():
    raw = _fake_raw_grid()
    idxs = sample_design_space(raw, n_re=3, n_h=2)
    assert len(idxs) == len(set(idxs))  # unique
    res = raw[idxs, 0, 2]
    hs = raw[idxs, 0, 3]
    assert res.min() >= 500.0 and res.max() <= 1000.0
    assert hs.min() >= 5.0 and hs.max() <= 6.0


def test_sample_includes_corners():
    raw = _fake_raw_grid()
    idxs = sample_design_space(raw, n_re=3, n_h=3)
    res = raw[idxs, 0, 2]
    hs = raw[idxs, 0, 3]
    pairs = set(zip(np.round(res).astype(int), np.round(hs * 10).astype(int)))
    assert (500, 50) in pairs
    assert (1000, 60) in pairs
    assert (500, 60) in pairs
    assert (1000, 50) in pairs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_design_space.py -q`
Expected: FAIL — `ImportError` (`sample_design_space` not defined).

- [ ] **Step 3: Write minimal implementation** (append; add the column import at the top)

```python
# add at the top imports:
from graph_diffusion.data.pOnEllipse import _H5_COL_H, _H5_COL_RE

_RE_RANGE = (500.0, 1000.0)
_H_RANGE = (5.0, 6.0)


def sample_design_space(
    raw: np.ndarray,
    n_re: int = 5,
    n_h: int = 3,
) -> list[int]:
    """Select sims spanning an (Re, H) grid, snapped to nearest real sims.

    A regular ``n_re`` x ``n_h`` grid is laid over the design space; each
    grid node is matched to the dataset sim nearest in normalised (Re, H)
    space. The four corners are included by construction. Duplicate matches
    are removed, preserving first-seen order.

    Args:
        raw (np.ndarray): Full HDF5 array, shape ``(N_sims, MAX_NODES, 105)``.
        n_re (int): Number of Reynolds grid points (>= 2 to hit corners).
        n_h (int): Number of channel-height grid points (>= 2).

    Returns:
        list[int]: Selected simulation indices.

    Raises:
        ValueError: If ``n_re < 2`` or ``n_h < 2``.
    """
    if n_re < 2 or n_h < 2:
        raise ValueError(f"n_re and n_h must be >= 2, got {n_re}, {n_h}")
    re = raw[:, 0, _H5_COL_RE].astype(np.float64)
    h = raw[:, 0, _H5_COL_H].astype(np.float64)
    re_n = (re - _RE_RANGE[0]) / (_RE_RANGE[1] - _RE_RANGE[0])
    h_n = (h - _H_RANGE[0]) / (_H_RANGE[1] - _H_RANGE[0])

    re_targets = np.linspace(0.0, 1.0, n_re)
    h_targets = np.linspace(0.0, 1.0, n_h)
    selected: list[int] = []
    for rt in re_targets:
        for ht in h_targets:
            d2 = (re_n - rt) ** 2 + (h_n - ht) ** 2
            idx = int(np.argmin(d2))
            if idx not in selected:
                selected.append(idx)
    return selected
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_design_space.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/graph_diffusion/data/design_space.py tests/test_design_space.py
git commit -m "feat(data): add (Re, H) grid sampler snapping to nearest sims"
```

---

### Task C3: Wire the sampler + naming into the export CLI

**Files:**
- Modify: `scripts/export_ellipse_for_openlb.py`

Add a `--grid` mode that selects indices via `sample_design_space`, and add a `name` column (from `make_case_name`) to the manifest so OpenLB/eval share the namespace. The existing `--indices` / `--n-sims` Re-only behaviour is preserved as a fallback.

- [ ] **Step 1: Add imports and CLI flags** to `scripts/export_ellipse_for_openlb.py`

At the top imports add:

```python
from graph_diffusion.data.design_space import make_case_name, sample_design_space
```

In `main()`'s argparse block add:

```python
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Select sims on an (Re, H) grid snapped to nearest real sims.",
    )
    parser.add_argument("--n-re", type=int, default=5, help="Re grid points for --grid.")
    parser.add_argument("--n-h", type=int, default=3, help="H grid points for --grid.")
```

- [ ] **Step 2: Use the sampler and add the name column**

Replace the index-selection line:

```python
    indices = args.indices if args.indices else select_indices(raw, args.n_sims)
```

with:

```python
    if args.indices:
        indices = args.indices
    elif args.grid:
        indices = sample_design_space(raw, n_re=args.n_re, n_h=args.n_h)
    else:
        indices = select_indices(raw, args.n_sims)
```

After `rows = [export_sim(raw, idx, args.out_dir) for idx in indices]`, add the name field and verify each row:

```python
    for row in rows:
        verify_manifest_row(raw, row)
        row["name"] = make_case_name(row["Re"], row["H"], int(row["idx"]))
```

Update the manifest writer fieldnames to include `name` first:

```python
        writer = csv.DictWriter(
            fh, fieldnames=["name", "idx", "Re", "H", "a", "b", "n_nodes"]
        )
```

- [ ] **Step 3: Run the quality gate + a real grid export**

Run:
```bash
uv run pytest tests/ -q && uv run ruff check src/ tests/ && uv run black --check src/ tests/ && uv run mypy src/
python scripts/export_ellipse_for_openlb.py --grid --n-re 4 --n-h 3 --out-dir outputs/eval_grid
head -3 outputs/eval_grid/manifest.csv
```
Expected: gate passes; manifest header is `name,idx,Re,H,a,b,n_nodes`; names look like `re0500_h53_idx0000`; Re/H span the corners.

- [ ] **Step 4: Commit**

```bash
git add scripts/export_ellipse_for_openlb.py
git commit -m "feat(integration): add (Re,H) grid export mode and case-name manifest column"
```

---

## Final verification

- [ ] **Run the full quality gate one more time**

Run: `uv run pytest tests/ -q && uv run ruff check src/ tests/ && uv run black --check src/ tests/ && uv run mypy src/`
Expected: all pass.

- [ ] **Confirm the three deliverables end-to-end**

1. `python scripts/plot_dataset_overview.py --max-sims 300` → 7 figures (PNG+PDF) in `outputs/dataset_overview/`.
2. OpenLB run without `--CHANNEL_HEIGHT` aborts with FATAL; with it, runs.
3. `python scripts/export_ellipse_for_openlb.py --grid` → manifest with `name` column spanning the (Re, H) corners; `cfd/ellipse2d/runs/run_from_manifest.sh outputs/eval_grid` launches per-sample Re/H runs.

---

## Notes on deferred work (Part D)

Part D (augmenting the Cp-conditioned model with Re/H and retraining) is intentionally NOT in this plan. It is independent of the validation loop (Re is imposed in OpenLB via viscosity, not inlet velocity — see spec) and requires a retrain. Pick it up from the spec's Part D section when there's training budget; it will get its own plan.
