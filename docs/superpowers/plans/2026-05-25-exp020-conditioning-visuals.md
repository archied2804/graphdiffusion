# EXP-020 Conditioning Experiments + Visualisations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable `graph_diffusion.visualisation` module + `scripts/postprocess_exp020.py` CLI that turns a trained EXP-020 checkpoint into Figure A (conditioning grid), Figure B (forward/reverse diffusion trajectory: PNG filmstrip + MP4 + GIF), and Figure C (CFG sweep). Run a full 200-epoch EXP-020 training and execute the post-processing.

**Architecture:** A new `src/graph_diffusion/visualisation/` package with two modules: `trajectory.py` (collect intermediate `x_t` from forward + reverse diffusion) and `plotting.py` (three figure builders + animation writer). A new `scripts/postprocess_exp020.py` CLI loads a checkpoint, picks targets via greedy farthest-first over `dataset.cond`, samples shapes, and emits all artefacts to `outputs/EXP-020_fourier_pressure_conditioning/`. The training run uses the existing `train.py` and the existing EXP-020 YAML config.

**Tech Stack:** PyTorch 2.8, PyTorch Geometric, NumPy, Matplotlib (incl. `matplotlib.animation` with ffmpeg + Pillow writers), pytest. No new pip dependencies.

---

## File Structure

**New files:**
- `src/graph_diffusion/visualisation/__init__.py` — package surface
- `src/graph_diffusion/visualisation/trajectory.py` — `collect_forward`, `collect_reverse`
- `src/graph_diffusion/visualisation/plotting.py` — `plot_conditioning_grid`, `plot_trajectory_filmstrip`, `write_trajectory_animation`
- `tests/test_visualisation.py` — unit tests for all five public functions
- `scripts/postprocess_exp020.py` — CLI that ties it all together

**Modified files:** none (no edits to `model/`, `data/`, or `train.py`). Existing `model.sample_with_trajectory` (used by `scripts/postprocess_ellipse.py`) is left untouched.

---

### Task 1: Create visualisation package skeleton

**Files:**
- Create: `src/graph_diffusion/visualisation/__init__.py`
- Create: `src/graph_diffusion/visualisation/trajectory.py` (empty module)
- Create: `src/graph_diffusion/visualisation/plotting.py` (empty module)

- [ ] **Step 1: Create the three files with module docstrings**

`src/graph_diffusion/visualisation/__init__.py`:
```python
"""
graph_diffusion.visualisation
==============================

Plotting and trajectory helpers for diffusion experiments. The package
is dependency-light: it consumes a trained ``GraphDiffusionModel`` and
a PyG ``Data`` template through their public APIs, never reaching into
model internals.
"""

from graph_diffusion.visualisation.plotting import (
    plot_conditioning_grid,
    plot_trajectory_filmstrip,
    write_trajectory_animation,
)
from graph_diffusion.visualisation.trajectory import (
    collect_forward,
    collect_reverse,
)

__all__ = [
    "collect_forward",
    "collect_reverse",
    "plot_conditioning_grid",
    "plot_trajectory_filmstrip",
    "write_trajectory_animation",
]
```

`src/graph_diffusion/visualisation/trajectory.py`:
```python
"""
graph_diffusion.visualisation.trajectory
==========================================

Capture intermediate ``x_t`` tensors from forward and reverse diffusion
runs for filmstrip and animation plotting.
"""

from __future__ import annotations
```

`src/graph_diffusion/visualisation/plotting.py`:
```python
"""
graph_diffusion.visualisation.plotting
========================================

Matplotlib figure builders for diffusion experiment writeups: a
conditioning grid, a forward+reverse trajectory filmstrip, and an
animation writer for the reverse pass.
"""

from __future__ import annotations
```

- [ ] **Step 2: Confirm imports work (placeholder package importable)**

Run:
```bash
uv run python -c "import graph_diffusion.visualisation as v; print(v.__all__)"
```

Expected: `ImportError` for the missing `collect_forward` etc. — this is the failing red baseline. We'll bring them online in subsequent tasks.

- [ ] **Step 3: Commit the skeleton**

```bash
git add src/graph_diffusion/visualisation/
git commit -m "feat(visualisation): scaffold visualisation package"
```

---

### Task 2: Implement `trajectory.collect_forward` (TDD)

**Files:**
- Test: `tests/test_visualisation.py`
- Modify: `src/graph_diffusion/visualisation/trajectory.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_visualisation.py`:
```python
"""Unit tests for the graph_diffusion.visualisation package."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from graph_diffusion.building_blocks.noise_schedule import NoiseSchedule
from graph_diffusion.model.graph_diffusion_model import GraphDiffusionModel
from graph_diffusion.model.score_network import ScoreNetwork
from graph_diffusion.visualisation.trajectory import collect_forward


def _tiny_model() -> GraphDiffusionModel:
    """Build the smallest possible model for fast tests."""
    torch.manual_seed(0)
    sn = ScoreNetwork(
        node_dim=8,
        edge_dim=2,
        global_dim=4,
        time_embed_dim=8,
        n_layers=1,
        input_dim=1,
        output_dim=1,
    )
    ns = NoiseSchedule(T=20, schedule_type="linear")
    return GraphDiffusionModel(score_network=sn, noise_schedule=ns)


def _tiny_template() -> Data:
    """8-node ring graph, radial features."""
    torch.manual_seed(0)
    n = 8
    theta = torch.linspace(0, 2 * 3.14159, n + 1)[:-1]
    pos = torch.stack([theta.cos(), theta.sin()], dim=1)
    x = torch.ones(n, 1)
    edge_index = torch.stack(
        [
            torch.arange(n),
            (torch.arange(n) + 1) % n,
        ],
        dim=0,
    )
    edge_attr = torch.zeros(n, 2)
    u = torch.zeros(1, 4)
    batch = torch.zeros(n, dtype=torch.long)
    return Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        u=u,
        batch=batch,
    )


def test_collect_forward_shapes() -> None:
    model = _tiny_model()
    template = _tiny_template()
    snapshots = collect_forward(
        model, template, snapshot_steps=[0, 5, 10, 19], seed=0
    )
    assert len(snapshots) == 4
    for snap in snapshots:
        assert snap.shape == (8, 1)
        assert snap.dtype == torch.float32
        assert torch.isfinite(snap).all()


def test_collect_forward_deterministic() -> None:
    model = _tiny_model()
    template = _tiny_template()
    a = collect_forward(model, template, snapshot_steps=[5, 15], seed=42)
    b = collect_forward(model, template, snapshot_steps=[5, 15], seed=42)
    for x, y in zip(a, b, strict=True):
        assert torch.allclose(x, y)
```

- [ ] **Step 2: Run the test and confirm it fails**

Run:
```bash
uv run pytest tests/test_visualisation.py::test_collect_forward_shapes -v
```

Expected: `ImportError: cannot import name 'collect_forward'`.

- [ ] **Step 3: Implement `collect_forward`**

In `src/graph_diffusion/visualisation/trajectory.py`:
```python
"""
graph_diffusion.visualisation.trajectory
==========================================

Capture intermediate ``x_t`` tensors from forward and reverse diffusion
runs for filmstrip and animation plotting.
"""

from __future__ import annotations

import copy

import torch
from torch_geometric.data import Data

from graph_diffusion.model.graph_diffusion_model import GraphDiffusionModel

__all__ = [
    "collect_forward",
    "collect_reverse",
]


def collect_forward(
    model: GraphDiffusionModel,
    template: Data,
    snapshot_steps: list[int],
    seed: int = 0,
) -> list[torch.Tensor]:
    """Run forward diffusion on ``template.x`` and snapshot at given ``t``.

    Args:
        model: Trained diffusion model providing the ``noise_schedule``.
        template: A graph ``Data`` whose ``x`` provides the clean signal
            ``x_0``. Must be on the same device as the model.
        snapshot_steps: 0-indexed timesteps to record (``t in [0, T-1]``).
            Each ``t`` corresponds to one entry in the returned list, in
            the same order as ``snapshot_steps``.
        seed: Seed for the additive Gaussian noise.

    Returns:
        A list of CPU ``torch.float32`` tensors, each of shape
        ``(N_nodes, n_noise_channels_or_x_cols)``, one per snapshot.
    """
    device = template.x.device
    generator = torch.Generator(device=device).manual_seed(seed)
    x_0 = template.x
    schedule = model.noise_schedule

    snapshots: list[torch.Tensor] = []
    for t in snapshot_steps:
        t_idx = torch.tensor([t], device=device)
        sqrt_alpha_bar = schedule.get_t(t_idx, "sqrt_alphas_cumprod")
        sqrt_one_minus = schedule.get_t(t_idx, "sqrt_one_minus_alphas_cumprod")
        # Fresh per-snapshot noise so each t is independent; matches what
        # the training-loop's forward_diffusion does sample-by-sample.
        epsilon = torch.randn(x_0.shape, generator=generator, device=device)
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus * epsilon
        snapshots.append(x_t.detach().cpu().to(torch.float32))
    return snapshots
```

- [ ] **Step 4: Run the test and confirm it passes**

Run:
```bash
uv run pytest tests/test_visualisation.py::test_collect_forward_shapes tests/test_visualisation.py::test_collect_forward_deterministic -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/graph_diffusion/visualisation/trajectory.py tests/test_visualisation.py
git commit -m "feat(visualisation): add trajectory.collect_forward"
```

---

### Task 3: Implement `trajectory.collect_reverse` (TDD)

**Files:**
- Modify: `src/graph_diffusion/visualisation/trajectory.py`
- Modify: `tests/test_visualisation.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_visualisation.py`:
```python
from graph_diffusion.visualisation.trajectory import collect_reverse


def test_collect_reverse_shapes() -> None:
    model = _tiny_model()
    template = _tiny_template()
    snapshots = collect_reverse(
        model,
        template,
        cond=None,
        snapshot_steps=[19, 10, 5, 0],
        seed=0,
    )
    assert len(snapshots) == 4
    for snap in snapshots:
        assert snap.shape == (8, 1)
        assert snap.dtype == torch.float32
        assert torch.isfinite(snap).all()


def test_collect_reverse_deterministic() -> None:
    model = _tiny_model()
    template = _tiny_template()
    a = collect_reverse(
        model, template, cond=None, snapshot_steps=[10, 0], seed=7
    )
    b = collect_reverse(
        model, template, cond=None, snapshot_steps=[10, 0], seed=7
    )
    for x, y in zip(a, b, strict=True):
        assert torch.allclose(x, y)
```

- [ ] **Step 2: Run and confirm failure**

Run:
```bash
uv run pytest tests/test_visualisation.py::test_collect_reverse_shapes -v
```

Expected: `ImportError: cannot import name 'collect_reverse'`.

- [ ] **Step 3: Implement `collect_reverse`**

Append to `src/graph_diffusion/visualisation/trajectory.py`:
```python
def collect_reverse(
    model: GraphDiffusionModel,
    template: Data,
    cond: torch.Tensor | None,
    snapshot_steps: list[int],
    guidance_scale: float = 1.0,
    seed: int = 0,
) -> list[torch.Tensor]:
    """Run reverse diffusion and snapshot ``x_t`` at the requested steps.

    This duplicates the inner loop of :meth:`GraphDiffusionModel.sample`
    so we can record intermediate ``x_t`` without modifying the model
    API. Supports classifier-free guidance via ``guidance_scale``; does
    NOT support DPS (kept simple — the spec only animates the standard
    CFG reverse path).

    Args:
        model: Trained diffusion model.
        template: Graph ``Data`` template. ``template.x`` is ignored;
            generation starts from ``x_T ~ N(0, I)``.
        cond: Optional conditioning vector of shape ``(B, cond_dim)``.
        snapshot_steps: 0-indexed timesteps to record. ``T - 1`` records
            the pure-noise initial state; ``0`` records the final
            denoised output.
        guidance_scale: CFG scale ``w``. ``1.0`` disables guidance.
        seed: Seed for the initial noise and per-step stochasticity.

    Returns:
        A list of CPU ``torch.float32`` tensors, each shape
        ``(N_nodes, n_noise_channels)``, one per snapshot in the input
        order of ``snapshot_steps``.
    """
    schedule = model.noise_schedule
    total_T = schedule.T
    device = template.edge_index.device

    if template.pos is not None:
        n_nodes = template.pos.size(0)
    else:
        n_nodes = template.x.size(0)

    if model.n_noise_channels is not None:
        noise_dim = model.n_noise_channels
    elif template.x is not None:
        noise_dim = template.x.size(1)
    else:
        noise_dim = template.u.size(1)

    generator = torch.Generator(device=device).manual_seed(seed)
    x_t = torch.randn(
        n_nodes, noise_dim, generator=generator, device=device
    )

    batch_vec = template.batch
    if batch_vec is None:
        batch_vec = torch.zeros(n_nodes, dtype=torch.long, device=device)
    n_graphs = int(batch_vec.max().item()) + 1

    p_cond = getattr(template, "p_cond", None)
    use_cfg = (
        guidance_scale != 1.0
        and cond is not None
        and getattr(model.score_network, "null_cond", None) is not None
    )

    # Record snapshots indexed by 0-based t. Keys are timesteps; values
    # are the *x_t* (i.e. the state *after* taking the step that lands
    # at index t). Index T-1 = pure noise initial; index 0 = clean.
    requested = set(snapshot_steps)
    captured: dict[int, torch.Tensor] = {}

    # Initial pure noise captured if requested.
    if total_T - 1 in requested:
        captured[total_T - 1] = x_t.detach().cpu().to(torch.float32)

    with torch.no_grad():
        for step in range(total_T, 0, -1):
            t_idx = torch.full(
                (n_graphs,), step - 1, dtype=torch.long, device=device
            )
            noisy_data = copy.copy(template)
            if model.n_noise_channels is not None and p_cond is not None:
                noisy_data.x = torch.cat([x_t, p_cond], dim=-1)
            else:
                noisy_data.x = x_t
            noisy_data.batch = batch_vec

            eps_pred = model.score_network(noisy_data, t_idx, cond=cond)
            if use_cfg:
                eps_null = model.score_network(
                    noisy_data, t_idx, cond=cond, force_uncond=True
                )
                eps_pred = (1.0 + guidance_scale) * eps_pred - (
                    guidance_scale * eps_null
                )

            beta_t = schedule.get_t(t_idx, "betas")[batch_vec]
            alpha_t = schedule.get_t(t_idx, "alphas")[batch_vec]
            sqrt_one_minus = schedule.get_t(
                t_idx, "sqrt_one_minus_alphas_cumprod"
            )[batch_vec]

            x_t = (1.0 / torch.sqrt(alpha_t)) * (
                x_t - (beta_t / sqrt_one_minus) * eps_pred
            )
            if step > 1:
                z = torch.randn(
                    x_t.shape, generator=generator, device=device
                )
                x_t = x_t + torch.sqrt(beta_t) * z

            new_t = step - 1
            if new_t in requested:
                captured[new_t] = x_t.detach().cpu().to(torch.float32)

    return [captured[t] for t in snapshot_steps]
```

- [ ] **Step 4: Run and confirm pass**

Run:
```bash
uv run pytest tests/test_visualisation.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/graph_diffusion/visualisation/trajectory.py tests/test_visualisation.py
git commit -m "feat(visualisation): add trajectory.collect_reverse with CFG support"
```

---

### Task 4: Implement `plotting.plot_conditioning_grid` (TDD)

**Files:**
- Modify: `tests/test_visualisation.py`
- Modify: `src/graph_diffusion/visualisation/plotting.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_visualisation.py`:
```python
import matplotlib

matplotlib.use("Agg")
import numpy as np

from graph_diffusion.visualisation.plotting import plot_conditioning_grid


def test_plot_conditioning_grid_axes_count() -> None:
    n_targets = 4
    n_samples = 3
    n_cp_points = 50
    n_shape_nodes = 16

    rng = np.random.default_rng(0)
    target_cps = [rng.standard_normal(n_cp_points) for _ in range(n_targets)]
    head_pred_cps = [rng.standard_normal(n_cp_points) for _ in range(n_targets)]
    head_pred_stds = [
        np.abs(rng.standard_normal(n_cp_points)) for _ in range(n_targets)
    ]
    sample_shapes = [
        [rng.standard_normal((n_shape_nodes, 2)) for _ in range(n_samples)]
        for _ in range(n_targets)
    ]
    row_labels = [f"target {i}" for i in range(n_targets)]

    fig = plot_conditioning_grid(
        target_cps=target_cps,
        sample_shapes=sample_shapes,
        head_pred_cps=head_pred_cps,
        head_pred_stds=head_pred_stds,
        row_labels=row_labels,
    )
    expected_axes = n_targets * (2 + n_samples)
    assert len(fig.axes) == expected_axes
```

- [ ] **Step 2: Run and confirm failure**

```bash
uv run pytest tests/test_visualisation.py::test_plot_conditioning_grid_axes_count -v
```

Expected: `ImportError: cannot import name 'plot_conditioning_grid'`.

- [ ] **Step 3: Implement `plot_conditioning_grid`**

In `src/graph_diffusion/visualisation/plotting.py`:
```python
"""
graph_diffusion.visualisation.plotting
========================================

Matplotlib figure builders for diffusion experiment writeups: a
conditioning grid, a forward+reverse trajectory filmstrip, and an
animation writer for the reverse pass.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "plot_conditioning_grid",
    "plot_trajectory_filmstrip",
    "write_trajectory_animation",
]


def plot_conditioning_grid(
    target_cps: list[np.ndarray],
    sample_shapes: list[list[np.ndarray]],
    head_pred_cps: list[np.ndarray],
    head_pred_stds: list[np.ndarray],
    row_labels: list[str],
    figsize: tuple[float, float] = (16, 10),
) -> matplotlib.figure.Figure:
    """Render the Figure A 4×(2+S) conditioning grid.

    Args:
        target_cps: One Cp curve per row, shape ``(N_cp_grid,)``.
        sample_shapes: Per-row list of generated boundary shapes, each
            shape ``(N_nodes, 2)`` in (x, y).
        head_pred_cps: Per-row mean head-predicted Cp, shape ``(N_cp_grid,)``.
        head_pred_stds: Per-row std across samples, shape ``(N_cp_grid,)``.
        row_labels: Per-row labels (left ylabel of column 0).
        figsize: Figure size in inches.

    Returns:
        The created figure. Caller is responsible for ``fig.savefig(...)``.
    """
    n_targets = len(target_cps)
    n_samples = len(sample_shapes[0])
    ncols = 2 + n_samples
    fig, axes = plt.subplots(
        n_targets, ncols, figsize=figsize, squeeze=False
    )
    x_grid = np.linspace(0.0, 1.0, target_cps[0].shape[0])

    for r in range(n_targets):
        # Column 0: target Cp
        ax = axes[r, 0]
        ax.plot(x_grid, target_cps[r], color="C0", lw=2)
        ax.set_ylabel(row_labels[r], fontsize=10)
        if r == 0:
            ax.set_title("target Cp(x/c)", fontsize=10)
        ax.grid(alpha=0.3)

        # Column 1: head-predicted Cp ± σ
        ax = axes[r, 1]
        mean = head_pred_cps[r]
        std = head_pred_stds[r]
        ax.plot(x_grid, mean, color="C1", lw=2, label="head pred")
        ax.fill_between(
            x_grid, mean - std, mean + std, alpha=0.3, color="C1"
        )
        ax.plot(x_grid, target_cps[r], color="C0", lw=1, ls="--", label="target")
        if r == 0:
            ax.set_title("predicted Cp ±σ", fontsize=10)
            ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)

        # Columns 2…2+S: sample shapes
        for s in range(n_samples):
            ax = axes[r, 2 + s]
            xy = sample_shapes[r][s]
            closed = np.vstack([xy, xy[:1]])
            ax.plot(closed[:, 0], closed[:, 1], color="C2", lw=1.5)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"sample {s + 1}", fontsize=10)

    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run and confirm pass**

```bash
uv run pytest tests/test_visualisation.py::test_plot_conditioning_grid_axes_count -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/graph_diffusion/visualisation/plotting.py tests/test_visualisation.py
git commit -m "feat(visualisation): add plot_conditioning_grid"
```

---

### Task 5: Implement `plotting.plot_trajectory_filmstrip` (TDD)

**Files:**
- Modify: `tests/test_visualisation.py`
- Modify: `src/graph_diffusion/visualisation/plotting.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_visualisation.py`:
```python
from graph_diffusion.visualisation.plotting import plot_trajectory_filmstrip


def test_plot_trajectory_filmstrip_axes_count() -> None:
    n_frames = 6
    n_nodes = 16
    rng = np.random.default_rng(1)
    forward = [rng.standard_normal((n_nodes, 2)) for _ in range(n_frames)]
    reverse = [rng.standard_normal((n_nodes, 2)) for _ in range(n_frames)]
    timesteps = [0, 5, 20, 50, 100, 199]
    target_cp = rng.standard_normal(50)

    fig = plot_trajectory_filmstrip(
        forward_snapshots=forward,
        reverse_snapshots=reverse,
        timesteps=timesteps,
        target_cp=target_cp,
    )
    # 2 rows × n_frames shape axes + 1 Cp panel.
    assert len(fig.axes) == 2 * n_frames + 1
```

- [ ] **Step 2: Run and confirm failure**

```bash
uv run pytest tests/test_visualisation.py::test_plot_trajectory_filmstrip_axes_count -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `plot_trajectory_filmstrip`**

Append to `src/graph_diffusion/visualisation/plotting.py`:
```python
def plot_trajectory_filmstrip(
    forward_snapshots: list[np.ndarray],
    reverse_snapshots: list[np.ndarray],
    timesteps: list[int],
    target_cp: np.ndarray,
    figsize: tuple[float, float] = (14, 5),
) -> matplotlib.figure.Figure:
    """Two-row diffusion-trajectory filmstrip + side Cp panel.

    Top row plots forward noising at the given timesteps; bottom row
    plots reverse denoising at the same timesteps (in reverse order
    visually so time flows left-to-right within each row). A side
    panel shows the target Cp curve.

    Args:
        forward_snapshots: F shapes captured during forward diffusion,
            each ``(N_nodes, 2)``.
        reverse_snapshots: F shapes captured during reverse diffusion,
            each ``(N_nodes, 2)``.
        timesteps: F timesteps the snapshots correspond to (used for
            column titles). Length must match the snapshot lists.
        target_cp: Target pressure curve, shape ``(N_cp_grid,)``.
        figsize: Figure size in inches.

    Returns:
        The created figure.
    """
    n_frames = len(timesteps)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, n_frames + 1, width_ratios=[1] * n_frames + [1.2])

    for col in range(n_frames):
        ax_fwd = fig.add_subplot(gs[0, col])
        xy = forward_snapshots[col]
        closed = np.vstack([xy, xy[:1]])
        ax_fwd.plot(closed[:, 0], closed[:, 1], color="C3", lw=1.2)
        ax_fwd.set_aspect("equal")
        ax_fwd.set_xticks([])
        ax_fwd.set_yticks([])
        ax_fwd.set_title(f"t={timesteps[col]}", fontsize=9)
        if col == 0:
            ax_fwd.set_ylabel("forward", fontsize=10)

        ax_rev = fig.add_subplot(gs[1, col])
        xy = reverse_snapshots[col]
        closed = np.vstack([xy, xy[:1]])
        ax_rev.plot(closed[:, 0], closed[:, 1], color="C2", lw=1.2)
        ax_rev.set_aspect("equal")
        ax_rev.set_xticks([])
        ax_rev.set_yticks([])
        if col == 0:
            ax_rev.set_ylabel("reverse", fontsize=10)

    ax_cp = fig.add_subplot(gs[:, n_frames])
    x_grid = np.linspace(0.0, 1.0, target_cp.shape[0])
    ax_cp.plot(x_grid, target_cp, color="C0", lw=2)
    ax_cp.set_title("target Cp(x/c)", fontsize=10)
    ax_cp.grid(alpha=0.3)

    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run and confirm pass**

```bash
uv run pytest tests/test_visualisation.py::test_plot_trajectory_filmstrip_axes_count -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/graph_diffusion/visualisation/plotting.py tests/test_visualisation.py
git commit -m "feat(visualisation): add plot_trajectory_filmstrip"
```

---

### Task 6: Implement `plotting.write_trajectory_animation` (TDD)

**Files:**
- Modify: `tests/test_visualisation.py`
- Modify: `src/graph_diffusion/visualisation/plotting.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_visualisation.py`:
```python
import shutil

import pytest

from graph_diffusion.visualisation.plotting import write_trajectory_animation


def test_write_trajectory_animation_writes_file(tmp_path) -> None:
    n_frames = 5
    n_nodes = 12
    rng = np.random.default_rng(2)
    frames = [rng.standard_normal((n_nodes, 2)) for _ in range(n_frames)]
    target_cp = rng.standard_normal(40)
    out_gif = tmp_path / "test_trajectory.gif"
    write_trajectory_animation(
        reverse_snapshots=frames,
        target_cp=target_cp,
        out_path_mp4=None,
        out_path_gif=out_gif,
        fps=10,
    )
    assert out_gif.exists()
    assert out_gif.stat().st_size > 1024


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed"
)
def test_write_trajectory_animation_writes_mp4(tmp_path) -> None:
    n_frames = 5
    n_nodes = 12
    rng = np.random.default_rng(3)
    frames = [rng.standard_normal((n_nodes, 2)) for _ in range(n_frames)]
    target_cp = rng.standard_normal(40)
    out_mp4 = tmp_path / "test_trajectory.mp4"
    write_trajectory_animation(
        reverse_snapshots=frames,
        target_cp=target_cp,
        out_path_mp4=out_mp4,
        out_path_gif=None,
        fps=10,
    )
    assert out_mp4.exists()
    assert out_mp4.stat().st_size > 1024
```

- [ ] **Step 2: Run and confirm failure**

```bash
uv run pytest tests/test_visualisation.py::test_write_trajectory_animation_writes_file -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `write_trajectory_animation`**

Append to `src/graph_diffusion/visualisation/plotting.py`:
```python
import matplotlib.animation as manim


def write_trajectory_animation(
    reverse_snapshots: list[np.ndarray],
    target_cp: np.ndarray,
    out_path_mp4: Path | None,
    out_path_gif: Path | None = None,
    fps: int = 25,
) -> None:
    """Render the reverse-diffusion trajectory as MP4 and/or GIF.

    Each frame is a side-by-side figure: the current shape on the left,
    the target Cp curve on the right. The shape line is replotted each
    frame; axis bounds are fixed to the global min/max across all frames
    so the animation doesn't jitter.

    Args:
        reverse_snapshots: T+1 shapes ``(N_nodes, 2)``, ordered from
            noise (frame 0) to clean (final frame).
        target_cp: Static target Cp curve shown alongside.
        out_path_mp4: Where to write the MP4. Skip MP4 if ``None`` or
            if ``ffmpeg`` is unavailable (logs a warning).
        out_path_gif: Where to write the GIF. Skip GIF if ``None``.
        fps: Frames per second.

    Returns:
        None. Both files are written to disk.
    """
    all_xy = np.concatenate(reverse_snapshots, axis=0)
    xy_min = all_xy.min(axis=0) - 0.1
    xy_max = all_xy.max(axis=0) + 0.1

    fig, (ax_shape, ax_cp) = plt.subplots(
        1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [1, 1.2]}
    )
    ax_shape.set_aspect("equal")
    ax_shape.set_xlim(xy_min[0], xy_max[0])
    ax_shape.set_ylim(xy_min[1], xy_max[1])
    ax_shape.set_xticks([])
    ax_shape.set_yticks([])
    (line,) = ax_shape.plot([], [], color="C2", lw=1.5)

    x_grid = np.linspace(0.0, 1.0, target_cp.shape[0])
    ax_cp.plot(x_grid, target_cp, color="C0", lw=2)
    ax_cp.set_title("target Cp(x/c)", fontsize=10)
    ax_cp.grid(alpha=0.3)

    title = ax_shape.set_title("", fontsize=10)

    def update(frame_idx: int) -> tuple:
        xy = reverse_snapshots[frame_idx]
        closed = np.vstack([xy, xy[:1]])
        line.set_data(closed[:, 0], closed[:, 1])
        title.set_text(f"step {frame_idx + 1}/{len(reverse_snapshots)}")
        return (line, title)

    anim = manim.FuncAnimation(
        fig,
        update,
        frames=len(reverse_snapshots),
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

- [ ] **Step 4: Run and confirm pass**

```bash
uv run pytest tests/test_visualisation.py -v
```

Expected: 6 passed (the ffmpeg test runs if ffmpeg is on PATH, else skip).

- [ ] **Step 5: Commit**

```bash
git add src/graph_diffusion/visualisation/plotting.py tests/test_visualisation.py
git commit -m "feat(visualisation): add write_trajectory_animation"
```

---

### Task 7: Run the full quality gate

**Files:** none

- [ ] **Step 1: Run pytest, ruff, black, mypy**

```bash
uv run pytest tests/ -q && \
uv run ruff check src/ tests/ && \
uv run black --check src/ tests/ && \
uv run mypy src/
```

Expected: 156 passed (150 prior + 6 new); ruff clean; black clean; mypy 0 issues.

- [ ] **Step 2: If any fail, fix in place and re-run until clean**

If ruff complains about unused imports or line length: fix the source.
If mypy complains about types: add precise annotations matching the existing model code.
If a test fails: fix the implementation, not the test.

- [ ] **Step 3: No commit needed if gate already green**

The quality gate is a verification step, not a code change.

---

### Task 8: Build `scripts/postprocess_exp020.py` — CLI scaffold + target picking

**Files:**
- Create: `scripts/postprocess_exp020.py`

This task and the next two build the CLI in three layers: scaffold + targets (T8), sampling + figures A/C (T9), trajectory + figure B (T10). Each layer is independently runnable.

- [ ] **Step 1: Write the scaffold**

Create `scripts/postprocess_exp020.py`:
```python
"""
postprocess_exp020.py — Inverse-design figures for the EXP-020 pipeline
=========================================================================

Loads a trained EXP-020 checkpoint and produces:
  * figure_a_conditioning_grid.png  — n_targets training + 1 OOD synthetic,
                                       n_samples shapes per target.
  * figure_b_trajectory.png         — forward + reverse filmstrip on the
                                       OOD synthetic target.
  * figure_b_reverse.mp4 / .gif     — animated reverse diffusion.
  * figure_c_cfg_sweep.png          — same OOD target, w ∈ {1, 3, 7}.

Usage:
    python scripts/postprocess_exp020.py \\
        --experiment-dir outputs/EXP-020_fourier_pressure_conditioning \\
        --config configs/EXP-020_fourier_pressure_conditioning.yaml \\
        --device cuda \\
        --n-samples 4 \\
        --n-targets 3 \\
        --target-seed 0
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import yaml
from torch_geometric.data import Data

from graph_diffusion.building_blocks.noise_schedule import NoiseSchedule
from graph_diffusion.data.pOnEllipseConditional import (
    dct_ii,
    pOnEllipseConditionalDataset,
)
from graph_diffusion.data.transforms import (
    ComputeAngularEdgeFeatures,
    ComputeArcLengthEdgeFeatures,
)
from graph_diffusion.model.graph_diffusion_model import GraphDiffusionModel
from graph_diffusion.model.pressure_head import PressurePredictionHead
from graph_diffusion.model.score_network import ScoreNetwork


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-dir", required=True, type=Path)
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--n-samples", default=4, type=int)
    p.add_argument("--n-targets", default=3, type=int)
    p.add_argument("--target-seed", default=0, type=int)
    p.add_argument(
        "--checkpoint",
        default=None,
        type=Path,
        help="Override checkpoint path (default: <experiment-dir>/checkpoint_best.pt)",
    )
    return p.parse_args()


def build_dataset(config: dict) -> pOnEllipseConditionalDataset:
    ds_cfg = config["ellipse_dataset"]
    feature_mode = ds_cfg.get("feature_mode", "radial_norm")
    pre_transform = (
        ComputeArcLengthEdgeFeatures()
        if feature_mode == "cartesian"
        else ComputeAngularEdgeFeatures()
    )
    return pOnEllipseConditionalDataset(
        root=ds_cfg.get("root", "data/ellipse"),
        cond_mode=ds_cfg.get("cond_mode", "fourier"),
        k_modes=ds_cfg.get("k_modes", 8),
        feature_mode=feature_mode,
        split=ds_cfg.get("split", "train"),
        n_samples=ds_cfg.get("n_samples", None),
        k_neighbors=ds_cfg.get("k_neighbors", 6),
        global_dim=ds_cfg.get("global_dim", 8),
        pre_transform=pre_transform,
    )


def build_model(config: dict, device: str) -> GraphDiffusionModel:
    ns_cfg = config["noise_schedule"]
    schedule = NoiseSchedule(
        T=ns_cfg["T"],
        schedule_type=ns_cfg.get("schedule_type", "cosine"),
        beta_start=ns_cfg.get("beta_start", 1.0e-4),
        beta_end=ns_cfg.get("beta_end", 0.02),
    )
    sn_cfg = config["score_network"]
    mlp_cfg = config["mlp"]
    sn = ScoreNetwork(
        node_dim=sn_cfg["node_dim"],
        edge_dim=sn_cfg["edge_dim"],
        global_dim=sn_cfg["global_dim"],
        time_embed_dim=sn_cfg["time_embed_dim"],
        n_layers=sn_cfg["n_layers"],
        hidden_dims=sn_cfg.get("hidden_dims", [64, 64]),
        activation=mlp_cfg.get("activation", "silu"),
        layer_norm=mlp_cfg.get("layer_norm", True),
        residual=mlp_cfg.get("residual", True),
        input_dim=sn_cfg.get("input_dim", None),
        cond_dim=sn_cfg.get("cond_dim", None),
        p_uncond=float(sn_cfg.get("p_uncond", 0.0)),
        output_dim=sn_cfg.get("output_dim", None),
    )
    ph_cfg = config["pressure_head"]
    head = PressurePredictionHead(
        in_dim=ph_cfg["in_dim"],
        out_dim=ph_cfg["out_dim"],
        node_hidden=ph_cfg.get("node_hidden", [64, 64]),
        global_hidden=ph_cfg.get("global_hidden", [64, 64]),
        node_embed_dim=ph_cfg.get("node_embed_dim", 64),
        activation=mlp_cfg.get("activation", "silu"),
        layer_norm=mlp_cfg.get("layer_norm", True),
    )
    model_cfg = config.get("model", {})
    return GraphDiffusionModel(
        score_network=sn,
        noise_schedule=schedule,
        n_noise_channels=model_cfg.get("n_noise_channels", None),
        pressure_head=head,
        lambda_pressure=float(model_cfg.get("lambda_pressure", 0.0)),
    ).to(device)


def pick_targets_farthest_first(
    dataset: pOnEllipseConditionalDataset,
    n_targets: int,
    seed: int,
) -> list[int]:
    """Greedy farthest-first picking over dataset.cond vectors."""
    conds = torch.stack(
        [g.cond.squeeze(0) for g in dataset], dim=0
    ).numpy()
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, conds.shape[0]))
    picked = [first]
    while len(picked) < n_targets:
        picked_vecs = conds[picked]
        dists = np.linalg.norm(
            conds[:, None, :] - picked_vecs[None, :, :], axis=-1
        )
        min_dists = dists.min(axis=1)
        next_idx = int(np.argmax(min_dists))
        picked.append(next_idx)
    return picked


def make_synthetic_target(
    dataset: pOnEllipseConditionalDataset, k_modes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Construct an asymmetric synthetic Cp curve and DCT-encode it.

    Returns:
        (target_cp_dense, target_cond) — the dense curve for plotting and
        the K-mode condition vector for sampling.
    """
    # Average raw Cp curve across dataset, then add asymmetric perturbation.
    n_grid = 128
    x_over_c = np.linspace(0.0, 1.0, n_grid)
    # Reconstruct dataset mean Cp by inverse-DCTing the average cond.
    mean_cond = (
        torch.stack([g.cond.squeeze(0) for g in dataset], dim=0)
        .mean(dim=0)
        .numpy()
    )
    cp_mean_dense = _inverse_dct(mean_cond, n_grid)
    cp_synth = cp_mean_dense + 0.3 * np.sin(np.pi * x_over_c)
    target_cond = dct_ii(cp_synth.astype(np.float32), k_modes)
    return cp_synth.astype(np.float32), target_cond


def _inverse_dct(modes: np.ndarray, n_grid: int) -> np.ndarray:
    """Inverse type-II DCT of K modes onto an n_grid sample grid."""
    k_modes = modes.shape[0]
    n_idx = np.arange(n_grid, dtype=np.float32)
    k_idx = np.arange(k_modes, dtype=np.float32)[:, None]
    basis = np.cos(np.pi * (2.0 * n_idx + 1.0) * k_idx / (2.0 * n_grid))
    norm = np.full(k_modes, np.sqrt(2.0 / n_grid), dtype=np.float32)
    norm[0] = np.sqrt(1.0 / n_grid)
    return (modes * norm) @ basis  # type: ignore[no-any-return]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = args.device

    dataset = build_dataset(config)
    model = build_model(config, device)

    ckpt_path = args.checkpoint or args.experiment_dir / "checkpoint_best.pt"
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint from {ckpt_path}")

    # --- Targets ---
    train_target_indices = pick_targets_farthest_first(
        dataset, args.n_targets, args.target_seed
    )
    print(f"Picked training target indices: {train_target_indices}")

    k_modes = config["ellipse_dataset"]["k_modes"]
    synth_cp_dense, synth_cond_np = make_synthetic_target(dataset, k_modes)
    print(f"Built synthetic asymmetric target ({k_modes} DCT modes)")

    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.experiment_dir / "exp020_targets.npz",
        train_indices=np.array(train_target_indices),
        synth_cp_dense=synth_cp_dense,
        synth_cond=synth_cond_np,
    )
    print(f"Saved target manifest to {args.experiment_dir / 'exp020_targets.npz'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run the scaffold**

Run:
```bash
uv run python scripts/postprocess_exp020.py \
    --experiment-dir outputs/EXP-020-smoke \
    --config configs/EXP-020_fourier_pressure_conditioning.yaml \
    --device cuda \
    --n-samples 4 \
    --n-targets 3
```

Expected stdout includes `Loaded checkpoint from outputs/EXP-020-smoke/checkpoint_best.pt`, `Picked training target indices: [...]`, `Built synthetic asymmetric target (8 DCT modes)`, `Saved target manifest to outputs/EXP-020-smoke/exp020_targets.npz`.

If the smoke training run from earlier has been replaced by a longer one, point `--experiment-dir` at it instead.

- [ ] **Step 3: Commit**

```bash
git add scripts/postprocess_exp020.py
git commit -m "feat(train): scaffold postprocess_exp020.py with target picking"
```

---

### Task 9: Add sampling + Figure A + Figure C to `postprocess_exp020.py`

**Files:**
- Modify: `scripts/postprocess_exp020.py`

- [ ] **Step 1: Add sampling helpers and Figure A/C generation**

Replace the body of `main()` from the `# --- Targets ---` comment onward with the version below; also add the new helper functions before `main()`.

Add these helpers above `main()`:
```python
def shape_template_from_dataset(
    dataset: pOnEllipseConditionalDataset, device: str
) -> Data:
    """Use dataset[0]'s topology as the sampling template."""
    template = copy.copy(dataset[0])
    template = template.to(device)
    return template


def radial_to_xy(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """(N,) r and (N,) theta → (N, 2) (x, y) Cartesian."""
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


def template_thetas(dataset: pOnEllipseConditionalDataset) -> np.ndarray:
    """Recover the (N,) theta vector from dataset[0].pos = (cosθ, sinθ)."""
    pos = dataset[0].pos.numpy()
    return np.arctan2(pos[:, 1], pos[:, 0])


def sample_shapes_for_target(
    model: GraphDiffusionModel,
    template: Data,
    cond_vec: torch.Tensor,
    n_samples: int,
    guidance_scale: float,
    device: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generate n_samples shapes for a target cond.

    Returns:
        shapes: list of (N, 2) Cartesian arrays.
        head_pred_cps_dense: list of (N_cp_grid,) head-predicted Cp curves
            (one per sample, dense via inverse DCT).
    """
    template_with_cond = copy.copy(template)
    template_with_cond.cond = cond_vec.unsqueeze(0).to(device)

    shapes: list[np.ndarray] = []
    head_preds_dense: list[np.ndarray] = []
    pos = template.pos
    theta = np.arctan2(pos[:, 1].cpu().numpy(), pos[:, 0].cpu().numpy())
    k_modes = cond_vec.shape[0]

    for i in range(n_samples):
        torch.manual_seed(i)
        out = model.sample(
            template_with_cond,
            clamp_range=tuple(template_with_cond.x.new_tensor([0.5, 2.0]).tolist()),
            guidance_scale=guidance_scale,
        )
        r = out.x[:, 0].detach().cpu().numpy()
        shapes.append(radial_to_xy(r, theta))

        # Head prediction on the generated x̂₀.
        assert model.pressure_head is not None
        batch_vec = torch.zeros(
            pos.size(0), dtype=torch.long, device=device
        )
        with torch.no_grad():
            pred = model.pressure_head(
                out.x, pos, batch_vec
            )  # (1, K)
        head_preds_dense.append(_inverse_dct(pred[0].cpu().numpy(), 128))
    return shapes, head_preds_dense
```

Then replace the trailing portion of `main()` (everything after the `np.savez(...)` block) with:
```python
    # --- Figure A: conditioning grid ---
    from graph_diffusion.visualisation.plotting import (
        plot_conditioning_grid,
    )

    template = shape_template_from_dataset(dataset, device)
    sampling_cfg = config.get("sampling", {})
    guidance_scale = float(sampling_cfg.get("guidance_scale", 1.0))

    target_cps_dense: list[np.ndarray] = []
    head_pred_cps: list[np.ndarray] = []
    head_pred_stds: list[np.ndarray] = []
    sample_shapes: list[list[np.ndarray]] = []
    row_labels: list[str] = []

    for rank, idx in enumerate(train_target_indices):
        cond_vec = dataset[idx].cond.squeeze(0)
        target_dense = _inverse_dct(cond_vec.numpy(), 128)
        target_cps_dense.append(target_dense)

        shapes, head_preds = sample_shapes_for_target(
            model, template, cond_vec, args.n_samples, guidance_scale, device
        )
        sample_shapes.append(shapes)
        stacked = np.stack(head_preds, axis=0)  # (S, 128)
        head_pred_cps.append(stacked.mean(axis=0))
        head_pred_stds.append(stacked.std(axis=0))
        row_labels.append(f"train #{rank}")

    # Synthetic OOD row.
    synth_cond_t = torch.tensor(synth_cond_np)
    target_cps_dense.append(synth_cp_dense)
    shapes, head_preds = sample_shapes_for_target(
        model, template, synth_cond_t, args.n_samples, guidance_scale, device
    )
    sample_shapes.append(shapes)
    stacked = np.stack(head_preds, axis=0)
    head_pred_cps.append(stacked.mean(axis=0))
    head_pred_stds.append(stacked.std(axis=0))
    row_labels.append("synth asym.")

    fig_a = plot_conditioning_grid(
        target_cps=target_cps_dense,
        sample_shapes=sample_shapes,
        head_pred_cps=head_pred_cps,
        head_pred_stds=head_pred_stds,
        row_labels=row_labels,
    )
    fig_a_path = args.experiment_dir / "figure_a_conditioning_grid.png"
    fig_a.savefig(fig_a_path, dpi=140, bbox_inches="tight")
    print(f"Saved Figure A to {fig_a_path}")

    # --- Figure C: CFG sweep on the synthetic target ---
    cfg_w_values = [1.0, 3.0, 7.0]
    cfg_shapes: list[np.ndarray] = []
    cfg_head_preds: list[np.ndarray] = []
    for w in cfg_w_values:
        shapes_w, head_preds_w = sample_shapes_for_target(
            model, template, synth_cond_t, n_samples=1, guidance_scale=w, device=device
        )
        cfg_shapes.append(shapes_w[0])
        cfg_head_preds.append(head_preds_w[0])

    import matplotlib.pyplot as plt

    fig_c, axes = plt.subplots(2, 3, figsize=(12, 6))
    x_grid = np.linspace(0.0, 1.0, 128)
    for col, w in enumerate(cfg_w_values):
        ax = axes[0, col]
        xy = cfg_shapes[col]
        closed = np.vstack([xy, xy[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color="C2", lw=1.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"w = {w}", fontsize=10)

        ax = axes[1, col]
        ax.plot(x_grid, cfg_head_preds[col], color="C1", lw=2, label="head pred")
        ax.plot(x_grid, synth_cp_dense, color="C0", lw=1, ls="--", label="target")
        ax.grid(alpha=0.3)
        if col == 0:
            ax.legend(fontsize=8)
    fig_c.tight_layout()
    fig_c_path = args.experiment_dir / "figure_c_cfg_sweep.png"
    fig_c.savefig(fig_c_path, dpi=140, bbox_inches="tight")
    plt.close(fig_c)
    print(f"Saved Figure C to {fig_c_path}")
```

- [ ] **Step 2: Smoke-run**

Run:
```bash
uv run python scripts/postprocess_exp020.py \
    --experiment-dir outputs/EXP-020-smoke \
    --config configs/EXP-020_fourier_pressure_conditioning.yaml \
    --device cuda
```

Expected: stdout includes both `Saved Figure A to ...` and `Saved Figure C to ...`. The figures will be visually noisy because the smoke checkpoint is only 5 epochs — but the pipeline must complete.

- [ ] **Step 3: Eyeball the figures**

Inspect:
```bash
ls -la outputs/EXP-020-smoke/figure_a_conditioning_grid.png \
       outputs/EXP-020-smoke/figure_c_cfg_sweep.png
```

Both files exist and are larger than 10 kB.

- [ ] **Step 4: Commit**

```bash
git add scripts/postprocess_exp020.py
git commit -m "feat(train): add Figure A + Figure C generation to postprocess_exp020"
```

---

### Task 10: Add trajectory snapshotting + Figure B + animation to `postprocess_exp020.py`

**Files:**
- Modify: `scripts/postprocess_exp020.py`

- [ ] **Step 1: Append trajectory + Figure B + animation to `main()`**

Append after the `# --- Figure C ---` block in `main()`:
```python
    # --- Figure B: forward + reverse trajectory + animation ---
    from graph_diffusion.visualisation.plotting import (
        plot_trajectory_filmstrip,
        write_trajectory_animation,
    )
    from graph_diffusion.visualisation.trajectory import (
        collect_forward,
        collect_reverse,
    )

    T = config["noise_schedule"]["T"]
    # Filmstrip timesteps: 6 log-spaced indices into [1, T-1] plus 0.
    log_steps = np.unique(
        np.round(
            np.logspace(np.log10(1), np.log10(T - 1), num=5)
        ).astype(int)
    ).tolist()
    snapshot_steps = [0, *log_steps]  # 6 frames
    snapshot_steps = sorted(set(snapshot_steps))[:6]
    print(f"Trajectory snapshot timesteps: {snapshot_steps}")

    # Forward needs an x_0. Use the first training-target's clean shape.
    fwd_template = copy.copy(dataset[train_target_indices[0]]).to(device)
    forward_snaps_t = collect_forward(
        model, fwd_template, snapshot_steps=snapshot_steps, seed=0
    )

    # Reverse on the synthetic target.
    template_with_synth = copy.copy(template)
    template_with_synth.cond = synth_cond_t.unsqueeze(0).to(device)
    reverse_snaps_t = collect_reverse(
        model,
        template_with_synth,
        cond=template_with_synth.cond,
        snapshot_steps=snapshot_steps,
        guidance_scale=guidance_scale,
        seed=0,
    )

    # Convert snapshots to (N, 2) xy using the template's theta grid.
    theta = template_thetas(dataset)
    forward_xy = [
        radial_to_xy(s[:, 0].numpy(), theta) for s in forward_snaps_t
    ]
    reverse_xy = [
        radial_to_xy(s[:, 0].numpy(), theta) for s in reverse_snaps_t
    ]

    fig_b = plot_trajectory_filmstrip(
        forward_snapshots=forward_xy,
        reverse_snapshots=reverse_xy,
        timesteps=snapshot_steps,
        target_cp=synth_cp_dense,
    )
    fig_b_path = args.experiment_dir / "figure_b_trajectory.png"
    fig_b.savefig(fig_b_path, dpi=140, bbox_inches="tight")
    print(f"Saved Figure B to {fig_b_path}")

    # Full-resolution reverse for animation.
    all_steps = list(range(T - 1, -1, -1))
    full_reverse_t = collect_reverse(
        model,
        template_with_synth,
        cond=template_with_synth.cond,
        snapshot_steps=all_steps,
        guidance_scale=guidance_scale,
        seed=0,
    )
    full_reverse_xy = [
        radial_to_xy(s[:, 0].numpy(), theta) for s in full_reverse_t
    ]
    write_trajectory_animation(
        reverse_snapshots=full_reverse_xy,
        target_cp=synth_cp_dense,
        out_path_mp4=args.experiment_dir / "figure_b_reverse.mp4",
        out_path_gif=args.experiment_dir / "figure_b_reverse.gif",
        fps=25,
    )
    print(
        f"Saved Figure B animation to {args.experiment_dir / 'figure_b_reverse.mp4'} "
        f"and {args.experiment_dir / 'figure_b_reverse.gif'}"
    )
```

- [ ] **Step 2: Smoke-run**

```bash
uv run python scripts/postprocess_exp020.py \
    --experiment-dir outputs/EXP-020-smoke \
    --config configs/EXP-020_fourier_pressure_conditioning.yaml \
    --device cuda
```

Expected: all four figure outputs (`figure_a_*.png`, `figure_b_trajectory.png`, `figure_b_reverse.{mp4,gif}`, `figure_c_*.png`) exist and are non-empty. Animation will take ~30 s to encode for 200 frames.

- [ ] **Step 3: Verify file sizes**

```bash
ls -la outputs/EXP-020-smoke/figure_*.{png,mp4,gif}
```

Each file > 10 kB.

- [ ] **Step 4: Final quality gate**

```bash
uv run pytest tests/ -q && \
uv run ruff check src/ tests/ scripts/ && \
uv run black --check src/ tests/ scripts/ && \
uv run mypy src/
```

Expected: 156 passed; ruff clean; black clean; mypy 0 issues.

If ruff or black complain about `scripts/postprocess_exp020.py`: fix in place.

- [ ] **Step 5: Commit**

```bash
git add scripts/postprocess_exp020.py
git commit -m "feat(train): add Figure B trajectory filmstrip + animation to postprocess_exp020"
```

---

### Task 11: Launch the 200-epoch EXP-020 training run

**Files:** none (uses existing train.py and config)

This task can run in parallel with Tasks 1–10 — kick it off as early as possible to overlap compute with implementation.

- [ ] **Step 1: Create the output directory**

```bash
mkdir -p outputs/EXP-020_fourier_pressure_conditioning
```

- [ ] **Step 2: Launch training in the background, with a Monitor for failures + per-epoch save lines**

Use the Bash tool's `run_in_background=true` to launch:
```bash
uv run python train.py \
    --config configs/EXP-020_fourier_pressure_conditioning.yaml \
    --epochs 200 \
    --device cuda \
    --output outputs/EXP-020_fourier_pressure_conditioning/generated_shapes.png \
    2>&1 | tee outputs/EXP-020_fourier_pressure_conditioning/train.log
```

Then use the Monitor tool with:
```bash
tail -F outputs/EXP-020_fourier_pressure_conditioning/train.log \
  | grep --line-buffered -E "val_loss|Saved best checkpoint|NaN|Error|Traceback|Killed"
```
description: "EXP-020 200-epoch training progress"

Expected events (one per epoch + checkpoint saves) until completion.

- [ ] **Step 3: When training completes, verify artefacts**

```bash
ls outputs/EXP-020_fourier_pressure_conditioning/
```

Expected: `checkpoint.pt`, `checkpoint_best.pt`, `loss_log.json`, `generated_shapes.png`, `tensorboard/`.

- [ ] **Step 4: Inspect final val loss**

Run:
```bash
uv run python -c "import json; d=json.load(open('outputs/EXP-020_fourier_pressure_conditioning/loss_log.json')); print('best val_loss:', min(e['val_loss'] for e in d), 'at epoch', min(d, key=lambda e: e['val_loss'])['epoch'])"
```

Expected: best val_loss < 0.02 (5-epoch smoke hit 0.020; a converged 200-epoch run should beat it; flag if it doesn't).

- [ ] **Step 5: No commit** — training outputs are gitignored.

---

### Task 12: Run postprocess on the 200-epoch checkpoint and produce final figures

**Files:** none (executes the CLI from Task 10 against the full checkpoint from Task 11)

- [ ] **Step 1: Run postprocess on the converged checkpoint**

```bash
uv run python scripts/postprocess_exp020.py \
    --experiment-dir outputs/EXP-020_fourier_pressure_conditioning \
    --config configs/EXP-020_fourier_pressure_conditioning.yaml \
    --device cuda \
    --n-samples 4 \
    --n-targets 3 \
    --target-seed 0
```

Expected: all four figures land in `outputs/EXP-020_fourier_pressure_conditioning/`.

- [ ] **Step 2: Eyeball each figure**

For Figure A: confirm the 3 training rows show ring-like shapes, and row 4 (synth asym) shows broken symmetry.
For Figure B: confirm the reverse filmstrip transitions from noise to recognisable ring.
For Figure C: confirm `w=7` head-prediction tracks the target Cp more tightly than `w=1`.

If any figure looks broken: stop and triage rather than committing.

- [ ] **Step 3: Update EXP-020 experiment log**

Edit `docs/experiments/EXP-020_fourier_pressure_conditioning.md`:
- Change `status: planned` to `status: complete`
- Add a `## Results` section that links to the four figure files and reports the final val_loss + epoch.

- [ ] **Step 4: Commit the doc + figures**

```bash
git add docs/experiments/EXP-020_fourier_pressure_conditioning.md
git commit -m "docs(experiments): record EXP-020 200-epoch results and figures"
```

(Figures themselves remain in `outputs/`, which is gitignored — only the doc references them.)

---

## Self-Review Summary

**Spec coverage:**
- ✅ Reusable `graph_diffusion.visualisation` module — Tasks 1–6.
- ✅ Figure A (4 × 6 grid) — Task 9.
- ✅ Figure B (filmstrip + MP4 + GIF) — Tasks 6, 10.
- ✅ Figure C (CFG sweep) — Task 9.
- ✅ 200-epoch training — Task 11.
- ✅ Unit tests for all public visualisation functions — Tasks 2, 3, 4, 5, 6.
- ✅ Quality gate — Tasks 7, 10.
- ✅ Greedy farthest-first target picking — Task 8.
- ✅ Synthetic asymmetric Cp — Task 8.
- ✅ DPS is correctly out of scope — not referenced in any task.

**Placeholder scan:** No `TBD`, `TODO`, `fill in details`, "appropriate error handling", or "similar to Task N" — every code step shows complete code.

**Type consistency:**
- `collect_forward` / `collect_reverse` both take `snapshot_steps: list[int]` and return `list[torch.Tensor]` — consistent.
- `plot_conditioning_grid` uses `target_cps: list[np.ndarray]` matching the `target_cps_dense` produced in Task 9.
- `_inverse_dct(modes, n_grid)` defined in Task 8 is reused in Task 9 sampling helper — same signature.
- `pick_targets_farthest_first` returns `list[int]` — consumed as indices into `dataset` in Task 9.

**Scope:** Single implementation plan with one module + one script + one training run. No further decomposition needed.
