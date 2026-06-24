# Systematic OOD Cp Sampling, Validation & Assessment (EXP-043) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a principled OOD inverse-design experiment — a PC-mode Cp-target sampler, model generation, OpenLB validation, and a results-visualisation suite — then run it and assess where inverse design breaks down.

**Architecture:** A shared dataset Cp-PCA primitive (`data/cp_manifold.py`) feeds both a systematic target sampler and a results-visualisation module (`visualisation/sweep_results.py`, filesystem-free). A generation script turns targets into shapes via the existing inverse-design checkpoint; the existing `run_from_manifest.sh` validates them; a CLI loads run outputs and renders five figures against the dataset backdrop. Strict one-way DAG preserved (`data/` and `visualisation/` depend only on `data/pOnEllipse`; `scripts/` may import anything).

**Tech Stack:** Python 3.13, NumPy, Matplotlib, PyTorch/PyG, h5py, pytest; OpenLB 1.9.0 case (`cfd/ellipse2d/`).

**Spec:** `docs/superpowers/specs/2026-06-23-systematic-ood-cp-validation-design.md`

**Conventions (verified in codebase):**
- `N_CP_GRID = 128` and helpers `shape_template_from_dataset`, `_resolve_clamp_range` live in `scripts/postprocess_exp020.py`. `dct_ii`, `dct_ii_inverse` in `graph_diffusion.data.pOnEllipseConditional`. `radial_to_xy`, `template_thetas` in `graph_diffusion.postprocessing.inference`. `load_config`, `build_dataset`, `build_model` in `graph_diffusion.postprocessing.loaders`.
- HDF5 columns: `_H5_COL_X=0, _H5_COL_Y=1, _H5_COL_RE=2, _H5_COL_H=3, _H5_COL_P_START=4, _H5_N_TIMESTEPS=101` (in `graph_diffusion.data.pOnEllipse`); reuse `_load_h5_raw`, `_node_count`.
- Tests build synthetic in-memory HDF5-shaped arrays (see `tests/test_dataset_overview.py::_fake_raw`); import via `from context import graph_diffusion`. `torch.manual_seed(0)` only where torch randomness is used; `np.random.default_rng(0)` for numpy.
- Plotting fns return `matplotlib.figure.Figure`, do no file I/O; module + Google-style docstrings; `from __future__ import annotations`; lowercase generics; fully-qualified imports.
- Quality gate per commit: `uv run pytest tests/ -q && uv run ruff check src/ tests/ && uv run black --check src/ tests/ && uv run mypy src/`.
- OpenLB driver `cfd/ellipse2d/runs/run_from_manifest.sh`: resolves columns by header name, needs `Re`/`H` (capitalised) and `idx`; prefers `<name>.dat`; auto-scales RES from Re.
- Measured Cp: `runs/<name>/tmp/ellipse2d_cp.csv` (`x_over_c,Cp,Cp_rms`); force: `ellipse2d_force.csv` (`Cd_mean,Cd_std,Cl_mean,Cl_std,n_samples,t_start,t_end`).

---

## Task 1: Dataset Cp-PCA primitive (`cp_manifold.py`)

**Files:**
- Create: `src/graph_diffusion/data/cp_manifold.py`
- Test: `tests/test_cp_manifold.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for graph_diffusion.data.cp_manifold."""

import numpy as np
import pytest
from context import graph_diffusion  # noqa: F401

from graph_diffusion.data.cp_manifold import (
    CpPcaBasis,
    cp_pca_basis,
    dataset_cp_curves,
    dataset_cp_pca,
)


def _fake_raw(n_sims: int = 20, max_nodes: int = 16) -> np.ndarray:
    rng = np.random.default_rng(0)
    raw = np.full((n_sims, max_nodes, 105), np.nan, dtype=np.float32)
    grid = np.linspace(0.0, 1.0, 12)
    for s in range(n_sims):
        raw[s, :12, 0] = grid - 0.5            # x in [-0.5, 0.5]
        raw[s, :12, 1] = 0.0
        raw[s, :12, 2] = 700.0
        raw[s, :12, 3] = 5.5
        # Cp curve = base + scaled mode, so PCA has a clear leading direction.
        base = np.cos(np.pi * grid)
        raw[s, :12, 4:105] = (base + 0.3 * s / n_sims * grid)[:, None]
    return raw


def test_dataset_cp_curves_shape():
    raw = _fake_raw()
    curves = dataset_cp_curves(raw, n_grid=64)
    assert curves.shape == (20, 64)
    assert np.isfinite(curves).all()


def test_cp_pca_basis_orthonormal_components():
    raw = _fake_raw()
    curves = dataset_cp_curves(raw, n_grid=64)
    basis = cp_pca_basis(curves, n_components=2)
    assert isinstance(basis, CpPcaBasis)
    assert basis.mean.shape == (64,)
    assert basis.components.shape == (2, 64)
    assert basis.proj_std.shape == (2,)
    # Components orthonormal.
    gram = basis.components @ basis.components.T
    assert np.allclose(gram, np.eye(2), atol=1e-5)
    # Projection of the mean curve is ~0.
    assert np.allclose(basis.project(basis.mean), 0.0, atol=1e-5)


def test_dataset_cp_pca_grid_matches():
    raw = _fake_raw()
    grid, basis = dataset_cp_pca(raw, n_grid=32, n_components=2)
    assert grid.shape == (32,)
    assert basis.components.shape == (2, 32)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cp_manifold.py -q`
Expected: FAIL — `ModuleNotFoundError` (module not defined).

- [ ] **Step 3: Write minimal implementation**

```python
"""
graph_diffusion.data.cp_manifold
=================================

The dataset surface-pressure (Cp) manifold: resample every simulation's
mean Cp(x/c) onto a common grid and fit a small PCA basis. Used to build
principled OOD Cp targets (perturbing the dataset mean along its own
principal modes) and to project samples into the dataset Cp plane for
context plots. Pure NumPy; depends only on the pOnEllipse column layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from graph_diffusion.data.pOnEllipse import (
    _H5_COL_P_START,
    _H5_COL_X,
    _H5_N_TIMESTEPS,
    _node_count,
)

__all__ = ["CpPcaBasis", "cp_pca_basis", "dataset_cp_curves", "dataset_cp_pca"]


@dataclass
class CpPcaBasis:
    """Mean curve + leading principal components of the dataset Cp manifold.

    Attributes:
        mean (np.ndarray): Mean Cp curve, shape ``(n_grid,)``.
        components (np.ndarray): Unit principal components, shape
            ``(n_components, n_grid)``.
        proj_std (np.ndarray): Std of the dataset projections along each
            component, shape ``(n_components,)`` — the natural amplitude
            unit for an OOD ladder.
    """

    mean: np.ndarray
    components: np.ndarray
    proj_std: np.ndarray

    def project(self, curve: np.ndarray) -> np.ndarray:
        """Project a Cp curve (or array of curves) onto the components.

        Args:
            curve (np.ndarray): Shape ``(n_grid,)`` or ``(M, n_grid)``.

        Returns:
            np.ndarray: Coordinates, shape ``(n_components,)`` or
                ``(M, n_components)``.
        """
        centred = np.asarray(curve, dtype=np.float64) - self.mean
        return centred @ self.components.T


def dataset_cp_curves(raw: np.ndarray, n_grid: int = 128) -> np.ndarray:
    """Every sim's mean Cp(x/c) resampled onto a shared ``n_grid`` grid.

    Args:
        raw (np.ndarray): Full HDF5 array ``(N_sims, MAX_NODES, 105)``.
        n_grid (int): Number of x/c grid points.

    Returns:
        np.ndarray: Shape ``(N_sims, n_grid)``, dtype float64.
    """
    grid = np.linspace(0.0, 1.0, n_grid)
    out = np.empty((raw.shape[0], n_grid), dtype=np.float64)
    for i in range(raw.shape[0]):
        n = _node_count(raw[i])
        data = raw[i, :n]
        xc = data[:, _H5_COL_X].astype(np.float64)
        chord = max(xc.max() - xc.min(), 1e-12)
        x_over_c = (xc - xc.min()) / chord
        cp_mean = data[
            :, _H5_COL_P_START : _H5_COL_P_START + _H5_N_TIMESTEPS
        ].mean(axis=1)
        order = np.argsort(x_over_c)
        out[i] = np.interp(grid, x_over_c[order], cp_mean[order])
    return out


def cp_pca_basis(curves: np.ndarray, n_components: int = 2) -> CpPcaBasis:
    """Fit a PCA basis to a set of Cp curves via thin SVD.

    Args:
        curves (np.ndarray): Shape ``(N, n_grid)``.
        n_components (int): Number of principal components to retain.

    Returns:
        CpPcaBasis: Mean, unit components, and per-component projection std.

    Raises:
        ValueError: If ``n_components`` exceeds the available rank.
    """
    curves = np.asarray(curves, dtype=np.float64)
    if n_components < 1 or n_components > min(curves.shape):
        raise ValueError(
            f"n_components must be in [1, {min(curves.shape)}], got {n_components}"
        )
    mean = curves.mean(axis=0)
    centred = curves - mean
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    components = vt[:n_components]
    proj_std = (centred @ components.T).std(axis=0)
    return CpPcaBasis(mean=mean, components=components, proj_std=proj_std)


def dataset_cp_pca(
    raw: np.ndarray, n_grid: int = 128, n_components: int = 2
) -> tuple[np.ndarray, CpPcaBasis]:
    """Convenience: resample dataset Cp curves and fit a PCA basis.

    Args:
        raw (np.ndarray): Full HDF5 array.
        n_grid (int): x/c grid resolution.
        n_components (int): Components to retain.

    Returns:
        tuple[np.ndarray, CpPcaBasis]: ``(grid, basis)`` where ``grid`` is
            ``np.linspace(0, 1, n_grid)``.
    """
    grid = np.linspace(0.0, 1.0, n_grid)
    curves = dataset_cp_curves(raw, n_grid=n_grid)
    return grid, cp_pca_basis(curves, n_components=n_components)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cp_manifold.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/graph_diffusion/data/cp_manifold.py tests/test_cp_manifold.py
git commit -m "feat(data): add dataset Cp-PCA manifold primitive"
```

---

## Task 2: Refactor `plot_cp_pca` onto the shared basis

**Files:**
- Modify: `src/graph_diffusion/visualisation/dataset_overview.py` (`plot_cp_pca`)

DRY: `plot_cp_pca` currently inlines its own SVD. Route it through `cp_pca_basis` so there is one PCA implementation. Behaviour (a 2-D scatter coloured by Re/b) is unchanged; the existing smoke test must stay green.

- [ ] **Step 1: Read the current `plot_cp_pca`** in `dataset_overview.py` to confirm it builds `curves` by `np.interp` onto a 100-pt grid, mean-centres, does `np.linalg.svd`, and scatters `coords[:,0]` vs `coords[:,1]`.

- [ ] **Step 2: Replace the SVD/centring block** inside `plot_cp_pca` (keep everything else — the grid build, the colour key, the scatter, labels). Add the import at the top of the file:

```python
from graph_diffusion.data.cp_manifold import cp_pca_basis
```

Replace the lines that build `centred`, call `np.linalg.svd`, and compute `coords` with:

```python
    basis = cp_pca_basis(curves, n_components=2)
    coords = basis.project(curves)
```

(`curves` is the `(N, 100)` stack already built in the function. `basis.project` returns `(N, 2)`.)

- [ ] **Step 3: Run the dataset-overview tests**

Run: `uv run pytest tests/test_dataset_overview.py -q`
Expected: PASS (all existing tests, including `test_extra_figures_smoke`).

- [ ] **Step 4: Quality gate on touched files**

Run: `uv run ruff check src/graph_diffusion/visualisation/dataset_overview.py && uv run black --check src/graph_diffusion/visualisation/dataset_overview.py && uv run mypy src/graph_diffusion/visualisation/dataset_overview.py`
Expected: clean. (If `np` becomes unused in the file, it won't — other functions use it.)

- [ ] **Step 5: Commit**

```bash
git add src/graph_diffusion/visualisation/dataset_overview.py
git commit -m "refactor(integration): route plot_cp_pca through shared cp_pca_basis"
```

---

## Task 3: PC-mode amplitude-ladder target builder

**Files:**
- Modify: `src/graph_diffusion/data/cp_manifold.py`
- Test: `tests/test_cp_manifold.py`

A pure function producing the systematic OOD Cp targets: for each principal mode and signed amplitude (in σ units), `target = mean + s·α·proj_std[m]·component[m]`, with a sortable name.

- [ ] **Step 1: Write the failing test** (append to `tests/test_cp_manifold.py`)

```python
from graph_diffusion.data.cp_manifold import pc_ladder_targets


def test_pc_ladder_targets_count_and_names():
    raw = _fake_raw()
    grid, basis = dataset_cp_pca(raw, n_grid=64, n_components=2)
    targets = pc_ladder_targets(basis, amplitudes=(1.0, 2.0, 3.0, 4.0), n_modes=2)
    assert len(targets) == 2 * 4 * 2  # modes x amps x signs = 16
    names = [t["name"] for t in targets]
    assert len(set(names)) == 16  # unique
    assert "pc1_p1sig" in names and "pc2_m4sig" in names
    for t in targets:
        assert t["cp_target"].shape == (64,)


def test_pc_ladder_amplitude_increases_deviation():
    raw = _fake_raw()
    _, basis = dataset_cp_pca(raw, n_grid=64, n_components=2)
    targets = pc_ladder_targets(basis, amplitudes=(1.0, 4.0), n_modes=1)
    by_name = {t["name"]: t for t in targets}
    dev1 = np.linalg.norm(by_name["pc1_p1sig"]["cp_target"] - basis.mean)
    dev4 = np.linalg.norm(by_name["pc1_p4sig"]["cp_target"] - basis.mean)
    assert dev4 > dev1  # larger sigma-amplitude => farther from the mean
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cp_manifold.py -q`
Expected: FAIL — `ImportError` (`pc_ladder_targets` not defined).

- [ ] **Step 3: Implement** (append to `cp_manifold.py`; add `"pc_ladder_targets"` to `__all__`; add `from typing import Any` at the top)

```python
def pc_ladder_targets(
    basis: CpPcaBasis,
    amplitudes: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0),
    n_modes: int = 2,
) -> list[dict[str, Any]]:
    """Systematic OOD Cp targets along the dataset's principal Cp modes.

    For each mode ``m`` in ``1..n_modes``, signed amplitude ``s in {+, -}``
    and ``alpha`` in ``amplitudes``::

        cp_target = mean + s * alpha * proj_std[m-1] * components[m-1]

    The amplitude is in units of the dataset projection std, so ``alpha=1``
    is interior, ``~2`` is the border of the dataset spread, and ``>=3`` is
    out of distribution.

    Args:
        basis (CpPcaBasis): Dataset Cp-PCA basis.
        amplitudes (tuple[float, ...]): Sigma-unit amplitudes to sweep.
        n_modes (int): Number of leading modes to perturb along.

    Returns:
        list[dict[str, Any]]: One dict per target with keys ``name``
            (e.g. ``"pc1_p2sig"``), ``mode`` (int, 1-based), ``sign``
            (``+1``/``-1``), ``amp_sigma`` (float), and ``cp_target``
            (``np.ndarray (n_grid,)``).

    Raises:
        ValueError: If ``n_modes`` exceeds the available components.
    """
    if n_modes < 1 or n_modes > basis.components.shape[0]:
        raise ValueError(
            f"n_modes must be in [1, {basis.components.shape[0]}], got {n_modes}"
        )
    targets: list[dict[str, Any]] = []
    for m in range(1, n_modes + 1):
        comp = basis.components[m - 1]
        sigma = float(basis.proj_std[m - 1])
        for alpha in amplitudes:
            for sign, tag in ((1.0, "p"), (-1.0, "m")):
                cp_target = basis.mean + sign * alpha * sigma * comp
                targets.append(
                    {
                        "name": f"pc{m}_{tag}{int(round(alpha))}sig",
                        "mode": m,
                        "sign": int(sign),
                        "amp_sigma": float(alpha),
                        "cp_target": cp_target.astype(np.float64),
                    }
                )
    return targets
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cp_manifold.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/graph_diffusion/data/cp_manifold.py tests/test_cp_manifold.py
git commit -m "feat(data): add PC-mode amplitude-ladder OOD Cp target builder"
```

---

## Task 4: Generation script (`export_ood_systematic.py`)

**Files:**
- Create: `scripts/export_ood_systematic.py`

Generates a shape for each PC-ladder target via the inverse-design checkpoint, writing a run-driver-compatible manifest. Mirrors `scripts/export_ood_sweep.py` (read it first) but draws targets from `pc_ladder_targets`. Script-level (needs checkpoint); verified by execution, not unit tests.

- [ ] **Step 1: Read `scripts/export_ood_sweep.py`** end-to-end to copy its model-loading, sampling, decode, and file-writing idioms exactly (imports from `postprocess_exp020`, `graph_diffusion.postprocessing.*`, `dct_ii`).

- [ ] **Step 2: Write the script**

```python
"""Generate the systematic PC-mode OOD Cp sweep for EXP-043.

Builds OOD Cp targets along the dataset's principal Cp modes (a controlled
in-distribution -> border -> OOD ladder via cp_manifold.pc_ladder_targets),
samples a shape for each with the inverse-design checkpoint (plain
classifier-free guidance; DPS hurts OOD per EXP-042), and exports each as an
OpenLB ``.dat`` plus its target Cp and a run-driver-compatible manifest
(name,idx,Re,H,ood_dist,mode,sign,amp_sigma). Run conditions are fixed at
Re/H = 754/5.5 so the Cp-target morphology is the only variable.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from postprocess_exp020 import (  # type: ignore[import-not-found]
    N_CP_GRID,
    _resolve_clamp_range,
    shape_template_from_dataset,
)

from graph_diffusion.data.cp_manifold import dataset_cp_pca, pc_ladder_targets
from graph_diffusion.data.pOnEllipse import _load_h5_raw
from graph_diffusion.data.pOnEllipseConditional import dct_ii
from graph_diffusion.postprocessing.inference import radial_to_xy, template_thetas
from graph_diffusion.postprocessing.loaders import (
    build_dataset,
    build_model,
    load_config,
)

_OOD_RE = 754.0
_OOD_H = 5.5


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/EXP-022_T1000.yaml")
    parser.add_argument(
        "--checkpoint", default="outputs/EXP-022_T1000/checkpoint_best.pt"
    )
    parser.add_argument(
        "--h5", type=Path, default=Path("data/ellipse/raw/pOnEllipseTrain.h5")
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("cfd/ellipse2d/shapes/ood_systematic")
    )
    parser.add_argument("--n-modes", type=int, default=2)
    parser.add_argument(
        "--amplitudes", type=float, nargs="*", default=[1.0, 2.0, 3.0, 4.0]
    )
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    dataset = build_dataset(config)
    model = build_model(config, args.device)
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()

    clamp_range = _resolve_clamp_range(config)
    k_modes = dataset.k_modes
    template = shape_template_from_dataset(dataset, args.device)
    theta = template_thetas(template)

    train_conds = torch.stack([g.cond.squeeze(0) for g in dataset], dim=0).numpy()

    # PC-mode targets on the N_CP_GRID grid (matches the generation pipeline).
    raw = _load_h5_raw(args.h5)
    _, basis = dataset_cp_pca(raw, n_grid=N_CP_GRID, n_components=args.n_modes)
    targets = pc_ladder_targets(
        basis, amplitudes=tuple(args.amplitudes), n_modes=args.n_modes
    )
    x_over_c = np.linspace(0.0, 1.0, N_CP_GRID)

    rows = []
    for i, t in enumerate(targets):
        cp_target = t["cp_target"].astype(np.float32)
        cond = dct_ii(cp_target, k_modes).astype(np.float32)
        ood_dist = float(np.linalg.norm(train_conds - cond, axis=1).min())

        template.cond = (
            torch.tensor(cond, dtype=torch.float32).unsqueeze(0).to(args.device)
        )
        torch.manual_seed(args.seed)
        out = model.sample(
            template, clamp_range=clamp_range, guidance_scale=args.guidance
        )
        radii = out.x[:, 0].detach().cpu().numpy()
        xy = radial_to_xy(radii, theta)

        name = t["name"]
        dat = args.out_dir / f"{name}.dat"
        with dat.open("w") as fh:
            fh.write(
                f"# EXP-043 systematic OOD: {name} mode={t['mode']} "
                f"amp={t['amp_sigma']}sig ood_dist={ood_dist:.4g}\n# x y\n"
            )
            for x, y in xy:
                fh.write(f"{x:.10f} {y:.10f}\n")

        with (args.out_dir / f"{name}_target_cp.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["x_over_c", "cp_target"])
            for xc, c in zip(x_over_c, cp_target, strict=True):
                writer.writerow([f"{xc:.6f}", f"{c:.6f}"])

        rows.append(
            {
                "name": name,
                "idx": i,
                "Re": _OOD_RE,
                "H": _OOD_H,
                "ood_dist": ood_dist,
                "mode": t["mode"],
                "sign": t["sign"],
                "amp_sigma": t["amp_sigma"],
            }
        )
        print(f"{name:>10}: ood_dist={ood_dist:.4f} -> {dat.name} ({len(xy)} pts)")

    fields = ["name", "idx", "Re", "H", "ood_dist", "mode", "sign", "amp_sigma"]
    with (args.out_dir / "manifest.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {len(rows)} systematic OOD targets to {args.out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Lint the script**

Run: `uv run ruff check scripts/export_ood_systematic.py && uv run black --check scripts/export_ood_systematic.py`
Expected: clean (run `uv run black scripts/export_ood_systematic.py` first if needed).

- [ ] **Step 4: Tiny generation smoke** (proves end-to-end wiring with the real checkpoint; 2 shapes only)

Run:
```bash
python scripts/export_ood_systematic.py --n-modes 1 --amplitudes 1 --out-dir /tmp/ood_sys_smoke
ls /tmp/ood_sys_smoke && head -3 /tmp/ood_sys_smoke/manifest.csv
```
Expected: writes `pc1_p1sig.dat`, `pc1_m1sig.dat`, their `_target_cp.csv`, and a `manifest.csv` with header `name,idx,Re,H,ood_dist,mode,sign,amp_sigma`. If the checkpoint/config is missing, report BLOCKED with the path — do not fake success.

- [ ] **Step 5: Commit**

```bash
git add scripts/export_ood_systematic.py
git commit -m "feat(integration): add systematic PC-mode OOD generation script"
```

---

## Task 5: Cp-match metric (`sweep_results.py`)

**Files:**
- Create: `src/graph_diffusion/visualisation/sweep_results.py`
- Test: `tests/test_sweep_results.py`

The corrected match metric: interpolate the **monotonic target** onto the (possibly non-monotonic) measured nodes, then report correlation, affine slope/offset, and RMS.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for graph_diffusion.visualisation.sweep_results."""

import matplotlib

matplotlib.use("Agg")  # headless backend, set before any pyplot import

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from context import graph_diffusion  # noqa: E402, F401

from graph_diffusion.visualisation.sweep_results import cp_match_metrics  # noqa: E402


def test_identical_curves_perfect_match():
    x = np.linspace(0.0, 1.0, 50)
    cp = np.cos(np.pi * x)
    m = cp_match_metrics(x, cp, x, cp)
    assert m["corr"] == pytest.approx(1.0, abs=1e-6)
    assert m["affine_a"] == pytest.approx(1.0, abs=1e-6)
    assert m["rms"] == pytest.approx(0.0, abs=1e-6)


def test_affine_slope_recovered_on_nonmonotonic_measured():
    # Target monotonic in x; measured nodes trace a non-monotonic path but
    # carry cp = 2*target. The interp-onto-measured direction must recover a=2.
    xt = np.linspace(0.0, 1.0, 64)
    cpt = np.cos(np.pi * xt)
    xm = np.r_[np.linspace(0.0, 1.0, 32), np.linspace(1.0, 0.0, 32)]  # non-monotonic
    cpm = 2.0 * np.cos(np.pi * xm)
    m = cp_match_metrics(xt, cpt, xm, cpm)
    assert m["corr"] == pytest.approx(1.0, abs=1e-3)
    assert m["affine_a"] == pytest.approx(2.0, rel=1e-2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sweep_results.py -q`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
"""
graph_diffusion.visualisation.sweep_results
============================================

Figures for assessing OpenLB validation sweeps (in-distribution and OOD
inverse-design samples) against the dataset distribution. Filesystem-free:
functions take loaded arrays / ``SampleRecord`` lists and return Figures.
The per-set on-disk loaders live in ``scripts/plot_sweep_results.py``.
"""

from __future__ import annotations

import numpy as np


def cp_match_metrics(
    target_x: np.ndarray,
    target_cp: np.ndarray,
    meas_x: np.ndarray,
    meas_cp: np.ndarray,
) -> dict[str, float]:
    """Match between a target Cp curve and an OpenLB-measured Cp.

    The OpenLB surface Cp is non-monotonic in x/c, so the monotonic target
    is interpolated onto the measured node positions (not the reverse), then
    compared. Reports Pearson correlation, the affine fit
    ``meas ≈ a*target_on_meas + b`` (``a`` is the magnitude metric), and RMS.

    Args:
        target_x (np.ndarray): Target x/c, strictly increasing, shape ``(K,)``.
        target_cp (np.ndarray): Target Cp, shape ``(K,)``.
        meas_x (np.ndarray): Measured x/c (may be non-monotonic), shape ``(N,)``.
        meas_cp (np.ndarray): Measured Cp, shape ``(N,)``.

    Returns:
        dict[str, float]: Keys ``corr``, ``affine_a``, ``affine_b``, ``rms``.
    """
    target_on_meas = np.interp(meas_x, target_x, target_cp)
    corr = float(np.corrcoef(meas_cp, target_on_meas)[0, 1])
    design = np.vstack([target_on_meas, np.ones_like(target_on_meas)]).T
    (a, b), *_ = np.linalg.lstsq(design, meas_cp, rcond=None)
    rms = float(np.sqrt(np.mean((meas_cp - target_on_meas) ** 2)))
    return {"corr": corr, "affine_a": float(a), "affine_b": float(b), "rms": rms}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sweep_results.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/graph_diffusion/visualisation/sweep_results.py tests/test_sweep_results.py
git commit -m "feat(integration): add corrected Cp-match metric for sweep results"
```

---

## Task 6: `SampleRecord` + sample gallery

**Files:**
- Modify: `src/graph_diffusion/visualisation/sweep_results.py`
- Test: `tests/test_sweep_results.py`

- [ ] **Step 1: Write the failing test** (append; `matplotlib.use("Agg")` is already set at the top of the file from Task 5)

```python
import matplotlib.figure  # noqa: E402

from graph_diffusion.visualisation.sweep_results import (  # noqa: E402
    SampleRecord,
    plot_sample_gallery,
)


def _fake_samples(n: int = 4) -> list:
    rng = np.random.default_rng(0)
    out = []
    xt = np.linspace(0.0, 1.0, 64)
    th = np.linspace(0.0, 2 * np.pi, 40, endpoint=False)
    for i in range(n):
        poly = np.stack([0.5 * np.cos(th), (0.25 + 0.02 * i) * np.sin(th)], axis=1)
        cpt = np.cos(np.pi * xt) + 0.1 * i
        out.append(
            SampleRecord(
                name=f"pc1_p{i + 1}sig",
                re=754.0,
                h=5.5,
                ood_dist=float(i),
                polygon=poly,
                target_x=xt,
                target_cp=cpt,
                meas_x=xt,
                meas_cp=cpt + 0.05 * rng.standard_normal(64),
            )
        )
    return out


def test_sample_gallery_smoke():
    fig = plot_sample_gallery(_fake_samples(4), ncols=2)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) >= 4
    matplotlib.pyplot.close("all")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sweep_results.py::test_sample_gallery_smoke -q`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement** (append to `sweep_results.py`; add imports `from dataclasses import dataclass`, `import matplotlib`, `import matplotlib.figure`, `import matplotlib.pyplot as plt` at the top)

```python
@dataclass
class SampleRecord:
    """One validated sample: generated/dataset shape + target & measured Cp.

    Attributes:
        name (str): Case name (matches the run dir).
        re (float): Reynolds number.
        h (float): Channel height.
        ood_dist (float | None): OOD distance (min-L2 in cond space) if known.
        polygon (np.ndarray): Body outline ``(N, 2)``.
        target_x (np.ndarray): Target Cp x/c, increasing, shape ``(K,)``.
        target_cp (np.ndarray): Target/reference Cp, shape ``(K,)``.
        meas_x (np.ndarray): OpenLB-measured x/c, shape ``(M,)``.
        meas_cp (np.ndarray): OpenLB-measured Cp, shape ``(M,)``.
    """

    name: str
    re: float
    h: float
    ood_dist: float | None
    polygon: np.ndarray
    target_x: np.ndarray
    target_cp: np.ndarray
    meas_x: np.ndarray
    meas_cp: np.ndarray


def plot_sample_gallery(
    samples: list[SampleRecord],
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
) -> matplotlib.figure.Figure:
    """Grid of panels: each sample's shape + measured vs target Cp + metrics.

    Args:
        samples (list[SampleRecord]): Samples to draw.
        ncols (int): Panels per row.
        figsize (tuple[float, float] | None): Figure size; auto if None.

    Returns:
        matplotlib.figure.Figure: The rendered gallery.
    """
    n = len(samples)
    nrows = int(np.ceil(n / ncols))
    figsize = figsize or (4.0 * ncols, 3.2 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for k, s in enumerate(samples):
        ax = axes[k // ncols][k % ncols]
        ax.axis("on")
        m = cp_match_metrics(s.target_x, s.target_cp, s.meas_x, s.meas_cp)
        order = np.argsort(s.meas_x)
        ax.plot(s.target_x, s.target_cp, "k-", lw=2, label="target")
        ax.plot(s.meas_x[order], s.meas_cp[order], "C1.-", ms=3, label="OpenLB")
        ax.invert_yaxis()
        ax.set_title(
            f"{s.name}\n" r"$\rho$=" f"{m['corr']:.2f}  a={m['affine_a']:.2f}",
            fontsize=8,
        )
        ax.set_xlabel("x/c", fontsize=7)
        ax.set_ylabel("Cp", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sweep_results.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/graph_diffusion/visualisation/sweep_results.py tests/test_sweep_results.py
git commit -m "feat(integration): add SampleRecord and sample gallery figure"
```

---

## Task 7: Context figures (design-space + Cp-manifold)

**Files:**
- Modify: `src/graph_diffusion/visualisation/sweep_results.py`
- Test: `tests/test_sweep_results.py`

- [ ] **Step 1: Write the failing test** (append; reuse `_fake_raw` shape — copy the helper from `tests/test_cp_manifold.py` into this file or import it; define a local one here)

```python
from graph_diffusion.visualisation.sweep_results import (  # noqa: E402
    plot_cp_manifold_context,
    plot_design_space_context,
)


def _fake_raw_ctx(n_sims: int = 30, max_nodes: int = 16) -> np.ndarray:
    rng = np.random.default_rng(1)
    raw = np.full((n_sims, max_nodes, 105), np.nan, dtype=np.float32)
    grid = np.linspace(0.0, 1.0, 12)
    for s in range(n_sims):
        raw[s, :12, 0] = grid - 0.5
        raw[s, :12, 1] = 0.0
        raw[s, :12, 2] = 500.0 + 500.0 * (s / n_sims)
        raw[s, :12, 3] = 5.0 + 1.0 * rng.random()
        raw[s, :12, 4:105] = (np.cos(np.pi * grid) + 0.2 * rng.standard_normal())[
            :, None
        ]
    return raw


def test_context_figures_smoke():
    raw = _fake_raw_ctx()
    samples = _fake_samples(4)
    f1 = plot_design_space_context(raw, samples)
    f2 = plot_cp_manifold_context(raw, samples)
    for fig in (f1, f2):
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) >= 1
    matplotlib.pyplot.close("all")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sweep_results.py::test_context_figures_smoke -q`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement** (append; add to the top imports: `from graph_diffusion.data.cp_manifold import dataset_cp_curves, dataset_cp_pca` and `from graph_diffusion.data.pOnEllipse import _H5_COL_H, _H5_COL_RE`)

```python
def plot_design_space_context(
    raw: np.ndarray,
    samples: list[SampleRecord],
    figsize: tuple[float, float] = (7.0, 5.0),
) -> matplotlib.figure.Figure:
    """Sample (Re, H) points over the full dataset cloud.

    Args:
        raw (np.ndarray): Full HDF5 array.
        samples (list[SampleRecord]): Samples to overlay.
        figsize (tuple[float, float]): Figure size.

    Returns:
        matplotlib.figure.Figure: The rendered figure.
    """
    re = raw[:, 0, _H5_COL_RE]
    h = raw[:, 0, _H5_COL_H]
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(re, h, s=8, c="0.7", alpha=0.5, label="dataset")
    ax.scatter(
        [s.re for s in samples],
        [s.h for s in samples],
        s=70,
        facecolors="none",
        edgecolors="red",
        linewidths=1.5,
        label="samples",
    )
    ax.set_xlabel("Reynolds number $Re$")
    ax.set_ylabel("channel height $H$")
    ax.set_title("Samples in dataset design space")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_cp_manifold_context(
    raw: np.ndarray,
    samples: list[SampleRecord],
    n_grid: int = 128,
    figsize: tuple[float, float] = (7.0, 5.0),
) -> matplotlib.figure.Figure:
    """Project sample target & measured Cp into the dataset Cp-PCA plane.

    The dataset cloud is the backdrop; an OOD sample sits outside it. Each
    sample's target (square) and measured (filled circle) are joined by a
    thin line so the generation+solver drift is visible.

    Args:
        raw (np.ndarray): Full HDF5 array.
        samples (list[SampleRecord]): Samples to project.
        n_grid (int): Cp grid for the basis.
        figsize (tuple[float, float]): Figure size.

    Returns:
        matplotlib.figure.Figure: The rendered figure.
    """
    grid, basis = dataset_cp_pca(raw, n_grid=n_grid, n_components=2)
    cloud = basis.project(dataset_cp_curves(raw, n_grid=n_grid))
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(cloud[:, 0], cloud[:, 1], s=8, c="0.75", alpha=0.5, label="dataset")
    for s in samples:
        tgt = basis.project(np.interp(grid, s.target_x, s.target_cp))
        mea = basis.project(np.interp(grid, np.sort(s.meas_x), s.meas_cp[np.argsort(s.meas_x)]))
        ax.plot([tgt[0], mea[0]], [tgt[1], mea[1]], "0.5", lw=0.6, zorder=1)
        ax.scatter(tgt[0], tgt[1], marker="s", s=45, c="C0", zorder=2)
        ax.scatter(mea[0], mea[1], marker="o", s=35, c="C3", zorder=2)
    ax.scatter([], [], marker="s", c="C0", label="target")
    ax.scatter([], [], marker="o", c="C3", label="measured")
    ax.set_xlabel("Cp PC 1")
    ax.set_ylabel("Cp PC 2")
    ax.set_title("Samples on the dataset Cp manifold")
    ax.legend()
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sweep_results.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/graph_diffusion/visualisation/sweep_results.py tests/test_sweep_results.py
git commit -m "feat(integration): add design-space and Cp-manifold context figures"
```

---

## Task 8: Cp-envelope overlay + results summary

**Files:**
- Modify: `src/graph_diffusion/visualisation/sweep_results.py`
- Test: `tests/test_sweep_results.py`

- [ ] **Step 1: Write the failing test** (append)

```python
from graph_diffusion.visualisation.sweep_results import (  # noqa: E402
    plot_cp_envelope_overlay,
    plot_results_summary,
)


def test_overlay_and_summary_smoke():
    raw = _fake_raw_ctx()
    samples = _fake_samples(5)
    f1 = plot_cp_envelope_overlay(raw, samples)
    f2 = plot_results_summary(samples)
    for fig in (f1, f2):
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) >= 1
    matplotlib.pyplot.close("all")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sweep_results.py::test_overlay_and_summary_smoke -q`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement** (append; add `from graph_diffusion.data.cp_manifold import dataset_cp_curves` to the top imports if not already present)

```python
def plot_cp_envelope_overlay(
    raw: np.ndarray,
    samples: list[SampleRecord],
    n_grid: int = 128,
    figsize: tuple[float, float] = (7.0, 5.0),
) -> matplotlib.figure.Figure:
    """Dataset Cp mean±std band with sample measured & target curves on top.

    Args:
        raw (np.ndarray): Full HDF5 array.
        samples (list[SampleRecord]): Samples to overlay.
        n_grid (int): Cp grid.
        figsize (tuple[float, float]): Figure size.

    Returns:
        matplotlib.figure.Figure: The rendered figure.
    """
    grid = np.linspace(0.0, 1.0, n_grid)
    curves = dataset_cp_curves(raw, n_grid=n_grid)
    mean = curves.mean(axis=0)
    std = curves.std(axis=0)
    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(
        grid, mean - std, mean + std, color="0.8", label=r"dataset mean$\pm\sigma$"
    )
    ax.plot(grid, mean, "k-", lw=1.5)
    for s in samples:
        order = np.argsort(s.meas_x)
        ax.plot(s.target_x, s.target_cp, "C0--", lw=0.8, alpha=0.7)
        ax.plot(s.meas_x[order], s.meas_cp[order], "C3-", lw=0.8, alpha=0.7)
    ax.plot([], [], "C0--", label="sample target")
    ax.plot([], [], "C3-", label="sample measured")
    ax.invert_yaxis()
    ax.set_xlabel("x/c")
    ax.set_ylabel("Cp")
    ax.set_title("Samples vs dataset Cp envelope")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_results_summary(
    samples: list[SampleRecord],
    figsize: tuple[float, float] = (8.0, 4.0),
) -> matplotlib.figure.Figure:
    """Correlation and affine slope ``a`` vs OOD distance across samples.

    Args:
        samples (list[SampleRecord]): Samples (need ``ood_dist`` set).
        figsize (tuple[float, float]): Figure size.

    Returns:
        matplotlib.figure.Figure: Two-panel figure (corr and a vs ood_dist).
    """
    ood = np.array([s.ood_dist if s.ood_dist is not None else np.nan for s in samples])
    corr = np.array(
        [cp_match_metrics(s.target_x, s.target_cp, s.meas_x, s.meas_cp)["corr"] for s in samples]
    )
    aff = np.array(
        [cp_match_metrics(s.target_x, s.target_cp, s.meas_x, s.meas_cp)["affine_a"] for s in samples]
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.scatter(ood, corr, c="C0")
    ax1.set_xlabel("OOD distance")
    ax1.set_ylabel(r"correlation $\rho$")
    ax1.set_title("Fidelity vs OOD distance")
    ax2.scatter(ood, aff, c="C3")
    ax2.axhline(1.0, color="0.6", ls="--", lw=1)
    ax2.set_xlabel("OOD distance")
    ax2.set_ylabel("affine slope $a$ (magnitude)")
    ax2.set_title("Magnitude creep vs OOD distance")
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sweep_results.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/graph_diffusion/visualisation/sweep_results.py tests/test_sweep_results.py
git commit -m "feat(integration): add Cp-envelope overlay and results-vs-OOD summary"
```

---

## Task 9: CLI with per-set loaders (`plot_sweep_results.py`)

**Files:**
- Create: `scripts/plot_sweep_results.py`

Holds the filesystem loaders (manifest schema + run-dir naming + target/reference Cp location differ per set) and renders the five figures. Two sets supported now: `ood_systematic` (target Cp + ood_dist) and `eval_grid` (dataset reference Cp). Verified on `eval_grid` (real data already on disk).

- [ ] **Step 1: Write the script**

```python
"""Render the sweep-results figure suite for a validated sample set.

Loads a shape set's manifest, each run's OpenLB-measured Cp, the target/
reference Cp, and the body polygon, then writes five figures (gallery,
design-space context, Cp-manifold context, Cp-envelope overlay, results-vs-
OOD summary) to outputs/sweep_results/<set>/. Read-only on the run outputs.

Usage:
    python scripts/plot_sweep_results.py --set eval_grid
    python scripts/plot_sweep_results.py --set ood_systematic
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

from graph_diffusion.data.pOnEllipse import _load_h5_raw  # noqa: E402
from graph_diffusion.visualisation.sweep_results import (  # noqa: E402
    SampleRecord,
    plot_cp_envelope_overlay,
    plot_cp_manifold_context,
    plot_design_space_context,
    plot_results_summary,
    plot_sample_gallery,
)

_CFD = Path("cfd/ellipse2d")


def _read_xy(dat_path: Path) -> np.ndarray:
    """Read an ``x y`` polygon .dat (skipping ``#`` comment lines)."""
    pts = []
    for line in dat_path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        x, y = line.split()
        pts.append((float(x), float(y)))
    return np.array(pts)


def _read_cp(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a 2-column Cp CSV (x_over_c, Cp); returns (x, cp)."""
    x, cp = [], []
    with csv_path.open() as fh:
        reader = csv.reader(fh)
        next(reader)  # header
        for row in reader:
            x.append(float(row[0]))
            cp.append(float(row[1]))
    return np.array(x), np.array(cp)


def _load_ood_systematic() -> list[SampleRecord]:
    shapes = _CFD / "shapes" / "ood_systematic"
    records = []
    with (shapes / "manifest.csv").open() as fh:
        for row in csv.DictReader(fh):
            name = row["name"]
            mx, mcp = _read_cp(_CFD / "runs" / name / "tmp" / "ellipse2d_cp.csv")
            tx, tcp = _read_cp(shapes / f"{name}_target_cp.csv")
            records.append(
                SampleRecord(
                    name=name,
                    re=float(row["Re"]),
                    h=float(row["H"]),
                    ood_dist=float(row["ood_dist"]),
                    polygon=_read_xy(shapes / f"{name}.dat"),
                    target_x=tx,
                    target_cp=tcp,
                    meas_x=mx,
                    meas_cp=mcp,
                )
            )
    return records


def _load_eval_grid() -> list[SampleRecord]:
    shapes = _CFD / "shapes" / "eval_grid"
    records = []
    with (shapes / "manifest.csv").open() as fh:
        for row in csv.DictReader(fh):
            name = row["name"]
            idx = row["idx"]
            mx, mcp = _read_cp(_CFD / "runs" / name / "tmp" / "ellipse2d_cp.csv")
            tx, tcp = _read_cp(shapes / f"sim_{idx}_ref_cp.csv")
            records.append(
                SampleRecord(
                    name=name,
                    re=float(row["Re"]),
                    h=float(row["H"]),
                    ood_dist=None,
                    polygon=_read_xy(shapes / f"sim_{idx}.dat"),
                    target_x=tx,
                    target_cp=tcp,
                    meas_x=mx,
                    meas_cp=mcp,
                )
            )
    return records


_LOADERS = {"ood_systematic": _load_ood_systematic, "eval_grid": _load_eval_grid}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", required=True, choices=sorted(_LOADERS))
    parser.add_argument(
        "--h5", type=Path, default=Path("data/ellipse/raw/pOnEllipseTrain.h5")
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or Path("outputs/sweep_results") / args.set
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = _LOADERS[args.set]()
    raw = _load_h5_raw(args.h5)

    figures = {
        "gallery": plot_sample_gallery(samples),
        "design_space_context": plot_design_space_context(raw, samples),
        "cp_manifold_context": plot_cp_manifold_context(raw, samples),
        "cp_envelope_overlay": plot_cp_envelope_overlay(raw, samples),
    }
    if any(s.ood_dist is not None for s in samples):
        figures["results_summary"] = plot_results_summary(samples)

    for name, fig in figures.items():
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"{name}.{ext}", dpi=200, bbox_inches="tight")
        print(f"wrote {name}.png / .pdf")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Lint**

Run: `uv run ruff check scripts/plot_sweep_results.py && uv run black --check scripts/plot_sweep_results.py`
Expected: clean (`uv run black scripts/plot_sweep_results.py` first if needed).

- [ ] **Step 3: Verify on the eval_grid real data** (already on disk)

Run:
```bash
python scripts/plot_sweep_results.py --set eval_grid
ls outputs/sweep_results/eval_grid/
```
Expected: prints `wrote …` for `gallery`, `design_space_context`, `cp_manifold_context`, `cp_envelope_overlay` (no `results_summary` — eval_grid has no ood_dist); 8 files (4 × png/pdf) present. If a run's `ellipse2d_cp.csv` is missing, report which — do not fake success.

- [ ] **Step 4: Full quality gate**

Run: `uv run pytest tests/ -q && uv run ruff check src/ tests/ && uv run black --check src/ tests/ && uv run mypy src/`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/plot_sweep_results.py
git commit -m "feat(integration): add sweep-results CLI with per-set loaders"
```

---

## Task 10: Run the pipeline + EXP-043 assessment

**Files:**
- Create: `docs/experiments/EXP-043_systematic_ood_validation.md` (copy structure from `docs/experiments/EXP-042_ood_fidelity_magnitude_creep.md`)

This task executes the experiment end-to-end and records it. No new library code.

- [ ] **Step 1: Generate the systematic OOD shapes** (full 16-target sweep)

Run:
```bash
python scripts/export_ood_systematic.py --n-modes 2 --amplitudes 1 2 3 4 \
  --out-dir cfd/ellipse2d/shapes/ood_systematic
cat cfd/ellipse2d/shapes/ood_systematic/manifest.csv
```
Expected: 16 `.dat` + `_target_cp.csv` + a manifest with `ood_dist` increasing with `amp_sigma`.

- [ ] **Step 2: Run the OpenLB validation sweep** (per-row RES auto-scales; Re fixed → RES≈151)

Run:
```bash
cd cfd/ellipse2d && bash runs/run_from_manifest.sh "$(pwd)/shapes/ood_systematic"
```
Expected: 16 runs, each printing a finite `ellipse2d_force.csv` tail; no NaN. (Long-running; the driver's `|| WARN` keeps the sweep alive on any single failure.)

- [ ] **Step 3: Render the figures for both sets**

Run:
```bash
python scripts/plot_sweep_results.py --set ood_systematic
python scripts/plot_sweep_results.py --set eval_grid
ls outputs/sweep_results/ood_systematic/ outputs/sweep_results/eval_grid/
```
Expected: five figures for `ood_systematic` (incl. `results_summary`), four for `eval_grid`.

- [ ] **Step 4: Write the EXP-043 log** at `docs/experiments/EXP-043_systematic_ood_validation.md` — read `EXP-042_ood_fidelity_magnitude_creep.md` for structure. Record: the PC-mode ladder design, the `ood_dist`/amplitude axis, per-sample ρ and affine `a` from the rendered `results_summary`, what the `cp_manifold_context` plot shows about the reachable Cp region, and a 2–4 sentence assessment of where inverse-design fidelity holds vs breaks down. Reference the figures by path. Do NOT invent numbers — read them from the produced figures / run `ellipse2d_force.csv` files.

- [ ] **Step 5: Update the changelog** — add a `2026-06-24 EXP-043` entry to `docs/experiments/changelog.md` (newest-first, matching the existing style) linking the spec, plan, and EXP-043 log.

- [ ] **Step 6: Commit**

```bash
git add docs/experiments/EXP-043_systematic_ood_validation.md docs/experiments/changelog.md
git commit -m "docs(experiments): add EXP-043 systematic OOD validation log"
```

---

## Final verification

- [ ] **Quality gate**: `uv run pytest tests/ -q && uv run ruff check src/ tests/ && uv run black --check src/ tests/ && uv run mypy src/` → all pass.
- [ ] **Deliverables**: `cp_manifold` primitive + PC-ladder; generation script; `sweep_results` module (5 figures) + CLI; 16-case OOD sweep run; figures for both sets under `outputs/sweep_results/`; EXP-043 log written from real numbers.

## Notes

- Tasks 1–3, 5–9 are pure code/test and can be done before any CFD runs (Tasks 5–9 are validated against the existing `eval_grid` data). Task 4 needs the checkpoint; Task 10 needs the GPU sweep (~30–45 min).
- The `EXP-022_T1000` checkpoint + plain CFG (DPS=0) are deliberate — continuity with EXP-041/042 and DPS degrades OOD (EXP-042).
