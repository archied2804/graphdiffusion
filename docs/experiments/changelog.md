---
tags: [changelog, features]
---

# Feature Changelog

Tracks code-level additions and modifications that support experiments.
Each entry links to the experiment that motivated it.

## 2026-04-13 — Circle experiment infrastructure

Motivated by: [[EXP-001_circle_radial_baseline]]

### New classes

| Class | Module | Purpose |
|-------|--------|---------|
| `UnitCircleDataset` | `graph_diffusion.data.circledataset` | Ring graphs with Fourier-perturbed radial profiles |
| `ComputeAngularEdgeFeatures` | `graph_diffusion.data.transforms` | `[sin(Δθ), cos(Δθ)]` edge features from `data.pos` |

### Modified classes

| Class | Change | Backward-compatible? |
|-------|--------|---------------------|
| `ScoreNetwork` | Added `input_dim: int \| None` — optional input/output linear projection | ✅ Yes (`None` = no change) |
| `GraphDiffusionModel` | Added `clamp_range: tuple \| None` to `sample()` | ✅ Yes (`None` = no clamping) |

### New files

| File | Purpose |
|------|---------|
| `src/graph_diffusion/data/circledataset.py` | `UnitCircleDataset` |
| `configs/circle_radial.yaml` | Circle experiment config |
| `train_circle.py` | Training + visualisation script |
| `tests/test_circledataset.py` | 15 unit tests |
| `tests/test_angular_features.py` | 7 unit tests |
| `tests/test_circle_integration.py` | 5 integration tests |

### Test impact

- Before: 109 tests
- After: 139 tests (+30)
- All passing, zero regressions

---

## 2026-05-07 — Full training infrastructure (EXP-005)

Motivated by: [[EXP-005_circle_radial_full]]

### Modified classes

| Class | Change | Backward-compatible? |
|-------|--------|---------------------|
| `train_circle.py` | Added `CosineAnnealingLR` scheduler; early stopping with best-checkpoint saving | ✅ Yes (opt-in via config) |

---

## 2026-05-07 — Rich node features (EXP-006)

Motivated by: [[EXP-006_circle_radial_rich-features]]

### Modified classes

| Class | Change | Backward-compatible? |
|-------|--------|---------------------|
| `UnitCircleDataset` | Added `include_curvature`, `include_arc_length` params; `_compute_curvature` and `_compute_arc_length_fraction` methods | ✅ Yes (both default to `False`) |
| `train_circle.py` | Passes `include_curvature`, `include_arc_length` to `UnitCircleDataset` from config | ✅ Yes |
| `scripts/postprocess_circle.py` | Passes same flags; evaluation uses `x[:, 0]` (r only) for comparability | ✅ Yes |
