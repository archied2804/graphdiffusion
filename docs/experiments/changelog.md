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
