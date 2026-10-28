# Aerodynamic Mesh Refactor — Design Spec

**Date:** 2026-05-14  
**Status:** Approved  
**Goal:** Refactor the repository from an accumulation of circle ablations and experimental conditional-diffusion code into a clean, aerodynamically-framed baseline that can be expanded upon.

---

## Motivation

The original goal is to learn a generative model over aerodynamic boundary mesh geometries from real CFD data (pOnEllipse, HuggingFace `mariolinov/Ellipse`), such that sampling produces novel, physically plausible surface mesh configurations that can be fed directly into a flow solver. The node positions are the output — the graph is not.

The codebase has drifted: it accumulated circle-series proof-of-concept code (EXP-001–006, now complete), a separate Fourier MLP ablation model, three dataset classes mixing pressure baselines with shape generation, and conditional diffusion features pre-built for experiments that have not yet run. The refactor removes this drift without touching the working DDPM machinery.

---

## Scope

This refactor is **organisational and naming-only** for the model layer. No DDPM logic, GN architecture, or data format changes. The conditional features (`n_noise_channels`, `cond_dim`, `smoothness_weight`) remain in place with clearer documentation.

---

## Section 1: Repository Structure

### Archive (new top-level folder, not a Python package)

```
archive/
  circle/
    data/circledataset.py
    data/dataset.py                    (SyntheticGraphDataset)
    model/fourier_score_network.py
    train_circle.py
    train_ddp.py
    configs/                           (all EXP-00x_*.yaml)
    tests/
      test_circledataset.py
      test_circledataset_rich.py
      test_circle_integration.py
      test_integration.py
  ellipse_experiments/
    data/ellipse_conditional.py        (EllipseConditionalDataset extracted)
    data/ellipse_pressure.py           (EllipseDataset extracted)
    configs/                           (EXP-013_*.yaml, EXP-013b_*.yaml)
    tests/
      test_ellipse_conditional.py
      test_ellipse_integration.py
```

### Active source tree after refactor

```
src/graph_diffusion/
  building_blocks/       (unchanged)
  model/                 (unchanged code; docstrings updated)
  data/
    base_dataset.py      (BaseGraphDataset ABC — extracted from dataset.py)
    transforms.py        (unchanged)
    dataloader.py        (unchanged)
    pOnEllipse.py        (renamed from ellipsedataset.py; see Section 2)
    __init__.py
  postprocessing/        (unchanged)

train.py                 (renamed from train_ellipse.py; see Section 3)
configs/
  EXP-010_*.yaml
  EXP-011_*.yaml
  EXP-012_*.yaml
```

---

## Section 2: Renames and Vocabulary

### `data/pOnEllipse.py`

| Before | After |
|--------|-------|
| `ellipsedataset.py` | `pOnEllipse.py` |
| `EllipseShapeDataset` | `pOnEllipseDataset` |
| `EllipseDataset` | archived → `archive/ellipse_experiments/data/ellipse_pressure.py` |
| `EllipseConditionalDataset` | archived → `archive/ellipse_experiments/data/ellipse_conditional.py` |

`pOnEllipseDataset` retains:
- `feature_mode`: `"radial"` | `"radial_norm"` | `"cartesian"` | `"normalised"` (unchanged — referenced in configs)
- `DatasetUrl` and `DatasetDownloader` (tightly coupled to the HuggingFace source, stay in the same file)
- `_build_ring_edge_index` module-level helper

### `data/base_dataset.py`

`BaseGraphDataset` extracted verbatim from `dataset.py`. `SyntheticGraphDataset` moves to `archive/circle/data/dataset.py`.

### `model/` — docstring additions only

`GraphDiffusionModel.__init__` and `ScoreNetwork.__init__` gain a `Note:` block on each conditional/experimental argument:

> **Note (future work — EXP-015+):** `n_noise_channels`, `feature_transform`, and `smoothness_weight` support conditional inverse design and bounded diffusion experiments. For the unconditional shape generation baseline leave these at their defaults (`None`, `None`, `0.0`).

Similarly for `ScoreNetwork`: `cond_dim` and `output_dim` get:

> **Note (future work — EXP-015+):** `cond_dim` enables global pressure conditioning (EXP-015); `output_dim` enables partial-channel noise prediction (EXP-016). Leave as `None` for the unconditional baseline.

No code changes to either class.

---

## Section 3: Training Script

`train_ellipse.py` → `train.py`

Changes:
- Module docstring rewritten to state aerodynamic purpose and list supported experiments
- `_DATASET_TYPES` reduced to `("shape",)` — `"pressure"` and `"conditional"` branches removed (both datasets archived; conditional will be re-added when EXP-015 is implemented)
- `_build_dataset` returns `pOnEllipseDataset` only
- Imports of archived `EllipseDataset` and `EllipseConditionalDataset` removed
- Everything else (training loop, TensorBoard, checkpointing, early stopping, plotting) unchanged

`train_circle.py` and `train_ddp.py` move to `archive/circle/`.

---

## Section 4: Tests

### Keep (update imports only)

```
tests/
  test_mlp.py
  test_graph_network.py
  test_noise_schedule.py
  test_score_network.py
  test_graph_diffusion_model.py
  test_transforms.py
  test_angular_features.py
  test_dataloader.py
  test_base_dataset.py        (renamed from test_dataset.py; remove SyntheticGraphDataset tests)
  test_pOnEllipse.py          (renamed from test_ellipsedataset.py; update class names)
  test_feature_transforms.py
  context.py                  (remove circle-specific fixtures)
```

### Archive

```
archive/circle/tests/
  test_circledataset.py
  test_circledataset_rich.py
  test_circle_integration.py
  test_integration.py

archive/ellipse_experiments/tests/
  test_ellipse_conditional.py
  test_ellipse_integration.py
```

---

## Section 5: CLAUDE.md Updates

The following sections of `CLAUDE.md` are updated to reflect the refactor:

- **Architecture / `data/`** subsection: replace references to `UnitCircleDataset`, `SyntheticGraphDataset` with `pOnEllipseDataset`; update file name `ellipsedataset.py` → `pOnEllipse.py`; add `base_dataset.py`
- **Data flow** subsection: replace `UnitCircleDataset → ComputeAngularEdgeFeatures` with `pOnEllipseDataset → ComputeAngularEdgeFeatures | ComputeArcLengthEdgeFeatures`
- **Commands** subsection: replace `train_circle.py` with `train.py`
- **Experiments & outputs**: update current roadmap line

---

## What Does Not Change

- The DDPM algorithm (`forward_diffusion`, `compute_loss`, `sample`, `sample_with_trajectory`)
- `GraphNetworkBlock`, `MLP`, `SinusoidalTimeEmbedding`, `NoiseSchedule`
- `ScoreNetwork` forward pass
- `ComputeAngularEdgeFeatures`, `ComputeArcLengthEdgeFeatures`
- `GraphDataLoader`
- `pOnEllipseDataset` internal logic (only the class name changes)
- All active experiment configs (EXP-010, 011, 012)
- `docs/experiments/` logs (no changes to experiment history)

---

## Success Criteria

1. `uv run pytest tests/ -q` passes with no failures on the active test suite
2. `uv run ruff check src/ tests/ && uv run mypy src/` passes clean
3. `src/graph_diffusion/` contains no references to `UnitCircleDataset`, `SyntheticGraphDataset`, `EllipseDataset`, `EllipseConditionalDataset`, or `FourierScoreNetwork`
4. `train.py --config configs/EXP-010_ellipse_data_pipeline.yaml --epochs 1` runs without error (smoke test)
5. A reader can open `src/graph_diffusion/` and immediately understand the aerodynamic purpose from class names and docstrings alone
