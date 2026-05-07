---
experiment_id: "EXP-006"
title: "Richer node features: [r, κ, s/L] on unit circle radial diffusion"
date: 2026-05-07
status: planned
parent: "EXP-005, EXP-004"
tags: [rich-features, curvature, arc-length, circle, radial]
config: "configs/EXP-006_circle_radial_rich-features.yaml"
output_dir: "outputs/EXP-006_circle_radial_rich/"
---

# EXP-006: richer node features

> **One-line summary:** Extend the node feature vector from [r] to [r, κ, s/L] and test whether providing geometric context improves generation quality.

## Motivation

[[EXP-001_circle_radial_baseline]] through [[EXP-005_circle_radial_full]] use a single node feature (radius r). A richer representation could help the GN block learn geometry:

- **κ (curvature):** measures local bending. High κ → tight corner. Low κ → smooth arc. Provides local shape context beyond just radius.
- **s/L (normalised arc-length fraction):** tells each node its position along the curve (0 = start, 1 = end). Provides positional context around the ring.

These are deterministic functions of r given the topology, so the model must learn to generate consistent triplets.

## Hypothesis

- Providing κ and s/L as additional features enriches the score network's understanding of local geometry
- Expected improvement in smoothness and circularity CV vs EXP-005
- Training may be harder (3× feature dimensionality, implicit geometric constraints)

## Changes from parent

- **Config diff:** `score_network.input_dim: 3`; `circle_dataset.include_curvature: true`, `include_arc_length: true`; `data_root: data/circle_rich`
- **Code changes:**
  - `src/graph_diffusion/data/circledataset.py` — `include_curvature`, `include_arc_length` params; `_compute_curvature` (2D finite differences), `_compute_arc_length_fraction` (cumulative chord length)
  - `train_circle.py` — passes `include_curvature`, `include_arc_length` to `UnitCircleDataset`
  - `scripts/postprocess_circle.py` — passes same flags; evaluation uses `x[:, 0]` (r only)
- **Data changes:** new processed dataset at `data/circle_rich/` (shape `(N, 3)` vs `(N, 1)`)

## Method

### Architecture

Same 4-layer GN as baseline, except `ScoreNetwork` projects from `input_dim=3` → `node_dim=32` via `input_proj` (linear) and decodes back to 3 via `output_decode`.

### Feature computation

Curvature and arc-length computed numerically per graph during `UnitCircleDataset._build_graphs()`:

**Curvature (2D, periodic central differences):**
```
dx = roll(x,-1) - roll(x,1)     # ∝ 2Δθ x'
dy = roll(y,-1) - roll(y,1)
d2x = roll(x,-1) - 2x + roll(x,1)
d2y = roll(y,-1) - 2y + roll(y,1)
κ = 4 |dx·d2y - dy·d2x| / (dx² + dy²)^(3/2)
```

**Arc-length fraction:**
```
ds_i = ||(x_{i+1}, y_{i+1}) - (x_i, y_i)||  (periodic)
s_i / L = cumsum(ds)[i] / total_arc_length
```

### Dataset

2000 ring graphs, 64 nodes, k=2. Node features: `x = [r, κ, s/L]`, shape (64, 3).

### Training

100 epochs, cosine LR, early stopping (patience=20). Same as EXP-005.

### Evaluation

Metrics computed on `x[:, 0]` (generated radii) only — same metrics as all prior experiments for comparability.

## Results

> **Status: planned** — run after EXP-004 and EXP-005 complete.

### Metrics

| Metric | Value |
|--------|-------|
| Best epoch (early stopping) | |
| Best val loss | |
| Smoothness | |
| Circularity CV | |
| Boundary violations | |
| KS statistic | |

### Key question

Do the generated κ and s/L columns remain geometrically consistent with the generated r values? (Compare `κ_generated` with `κ_recomputed_from_r`.)

## Conclusions

*Fill in after results are available.*

## Next steps

- [ ] Geometric consistency check: compare κ_generated vs κ_recomputed
- [ ] If richer features help: apply to [[EXP-007_naca_radial_baseline]] (NACA aerofoil)
