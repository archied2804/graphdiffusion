---
experiment_id: "EXP-010"
title: "Ellipse radial mesh movement baseline (radial_norm + positional conditioning)"
date: 2026-05-13
status: completed
parent: "EXP-005"
tags: [ellipse, aerodynamic, radial, radial_norm, positional-conditioning, shape-generation, pOnEllipse, node-movement]
config: "configs/EXP-010_ellipse_data_pipeline.yaml"
output_dir: "outputs/EXP-010_ellipse_radial_mesh/"
---

# EXP-010: Ellipse radial mesh movement baseline

> **One-line summary:** Apply DDPM to pOnEllipse boundary meshes using per-graph normalised radii `x = r/r̄` with absolute angular position `[cos θ, sin θ]` as fixed node-level conditioning — breaking rotational symmetry and enabling θ-dependent shape generation.

## Motivation

The EXP-00x circle series validated DDPM on synthetic radial shapes. EXP-010 is the direct analogue on real CFD data: switch in the actual pOnEllipse boundary meshes and train without any pressure conditioning.

### Why radial_norm

Diffusing absolute `r` conflates scale (mean radius, ~0.41, large, trivial) with shape (eccentricity profile r(θ)−r̄, std ≈ 0.06, the signal of interest). By t=50 in the cosine schedule the shape SNR is only 0.14. Normalising per graph by `r̄` removes scale and amplifies the shape signal.

### Why positional conditioning

`r_norm(θ)` varies from 0.80 (minor axis) to 1.20 (major axis) — a CV of 0.155 across θ bins. This is a function of **absolute θ**, but with node feature `x = [r/r̄]` and approximately-constant ring-edge features (Δθ ≈ 0.1 rad, nearly identical for all nodes), the GN is **rotationally symmetric**: it cannot distinguish nodes at different angular positions and collapses to predicting circular (uniform-r) noise. This was confirmed experimentally — the pre-fix run produced circularity CV = 0.099–0.119 vs reference 0.153.

Fix: store `p_cond = [cos θ, sin θ]` per node (shape `[N, 2]`). The score network receives `[r_t, cos θ, sin θ]` at each step with `n_noise_channels=1` (only r is noised; angular skeleton is fixed from the template). This is the same `node_concat` mechanism as EXP-012, but with geometry as context instead of pressure.

## Hypothesis

- Shape-space KS < 0.15 (r/r̄ distribution well-matched)
- Circularity CV ≥ 0.15 (matching reference eccentricity)
- Smoothness < 0.01 (well-formed closed curves)

## Changes from parent (EXP-005)

- **Dataset:** `EllipseShapeDataset(feature_mode="radial_norm")` — 5 701 CFD simulations
  - `x = [r/r̄]`, `p_cond = [cos θ, sin θ]`, `r_scale` stores `r̄` for reconstruction
- **Score network:** `input_dim=3` (r_t + cos θ + sin θ), `output_dim=1`, `n_noise_channels=1`
- **Clamp range:** `[0.5, 2.0]` (normalised data range [0.68, 1.37] with margin)
- **k=6 ring connectivity**

## Dataset

Per-ellipse CV(r): mean=0.153, range=[0.08, 0.23] — significant eccentricity variation.

## Method

### Architecture

4-layer GN, node_dim=32, edge_dim=2, global_dim=8, T=200 cosine schedule.  
`input_dim=3` ([r_t, cos θ, sin θ]), `output_dim=1`, `n_noise_channels=1`.

### Training

300 epochs (limit), CosineAnnealingLR (lr=1e-3→1e-5), early stopping patience=20.

## Results

| Metric | Value |
|--------|-------|
| Best epoch | 29 |
| Best val loss | 0.0029 |
| Early stop at epoch | 49 (patience=20) |
| KS statistic (shape-space r/r̄) | 0.1294 |
| KS statistic (physical r) | 0.2138 |
| Smoothness (2nd-order diff) | 0.0071 |
| Closure error | 0.0048 |
| Circularity (CV of radii) | 0.2033 |
| Boundary violation rate | 0.0000 |

Reference: CV(r) = 0.153

**Ablation summary** (same architecture, same dataset):

| Representation | Best epoch | val_loss | Circularity CV | Smoothness |
|---------------|-----------|----------|----------------|------------|
| Raw `r` (100 ep) | 99 | 0.0113 | 0.099 | 0.036 |
| `r/r̄` no pos (49 ep) | 29 | 0.0259 | 0.119 | 0.036 |
| `r/r̄` + pos (this run) | **29** | **0.0029** | **0.203** | **0.007** |
| Reference | — | — | 0.153 | — |

## Conclusions

Adding `[cos θ, sin θ]` as fixed node-level conditioning resolves the rotational symmetry failure. Circularity CV went from 0.099 (raw-radial baseline) to 0.203, exceeding the training-set reference of 0.153 — the model now captures the full eccentricity range. Smoothness improved 5× (0.036 → 0.007), indicating well-formed smooth closed curves. Convergence was also dramatically faster (best epoch 29 vs 99). The val_loss of 0.0029 is an order of magnitude below the previous runs, consistent with the model correctly learning the θ-dependent noise field rather than averaging it out.

## Next steps

- [ ] Proceed to [[EXP-011_ellipse_shape_method_A]] (radial_norm + global pressure conditioning)
