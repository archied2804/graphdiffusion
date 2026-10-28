---
experiment_id: "EXP-012"
title: "Radial mesh movement + per-node pressure conditioning"
date: 2026-05-13
status: completed
parent: "EXP-011"
tags: [ellipse, aerodynamic, radial_norm, positional-conditioning, conditional, pressure, inverse-design, node-concat, pOnEllipse]
config: "configs/EXP-012_ellipse_shape_method_B.yaml"
output_dir: "outputs/EXP-012_ellipse_radial_node_cond/"
---

# EXP-012: Radial mesh movement + per-node pressure conditioning

> **One-line summary:** Extend EXP-011 with full per-node Cp concatenation — preserving all spatial pressure information — while retaining the radial_norm + positional conditioning foundation.

## Motivation

EXP-011 conditioned on four global pressure scalars. This is information-lossy: two ellipses with different spatial distributions can share identical statistics. EXP-012 gives the score network the full per-node normalised pressure field concatenated to the noisy radial feature at each reverse step. The question is whether this richer spatial signal improves shape accuracy and/or allows better pressure-targeted inverse design.

## Hypothesis

- val_loss ≤ EXP-011 (model has strictly more information)
- Circularity CV ≥ EXP-011 (positional fix still in place; pressure adds information, not noise)
- Inverse design accuracy improves — per-node conditioning preserves pressure structure that global stats discard

## Changes from parent (EXP-011)

- **Dataset:** `EllipseConditionalDataset(feature_mode="radial_norm", cond_type="node_concat")`
  - `p_cond = [cos θ, sin θ, p_node]` (shape `(N, 3)`) — positional context prepended to per-node pressure
  - No global `cond` attribute (all conditioning is per-node)
- **Score network:** `input_dim=4` ([r_t, cos θ, sin θ, p_node]), `output_dim=1`, `n_noise_channels=1`
- **At inference:** Store target pressure field as `graph_template.p_cond[:, 2:]` (channel 2); pos channels 0–1 come from the template geometry automatically

## Method

### Architecture

Same 4-layer GN backbone. `input_dim=4` lifts `[r_t, cos θ, sin θ, p_node]` into node_dim=32; output projects to `output_dim=1` (noise for r only).

### Dataset construction

`EllipseConditionalDataset(feature_mode="radial_norm", cond_type="node_concat")` stores:
- `x = [r/r̄]` — noised channel
- `p_cond = [cos θ, sin θ, p_norm]` — 3-channel fixed conditioning
- `r_scale` — per-graph mean radius for reconstruction

### Training

300 epochs (limit), CosineAnnealingLR (lr=1e-3→1e-5), early stopping patience=20.

## Results

| Metric | Value |
|--------|-------|
| Best epoch | 99 (early stop at 119, patience=20) |
| Best val loss | 0.00101 |
| KS statistic (shape-space) | 0.1696 |
| Circularity CV | 0.2268 |
| Smoothness (2nd-order diff) | 0.0068 |
| Boundary violation rate | 0.000 |

### Comparison across EXP-010 → EXP-011 → EXP-012

| Metric | EXP-010 (unconditional) | EXP-011 (global Cp) | EXP-012 (per-node Cp) |
|--------|------------------------|--------------------|-----------------------|
| Val loss | 0.00290 | 0.00128 | **0.00101** |
| Circularity CV | 0.203 | 0.225 | **0.227** |
| Smoothness | 0.0070 | **0.0061** | 0.0068 |
| KS shape-space | — | **0.1652** | 0.1696 |

## Conclusions

- **val_loss hypothesis confirmed**: 0.00101 < EXP-011's 0.00128 — per-node conditioning strictly reduces loss as expected from the information argument.
- **Circularity CV hypothesis confirmed**: 0.2268 ≥ 0.2253 — positional fix remains effective; pressure adds signal.
- **Shape quality metrics mixed**: smoothness (0.0068 vs 0.0061) and KS shape-space (0.1696 vs 0.1652) are marginally worse than EXP-011 despite lower val_loss. The richer per-node conditioning may introduce slightly higher-frequency noise in the r(θ) profile that the loss doesn't penalise but the post-hoc metrics detect.
- The val_loss gap between EXP-011 and EXP-012 is smaller than EXP-010→EXP-011 (−21% vs −55%), suggesting diminishing returns from conditioning richness for unconditional generation quality. Per-node conditioning's advantage is expected to be most visible in **conditional inverse design** tasks (not tested here).
- Next focus: evaluate inverse design accuracy — whether providing a target pressure field at inference produces shapes with the correct aerodynamic profile.

## Next steps

- [x] Run postprocessing
- [x] Compare val_loss, circularity CV and KS shape-space to EXP-011
- [ ] Implement inverse design evaluation: given a target `cond_pressure`, generate shapes and check Cp distribution match
- [ ] Consider whether EXP-013 (e.g. conditional inverse design benchmark) is warranted
