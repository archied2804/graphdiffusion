---
experiment_id: "EXP-013"
title: "Radial shape generation + smoothness regularisation"
date: 2026-05-13
status: completed
parent: "EXP-011"
tags: [ellipse, aerodynamic, radial_norm, positional-conditioning, conditional, pressure, global-summary, smoothness, regularisation]
config: "configs/EXP-013_ellipse_smoothness_reg.yaml"
output_dir: "outputs/EXP-013_ellipse_smoothness_reg/"
---

# EXP-013: Radial shape generation + smoothness regularisation

> **One-line summary:** Add a second-order finite difference penalty on the x̂₀ reconstruction to EXP-011, targeting smoother generated boundary shapes without changing the architecture.

## Motivation

EXP-011 showed a smoothness metric (mean squared 2nd-order finite difference of generated radii) of 0.0061. While the shapes were plausible, high-frequency oscillations in the generated boundary degrade aerodynamic validity — rough profiles create non-physical pressure spikes in panel methods. The hypothesis is that a small loss penalty, rather than an architectural change, is sufficient to suppress this roughness because the score network already has access to neighbour radii through message passing and can suppress curvature given the right training signal.

## Hypothesis

- Smoothness metric decreases vs EXP-011 (0.0061) — the primary target
- Val loss within ±10% of EXP-011 (0.00128) — penalty should not dominate
- Circularity CV remains ≥ 0.20 — shapes should not collapse to circles

## Changes from parent (EXP-011)

- **Config:** `model.smoothness_weight: 1.0e-4` added
- **Code:** `GraphDiffusionModel.__init__` gains `smoothness_weight` param; `compute_loss` reconstructs `x̂₀ = (x_t − √(1−ᾱ_t)·ε_pred) / √ᾱ_t` and adds `λ · mean(Δ²x̂₀)²` to MSE; `_smoothness_loss` is fully vectorised (no Python loops over graphs)
- Everything else identical to EXP-011

## Method

### Architecture

4-layer GN, node_dim=32, edge_dim=2, global_dim=8, T=200 cosine schedule.
`input_dim=3` ([r_t, cos θ, sin θ]), `output_dim=1`, `cond_dim=4`, `n_noise_channels=1`.
`smoothness_weight=1e-4`.

### Dataset

`EllipseConditionalDataset(feature_mode="radial_norm", cond_type="global_summary")` — identical to EXP-011.

### Training

300 epochs (limit), CosineAnnealingLR (lr=1e-3→1e-5), early stopping patience=20.

## Results

| Metric | Value |
|--------|-------|
| Best epoch | 68 (early stop at 88, patience=20) |
| Best val loss (composite) | 0.00212 |
| KS statistic (shape-space) | 0.1513 |
| Circularity CV | 0.2156 |
| Smoothness (2nd-order diff) | 0.00723 |
| Boundary violation rate | 0.000 |

### Comparison vs EXP-011 (parent baseline)

| Metric | EXP-011 | EXP-013 | Δ |
|--------|---------|---------|---|
| Val loss | 0.00128 | 0.00212 | +66% (includes smoothness term) |
| Circularity CV | 0.2253 | 0.2156 | −0.010 |
| Smoothness | 0.0061 | 0.0072 | +18% ❌ |
| KS shape-space | 0.165 | 0.151 | −0.014 ✅ |
| Boundary violation rate | 0.000 | 0.000 | — |

### Observations

- The composite val_loss is higher than EXP-011's pure MSE loss because the smoothness penalty is included in the reported figure; the true denoising quality cannot be directly compared from these numbers alone.
- **Primary hypothesis not confirmed**: the smoothness metric (mean squared 2nd-order finite difference of generated radii) increased from 0.0061 → 0.0072 (+18%). The regularisation did not suppress boundary roughness.
- KS shape-space improved slightly (0.165 → 0.151), suggesting a marginally better match to the training distribution.
- Circularity CV fell from 0.225 → 0.216, moving slightly closer to the reference value of 0.153.
- Early stopping triggered at epoch 88 vs 118 in EXP-011 — the added loss term raised the absolute loss values, making it harder to improve against the patience counter.

## Conclusions

- **Negative result**: `smoothness_weight=1e-4` with SNR weighting (`×ᾱ_t`) is insufficient to improve boundary smoothness. Two likely causes: (a) the effective gradient contribution is suppressed too aggressively at high noise levels by the `ᾱ_t` factor, and (b) the weight itself may be too small once that suppression is applied.
- The SNR weighting was necessary to prevent loss spikes (up to 445 without it), but it may have reduced the penalty to a negligible signal.
- The approach remains worth pursuing at a higher weight (`smoothness_weight=1e-3`) or with a less aggressive weighting scheme (e.g., clip `ᾱ_t` from below at 0.1 rather than allowing it to reach near-zero).
- Alternatively, per-epoch annealing of the weight (start at 0, ramp to 1e-3) may allow stable early training while building up the smoothness signal.

## Next steps

- [ ] EXP-013b: repeat with `smoothness_weight=1e-3` and `ᾱ_t.clamp(min=0.1)` to give the penalty a stronger floor
- [ ] If still no improvement, consider post-hoc Fourier low-pass filtering of generated radii as a simpler baseline
- [ ] Proceed to XFOIL aerodynamic evaluation pipeline regardless (orthogonal track)
