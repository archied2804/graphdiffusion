---
experiment_id: "EXP-013b"
title: "Smoothness regularisation — stronger weight (λ=1e-3)"
date: 2026-05-13
status: completed
parent: "EXP-013"
tags: [ellipse, aerodynamic, radial_norm, positional-conditioning, conditional, pressure, global-summary, smoothness, regularisation]
config: "configs/EXP-013b_ellipse_smoothness_reg_strong.yaml"
output_dir: "outputs/EXP-013b_ellipse_smoothness_reg_strong/"
---

# EXP-013b: Smoothness regularisation — stronger weight (λ=1e-3)

> **One-line summary:** Repeat EXP-013 with `smoothness_weight` raised 10× (1e-4 → 1e-3) to test whether a stronger penalty signal can reduce boundary roughness.

## Motivation

EXP-013 produced a negative result: `smoothness_weight=1e-4` with SNR weighting (`×ᾱ_t`) failed to improve the smoothness metric (0.0061 → 0.0072, +18%). Analysis suggested the gradient contribution from the smoothness term was O(5e-7) per element — roughly 10,000× smaller than the MSE gradient — making it negligible. Raising λ to `1e-3` increases that contribution 10×. Loss remains stable (smoke-tested: max batch loss 1.01 with λ=1e-3 vs 1.00 baseline).

## Hypothesis

- Smoothness metric decreases vs EXP-013 (0.0072) and ideally vs EXP-011 (0.0061)
- Val loss within ±20% of EXP-011 (0.00128) — penalty is still subordinate to MSE
- Circularity CV ≥ 0.20

## Changes from parent (EXP-013)

- `model.smoothness_weight`: `1.0e-4` → `1.0e-3`
- All other hyperparameters, architecture, and dataset identical

## Method

Identical to EXP-013 except `smoothness_weight=1e-3`. SNR weighting (`×ᾱ_t`) is retained for training stability — verified stable via smoke test (30 random batches, max loss 1.01, no spikes).

## Results

| Metric | Value |
|--------|-------|
| Best epoch | 41 (early stop at 61, patience=20) |
| Best val loss (composite) | 0.00316 |
| KS statistic (shape-space) | 0.1530 |
| Circularity CV | 0.2195 |
| Smoothness (2nd-order diff) | 0.00865 |
| Boundary violation rate | 0.000 |

### Full series comparison

| Metric | EXP-011 (no reg) | EXP-013 (λ=1e-4) | EXP-013b (λ=1e-3) | Trend |
|--------|-----------------|------------------|-------------------|-------|
| Val loss | 0.00128 | 0.00212 | 0.00316 | ↑ monotone |
| Smoothness | 0.0061 | 0.0072 | 0.0087 | ↑ monotone ❌ |
| Circularity CV | 0.2253 | 0.2156 | 0.2195 | ↓ slightly |
| KS shape-space | 0.165 | 0.151 | 0.153 | similar |
| Boundary violations | 0.000 | 0.000 | 0.000 | — |

### Observations

- Smoothness metric worsens monotonically with increasing λ: 0.0061 → 0.0072 → 0.0087. Doubling λ worsened smoothness by +20%.
- The penalty is not just failing to help — it is actively making shapes rougher.
- Earlier convergence (epoch 41 vs 68 vs 98) suggests the composite loss landscape changes enough to land in a worse basin.
- The effective gradient from the smoothness term at meaningful timesteps is O(10,000×) smaller than the MSE gradient even at λ=1e-3 with SNR weighting, so it cannot provide directional guidance — it only adds noise to training.

## Conclusions

**Approach abandoned.** The SNR-weighted x̂₀ curvature penalty (`×ᾱ_t`) is fundamentally insufficient as a smoothness regulariser at any weight that keeps training stable:

- Small λ (1e-4): gradient contribution negligible, no effect.
- Large λ (1e-3): gradient contribution still negligible, but the composite loss landscape is distorted enough to drive the model to a worse optimum — producing *rougher* shapes.

The root cause: SNR weighting `ᾱ_t` is necessary to prevent loss spikes (the x̂₀ reconstruction has `1/√ᾱ_t` amplification at high t), but this same weighting kills the gradient signal at all timesteps. There is no λ in the stable range that produces a meaningful smoothness gradient.

**Recommended alternative**: Fourier low-pass post-processing — truncate the Fourier series of generated radii to the lowest K modes after sampling. This is architecture-free, training-free, and directly controllable.

## Next steps

- [x] Abandon SNR-weighted x̂₀ curvature penalty
- [ ] Implement Fourier low-pass filter as a post-processing step in `scripts/postprocess_ellipse.py`; add `fourier_modes` parameter to config
- [ ] Proceed to XFOIL aerodynamic evaluation pipeline (higher priority)
