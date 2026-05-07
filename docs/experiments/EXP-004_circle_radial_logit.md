---
experiment_id: "EXP-004"
title: "Logit-transform bounded diffusion on unit circle"
date: 2026-05-07
status: complete
parent: "EXP-001"
tags: [bounded-diffusion, logit-transform, circle, radial]
config: "configs/EXP-004_circle_radial_logit.yaml"
output_dir: "outputs/EXP-004_circle_radial_logit/"
---

# EXP-004: logit-transform bounded diffusion

> **One-line summary:** Replace post-hoc clamping with a logit/sigmoid transform so the model diffuses in an unconstrained space and boundary satisfaction is guaranteed by construction.

## Motivation

[[EXP-001_circle_radial_baseline]] uses `clamp_range=[0.5, 1.5]` applied after each reverse diffusion step. This is a heuristic that:
1. Breaks the DDPM reverse process (modifying x_t mid-trajectory is not theoretically grounded)
2. Can cause artefacts near the boundary — clamped samples are biased towards r_min/r_max

A principled alternative: apply a bijection `logit((r - r_min)/(r_max - r_min))` before diffusion (mapping [0.5, 1.5] → ℝ) and `sigmoid` after sampling (recovering [0.5, 1.5]). The model operates in an unconstrained Gaussian space throughout.

## Hypothesis

- Boundary violation rate = 0.0 (guaranteed by transform, not heuristic)
- Potentially better distribution fidelity (KS stat) and smoother shapes
- Loss convergence may differ slightly due to the change in feature scale

## Changes from parent

- **Config diff:** `feature_transform: {type: logit_norm, r_min: 0.5, r_max: 1.5}`; `clamp_range` removed
- **Code changes:**
  - `src/graph_diffusion/building_blocks/feature_transforms.py` — `FeatureTransform` ABC + `LogitNormTransform`
  - `src/graph_diffusion/model/graph_diffusion_model.py` — `feature_transform` arg; applied in `compute_loss` and `sample`
  - `train_circle.py` — wires `feature_transform` from config into `GraphDiffusionModel`
  - `scripts/postprocess_circle.py` — `build_model` reconstructs `LogitNormTransform` from config
- **Data changes:** same dataset as EXP-001 (reuses `data/circle`)

## Method

### Architecture

Identical to EXP-001 except: node features pass through `LogitNormTransform.forward` before entering the diffusion process and `LogitNormTransform.inverse` (sigmoid) is applied after sampling.

Transform: `z = logit((r - 0.5) / 1.0)` → diffuse in z-space → `r = 0.5 + sigmoid(z) * 1.0`

### Dataset

Same as [[EXP-001_circle_radial_baseline]]: 2000 ring graphs, 64 nodes, k=2, amplitude=0.15.

### Training

50 epochs, Adam lr=1e-3, batch=64. No clamp_range.

## Results

### Metrics

| Metric | Value |
|--------|-------|
| Final train loss | 0.1196 |
| Final val loss | 0.1197 |
| Smoothness | 0.0275 |
| Circularity CV | 0.2636 |
| **Boundary violations** | **0.0000** |
| KS statistic | 0.8929 |

### Observations

- **Boundary violations = 0.0** ✓ — the sigmoid inverse guarantees r ∈ [0.5, 1.5] by construction. This hypothesis is confirmed.
- **KS statistic = 0.8929** — near-maximum divergence. The generated radii distribution is qualitatively different from the reference. This is a critical failure for distribution fidelity.
- **Val loss (0.1197) is 2.6× higher** than the k=2 baseline (0.0403 from EXP-002b). The logit transform significantly changes the learning dynamics.
- **Smoothness (0.0275) and circularity CV (0.2636)** are both worse than the clamped baseline, suggesting the model generates noisier, less circular shapes.
- The DDPM score network must now learn to denoise in the logit-transformed space where the noise distribution is no longer a simple Gaussian — the bounded nature of the original data creates a non-standard transformed distribution that may require architectural changes or much longer training.

## Conclusions

**EXP-004 result: logit-transform does not improve generation quality at 50 epochs.** While it achieves the desired boundary guarantee (BVR=0.0), the distribution fidelity (KS=0.8929) is dramatically worse than clamped diffusion. The underlying issue is that r ∈ [0.5, 1.5] maps to a bounded distribution in logit-space (since r is not uniformly distributed in [0.5, 1.5] — it peaks around r=1.0), creating a non-Gaussian target that standard DDPM assumes is Gaussian.

**Recommendation:** Do not use logit-transform for EXP-005/006. The clamp_range heuristic, while theoretically impure, achieves boundary violation rates of 0.0 in practice (evidenced by EXP-002/003) and much better distribution fidelity. Logit-transform could be revisited with:
- A normalising flow or score SDE formulation designed for bounded domains
- Longer training (200+ epochs) to allow convergence in the harder transformed space
- A different transform that better preserves the Gaussian structure (e.g. Box-Cox)

## Next steps

- [x] Compare boundary_violation_rate with EXP-001 (dropped to 0, as expected)
- [ ] ~~If successful, consider adopting logit-transform in [[EXP-006_circle_radial_rich-features]]~~ — Not recommended given KS=0.8929 failure. EXP-006 will use clamp_range.
