---
experiment_id: "EXP-005"
title: "Full 100-epoch training with cosine LR schedule and early stopping"
date: 2026-05-07
status: complete
parent: "EXP-002, EXP-003"
tags: [full-training, cosine-annealing, early-stopping, circle, radial]
config: "configs/EXP-005_circle_radial_full.yaml"
output_dir: "outputs/EXP-005_circle_radial_full/"
---

# EXP-005: full training run

> **One-line summary:** Train to convergence (100 epochs, cosine LR, early stopping) using k=6 from EXP-002 and amplitude=0.15 from EXP-003.

## Motivation

[[EXP-001_circle_radial_baseline]] was a 3-epoch validation run and [[EXP-002]] / [[EXP-003]] used 50-epoch ablation runs. Neither establishes the true converged performance of the model. This experiment runs to convergence with a principled LR schedule to produce the reference result for the circle experiment.

## Hypothesis

- Loss converges meaningfully below the 50-epoch ablation checkpoints
- Cosine annealing prevents oscillation near the minimum
- Early stopping (patience=20) prevents overfitting without manual epoch tuning

## Changes from parent

Config updated after reviewing EXP-002/003 results:

- **k_neighbors = 6** (chosen from EXP-002: best val_loss=0.0378, KS=0.0944)
- **amplitude_scale = 0.15** (chosen from EXP-003: meaningful complexity, KS=0.0653)
- **Config diff:** `training: {scheduler: cosine_annealing, eta_min: 1e-5, early_stopping_patience: 20}`; epochs=100; root="data/circle_k6"
- **Code changes:**
  - `train_circle.py` — `CosineAnnealingLR` scheduler; early stopping with best-checkpoint saving

## Method

### Architecture

Same 4-layer GN architecture as EXP-001 baseline.

### Dataset

2000 ring graphs, 64 nodes, k=6, amplitude=0.15. Reuses cached data from EXP-002d (`data/circle_k6`).

### Training

Up to 100 epochs. CosineAnnealingLR from lr=1e-3 to eta_min=1e-5. Early stopping: halt if val_loss does not improve for 20 consecutive epochs; save best checkpoint.

## Results

### Metrics

| Metric | Value |
|--------|-------|
| Best epoch (early stopping) | 89 |
| Best val loss | **0.0303** |
| Final train loss | 0.0425 |
| Smoothness | **0.0092** |
| Circularity CV | 0.1137 |
| Boundary violations | **0.0000** |
| KS statistic | 0.1049 |

### Observations

- **Early stopping did not trigger** — best checkpoint is epoch 89, but val_loss at epoch 100 (0.0408) is still within patience window from epoch 89 (89 + 20 = 109 > 100). All 100 epochs ran; the best checkpoint at epoch 89 (val_loss=0.0303) was saved as `checkpoint.pt`.
- **Val loss improved significantly** vs 50-epoch ablation baseline: 0.0378 (EXP-002d, 50 epochs) → 0.0303 (EXP-005, 89 best epoch) — a 20% improvement. The cosine LR schedule enabled continued improvement in the second half of training.
- **KS stat = 0.1049** — slightly worse than the 50-epoch k=6 ablation (0.0944). Evaluated with 50 samples (vs 16 in ablations), so the sample size difference may account for some variance. The distribution fidelity is still strong.
- **Smoothness = 0.0092** matches EXP-002b (k=2) closely despite the wider k=6 connectivity — the additional training epochs appear to have recovered the smoothness advantage.
- **Boundary violations = 0.0** — consistent with all prior experiments; clamp_range is effective.
- The slight val_loss degradation after epoch 89 (0.0303 → 0.0408) suggests mild overfitting in the final epochs that cosine annealing could not fully prevent. Patience=20 is correct but the model would benefit from earlier stopping at epoch 89.

## Conclusions

**EXP-005 establishes the circle radial reference result:** k=6, amplitude=0.15, 89 best epochs, val_loss=0.0303, KS=0.1049, smoothness=0.0092, BVR=0.0.

Compared to the best 50-epoch ablation (EXP-002d, k=6): val_loss improved by 20%, smoothness improved from 0.0153 to 0.0092. The full training run confirms that k=6 + cosine LR + best-checkpoint saving is the right training recipe for this circle task.

This checkpoint is the parent for [[EXP-006_circle_radial_rich-features]].

## Next steps

- [x] Use this checkpoint as parent for [[EXP-006_circle_radial_rich-features]]
