---
experiment_id: "EXP-005"
title: "Full 100-epoch training with cosine LR schedule and early stopping"
date: 2026-05-07
status: running
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

> **Status: running** — fill in after training completes.

### Metrics

| Metric | Value |
|--------|-------|
| Best epoch (early stopping) | |
| Best val loss | |
| Final train loss | |
| Smoothness | |
| Circularity CV | |
| Boundary violations | |
| KS statistic | |

## Conclusions

*Fill in after results are available.*

## Next steps

- [ ] Use this checkpoint as parent for [[EXP-006_circle_radial_rich-features]]
