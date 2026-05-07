---
experiment_id: "EXP-003"
title: "amplitude-scale ablation on unit circle radial diffusion"
date: 2026-05-07
status: complete
parent: "EXP-001"
tags: [ablation, circle, amplitude, radial]
config: "configs/EXP-003a_circle_radial_amp005.yaml ... EXP-003c_circle_radial_amp030.yaml"
output_dir: "outputs/EXP-003a_circle_radial_amp005/ ... outputs/EXP-003c_circle_radial_amp030/"
---

# EXP-003: amplitude-scale ablation

> **One-line summary:** Vary Fourier perturbation magnitude amplitude_scale ∈ {0.05, 0.15, 0.30} to map how shape complexity affects model learnability.

## Motivation

[[EXP-001_circle_radial_baseline]] used amplitude_scale=0.15, which produces shapes with ≈15% radial variation per Fourier mode. Lower amplitudes produce near-circles (easier to learn); higher amplitudes produce more irregular shapes (harder). Understanding this sensitivity informs the difficulty of the eventual NACA aerofoil task.

## Hypothesis

- amplitude=0.05: near-perfect circles — model should converge quickly, low KS stat, low smoothness cost
- amplitude=0.15: moderate perturbation — balanced learnability (baseline)
- amplitude=0.30: strong perturbation — harder to learn; expect higher loss and possibly higher boundary violations

## Changes from parent

- **Config diff:** `circle_dataset.amplitude_scale` ∈ {0.05, 0.15, 0.30}; all other params identical to EXP-001
- **Code changes:** none
- **Data changes:** unique `data_root` per variant; EXP-003b reuses `data/circle` (same as baseline)

## Variants

| Variant | Config | amplitude_scale | Output dir |
|---------|--------|----------------|------------|
| EXP-003a | `configs/EXP-003a_circle_radial_amp005.yaml` | 0.05 | `outputs/EXP-003a_circle_radial_amp005/` |
| EXP-003b | `configs/EXP-003b_circle_radial_amp015.yaml` | 0.15 | `outputs/EXP-003b_circle_radial_amp015/` |
| EXP-003c | `configs/EXP-003c_circle_radial_amp030.yaml` | 0.30 | `outputs/EXP-003c_circle_radial_amp030/` |

## Method

### Architecture

4-layer GN, node_dim=32, edge_dim=2, global_dim=8, T=200 cosine schedule. Input projection 1→32. Identical to [[EXP-001_circle_radial_baseline]].

### Dataset

2000 ring graphs, 64 nodes, k_neighbors=2. Radii: r(θ) = 1 + Σ Fourier modes with σ = amplitude_scale / n. r ∈ [0.5, 1.5] clamped.

### Training

50 epochs, Adam lr=1e-3, batch=64, clamp_range=[0.5, 1.5].

## Results

### Metrics

| Variant | amplitude | Final train loss | Final val loss | Smoothness | Circularity CV | Boundary violations | KS stat |
|---------|-----------|-----------------|----------------|------------|----------------|---------------------|---------|
| EXP-003a | 0.05 | **0.0220** | **0.0189** | 0.0142 | **0.0507** | 0.000 | **0.0452** |
| EXP-003b | 0.15 | 0.0506 | 0.0605 | **0.0098** | 0.1330 | 0.000 | 0.0653 |
| EXP-003c | 0.30 | 0.0708 | 0.0677 | 0.0223 | 0.1223 | 0.000 | 0.2356 |

### Observations

- **Hypothesis confirmed:** amplitude=0.05 is trivially easy — KS=0.0452 and val_loss=0.0189 after 50 epochs. The model learns the near-circular distribution almost perfectly.
- **amplitude=0.30 is significantly harder** — KS=0.2356 suggests the 50-epoch model fails to capture the full radial variance of high-amplitude shapes. Val loss (0.0677) is 3.6× higher than amp=0.05.
- **amplitude=0.15 achieves the best balance** for a meaningful learning challenge: KS=0.0653 (slightly worse than 0.05 but far better than 0.30), best smoothness (0.0098), and reasonable val loss (0.0605).
- Boundary violations = 0.0 for all variants — amplitude alone does not drive boundary violations at 50 epochs.
- The large gap between amp=0.15 and amp=0.30 in KS (0.0653 → 0.2356) mirrors the jump between k=4 and k=6 in EXP-002 and confirms that shape complexity scales non-linearly with amplitude.

## Conclusions

**Chosen amplitude for EXP-005: amplitude_scale=0.15.**

While amp=0.05 achieves the best raw metrics, it represents a task that is too easy to be informative — shapes are nearly perfect circles and the model's performance is ceiling-capped. amplitude=0.15 provides a meaningful generative challenge (KS~0.065, diverse radial variation) that scales to the intended future NACA aerofoil task. amplitude=0.30 degrades generation quality significantly in 50 epochs, suggesting it may require longer training or architectural changes beyond the scope of this ablation.

## Next steps

- [x] Choose best amplitude → update `EXP-005_circle_radial_full.yaml` (`circle_dataset.amplitude_scale = 0.15`)
- [ ] Cross-reference with [[EXP-002_circle_radial_k-neighbors]] before configuring [[EXP-005_circle_radial_full]]
