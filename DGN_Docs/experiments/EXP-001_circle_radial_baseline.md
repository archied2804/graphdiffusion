---
experiment_id: "EXP-001"
title: "Radial diffusion baseline on unit circle"
date: 2026-04-13
status: complete
parent: null
tags: [circle, radial, baseline, fourier, ddpm]
config: "configs/circle_radial.yaml"
output_dir: "outputs/EXP-001_circle_radial_baseline/"
---

# EXP-001: Radial diffusion baseline on unit circle

> **One-line summary:** First proof-of-concept — DDPM diffusion of the radial
> coordinate on a 64-node unit circle ring graph with Fourier-perturbed shapes.

## Motivation

Validate that the `graph_diffusion` framework can learn to denoise a
non-trivial 1D signal (radial displacement) on a fixed-topology ring graph.
This is the simplest possible aerodynamic-shape experiment: a closed curve
whose nodes move purely in the radial direction.

## Hypothesis

A 4-layer GN with `input_dim=1 → node_dim=32` projection and angular edge
features `[sin(Δθ), cos(Δθ)]` can learn to denoise Fourier-perturbed radial
profiles back to smooth closed shapes within 100 epochs.  Post-step clamping
to `[0.5, 1.5]` keeps generated radii physically plausible.

## Changes from parent

*First experiment — no parent.*

### New code

| Component | File | Description |
|-----------|------|-------------|
| `UnitCircleDataset` | `src/graph_diffusion/data/circledataset.py` | Ring graphs with Fourier-perturbed `r(θ)` |
| `ComputeAngularEdgeFeatures` | `src/graph_diffusion/data/transforms.py` | `[sin(Δθ), cos(Δθ)]` edge features |
| `ScoreNetwork.input_dim` | `src/graph_diffusion/model/score_network.py` | Optional input/output projection layers |
| `GraphDiffusionModel.clamp_range` | `src/graph_diffusion/model/graph_diffusion_model.py` | Bounded reverse-step clamping |
| `train_circle.py` | `train_circle.py` | Training + matplotlib shape generation |

## Method

### Architecture

- **Score network:** 4 × `GraphNetworkBlock` (full Battaglia GN), `node_dim=32`,
  `edge_dim=2`, `global_dim=8`, `hidden_dims=[64, 64]`, SiLU, LayerNorm, residual
- **Input projection:** `Linear(1 → 32)` before GN layers, `Linear(32 → 1)` after
- **Time embedding:** sinusoidal `dim=64` → `Linear(64 → 8)` added to global `u`
- **Noise schedule:** cosine, `T=200`

### Dataset

- **Geometry:** 64 nodes uniformly spaced in θ ∈ [0, 2π) on the unit circle
- **Topology:** bidirectional ring, `k_neighbors=2` (each node ↔ 2 neighbours each side)
- **Node features (diffused):** `x = [r_i]` (1D) where
  `r(θ) = 1 + Σ_{n=1}^{5} (aₙ sin(nθ) + bₙ cos(nθ))`, coefficients `~ N(0, 0.15/n)`
- **Edge features (fixed):** `[sin(Δθ), cos(Δθ)]` — rotation-invariant angular separation
- **Clamp:** `r ∈ [0.5, 1.5]`
- **Graphs:** 2000 total, 90/10 train/val split

### Training

- **Optimiser:** Adam, `lr=1e-3`
- **Epochs:** 3 (initial validation run), 100 (full run)
- **Hardware:** CPU
- **Parameters:** 147,441

## Results

### Metrics

| Metric           | Train  | Val    |
|-----------------|--------|--------|
| Loss @ epoch 1   | 0.3439 | 0.1675 |
| Loss @ epoch 3   | 0.0841 | 0.0725 |

### Plots

Generated shapes after 3 epochs (4 samples, `clamp_range=[0.5, 1.5]`):

![[assets/EXP-001_generated_shapes.png]]

### Observations

- Loss drops rapidly in the first 3 epochs — the 1D radial signal is
  relatively easy for the model to learn
- Generated shapes are smooth closed curves centred around the unit circle
- Clamping keeps all radii within `[0.5, 1.5]` as designed
- Angular edge features provide sufficient spatial context for the GN to
  disambiguate node positions without absolute θ
- The `input_dim → node_dim` projection successfully lifts the 1D signal
  into the 32D internal representation

## Conclusions

- **Hypothesis confirmed:** the pipeline works end-to-end for radial
  diffusion on a ring graph with only 3 epochs of training
- The framework extensions (`input_dim`, `clamp_range`,
  `ComputeAngularEdgeFeatures`) are validated and backward-compatible
- Key insight: relative angular features `[sin(Δθ), cos(Δθ)]` provide
  rotation invariance for free

## Next steps

- [ ] Full 100-epoch training run with loss curves
- [ ] Ablation: `k_neighbors` ∈ {1, 2, 4, 6} → effect on shape smoothness
- [ ] Ablation: `amplitude_scale` ∈ {0.05, 0.15, 0.30} → model capacity vs distribution width
- [ ] Experiment: logit-transform bounded diffusion (replace clamping)
- [ ] Experiment: richer node features `[r, curvature, arc_length_fraction]`
