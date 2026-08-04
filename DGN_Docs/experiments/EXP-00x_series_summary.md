---
experiment_id: "EXP-00x"
title: "EXP-00x Circle Radial Series — Summary & Recommendations"
date: 2026-05-07
status: complete
tags: [summary, circle, radial, ablation, series]
---

# EXP-00x Series Summary: Circle Radial Diffusion

> The EXP-00x series validates the `graph_diffusion` DDPM pipeline on a synthetic circle radial benchmark (Fourier-perturbed unit circles). Six experiments ran from April–May 2026, establishing a stable training recipe and characterising the limits of standard DDPM on this geometry.

## Experiments in this series

| ID | Name | Status | Key result |
|----|------|--------|------------|
| [[EXP-001_circle_radial_baseline]] | Circle baseline | ✅ | Pipeline validated; 3-epoch proof-of-concept |
| [[EXP-002_circle_radial_k-neighbors]] | k-neighbors ablation | ✅ | k=6 best: KS=0.0944 vs 0.2150 for k=1 |
| [[EXP-003_circle_radial_amplitude]] | Amplitude ablation | ✅ | amplitude=0.15 optimal; 0.30 needs longer training |
| [[EXP-004_circle_radial_logit]] | Logit-transform | ✅ | BVR=0.0 guaranteed, but KS=0.8929 — not adopted |
| [[EXP-005_circle_radial_full]] | Full 100-epoch training | ✅ | **Reference result:** val_loss=0.0303, KS=0.1049 |
| [[EXP-006_circle_radial_rich-features]] | Rich features [r,κ,s/L] | ✅ | Marginal shape gain at 3× higher loss |

---

## Reference result (EXP-005)

The best configuration found in this series:

| Hyperparameter / metric | Value |
|------------------------|-------|
| Architecture | 4-layer GraphNetworkBlock, node_dim=32, edge_dim=2, global_dim=8 |
| Diffusion schedule | Cosine, T=200 |
| Graph connectivity | k=6 ring neighbors |
| Node features | `[r]` (single radius) |
| Fourier amplitude | 0.15 |
| Training | 100 epochs, CosineAnnealingLR (1e-3→1e-5), early stopping patience=20 |
| Best epoch | 89 |
| **Best val loss** | **0.0303** |
| **KS statistic** | **0.1049** |
| **Smoothness** | **0.0092** |
| **Boundary violations** | **0.0000** |

Config: `configs/EXP-005_circle_radial_full.yaml`
Checkpoint: `outputs/EXP-005_circle_radial_full/checkpoint.pt`

---

## Key findings

### 1. Graph connectivity matters more than expected (EXP-002)

Wider ring connectivity dramatically improves distribution fidelity: k=6 achieves KS=0.0944 vs KS=0.2150 for k=1. The KS gap between k=4 and k=6 (0.160→0.094) suggests k=6 crosses a structural threshold in the ring topology. **Recommendation for aerodynamic meshes:** use higher-order connectivity (k≥4) where the mesh supports it.

### 2. Perturbation complexity must match model capacity (EXP-003)

amplitude=0.30 degrades generation quality significantly (KS=0.2356 vs 0.0653 for amplitude=0.15) at 50 epochs. This is likely a training-time issue, not an architectural one. **Implication for aerofoil data:** pressure fields with large dynamic range may need longer training or normalisation.

### 3. Principled bounded diffusion is currently too costly (EXP-004)

Logit-transform guarantees BVR=0.0 by construction but collapses distribution fidelity (KS=0.8929). Standard DDPM assumes Gaussian-distributed features; the logit-mapped distribution is bounded and non-Gaussian for peaked data like r∼N(1, 0.15). **Decision:** use clamp_range heuristic for now; revisit with normalising flows or score SDE.

### 4. Standard DDPM cannot enforce geometric coupling (EXP-006)

Adding curvature (κ) and arc-length fraction (s/L) as independent diffusion channels does not improve distribution fidelity (KS 0.1049→0.1298) despite marginal smoothness gains (−8%). DDPM treats all channels as independently noised; κ and s/L are deterministic functions of r, creating inconsistent training signals. **Implication for multi-field aerofoil data (u, v, p):** fields that are physically coupled under the Navier–Stokes equations will likely suffer the same issue. Conditional diffusion or joint score estimation will be needed.

---

## Recommended EXP-01x starting configuration

Based on these results, the EXP-010 baseline for pOnEllipse should start from:

- **Architecture:** EXP-005 (4-layer GN, node_dim=32, cosine schedule T=200)
- **Connectivity:** k=6 (or graph adjacency derived from mesh topology)
- **Feature space:** single pressure feature `[p]` first; multi-field (u,v,p) in later ablation
- **Training:** cosine LR, early stopping patience=20, 100 epochs
- **Boundary handling:** clamp_range on normalised pressure

---

## Open questions for EXP-01x

1. **Geometric consistency:** How should the node positions (2D coordinates on ellipse surface) be incorporated — as fixed positional encodings, additional node features, or edge features from mesh edges?
2. **Temporal dimension:** The pOnEllipse dataset includes time-varying pressure fields (multiple snapshots per geometry). Should the diffusion model treat snapshots as i.i.d. or incorporate time?
3. **Multi-field coupling:** For uvpAroundEllipse, how do we handle the physically coupled (u,v,p) fields without the channel-independence failure seen in EXP-006?
4. **Out-of-distribution generalisation:** Can a model trained on `pOnEllipseTrain.h5` generalise to `pOnEllipseLowRe.h5` / `pOnEllipseHighRe.h5`?

---

## Series closed

EXP-007 (NACA aerofoil geometry) has been superseded by the pOnEllipse pathway. The pOnEllipse dataset (HuggingFace `mariolinov/Ellipse`) provides labelled pressure data with Reynolds-number and geometry variants, making it a more tractable first step towards real aerodynamic generation.

Next: [[EXP-010_ellipse_radial_baseline]]
