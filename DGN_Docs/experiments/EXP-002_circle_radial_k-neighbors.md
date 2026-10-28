---
experiment_id: "EXP-002"
title: "k-neighbors ablation on unit circle radial diffusion"
date: 2026-05-07
status: complete
parent: "EXP-001"
tags: [ablation, circle, k-neighbors, radial]
config: "configs/EXP-002a_circle_radial_k1.yaml ... EXP-002d_circle_radial_k6.yaml"
output_dir: "outputs/EXP-002a_circle_radial_k1/ ... outputs/EXP-002d_circle_radial_k6/"
---

# EXP-002: k-neighbors ablation

> **One-line summary:** Vary ring connectivity k ∈ {1, 2, 4, 6} to determine whether wider neighbourhoods improve shape smoothness or generation quality.

## Motivation

[[EXP-001_circle_radial_baseline]] used k=2 (each node connects to its 2 nearest ring neighbours on each side). It is unclear whether this is the optimal choice. Wider connectivity may allow smoother gradient flow during the reverse diffusion and produce smoother shapes. Narrower connectivity is computationally cheaper.

## Hypothesis

- k=1 (minimal ring): noisier shapes, weaker spatial coherence
- k=2 (baseline): balanced — good quality without excess computation
- k=4: marginal improvement in smoothness, minimal cost increase
- k=6: diminishing returns; may over-smooth or slow training

## Changes from parent

- **Config diff:** `circle_dataset.k_neighbors` ∈ {1, 2, 4, 6}; all other params identical to EXP-001
- **Code changes:** none
- **Data changes:** unique `data_root` per variant to avoid cache collisions

## Variants

| Variant | Config | k | Output dir |
|---------|--------|---|------------|
| EXP-002a | `configs/EXP-002a_circle_radial_k1.yaml` | 1 | `outputs/EXP-002a_circle_radial_k1/` |
| EXP-002b | `configs/EXP-002b_circle_radial_k2.yaml` | 2 | `outputs/EXP-002b_circle_radial_k2/` |
| EXP-002c | `configs/EXP-002c_circle_radial_k4.yaml` | 4 | `outputs/EXP-002c_circle_radial_k4/` |
| EXP-002d | `configs/EXP-002d_circle_radial_k6.yaml` | 6 | `outputs/EXP-002d_circle_radial_k6/` |

## Method

### Architecture

4-layer GN, node_dim=32, edge_dim=2 (angular features), global_dim=8, T=200 cosine schedule. Input projection 1→32.

### Dataset

2000 ring graphs, 64 nodes, Fourier perturbation (5 modes, amplitude=0.15), r ∈ [0.5, 1.5]. Pre-transform: `ComputeAngularEdgeFeatures` (edge features = [sin(Δθ), cos(Δθ)]).

### Training

50 epochs, Adam lr=1e-3, batch=64, clamp_range=[0.5, 1.5].

## Results

### Metrics

| Variant | k | Final train loss | Final val loss | Smoothness | Circularity CV | Boundary violations | KS stat |
|---------|---|-----------------|----------------|------------|----------------|---------------------|---------|
| EXP-002a | 1 | 0.0529 | 0.0642 | 0.0135 | 0.0727 | 0.000 | 0.2150 |
| EXP-002b | 2 | 0.0465 | 0.0403 | **0.0082** | 0.0937 | 0.000 | 0.1930 |
| EXP-002c | 4 | 0.0479 | 0.0414 | 0.0141 | 0.1090 | 0.000 | 0.1602 |
| EXP-002d | 6 | **0.0443** | **0.0378** | 0.0153 | 0.1275 | 0.000 | **0.0944** |

### Observations

- **KS statistic falls monotonically** with k (0.2150 → 0.0944), confirming that wider graph connectivity strongly improves distribution fidelity. k=6 achieves the best match to the reference radius distribution.
- **Smoothness is best at k=2** (0.0082). Both narrower (k=1) and wider (k≥4) neighbourhoods produce slightly rougher shapes, suggesting k=2 is a local optimum for per-sample smoothness.
- **Circularity CV increases with k**: k=1 produces the most circular shapes (CV=0.073) while k=6 produces more varied shapes (CV=0.128). This may reflect k=6 capturing more genuine radial diversity from the training data rather than collapsing to near-circles.
- **Boundary violations = 0.0** across all variants; the clamp_range heuristic is effective enough for 50-epoch ablations.
- The gap between k=4 and k=6 in KS stat is large (0.1602 → 0.0944), suggesting k=6 crosses an important structural threshold in the ring graph.
- k=2 hypothesis (baseline is balanced) is **partially confirmed**: k=2 has best smoothness but k=6 dominates on distribution fidelity and val loss.

## Conclusions

**Chosen k for EXP-005: k=6.**

The KS statistic is the primary quality metric, as it directly measures how well the generative model reproduces the reference distribution shape. k=6 achieves 0.0944 vs 0.1602 for k=4 — a 41% improvement. The trade-off in smoothness (0.0153 vs 0.0082) is acceptable given the much better distribution fidelity. The consistently lower val loss (0.0378) further confirms k=6 gives a better-fitting model.

If smoothness were the primary concern (e.g. for aerodynamic applications), k=2 would be preferred.

## Next steps

- [x] Choose best k → update `EXP-005_circle_radial_full.yaml` (`circle_dataset.k_neighbors = 6`)
- [ ] Cross-reference with [[EXP-003_circle_radial_amplitude]] results before configuring [[EXP-005_circle_radial_full]]
