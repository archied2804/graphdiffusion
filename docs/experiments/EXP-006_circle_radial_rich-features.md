---
experiment_id: "EXP-006"
title: "Richer node features: [r, κ, s/L] on unit circle radial diffusion"
date: 2026-05-07
status: complete
parent: "EXP-005, EXP-004"
tags: [rich-features, curvature, arc-length, circle, radial]
config: "configs/EXP-006_circle_radial_rich-features.yaml"
output_dir: "outputs/EXP-006_circle_radial_rich/"
---

# EXP-006: richer node features

> **One-line summary:** Extend the node feature vector from [r] to [r, κ, s/L] and test whether providing geometric context improves generation quality.

## Motivation

[[EXP-001_circle_radial_baseline]] through [[EXP-005_circle_radial_full]] use a single node feature (radius r). A richer representation could help the GN block learn geometry:

- **κ (curvature):** measures local bending. High κ → tight corner. Low κ → smooth arc. Provides local shape context beyond just radius.
- **s/L (normalised arc-length fraction):** tells each node its position along the curve (0 = start, 1 = end). Provides positional context around the ring.

These are deterministic functions of r given the topology, so the model must learn to generate consistent triplets.

## Hypothesis

- Providing κ and s/L as additional features enriches the score network's understanding of local geometry
- Expected improvement in smoothness and circularity CV vs EXP-005
- Training may be harder (3× feature dimensionality, implicit geometric constraints)

## Changes from parent

- **Config diff:** `score_network.input_dim: 3`; `circle_dataset.include_curvature: true`, `include_arc_length: true`; `data_root: data/circle_rich`
- **Code changes:**
  - `src/graph_diffusion/data/circledataset.py` — `include_curvature`, `include_arc_length` params; `_compute_curvature` (2D finite differences), `_compute_arc_length_fraction` (cumulative chord length)
  - `train_circle.py` — passes `include_curvature`, `include_arc_length` to `UnitCircleDataset`
  - `scripts/postprocess_circle.py` — passes same flags; evaluation uses `x[:, 0]` (r only)
- **Data changes:** new processed dataset at `data/circle_rich/` (shape `(N, 3)` vs `(N, 1)`)

## Method

### Architecture

Same 4-layer GN as baseline, except `ScoreNetwork` projects from `input_dim=3` → `node_dim=32` via `input_proj` (linear) and decodes back to 3 via `output_decode`.

### Feature computation

Curvature and arc-length computed numerically per graph during `UnitCircleDataset._build_graphs()`:

**Curvature (2D, periodic central differences):**
```
dx = roll(x,-1) - roll(x,1)     # ∝ 2Δθ x'
dy = roll(y,-1) - roll(y,1)
d2x = roll(x,-1) - 2x + roll(x,1)
d2y = roll(y,-1) - 2y + roll(y,1)
κ = 4 |dx·d2y - dy·d2x| / (dx² + dy²)^(3/2)
```

**Arc-length fraction:**
```
ds_i = ||(x_{i+1}, y_{i+1}) - (x_i, y_i)||  (periodic)
s_i / L = cumsum(ds)[i] / total_arc_length
```

### Dataset

2000 ring graphs, 64 nodes, k=6 (from EXP-005), amplitude=0.15. Node features: `x = [r, κ, s/L]`, shape (64, 3).

### Training

100 epochs, cosine LR (lr=1e-3 → 1e-5), early stopping (patience=20). Same as EXP-005.

### Evaluation

Metrics computed on `x[:, 0]` (generated radii) only — same metrics as all prior experiments for comparability.

## Results

### Metrics

| Metric | EXP-005 (baseline) | EXP-006 (rich) | Δ |
|--------|-------------------|----------------|---|
| Best epoch | 89 | 77 | −12 |
| Best val loss | 0.0303 | 0.0915 | +0.0612 ↑ |
| Final train loss | 0.0425 | 0.1075 | +0.0650 ↑ |
| Smoothness | 0.0092 | **0.0085** | −0.0007 ✓ |
| Circularity CV | 0.1137 | **0.1041** | −0.0096 ✓ |
| Boundary violations | 0.0000 | 0.0000 | — |
| KS statistic | **0.1049** | 0.1298 | +0.0249 ↑ |

### Observations

- **Val loss is 3× higher** than EXP-005 (0.0915 vs 0.0303). This is expected: the score network must now predict 3D noise (r, κ, s/L channels) instead of 1D, and the added geometric channels have implicit consistency constraints that are harder to learn under standard DDPM.
- **Shape quality improves marginally:** smoothness 0.0092→0.0085 (−8%), circularity CV 0.1137→0.1041 (−8%). Both improvements are consistent with the hypothesis that geometric context helps the model generate smoother, more circular shapes.
- **Distribution fidelity (KS) degrades:** 0.1049→0.1298 (+24%). Evaluated on the r column only, the rich-feature model is slightly worse at reproducing the reference radius distribution. The model may be spending capacity on κ and s/L consistency at the expense of r fidelity.
- **Best epoch at 77** vs 89 in EXP-005. Richer features cause the model to peak earlier, suggesting it gains fast geometric structure then stalls. The final val_loss (0.1012) is 11% higher than the best (0.0915), indicating mild overfitting in the last ≈20 epochs.
- **Boundary violations = 0.0** — clamp_range remains effective even in the 3D feature space (applied to r column only).

### Key question: geometric consistency

The generated κ and s/L columns have not been compared against κ and s/L recomputed from the generated r values. This check requires manual analysis of `generated_samples.pt`. If the generated triplets are geometrically inconsistent (κ_generated ≠ κ_recomputed_from_r), it would confirm that DDPM treats the three channels as independent rather than geometrically coupled — a known limitation of standard DDPM for structured multi-channel outputs.

## Conclusions

**Richer features deliver marginal shape quality improvements but at significant cost.** The 3× increase in val loss and +24% KS degradation mean EXP-006 is not a clear improvement over EXP-005 for the circle task as currently formulated.

The core problem: DDPM assumes independent Gaussian noise across all channels. The κ and s/L channels are deterministic functions of r, so injecting them as independent noised features creates an inconsistent training signal — the model must simultaneously denoise three channels that are geometrically coupled but treated as uncorrelated by the diffusion process.

**Recommendations for follow-on work:**
1. **Geometric consistency check:** Compute `κ_recomputed_from_r` for all generated samples and compare to `κ_generated`. Large discrepancy confirms the coupling issue.
2. **Conditional diffusion:** Diffuse r only; compute κ and s/L analytically from the generated r at each reverse step. This preserves geometric consistency by construction.
3. **Feature normalisation:** κ values are on a different scale from r and s/L. Standardising each channel before diffusion may improve training stability.
4. **Proceed to NACA:** For [[EXP-007_naca_radial_baseline]], use the EXP-005 recipe (single r feature, k=6, cosine LR) as the starting point.

## Next steps

- [ ] Geometric consistency check: compare κ_generated vs κ_recomputed from `generated_samples.pt`
- [ ] If richer features help: apply conditional diffusion approach to [[EXP-007_naca_radial_baseline]]
- [ ] Proceed to [[EXP-007_naca_radial_baseline]] using EXP-005 checkpoint/config as parent
