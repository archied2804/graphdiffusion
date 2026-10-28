---
experiment_id: "EXP-021"
title: "Longer training (1000 epochs, patience=100)"
date: 2026-05-27
status: complete
parent: "EXP-020"
tags: [ellipse, aerodynamic, pressure-conditioning, fourier-dct, classifier-free-guidance, surface-quality, under-training]
config: "configs/EXP-021_long_train.yaml"
output_dir: "outputs/EXP-021_long_train/"
---

# EXP-021: Longer training (1000 epochs, patience=100)

> **One-line summary:** Repeat EXP-020 verbatim except let it train until convergence (1000 epoch ceiling, patience=100) to isolate whether under-training is the cause of the visible surface imperfections in `figure_a_conditioning_grid.png` row 1.

## Motivation

EXP-020 produced visibly wobbly boundaries for **train #1** (overall roughness 0.0338, train #1 alone 0.0759 by the [boundary-roughness metric](../../src/graph_diffusion/postprocessing/metrics.py)). Training early-stopped at epoch 79 with `patience=20`, best val_loss=0.00306 at epoch 59. Before reaching for more invasive levers (T=1000, Min-SNR-γ, EMA, v-prediction), the cheapest control variable to vary is wall-clock training time — if the score network simply hadn't seen enough of the low-noise regime, more epochs should clean up the boundary.

[[EXP-013_ellipse_smoothness_reg]] and [[EXP-013b_ellipse_smoothness_reg_strong]] previously established that a smoothness regulariser on `x̂₀` makes shapes *worse*, not better, so that lever is off the menu and the EXP-021..025 series uses different controls instead.

## Hypothesis

- Overall boundary roughness drops from EXP-020's 0.0338 toward 0.024 or below (the ≥30% reduction we set as the stopping criterion for the EXP-021..025 series).
- Train #1 in particular — currently 0.0759 — sees the largest reduction.
- Head-prediction MSE stays within 50% of EXP-020's value (we don't want better-looking shapes that no longer match the target Cp).
- Val loss improves modestly vs EXP-020's 0.00306 but is not expected to halve — the network is small (32-dim nodes, 4 layers) so capacity caps how much extra epochs can help.

## Changes from parent (EXP-020)

- **Config diff:** `training.early_stopping_patience` 20 → 100; training run with `--epochs 1000` instead of 200.
- **Code changes:** none.
- **Data changes:** none.

## Method

### Architecture

Identical to EXP-020 — same `ScoreNetwork`, same `PressurePredictionHead`, same `NoiseSchedule(T=200, cosine)`, same `p_uncond=0.15`, same `λ_pressure=0.1`. See [`configs/EXP-021_long_train.yaml`](../../configs/EXP-021_long_train.yaml).

### Dataset

Identical to EXP-020: `pOnEllipseConditionalDataset(cond_mode="fourier", k_modes=8)` on `pOnEllipseTrain.h5`, with `Data.cond ∈ ℝ^{1×8}` attached per graph.

### Training

- Up to 1000 epochs, `patience=100` (5× EXP-020's budget).
- Optimiser, LR, scheduler unchanged.
- Sampler at evaluation time: DDPM, CFG `w=3.0`.

### Reproduce

```bash
python train.py \
    --config configs/EXP-021_long_train.yaml \
    --epochs 1000 \
    --device cuda \
    --output outputs/EXP-021_long_train/generated_shapes.png

python scripts/postprocess_exp020.py \
    --experiment-dir outputs/EXP-021_long_train \
    --config configs/EXP-021_long_train.yaml \
    --device cuda \
    --n-samples 4 --n-targets 3 --target-seed 0
```

## Results

| Metric | EXP-020 (baseline) | EXP-021 | Δ vs EXP-020 |
|---|---|---|---|
| Best val loss | 0.00306 (ep 59) | **0.00010** (ep 986) | **−97%** |
| Boundary roughness (overall) | 0.0338 | **0.02125** | **−37%** ✅ |
| Boundary roughness (train #0) | 0.0125 | 0.0155 | +24% |
| Boundary roughness (train #1, worst row) | **0.0759** | **0.0263** | **−65%** |
| Boundary roughness (train #2) | 0.0174 | 0.0103 | −41% |
| Boundary roughness (synth asym.) | 0.0293 | 0.0330 | +12% |
| Epochs run | 79 (early-stop, patience=20) | 1000 (full budget, patience=100) | — |

Best checkpoint at epoch 986 — val loss was still slowly improving when the run hit the 1000-epoch ceiling, suggesting the network had real headroom that EXP-020's `patience=20` cut short.

### Plots

`outputs/EXP-021_long_train/figure_a_conditioning_grid.png` shows the headline improvement: the train #1 row that was visibly jagged in EXP-020 is now a clean ellipse, and the head-predicted Cp curve tracks the target's absolute magnitude (EXP-020's reported ~0.1 magnitude bias is essentially gone).

### Observations

- **Stop criterion hit.** Overall roughness 0.02125 < target 0.0237 (≥30% reduction); EXP-022..025 in the series are not run.
- **Train #1 was under-trained, not architecturally limited.** The worst row in EXP-020 (0.0759) drops by 65% to 0.0263. The wobble was an artefact of stopping training too early, not a capacity ceiling.
- **Mild regression on already-clean rows.** Train #0 and synth asymmetric got slightly rougher (+24%, +12%). The overall mean still improves because the dominant contributor (train #1) collapsed. Worth verifying with a different `target-seed` that the easy-target regression isn't seed-specific before declaring this fully won.
- **Val loss kept falling for 900+ epochs.** The 30× val-loss reduction shows the original early-stopping was firing on noise, not on convergence.
- **Implication for the notebook.** Swap [`notebooks/EXP-020_interactive_cp.ipynb`](../../notebooks/EXP-020_interactive_cp.ipynb) cell 2 to point at `outputs/EXP-021_long_train/`.

## Conclusions

- Hypothesis confirmed: under-training was the dominant cause of EXP-020's boundary imperfections.
- The cheapest control variable in the EXP-021..025 menu was sufficient.
- EXP-022 (T=1000), EXP-023 (Min-SNR-γ), EXP-024 (EMA), and EXP-025 (v-pred) remain as code-complete-and-ready experiments for future work that needs to push roughness further or chase the train #0 / synth-asym mild regressions.

## Next steps

- [x] Point the interactive notebook at this checkpoint.
- [ ] Re-run postprocess with `--target-seed 1` and `--target-seed 2` to confirm the train #0 regression isn't seed-specific.
- [ ] If a further roughness reduction is wanted later (e.g. for aerofoil OOD targets), start EXP-024 (EMA) on top of this 1000-epoch baseline — it's the cheapest compounding lever.
