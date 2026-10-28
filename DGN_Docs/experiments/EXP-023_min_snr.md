---
experiment_id: "EXP-023"
title: "Min-SNR-γ loss weighting (γ=5)"
date: 2026-05-27
status: complete
parent: "EXP-022"
tags: [ellipse, aerodynamic, pressure-conditioning, loss-weighting, min-snr, surface-quality]
config: "configs/EXP-023_min_snr.yaml"
output_dir: "outputs/EXP-023_min_snr/"
---

# EXP-023: Min-SNR-γ loss weighting (γ=5)

> **One-line summary:** Replace the uniform DDPM MSE with the Min-SNR-γ loss of [Hang et al. 2023](https://arxiv.org/abs/2303.09556), reweighting per-step MSE by `min(SNR_t, γ) / SNR_t` (γ=5) so the optimiser spends more capacity on low-noise timesteps — the regime that determines boundary smoothness.

## Motivation

Uniform-MSE DDPM training treats every timestep equally, but the optimisation problem is *not* equally hard at every t. High-t (high-noise) loss is dominated by easy near-Gaussian targets; low-t (low-noise) loss requires the network to recover fine structural detail. With uniform weighting the easy gradients dominate. Min-SNR-γ clamps the relative weight at γ to prevent low-t exploding the loss while still up-weighting it relative to vanilla.

The relevant SNR range here: T=200 cosine schedule gives `SNR_t = ᾱ_t / (1 − ᾱ_t)` from ~10³ at t=0 (clean) down to ~10⁻³ at t=T-1 (pure noise). γ=5 means timesteps with SNR > 5 (the lower-noise half) get up-weighted, capped at a 5× boost. This is well-studied territory; γ=5 is Hang et al.'s recommended default and works robustly across image diffusion benchmarks.

## Hypothesis

- Boundary roughness drops by ≥30% versus EXP-022 (or versus whichever of EXP-021/022 ran first if both ran).
- Head-prediction MSE stays within 50% of EXP-020 — the head loss is unchanged, only the noise-prediction loss is reweighted.
- Convergence is faster (Min-SNR-γ is reported to reduce DDPM training time by ~3×).

## Changes from parent (EXP-022)

- **Config diff:** `model.min_snr_gamma: 5.0` added.
- **Code changes:** `GraphDiffusionModel.__init__` gains `min_snr_gamma` arg; `compute_loss` branches on it. Tests in [`tests/test_min_snr_loss.py`](../../tests/test_min_snr_loss.py).
- **Data changes:** none.

## Method

Identical to [[EXP-022_T1000]] except for the loss term. The Min-SNR weight is computed per-graph from `ᾱ_t` already on the `NoiseSchedule` buffer, broadcast back to per-node via `batch_vec`, and applied to the squared-error before mean-reduction.

### Reproduce

```bash
python train.py \
    --config configs/EXP-023_min_snr.yaml \
    --epochs 1000 \
    --device cuda \
    --output outputs/EXP-023_min_snr/generated_shapes.png

python scripts/postprocess_exp020.py \
    --experiment-dir outputs/EXP-023_min_snr \
    --config configs/EXP-023_min_snr.yaml \
    --device cuda \
    --n-samples 4 --n-targets 3 --target-seed 0
```

## Results

| Metric | EXP-020 | EXP-021 (T=200) | EXP-022 (T=1000) | EXP-023 (T=200 + Min-SNR) |
|---|---|---|---|---|
| Best val loss | 0.00306 | 0.000104 | 0.000090 | **0.000028** |
| Boundary roughness (overall) | 0.0338 | 0.02125 | **0.01523** | 0.02700 |
| Boundary roughness (train #1) | 0.0759 | 0.02629 | 0.00625 | 0.03666 |
| Epochs (best) | 59 | 986 | 988 | 972 |

### Observations

- **Lowest val loss of the entire series (0.000028), but worse roughness.** Min-SNR-γ=5 cut the training MSE to a third of EXP-021's, yet boundary roughness *rose* from 0.02125 to 0.02700 (+27% vs the like-for-like T=200 EXP-021 baseline). Cleanest evidence in the study that **MSE and boundary roughness are decoupled** — optimising the loss harder does not smooth shapes.
- Min-SNR up-weights high-noise (low-SNR) timesteps relative to vanilla MSE — the opposite of what helps fine geometric detail, which lives in the low-noise tail.
- Built on EXP-021's recipe (T=200), so the fair comparison is against EXP-021's 0.02125, not EXP-022's T=1000.

## Conclusions

- **Net-negative for surface quality; not adopted.** If revisited, compose with T=1000 — but EXP-022 alone already hits 0.015, so marginal value is low.

## Next steps

- [x] Recorded in [[EXP-02x_methodology_summary]]; EXP-022 remains the winner.
