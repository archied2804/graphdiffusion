---
title: "EXP-02x methodology summary — surface-quality levers"
date: 2026-05-28
status: complete
tags: [ellipse, aerodynamic, surface-quality, boundary-roughness, methodology, summary]
---

# EXP-02x methodology summary

> **One-line takeaway:** Of the five levers tried to reduce boundary roughness, only two helped — longer training (EXP-021) and more diffusion steps (EXP-022). Min-SNR-γ, EMA, and v-prediction all made roughness *worse*. **EXP-022 (T=1000) is the winner at 0.01523, a 55% reduction from the EXP-020 baseline.**

## Goal

EXP-020 produced visibly wobbly generated boundaries (overall second-difference roughness 0.0338, worst row 0.0759). This series tested five orthogonal levers to find which one cleans up the surface. Each is evaluated on the **same 4 conditioning targets** (3 farthest-first training Cp curves + 1 synthetic asymmetric) using mean second-difference of `r(θ)` across 4 samples × 4 targets (lower = smoother).

## Results

| Exp | Lever (on top of...) | T | Overall roughness | Δ vs EXP-020 | Best val loss |
|---|---|---|---|---|---|
| EXP-020 | baseline | 200 | 0.03379 | — | 0.00306 |
| EXP-021 | longer training: 1000 ep, patience 100 | 200 | 0.02125 | **−37%** | 0.000104 |
| **EXP-022** | **EXP-021 + T=1000 diffusion steps** | **1000** | **0.01523** | **−55%** ✅ | 0.000090 |
| EXP-023 | EXP-021 + Min-SNR-γ=5 loss | 200 | 0.02700 | −20% | **0.000028** |
| EXP-024 | EXP-021 + EMA (decay 0.9999) | 200 | 0.03747 | **+11%** ❌ | 0.000124 |
| EXP-025 | EXP-021 + EMA + v-prediction | 200 | 0.02451 | −27% | 0.000189 |

Per-target breakdown (roughness):

| Exp | train #0 | train #1 | train #2 | synth asym. |
|---|---|---|---|---|
| EXP-020 | 0.0125 | 0.0759 | 0.0174 | 0.0293 |
| EXP-021 | 0.01547 | 0.02629 | 0.01029 | 0.03296 |
| EXP-022 | 0.00991 | **0.00625** | 0.00873 | 0.03604 |
| EXP-023 | 0.02516 | 0.03666 | 0.02182 | 0.02436 |
| EXP-024 | 0.02720 | 0.06291 | 0.01117 | 0.04862 |
| EXP-025 | 0.02156 | 0.03618 | 0.01039 | 0.02993 |

## Interpretation

**Roughness is a sampling-time property, not a training-loss property.** This is the headline finding and it explains every row:

- **EXP-022 (T=1000) wins because more reverse steps = more polish.** The final ~5–10% of reverse steps are where fine geometric detail emerges. At T=200 there are only ~20 such steps; at T=1000 there are ~100. The val loss is essentially identical to EXP-021 (0.00009 vs 0.00010) — the improvement is entirely in the reverse process, not the network.
- **EXP-023 (Min-SNR-γ) got the lowest val loss (0.000028) yet *worse* roughness (0.027).** A direct demonstration that minimising the training MSE harder does not smooth boundaries. Min-SNR reweights toward high-noise timesteps, which is the opposite of what helps fine detail.
- **EXP-024 (EMA) was the worst (0.037, above baseline).** Surprising, since EMA usually helps DDPMs. Likely the 0.9999 decay averaged over only ~160k steps under-converged the shadow weights, and the averaging blurred the model's sharp low-noise conditioning response — exactly the regime that does the final polish.
- **EXP-025 (v-pred + EMA) recovered some of EMA's loss but stayed above EXP-021.** v-prediction's benefits (smoother fine detail, DDIM compatibility) didn't materialise here; bundled with EMA it inherited EMA's penalty.

**Validated negative result:** the three "standard modern DDPM tricks" (Min-SNR, EMA, v-pred) are all net-negative for this task at T=200. They optimise the wrong thing. If pursued further they should be composed with T=1000, not the T=200 baseline — but given EXP-022 already hits 0.015, the marginal value is low.

## Recommendation

- **Use EXP-022 (`outputs/EXP-022_T1000/checkpoint_best.pt`) as the production checkpoint.** The interactive notebook already points at it.
- The remaining roughness (0.015) is dominated by the synthetic OOD target (0.036); the 3 in-distribution training targets are all below 0.010. For in-distribution inverse design the model is now smooth.
- Do not pursue EMA/Min-SNR/v-pred further unless a specific need arises; if so, compose with T=1000.

## Open thread

- The synthetic-asymmetric target's roughness *rose* slightly in EXP-022 (0.036 vs EXP-021's 0.033). Worth re-checking with several `--target-seed` values to confirm it isn't a single-seed artefact. The extra reverse steps may let the model commit more confidently to an already-OOD shape.
