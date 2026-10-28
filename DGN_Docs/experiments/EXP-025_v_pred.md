---
experiment_id: "EXP-025"
title: "v-prediction parameterisation"
date: 2026-05-27
status: complete
parent: "EXP-024"
tags: [ellipse, aerodynamic, pressure-conditioning, v-prediction, salimans-ho, surface-quality]
config: "configs/EXP-025_v_pred.yaml"
output_dir: "outputs/EXP-025_v_pred/"
---

# EXP-025: v-prediction parameterisation

> **One-line summary:** Reparameterise the score network's training target from ε to v = √ᾱ_t · ε − √(1−ᾱ_t) · x_0 (Salimans & Ho 2022). v-prediction is empirically smoother at fine detail than ε-prediction and is the parameterisation of choice for fast DDIM-style sampling.

## Motivation

ε-prediction (the EXP-020 default) is the original DDPM formulation but is known to over-train on the easy high-noise regime: when t ≈ T, x_t ≈ ε, so the network can solve the target almost trivially. v-prediction rotates the target so it interpolates between predicting x_0 (at low t) and predicting ε (at high t), giving better-conditioned learning across timesteps. Salimans & Ho originally introduced this for progressive distillation; subsequent work (Karras et al. 2024, latent diffusion follow-ups) reports v-prediction usually wins on visual quality of fine detail — exactly the surface-roughness metric we care about.

The conversion is purely algebraic: the score network still outputs an N-d tensor, the loss target changes, and sampling adds a one-line conversion `ε = √(1−ᾱ_t) · x_t + √ᾱ_t · v_pred` before the DDPM reverse step. CFG, DPS, and the pressure-head loss all keep working unchanged.

This is the most code-invasive experiment in the series and runs **last** — only if EXP-021..024 collectively fail to hit the roughness target.

## Hypothesis

- Boundary roughness drops below the winner of EXP-021..024 by an additional 15–25%.
- Train #1's pathological 0.0759 case is fully cleaned up (target: roughness < 0.03 on that row).
- Head MSE on synthetic targets matches the parent — v-prediction doesn't disrupt the forward shape→Cp learning, since the pressure-head loss operates on `x̂₀` regardless of parameterisation.
- Optional follow-up: with v-prediction the door is open to ditching DDPM sampling for DDIM, which would make the interactive notebook ~10× faster.

## Changes from parent (EXP-024)

- **Config diff:** `model.prediction_type: v` added.
- **Code changes:** `GraphDiffusionModel.__init__` gains `prediction_type` arg; `compute_loss` computes the v-target when in v-mode and reconstructs `eps_pred` for downstream auxiliary losses; `sample` converts v_pred → eps_pred before the DDPM reverse step. Tests in [`tests/test_v_prediction.py`](../../tests/test_v_prediction.py) cover the round-trip identity, loss divergence vs ε, and end-to-end sampling.
- **Data changes:** none.

## Method

Identical training run to [[EXP-024_ema]] except the score network learns v instead of ε. The total loss is

```
L_total = MSE(v_pred, v_target)            (or Min-SNR-γ-weighted if that won earlier)
        + λ_pressure · L_head(x̂₀(v_pred), cond)
```

where `x̂₀ = √ᾱ_t · x_t − √(1−ᾱ_t) · v_pred` is the Tweedie-style clean-shape estimate from v-pred. Sampling rebuilds ε from v at every reverse step and otherwise uses the existing DDPM update.

### Reproduce

```bash
python train.py \
    --config configs/EXP-025_v_pred.yaml \
    --epochs 1000 \
    --device cuda \
    --output outputs/EXP-025_v_pred/generated_shapes.png

python scripts/postprocess_exp020.py \
    --experiment-dir outputs/EXP-025_v_pred \
    --config configs/EXP-025_v_pred.yaml \
    --device cuda \
    --n-samples 4 --n-targets 3 --target-seed 0 \
    --checkpoint outputs/EXP-025_v_pred/checkpoint_ema.pt
```

## Results

| Metric | EXP-020 | EXP-021 (T=200) | EXP-022 (T=1000, winner) | EXP-025 (T=200 + EMA + v-pred) |
|---|---|---|---|---|
| Best val loss | 0.00306 | 0.000104 | 0.000090 | 0.000189 |
| Boundary roughness (overall) | 0.0338 | 0.02125 | **0.01523** | 0.02451 |
| Boundary roughness (train #1) | 0.0759 | 0.02629 | 0.00625 | 0.03618 |
| Epochs (best) | 59 | 986 | 988 | 802 (early-stopped at 902) |

Evaluated on the EMA copy (`checkpoint_ema.pt`) since the config keeps EMA on.

### Observations

- **Roughness 0.02451 — better than EMA alone (EXP-024's 0.0375) but still above the T=200 baseline EXP-021 (0.02125).** v-prediction recovered most of the damage EMA did, but didn't get below the plain longer-training result.
- Highest val loss of the 021–025 group (0.000189) and the only run to early-stop (epoch 902). v-prediction changes the loss scale/landscape; with the same patience it converged to a shallower optimum here.
- The expected v-prediction benefits (smoother fine detail, DDIM compatibility) didn't show up at T=200 on this task. Bundled with EMA, it inherited EMA's penalty.

## Conclusions

- **Net-negative vs EXP-021; not adopted.** Confirms the series-wide finding: only T=1000 (EXP-022) beats plain longer training. The three modern tricks (Min-SNR, EMA, v-pred) are all the wrong tool for boundary roughness, which is set in the sampling stage.
- v-prediction would still be the right choice *if* a fast DDIM sampler were needed for the interactive notebook — but at ~7.5 s/4-shapes EXP-022 is already usable, so it's not pursued.

## Next steps

- [x] Series complete; see [[EXP-02x_methodology_summary]]. EXP-022 is the production checkpoint.
- [ ] Remaining roughness (~0.015, dominated by OOD targets) — future levers if needed: larger network, dataset augmentation, or a Fourier-shape-space boundary loss. Not currently warranted.
