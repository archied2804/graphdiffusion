---
experiment_id: "EXP-024"
title: "EMA weights (decay=0.9999)"
date: 2026-05-27
status: complete
parent: "EXP-023"
tags: [ellipse, aerodynamic, pressure-conditioning, ema, swa, surface-quality]
config: "configs/EXP-024_ema.yaml"
output_dir: "outputs/EXP-024_ema/"
---

# EXP-024: EMA weights (decay=0.9999)

> **One-line summary:** Maintain an exponential moving average of the model weights during training (decay = 0.9999) and evaluate using the EMA copy. This is the canonical DDPM quality booster — virtually free at training time, well-documented to produce smoother and more consistent samples.

## Motivation

DDPM training loss landscapes are noisy: SGD bounces around a basin in late training rather than converging to a single point. EMA weights average those late-training oscillations, giving a more stable representation at inference. The technique is universal in image-diffusion work (Ho et al. 2020 already used it; Karras et al. 2024 study its dynamics in detail) and routinely accounts for a chunk of the visible quality difference between published samples and bare-bones reimplementations.

EXP-024 is the second-cheapest experiment in this series after EXP-021. Implementation cost: one `AveragedModel` wrapper + one `update_parameters` call per optimiser step + one extra checkpoint file. No new hyperparameters beyond the decay.

The EMA lever **composes** with the EXP-021..023 levers. If Min-SNR-γ wins, run EXP-024 by setting `model.min_snr_gamma: 5.0` in this config too. If T=1000 wins, copy that schedule in. The config currently shipped is a clean EMA-on-EXP-020-baseline starting point; tune it based on the winner of B.1–B.3.

## Hypothesis

- Boundary roughness drops below the winner of EXP-021..023 by an additional 10–20%.
- The reduction is most pronounced on the worst-roughness target (currently train #1 in EXP-020 at 0.0759).
- Val loss on the EMA checkpoint is similar to or slightly better than the raw checkpoint.
- Sample-to-sample variance (visible as the spread across the 4 samples per target in figure_a) drops noticeably.

## Changes from parent (EXP-023)

- **Config diff:** `training.ema_decay: 0.9999` added.
- **Code changes:** new [`src/graph_diffusion/model/ema.py`](../../src/graph_diffusion/model/ema.py) with `build_ema` and `save_ema_state_dict`; `train.py` builds an EMA copy when `ema_decay` is set, updates it every optimiser step, and saves a parallel `checkpoint_ema.pt` alongside `checkpoint_best.pt`. Tests in [`tests/test_ema.py`](../../tests/test_ema.py).
- **Data changes:** none.

## Method

Identical training run to whichever parent we choose, plus EMA tracking. Evaluation uses the EMA checkpoint via `--use-ema`-style loading; in this codebase that means passing `checkpoint_name="checkpoint_ema.pt"` to `load_exp020`.

### Reproduce

```bash
python train.py \
    --config configs/EXP-024_ema.yaml \
    --epochs 1000 \
    --device cuda \
    --output outputs/EXP-024_ema/generated_shapes.png

# Evaluate the EMA copy specifically.
python scripts/postprocess_exp020.py \
    --experiment-dir outputs/EXP-024_ema \
    --config configs/EXP-024_ema.yaml \
    --device cuda \
    --n-samples 4 --n-targets 3 --target-seed 0 \
    --checkpoint outputs/EXP-024_ema/checkpoint_ema.pt
```

## Results

| Metric | EXP-020 | EXP-021 (T=200) | EXP-024 (T=200 + EMA) |
|---|---|---|---|
| Best val loss | 0.00306 | 0.000104 | 0.000124 |
| Boundary roughness (overall) | 0.0338 | 0.02125 | **0.03747** |
| Boundary roughness (train #1) | 0.0759 | 0.02629 | 0.06291 |
| Epochs (best) | 59 | 986 | 986 |

Evaluated on the EMA copy (`checkpoint_ema.pt`).

### Observations

- **Worst result in the series — roughness 0.03747, above even the EXP-020 baseline (0.0338).** EMA with decay 0.9999 made shapes *rougher*, the opposite of the usual DDPM expectation.
- Likely cause: at decay 0.9999 the EMA shadow weights have a ~10k-step time constant; over ~160k steps they track but lag the live weights, and the averaging blurs the sharp low-noise conditioning response that does the final boundary polish. The val loss also regressed slightly (0.000124 vs 0.000104), consistent with the shadow weights being a slightly-stale average rather than a better optimum.
- Built on EXP-021's recipe (T=200); fair comparison is against EXP-021's 0.02125.

## Conclusions

- **Net-negative; EMA not adopted.** A lower decay (e.g. 0.999) or warm-up might behave differently, but given EXP-022 (T=1000) already hits 0.015 without any EMA, there's no motivation to tune it.

## Next steps

- [ ] If roughness target hit, halt; otherwise advance to [[EXP-025_v_pred]].
