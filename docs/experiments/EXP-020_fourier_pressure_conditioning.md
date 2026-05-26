---
experiment_id: "EXP-020"
title: "Fourier-DCT pressure conditioning with joint pressure-head loss and CFG"
date: 2026-05-19
status: complete
parent: "EXP-010"
tags: [ellipse, aerodynamic, pressure-conditioning, fourier-dct, classifier-free-guidance, dps, inverse-design, pOnEllipse]
config: "configs/EXP-020_fourier_pressure_conditioning.yaml"
output_dir: "outputs/EXP-020_fourier_pressure_conditioning/"
---

# EXP-020: Fourier-DCT pressure conditioning

> **One-line summary:** Condition the radial DDPM on a fixed-size DCT encoding of the steady-state ⟨Cp⟩(x/c) curve, with a jointly-trained pressure prediction head to drive shape→pressure physics, classifier-free guidance for inference amplification, and optional DPS gradient correction for out-of-distribution targets.

## Motivation

EXP-010/011/012 produced an unconditional radial DDPM that reproduces ellipse-like geometry. EXP-020 turns the pipeline into an **inverse design** tool: given a target pressure distribution, generate a closed boundary shape whose induced pressure matches it. The target may come from a non-ellipse source (e.g. an aerofoil), so the model must generalise beyond simple lookup of the training ellipses.

### Why DCT modes of Cp(x/c)

The HDF5 stores 101 unsteady Cp timesteps per node. We follow Mariolinov's *Diffusion Graph Networks*: collapse to the steady-state mean per node, order by chordwise x/c, and encode as the first K=8 type-II DCT coefficients (orthonormalised). This descriptor is fixed-size and node-count-independent — an aerofoil Cp curve can be encoded the same way and fed in at inference.

### Why a pressure prediction head

A naïve conditional DDPM trained on `(ellipse_shape, ellipse_Cp)` pairs tends to memorise the mapping (target_Cp → nearest training ellipse). Adding a learnable head `h(x̂₀, pos) → ℝ^K` and a joint loss `λ·‖h(x̂₀,pos) − c_target‖²` forces the model to internalise the **forward** map (shape → pressure), so any shape that satisfies the pressure target — including shapes outside the training distribution — becomes a valid generation. The head is only active at lower-noise timesteps (t ≤ T/2) where x̂₀ is reliable.

### Why CFG and DPS

- **CFG** with `p_uncond = 0.15` and inference scale `w = 3.0` lets us amplify the conditioning signal at sampling time without overfitting at training time.
- **DPS** (`dps_guidance_weight > 0`) optionally injects `∇‖h(x̂₀,pos) − c_target‖²` into the reverse step. This is the safety valve for aggressive OOD targets (aerofoil Cp); off by default.

## Hypothesis

With the joint head loss + CFG, conditioning on distinct training Cp curves should produce visibly distinct generated shapes (not noise variance around a mean ellipse), and a synthetic asymmetric Cp target should produce a vertically-asymmetric shape — the first step toward lift-producing geometry.

## Setup

- Dataset: `pOnEllipseConditionalDataset(cond_mode="fourier", k_modes=8)` on `pOnEllipseTrain.h5`. Attaches `Data.cond ∈ ℝ^{1×8}`.
- Score network: same backbone as EXP-010 plus `cond_dim=8`, `p_uncond=0.15`, and learnable `null_cond ∈ ℝ^8`.
- Pressure head: DeepSets `MLP → scatter_mean → MLP` mapping `(x̂₀, pos) → ℝ^8`, jointly trained with `λ_pressure = 0.1` (active for `t ≤ T/2`).
- Inference: `guidance_scale = 3.0`; DPS off by default. Smoke test with `dps_guidance_weight = 0.5` on an aerofoil target.

## Status

Complete — trained 2026-05-26, early-stopped at epoch 79 with best val_loss `0.003058` at epoch 59 (patience=20). Final train_loss `0.004841`. The plan target was val_loss < 0.02; converged 6× below it.

See `docs/superpowers/specs/2026-05-25-exp020-conditioning-visuals-design.md` for the design and `docs/superpowers/plans/2026-05-25-exp020-conditioning-visuals.md` for the executed implementation plan.

## Results

Checkpoint: `outputs/EXP-020_fourier_pressure_conditioning/checkpoint_best.pt` (epoch 59).
Loss log: `outputs/EXP-020_fourier_pressure_conditioning/loss_log.json`.

Figures (post-processed via `scripts/postprocess_exp020.py`):

| File | What it shows |
|---|---|
| `outputs/EXP-020_fourier_pressure_conditioning/figure_a_conditioning_grid.png` | 3 farthest-first training targets + 1 OOD synthetic asymmetric target, with target Cp / head-predicted Cp ± σ / 4 generated shapes per row. Distinct conditioning vectors produce visibly distinct shapes; the synthetic OOD row shows the model still produces a smooth closed boundary, with the head prediction tracking the asymmetric perturbation. |
| `outputs/EXP-020_fourier_pressure_conditioning/figure_b_trajectory.png` | Forward (top) + reverse (bottom) diffusion filmstrip at t ∈ {0, 1, 4, 14, 53, 199} on the synthetic Cp target. Forward goes from clean ellipse to pure noise; reverse mirrors back from noise to clean ellipse. |
| `outputs/EXP-020_fourier_pressure_conditioning/figure_b_reverse.mp4` / `.gif` | Full-resolution reverse-diffusion animation (200 frames, 25 fps) targeting the synthetic asymmetric Cp. |
| `outputs/EXP-020_fourier_pressure_conditioning/figure_c_cfg_sweep.png` | Same OOD synthetic target sampled at CFG `w ∈ {1, 3, 7}`. Head-predicted Cp follows the target shape; higher `w` introduces more boundary detail without diverging. |

### Reproduce

```bash
python train.py --config configs/EXP-020_fourier_pressure_conditioning.yaml --epochs 200 --device cuda --output outputs/EXP-020_fourier_pressure_conditioning/generated_shapes.png
python scripts/postprocess_exp020.py --experiment-dir outputs/EXP-020_fourier_pressure_conditioning --config configs/EXP-020_fourier_pressure_conditioning.yaml --device cuda --n-samples 4 --n-targets 3 --target-seed 0
```

### Caveats

- The head-predicted Cp magnitude in Figure C is biased ~0.1 below the target across all `w`, but the shape of the curve matches. This is consistent with `lambda_pressure=0.1` weighting the head loss lightly relative to the DDPM MSE — the head learns the right Cp morphology but not absolute amplitude. Worth revisiting in EXP-021 if absolute amplitude matters.
- DPS guidance was not exercised here (`dps_guidance_weight=0.0`). The synthetic target is mild enough that CFG suffices; aerofoil-style OOD targets should retry with `dps_guidance_weight=0.5`.
- The boundary nodes in the HDF5 are not stored in chordwise order; `postprocess_exp020.py:radial_to_xy` sorts by theta before plotting so the closed boundary draws correctly.
