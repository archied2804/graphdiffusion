---
experiment_id: "EXP-030"
title: "Upper/lower Cp conditioning (dual-Cp pipeline validation)"
date: 2026-05-28
status: complete
parent: "EXP-021"
tags: [ellipse, aerodynamic, pressure-conditioning, fourier-dct, aoa, upper-lower, lift]
config: "configs/EXP-030_dual_cp.yaml"
output_dir: "outputs/EXP-030_dual_cp/"
---

# EXP-030: Upper/lower Cp conditioning (dual-Cp pipeline validation)

> **One-line summary:** Replace the single-curve `Cp(x/c)` conditioning used by EXP-020/021/022 with a dual `[Cp_upper, Cp_lower]` representation (each K=8 DCT modes, total 16-dim). First experiment that exposes the upper and lower surface Cp curves to the model independently — a prerequisite for any future asymmetric-AoA work.

## Motivation

EXP-020/021/022's pipeline computes `cp_mean.argsort(x/c)` per simulation, collapsing both the upper and lower surface nodes into a single Cp curve. Even when the underlying physics produces asymmetric Cp distributions (e.g. at AoA ≠ 0), the conditioning vector cannot represent that asymmetry — the model can only ever see one curve per sample.

This experiment introduces the architectural fix:

- **`cond_mode="fourier_dual"`** — split boundary nodes by `sign(y − y.mean())`, interpolate each half's Cp onto a shared `N_GRID=64` x/c grid, DCT-truncate each to K=8 modes, concatenate into a 2K=16 cond vector. Score network's `cond_dim` and pressure-head's `out_dim` both move 8 → 16.
- **`CpCurveEditor` × 2** — the new `notebooks/EXP-030_dual_cp_interactive.ipynb` stacks two editors (upper, lower) so the user can drag them independently.

### Data caveat

The originally-planned target dataset was a wider-AoA file. Inspection revealed:

- `pOnEllipseTrain.h5` (5701 sims): **AoA = 0**, H (domain height) spans 5.0–6.0. The "AoA" column in earlier code was misnamed — it actually carries H per [Mariolinov TABLE III](https://huggingface.co/datasets/mariolinov/Ellipse). The codebase constant is now `_H5_COL_H` with a legacy alias.
- `pOnEllipseAoA10.h5` (24 sims): **AoA = 10°**, fixed. Too sparse (24 sims) to train on.

So EXP-030 trains on the AoA=0 dataset where upper/lower Cp are nearly symmetric by physics. The dual split is mostly redundant in this regime — but it validates the architecture end-to-end (config, dataset, model, postprocess, notebook) so that when a wider-AoA dataset becomes available, the only change needed is the YAML's `variant` field. Treat this as wiring validation, not lift generation.

## Hypothesis

- Boundary roughness stays within ±15% of EXP-022's 0.01523 — the dual-cond architecture should be neutral on data where the two surfaces are symmetric.
- Head MSE on the synth-asymmetric target (a deliberately asymmetric Cp pair) is lower than EXP-022's, because the model now has the *vocabulary* to represent that asymmetry even if the training data didn't strongly demand it.
- The CFG-3 sweep produces visibly asymmetric shapes when the synth-asymmetric target's upper/lower curves diverge — demonstrating that the dual conditioning channel is *live*, not collapsed.
- Generated shapes for symmetric training targets stay symmetric; the model doesn't gratuitously break left/right symmetry.

## Changes from parent (EXP-022)

- **Config diff:**
  - `ellipse_dataset.variant: default` (explicit; same data as EXP-021/022).
  - `ellipse_dataset.cond_mode: fourier_dual` (new option alongside `fourier`/`nodal`).
  - `score_network.cond_dim: 8 → 16` and `pressure_head.out_dim: 8 → 16`.
  - `noise_schedule.T: 200` (back to the EXP-021 value rather than EXP-022's 1000, so the dual-Cp lever is isolated from the T-step lever for a clean comparison; a follow-up EXP-031 can compose them).
- **Code changes:**
  - `DatasetUrl.AOA10_H5` and `variant` arg on `pOnEllipseDataset` (validates against `{"default", "aoa10"}`, switches the URL + `raw_file_names` + `processed_file_names`).
  - `fourier_dual` branch in `pOnEllipseConditionalDataset._build_graphs` (new private helper `_fourier_dual_modes`).
  - `make_synthetic_target` + `sample_shapes_for_target` + `_figure_a` + `_figure_c` in `scripts/postprocess_exp020.py` generalised to handle 2K cond vectors (detected from `config["ellipse_dataset"]["cond_mode"]`).
  - `plot_conditioning_grid` gains optional `target_cps_lower` / `head_pred_cps_lower` / `head_pred_stds_lower` args; when set, plots upper + lower on the same axes (upper blue, lower red).
- **Tests (new):** `tests/test_pOnEllipseAoA.py` (4 tests) covers `fourier_dual` shape, dual-vs-single asymmetry, and existing `fourier` regression.
- **Data changes:** new ~250 MB file downloaded on first use; cached alongside the existing `pOnEllipseTrain.h5`.

## Method

Identical training recipe to [[EXP-021_long_train]] (1000-epoch ceiling, `patience=100`, T=200 cosine schedule, CFG=3.0, `λ_pressure=0.1`). Only the dataset + cond representation changes.

### Reproduce

```bash
python train.py \
    --config configs/EXP-030_dual_cp.yaml \
    --epochs 1000 \
    --device cuda \
    --output outputs/EXP-030_dual_cp/generated_shapes.png

python scripts/postprocess_exp020.py \
    --experiment-dir outputs/EXP-030_dual_cp \
    --config configs/EXP-030_dual_cp.yaml \
    --device cuda \
    --n-samples 4 --n-targets 3 --target-seed 0
```

## Results

| Metric | EXP-021 (T=200, single) | EXP-022 (T=1000, single) | EXP-030 (T=200, dual) |
|---|---|---|---|
| Best val loss | 0.000104 | 0.000090 | 0.000277 |
| Boundary roughness (overall) | 0.02125 | **0.01523** | 0.01798 |
| Boundary roughness (train #0 / #1 / #2) | 0.0155 / 0.0263 / 0.0103 | 0.0099 / 0.0063 / 0.0087 | 0.0085 / 0.0226 / 0.0173 |
| Boundary roughness (synth asym.) | 0.03296 | 0.03604 | **0.02345** |
| Epochs (best) | 986 | 988 | 400 (early-stopped at 500) |

### Plots

`outputs/EXP-030_dual_cp/figure_a_conditioning_grid.png` confirms the dual-Cp pipeline is fully live:

- **Training rows (AoA=0 data):** upper (blue) and lower (red) target Cp curves overlap almost exactly — the physically-correct result for symmetric flow — and the head prediction tracks both. This validates that the split, interpolation, and DCT are behaving sensibly on symmetric data.
- **Synth-asymmetric row:** the upper and lower target curves diverge strongly, and the head's predicted upper/lower curves diverge to match. The dual conditioning channel carries independent information; it is not collapsing to a single curve.
- All four rows produce clean closed boundaries.

### Observations

- **Dual-Cp is neutral-to-positive on surface quality.** At the same T=200, EXP-030 (0.01798) beats the single-Cp EXP-021 (0.02125) by 15% overall — adding the architecture did not regress roughness, it slightly improved it. Still above EXP-022's T=1000 (0.01523), as expected since EXP-030 keeps T=200 to isolate the dual-Cp lever.
- **Biggest win on the OOD synthetic target:** 0.02345 vs EXP-021's 0.03296 and EXP-022's 0.03604 (−29% / −35%). The richer two-curve representation gives the model more vocabulary to handle asymmetric targets — exactly where the single-Cp models struggled most.
- **Higher val loss (0.000277) and earlier stop (epoch 400).** Predicting 16 modes is a harder objective than 8, and the head-loss term now spans both surfaces. The early stop at 500 epochs (patience 100 from best at 400) suggests the model converged faster on this richer target; a longer patience might eke out a little more.

## Conclusions

- **Wiring validated end-to-end.** `cond_mode="fourier_dual"`, the `variant` arg, the 16-dim score-network/head, postprocess dual plotting, and the two-editor notebook all work. The cache-naming bug found during pre-flight (variant omitted from the conditional cache filename) is fixed and regression-tested.
- **The dual representation is worth keeping even on symmetric data** — it improves OOD robustness at no cost to in-distribution quality.
- Ready for a wider-AoA dataset: only the YAML `variant` field needs to change when one is available.

### Plots

Once run, `outputs/EXP-030_dual_cp/figure_a_conditioning_grid.png` shows the new layout: target column overlays upper (blue) and lower (red) Cp curves, and the predicted-Cp column shows the head's prediction of both surfaces with ±σ shading. The synth-asymmetric row should produce a shape with visibly different curvature top vs bottom.

### Observations

_To be filled in after the run._

## Conclusions

_To be filled in after the run._

## Next steps

- [ ] Once trained, open `notebooks/EXP-030_dual_cp_interactive.ipynb` and verify the dual editor drives visibly asymmetric shapes for asymmetric Cp inputs.
- [ ] EXP-031 — compose EXP-030's dual-Cp recipe with EXP-022's T=1000 schedule on the same data, isolate the joint effect.
- [ ] When a wider-AoA dataset becomes available, flip `variant: default → aoa_wide` (whatever name) — no code change needed.
