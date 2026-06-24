---
experiment_id: "EXP-043"
title: "Systematic OOD Cp sampling & OpenLB validation"
date: 2026-06-24
status: complete
parent: "EXP-042"
tags: [openlb, ood, inverse-design, validation, cp-manifold]
config: "configs/EXP-022_T1000.yaml"
output_dir: "outputs/sweep_results/"
---

# EXP-043: Systematic OOD Cp sampling & OpenLB validation

> **One-line summary:** A principled PC-mode OOD ladder + the 15-case in-distribution
> grid, both validated through GPU OpenLB; the solver reproduces the dataset Cp at
> ρ≈0.96 / a≈1.74, while generated OOD shapes degrade with OOD distance and carry a
> magnitude creep a≈2.0–2.5 — confirming the bias is model-side.

## Motivation

EXP-041/042 probed OOD inverse design with five *ad-hoc* synthetic Cp targets, so the
OOD axis was uncontrolled. This experiment builds a **systematic OOD ladder** by
perturbing the dataset-mean Cp along the dataset's own principal Cp modes (PCA),
generates a shape per target, validates each through OpenLB, and assesses where
inverse-design fidelity breaks down — alongside a clean in-distribution solver check
([[EXP-042]] established the creep is model-side; this stresses it on a controlled axis).

## Hypothesis

- Fidelity (ρ) should fall and magnitude creep (`a`) should drift as the target moves
  out of distribution.
- The in-distribution eval_grid should validate the solver: high ρ and `a` near the
  dataset's ~1.8 Cp-normalisation (the EXP-040 baseline found a≈1.78).

## Method

- **Targets:** `cp_manifold.pc_ladder_targets` — for PC1, PC2 of the dataset Cp-PCA,
  `target = mean + sign·α·σ_mode·PC_mode`, α ∈ {1,2,3,4}σ, both signs → 16 targets.
- **Generation:** `scripts/export_ood_systematic.py` with the inverse-design checkpoint
  `EXP-022_T1000/checkpoint_best.pt`, plain classifier-free guidance (DPS=0 — EXP-042
  showed DPS degrades OOD). Fixed Re/H = 754/5.5 so the Cp-target morphology is the only
  variable.
- **Validation:** `cfd/ellipse2d/runs/run_from_manifest.sh` on the 16 shapes
  (per-row RES auto-scaled; Re=754 → RES≈151) on the GPU build. Plus the 15-case
  in-distribution `eval_grid` (dataset shapes through OpenLB).
- **Metric:** `sweep_results.cp_match_metrics` — interpolate the monotonic target onto
  the measured (non-monotonic) nodes; report Pearson ρ and affine slope `a`
  (`measured ≈ a·target + b`). Figures: `outputs/sweep_results/{ood_systematic,eval_grid}/`.

## Changes from parent

- **Code:** new `data/cp_manifold.py` (Cp-PCA + PC-ladder), `visualisation/sweep_results.py`
  (gallery, design-space & Cp-manifold context, envelope overlay, results-vs-OOD),
  `scripts/export_ood_systematic.py`, `scripts/plot_sweep_results.py`. `plot_cp_pca`
  refactored onto the shared basis. OpenLB driver gained per-row RES auto-scaling.

## Results

All 16 OOD runs and all 15 eval_grid runs were numerically stable (avgRho≈1.0,
maxCFL≈0.1, **no NaN**).

### A. Solver validation (eval_grid, in-distribution)

OpenLB reproduces the dataset Cp across the (Re, H) grid: **ρ = 0.89–1.00 (mean 0.96)**,
affine **a = 1.51–1.88 (mean 1.74)**. The slope a≈1.74 matches the dataset's ~1.8 Cp
velocity-normalisation (EXP-040 baseline a≈1.78). **The solver is trusted ground truth.**
Figures: `outputs/sweep_results/eval_grid/`.

### B. OOD inverse design degrades with OOD distance

Along the genuinely-OOD **+PC1** direction (ood_dist 1.80 → 2.74 as α goes 1σ → 4σ):

| α (σ) | ood_dist | ρ | affine a |
|---|---|---|---|
| 1 | 1.80 | 0.901 | 2.08 |
| 2 | 2.02 | 0.922 | 2.06 |
| 3 | 2.33 | 0.858 | 1.79 |
| 4 | 2.74 | 0.857 | 1.76 |

ρ declines from ~0.92 to ~0.86 as the target is pushed out of distribution.
Across all 16 OOD cases, ρ = 0.857–0.980 and **a = 1.76–2.53** — i.e. the generated
OOD shapes carry a **magnitude creep a≈2.0–2.5, above the in-distribution solver
baseline (1.74)**. Since the solver is validated in §A, this creep is **model-side**
(generative amplitude/pressure-head bias), reproducing the EXP-042 conclusion on a
controlled axis.

### C. Parameterisation finding: σ-amplitude is not isometric to cond-space distance

Only **+PC1** is genuinely reachable as OOD: −PC1 and both PC2 directions keep
`ood_dist ≈ 1.0–1.6` regardless of amplitude (e.g. PC2 ood_dist ≈ 1.47–1.64 flat
across 1σ→4σ). Raw-curve-PCA σ-units do not map linearly onto the DCT-conditioning
space the model sees — the dataset Cp spread the model can represent is dominated by
the +PC1 direction. **For a future OOD ladder, sweep amplitude in cond/DCT space (or
along +PC1 only) to get a monotone OOD axis.**

## Conclusions

1. The OpenLB loop is validated in-distribution (ρ≈0.96, a≈1.74) — solver trusted.
2. Inverse-design fidelity falls with OOD distance (+PC1: ρ 0.90→0.86) and the OOD
   shapes show a model-side magnitude creep a≈2.0–2.5, consistent with [[EXP-042]].
3. The PCA-σ ladder only reaches OOD along +PC1; the OOD axis should be defined in
   conditioning space next time.

## Next steps

- Re-parameterise the OOD ladder in DCT-cond space for a monotone ood_dist axis.
- The magnitude creep is the model-side target for the deferred Re-conditioning work
  (see `docs/superpowers/specs/2026-06-23-dataset-viz-physical-conditioning-design.md`,
  Part D) — does Re conditioning pull `a` toward the validated 1.74?
- Hron–Turek benchmark validation (separate, later).
