---
experiment_id: "EXP-042"
title: "OOD fidelity & Cp-magnitude creep — DPS guidance, boundary discretisation, surface roughness"
date: 2026-06-17
status: complete
parent: "EXP-041"
tags: [ellipse, aerodynamic, cfd, openlb, lattice-boltzmann, physical-validation, cp, inverse-design, dps, magnitude-creep, discretisation, roughness, resolution]
config: "cfd/ellipse2d/ (no YAML — OpenLB case); generator: configs/EXP-022_T1000.yaml"
output_dir: "cfd/ellipse2d/runs/"
---

# EXP-042: OOD fidelity & Cp-magnitude creep

> **One-line summary:** [[EXP-041_openlb_generated_inverse_design]] left two loose ends — out-of-distribution fidelity stuck at ρ≈0.85, and an absolute-`Cp` "magnitude creep" (affine slope `a`≈2.2–3.6 vs the dataset's 1.78). This experiment chases both from the **CFD side** and rules the CFD side out: posterior guidance (DPS) *degrades* OOD fidelity rather than helping; boundary node density, lattice resolution, and surface roughness do **not** move the creep or the drag. The creep is present **in-distribution** too (`a`≈2.8–3.0), so it is a **generative-model amplitude bias**, not a solver, mesh, or OOD-target artefact.

## Motivation

EXP-041 validated inverse design in-distribution (ρ≈0.95) with an independent OpenLB solver, but flagged two limitations as next steps:

1. **OOD fidelity** dropped to ρ≈0.85 on a synthetic asymmetric target — could Diffusion Posterior Sampling (DPS), the EXP-020 safety valve, lift it?
2. **Magnitude creep** — the affine slope `a` relating measured to target `Cp` sat at 2.2–3.6, well above the `a`≈1.78 that EXP-040 pinned on *known* dataset ellipses. Is that a real model bias, or an artefact of the coarse exported boundary polygon / lattice resolution / surface roughness?

Before Phase 2 (optimisation in the model's gradient/latent space with OpenLB as ground truth), we need to know **where the error lives**. If it is a CFD/discretisation artefact, the solver loop needs work; if it is model-side, Phase 2 rests on a trustworthy solver and the work is on the generator.

Related: [[EXP-041_openlb_generated_inverse_design]] (parent) · [[EXP-040_openlb_cp_baseline]] (validated loop) · [[EXP-020_fourier_pressure_conditioning]] (amplitude/roughness lineage) · [[validation_report]].

## Hypothesis

- **DPS** should *improve* OOD fidelity — it adds a likelihood gradient toward the target at each reverse step.
- **More boundary nodes** (higher-fidelity polygon) should round the facets the lattice sees, reduce spurious separation, and lower `Cd` (and perhaps pull `a` toward 1.78).
- **Surface roughness** (the diffusion wobble) might add drag once the lattice is fine enough to resolve it; **under-resolving** the boundary might *mask* that effect.

All three were wrong or null — see Conclusions.

## Method

No `src/graph_diffusion/` and no `cfd/ellipse2d/` solver code changed. Three new analysis tools were added:

- `scripts/dps_sweep_export.py` — samples the EXP-022 generator on the synthetic OOD target at several `dps_guidance_weight` values (same seed, so shapes differ *only* by DPS), exports each as an OpenLB `.dat`.
- `scripts/resample_polygon.py` — resamples a generated boundary to a chosen node count with a **periodic cubic spline** in arc length (linear subdivision only inserts collinear points the SDF ignores; a smooth spline genuinely rounds the polygon), plus a Fourier-smoothed variant.
- `scripts/compare_dps_sweep.py`, `scripts/analyze_node_sweep.py`, `scripts/analyze_creep_followup.py` — score each run's OpenLB-measured `Cp` against the shared target. Because the surface `Cp` traces the full body boundary (non-monotonic `x/c`), the **monotonic target is interpolated onto the measured nodes** (the robust direction), and we report correlation (affine-invariant) plus the affine slope `a` (the magnitude metric).

All geometries run through the validated EXP-040 OpenLB GPU case (rest-init + sponge outlet), `MAX_PHYS_T 50`, averaging window 25 chords. Stable runs held `avgRho` quiescent, as in EXP-040/041.

### A. DPS sweep — synthetic OOD target, `dps ∈ {0, 0.3, 0.5, 1.0}`, Re 754, RES 160.
### B. Node-density sweep — synth shape resampled to `n ∈ {60, 120, 240, 480}` at RES 160, plus a Fourier-smoothed `n=240` and a RES-240 control.
### C. Roughness vs resolution — synth `raw_n240` (rough) and `smooth_n240` at RES `{60, 80, 120, 160, 240}`.
### D. In-distribution creep — gen_4849 (Re 960) at `n ∈ {60, 240}` and RES `{160, 240}`.

## Results

### A. DPS *degrades* OOD fidelity

![[assets/EXP-041_dps_sweep.png]]

| DPS weight | `Cp` corr | affine `a` | `Cd` |
|---|---|---|---|
| 0.0 (plain CFG) | **0.860** | 3.46 | 1.59 |
| 0.3 | 0.818 | 5.22 | 2.46 |
| 0.5 | 0.807 | 5.32 | 2.33 |
| 1.0 | 0.762 | 6.48 | 3.10 |

Every metric degrades monotonically with guidance strength: correlation falls, the magnitude slope blows up (3.5→6.5), the body inflates (drag nearly doubles). DPS steers toward the model's **self-trained pressure head**, which is unreliable OOD, so stronger guidance optimises the surrogate's own error. **Plain CFG (`dps=0`) is the correct OOD configuration.**

### B. Node density does not move `Cd` or the creep

![[assets/EXP-042_node_sweep.png]]

| variant | nodes | RES | `Cd` | `Cd` std | `a` |
|---|---|---|---|---|---|
| orig | 60 | 160 | 1.593 | 0.182 | 3.46 |
| raw spline | 120 | 160 | 1.601 | 0.183 | 3.60 |
| raw spline | 240 | 160 | 1.600 | 0.182 | 3.56 |
| raw spline | 480 | 160 | 1.610 | 0.183 | 3.49 |
| raw spline | 240 | **240** | 1.604 | 0.160 | 3.50 |
| Fourier-smooth | 240 | 160 | 1.633 | 0.178 | 3.43 |

`Cd` is flat (1.59→1.61 from 60→480 nodes, all inside the ±0.18 shedding fluctuation); `a` stays ~3.5 regardless of node count, smoothing, or resolution. The Bouzidi curved no-slip wall already interpolates the boundary sub-grid, so refining the polygon changes nothing the flow cares about. Lattice RES 160→240 leaves `Cd` unchanged (1.600→1.604), only damping the fluctuation amplitude (std 0.182→0.160) — the lattice was not the bottleneck.

### C. Roughness has no drag signature; under-resolution can't be reached

![[assets/EXP-042_roughness_resolution.png]]

| RES | `Cd` raw | `Cd` smooth | gap |
|---|---|---|---|
| 60, 80 | **NaN** | **NaN** | unstable |
| 120 | 1.658 | 1.676 | −0.018 |
| 160 | 1.600 | 1.633 | −0.033 |
| 240 | 1.604 | 1.607 | **−0.003** |

Two findings. **(i)** You cannot stably under-resolve at this Re: below RES≈100 the lattice Mach number blows up and the solve goes NaN — under-resolution manifests as numerical destruction, not as benign smoothing. **(ii)** Across the entire *stable* window the raw-minus-smooth `Cd` gap is < 2% and well inside the ±0.16–0.32 shedding noise — statistically zero — and at the **finest, lowest-noise resolution (RES 240) the gap is ≈ 0**. Higher fidelity makes rough and smooth *converge*, the opposite of revealing a masked roughness effect. Physically, at Re 500–1000 the laminar boundary layer is thick relative to the sub-boundary-layer surface wobble, so the flow never "feels" it.

The vorticity wake makes this visual — the rough (raw) and Fourier-smoothed bodies shed an essentially identical von Kármán street at the same width and spacing:

![[assets/EXP-042_vorticity_raw.png]]
![[assets/EXP-042_vorticity_smooth.png]]

### D. The creep is present in-distribution

| in-dist run (gen_4849) | nodes | RES | `Cd` | corr | `a` |
|---|---|---|---|---|---|
| orig | 60 | 160 | 1.456 | 0.977 | 2.97 |
| raw spline | 240 | 160 | 1.462 | 0.959 | 2.93 |
| raw spline | 240 | 240 | 1.379 | 0.966 | 2.82 |

In-distribution `a` is 2.8–3.0 — far above the dataset-shape baseline (~1.78) and unmoved by node count or resolution (node-density null reproduces here too). The two EXP-041 in-distribution targets independently show `a` = 2.92 and 2.48. So the ~1.6× amplitude overshoot exists *where the model has data*; OOD merely worsens it to ~1.9×.

## Conclusions

1. **DPS is the wrong knob OOD.** Posterior guidance through the self-trained head degrades true fidelity monotonically. Phase 2 should sample with plain CFG and put the optimisation pressure on **CFD-in-the-loop**, not the surrogate head.
2. **The magnitude creep is model-side.** Neither boundary node density, lattice resolution, nor surface roughness moves `a`, and the creep is present in-distribution (`a`≈2.8–3.0 vs the solver's validated 1.78). It is a **generative amplitude bias** — a pressure-head amplitude-calibration issue (and, OOD, target realizability) — to be fixed on the model side.
3. **Roughness is aerodynamically inert in this regime**, confirming EXP-041 from the resolution angle: rough and smooth converge as the lattice is refined; smoothing is unnecessary and marginally counterproductive.
4. **The solver and its discretisation are trustworthy.** The validation loop is not the source of any residual error, so Phase 2 gradient-space optimisation rests on solid ground.

> [!NOTE]
> **Design note.** The RES 60/80 NaNs were expected from EXP-040's `RES ≳ 0.2·Re` stability floor — they bound the accessible resolution window rather than being failures, and make the "under-resolution is unreachable" point concretely.

## Next steps

- [ ] **Pressure-head amplitude calibration** — investigate `λ_pressure` / head normalisation so generated shapes carry the requested `Cp` *magnitude*, not just morphology. This is the lever the CFD side just ruled itself out of.
- [ ] **OOD target realizability** — characterise which synthetic `Cp` targets correspond to physically achievable bodies (the OOD increment in `a` may partly reflect unrealizable targets).
- [ ] **Phase 2** — optimisation in the generator's gradient/latent space with OpenLB as ground truth, using plain CFG sampling. → see [[validation_report]] §5.
