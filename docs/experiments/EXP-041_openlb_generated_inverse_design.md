---
experiment_id: "EXP-041"
title: "Inverse-design validation — generated shapes through the OpenLB Cp loop"
date: 2026-06-17
status: complete
parent: "EXP-040"
tags: [ellipse, aerodynamic, cfd, openlb, lattice-boltzmann, physical-validation, cp, inverse-design, conditional, generated-shapes]
config: "cfd/ellipse2d/ (no YAML — OpenLB case); generator: configs/EXP-022_T1000.yaml"
output_dir: "cfd/ellipse2d/runs/"
---

# EXP-041: Inverse-design validation — generated shapes through the OpenLB Cp loop

> **One-line summary:** Take shapes the conditional generator produced for target `Cp` curves ([[EXP-022_T1000]]), run them through the *independent* 2D OpenLB loop validated in [[EXP-040_openlb_cp_baseline]], and ask the only question that matters: does the OpenLB-measured surface `Cp` match the **target** `Cp` each shape was conditioned on? **It does, in-distribution (corr ≈ 0.95)** — the first independent confirmation of the inverse-design claim. It degrades out-of-distribution (≈ 0.85), and boundary roughness turns out to be cosmetic (raw vs smoothed give near-identical physics).

## Motivation

The conditional generator ([[EXP-020_fourier_pressure_conditioning]] → [[EXP-021_long_train]] → [[EXP-022_T1000]]) claims **inverse design**: condition on a target `Cp(x/c)` and receive a shape that produces it. Until now that claim was graded only by the model's **own jointly-trained pressure head** — the head learned from the same data and objective that produced the shape, so it was effectively grading itself. Circular.

[[EXP-040_openlb_cp_baseline]] removed the circularity by building and validating an **independent** 2D OpenLB (lattice-Boltzmann) solver that measures surface `Cp(x/c)` from first principles, and proving on *known* dataset ellipses that it reproduces the dataset `Cp` (cleanest case corr 0.998). EXP-041 is the payoff: feed the loop *generated* shapes and grade them against their conditioned targets with a solver that has never seen the model.

This is the real Phase-1 step that EXP-040 was a prerequisite for.

Related: [[EXP-040_openlb_cp_baseline]] (the validated loop) · [[EXP-022_T1000]] (the generator) · [[changelog]].

## Hypothesis

- **In-distribution targets** (Cp curves drawn from the training set) will reproduce well under OpenLB — high correlation between measured and target `Cp` — because the generator has seen the shape→`Cp` map there.
- **An out-of-distribution synthetic target** will reproduce less faithfully: the generator must extrapolate the shape→`Cp` map, which is exactly the failure mode this loop was built to expose.
- Consistent with [[EXP-020_fourier_pressure_conditioning]], absolute `Cp` **amplitude** will not match (the model captures morphology, not magnitude), so **correlation (shape)** — not RMS — is the right inverse-design metric.

## Changes from parent

- **Code changes:** new export script `scripts/export_generated_for_openlb.py` (samples the generator, reconstructs the boundary polygon, exports raw + Fourier-smoothed `.dat`, decoded target `Cp`, and a manifest). Comparison reuses the [[EXP-040_openlb_cp_baseline]] machinery in `scripts/compare_cp_baseline.py`. No `src/graph_diffusion/` code changed; no `cfd/ellipse2d/` solver code changed (the EXP-040 build is reused as-is).
- **Inputs:** EXP-040 fed *known dataset ellipses*; EXP-041 feeds *generated shapes* sampled from the EXP-022 checkpoint.

## Method

### Generator and targets

The generator is the EXP-022 checkpoint (`outputs/EXP-022_T1000/checkpoint_best.pt`): single-`Cp` conditioning via a K=8 DCT of the target curve, T=1000 DDPM schedule, classifier-free guidance at scale 3.0. Three targets were chosen:

- **Two in-distribution targets**, picked **farthest-first** across the training `Cp` curves so they are maximally different from one another: sim #4849 (Re=960) and sim #4066 (Re=884).
- **One synthetic, asymmetric out-of-distribution target** (run at Re=754), deliberately outside the dataset's `Cp` morphology to probe extrapolation.

### Export — `scripts/export_generated_for_openlb.py`

For each target the script: samples a shape from the generator; reconstructs the closed boundary polygon; and writes it as an `x y` `.dat` in **two variants** —

- **raw** — the polygon exactly as decoded from the sampled node field, and
- **Fourier-smoothed** — a low-pass of `r(θ)` keeping only the first **6 modes**, to test whether the characteristic boundary wobble (the long-running smoothness thread from [[EXP-020_fourier_pressure_conditioning]] / EXP-013) corrupts the physics.

It also exports the decoded **target** `Cp(x/c)` for each shape and a manifest carrying each shape's Re and channel height `H`. Six `.dat` files result (3 targets × {raw, smoothed}).

### Solver — the EXP-040 loop, unchanged

All six geometries run through the validated `cfd/ellipse2d/` OpenLB GPU case (same BC setup, rest-initialisation, and sponge outlet established in [[EXP-040_openlb_cp_baseline]]). Resolution is Re-scaled for stability, following EXP-040's `RES ≈ 0.2·Re` rule: in-distribution targets at RES 180–200, the synthetic target at RES 160.

> [!NOTE]
> **The sponge fix generalises.** Every one of the six runs was numerically stable — `avgRho` amplitude stayed at ≈ 0.0001–0.0006, the same quiescent regime EXP-040 reached after the acoustic-resonance fix. The rest-init + sponge-outlet treatment holds for *generated* geometries too, not just clean dataset ellipses.

### Compare — `scripts/compare_cp_baseline.py`

The OpenLB-measured `Cp` is compared to the conditioned target `Cp` via correlation plus a least-squares affine fit (`Cp_olb ≈ a·Cp_target + b`), reusing the EXP-040 comparison machinery. The figure overlays target vs measured `Cp` (rescaled to target units), for raw and smoothed variants, per target.

## Results

### Inverse-design correlation — OpenLB-measured `Cp` vs target `Cp`

| Target | Type | Re | `Cp` corr (raw) | `Cp` corr (smoothed) | Δcorr |
|---|---|---|---|---|---|
| #4849 | in-distribution | 960 | **0.964** | 0.952 | −0.012 |
| #4066 | in-distribution | 884 | **0.966** | 0.963 | −0.003 |
| synthetic | out-of-distribution | 754 | **0.860** | 0.876 | +0.016 |

![[assets/EXP-041_correlation_bars.png]]

> [!NOTE]
> **Metric (updated).** These correlations use the *robust* convention — the
> monotonic target `Cp` is interpolated onto the OpenLB-measured surface nodes,
> not the reverse. The OpenLB surface `Cp` is non-monotonic in `x/c` (it traces
> the full body boundary, folding over the top and bottom surfaces), so the
> earlier "measured-onto-reference" interpolation in `compare_cp_baseline.py`
> was the buggy direction; this is the convention standardised in EXP-042 / the
> DPS sweep. The headline is unchanged (in-dist ≈ 0.96, OOD ≈ 0.86), but the
> smooth-OOD value moves from 0.822 to 0.876 — see conclusion 3. The
> `EXP-041_cp_inverse_design.png` overlay panel titles below predate this fix
> and show the legacy values.

### Drag — raw vs smoothed agree

| Target | `Cd` (raw, approx) | `Cd` (smoothed, approx) | within |
|---|---|---|---|
| #4849 | ≈ 1.45 | ≈ 1.45 | ~5% |
| #4066 | ≈ 1.9 | ≈ 1.9 | ~5% |
| synthetic | ≈ 1.6 | ≈ 1.6 | ~5% |

Raw and Fourier-smoothed variants give `Cd` within ~5% for every target.

### Affine scale

The affine slope `a` from the target-vs-measured fit lands in the range **2.2–3.6**, *larger* than the dataset-normalisation factor `a ≈ 1.78` that EXP-040 pinned on known ellipses. This is consistent with [[EXP-020_fourier_pressure_conditioning]]'s known finding that the model reproduces `Cp` *morphology* but not absolute *amplitude* — so the generated targets sit at a different magnitude, inflating the slope. Correlation (morphology) is therefore the meaningful inverse-design metric here, not the slope or RMS.

### Figures

Target (conditioned) vs OpenLB-measured `Cp(x/c)`, rescaled to target units, raw and smoothed overlaid per target:

![[assets/EXP-041_cp_inverse_design.png]]

Per-run OpenLB outputs (force CSV, `Cp` CSV, VTK fields) live under `cfd/ellipse2d/runs/`. The generated `.dat` polygons, decoded target `Cp` curves, and the manifest are produced by `scripts/export_generated_for_openlb.py`.

## Conclusions

1. **Inverse design is validated in-distribution.** Both in-distribution targets reproduce at corr ≈ 0.95, measured by an *independent* solver — not the self-trained pressure head that graded every prior `Cp` result. The generator's central claim — condition on a target `Cp`, get a shape that produces it — holds where the model has data.

2. **Fidelity degrades out-of-distribution.** The synthetic, asymmetric target drops to corr ≈ 0.85: beyond the training manifold the model extrapolates the shape→`Cp` map less faithfully. This is precisely the limitation the independent loop was built to surface, and it now has a number.

3. **Boundary roughness is cosmetic, not physical.** Raw and Fourier-smoothed variants give near-identical `Cp` (|Δcorr| ≤ 0.016) and `Cd` (~5%). The wobble does not corrupt the physics, so post-hoc smoothing is essentially **neutral** — marginally lower correlation in-distribution (Δcorr −0.012, −0.003) and marginally *higher* out-of-distribution (+0.016), all within the run-to-run shedding noise. (The original log read these the other way — "slightly counterproductive" — but that was an artefact of the legacy interpolation direction; under the robust metric smoothing is a wash.) This resolves the long-running EXP-013 / EXP-02x smoothness thread from the **physics** side: the roughness was never an aerodynamic problem. EXP-042 confirms this from the resolution angle.

> [!WARNING]
> **Amplitude caveat.** The affine scale `a` (2.2–3.6) exceeds EXP-040's dataset-normalisation factor (~1.78). The model captures `Cp` *shape*, not absolute *magnitude* — a known result from [[EXP-020_fourier_pressure_conditioning]]. Treat correlation as the inverse-design verdict; do not read absolute `Cp` amplitude off these runs.

The hypothesis held on all three counts: strong in-distribution correlation, an OOD drop, and a morphology-not-magnitude amplitude story.

## Next steps

- [ ] **OOD fidelity:** test DPS (diffusion posterior sampling) guidance — EXP-020's safety valve — on the synthetic / aerofoil targets to see whether it lifts the OOD correlation above ≈ 0.85.
- [ ] **Amplitude bias:** revisit `λ_pressure` / pressure-head weighting if absolute `Cp` magnitude ever becomes a requirement (currently only morphology matters for inverse design).
- [ ] **Extend the loop:** run [[EXP-030_dual_cp]] dual-`Cp` targets, and lift-producing (asymmetric) targets, once a wider-AoA dataset exists.
