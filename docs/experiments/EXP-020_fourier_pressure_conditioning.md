---
experiment_id: "EXP-020"
title: "Fourier-DCT pressure conditioning with joint pressure-head loss and CFG"
date: 2026-05-19
status: planned
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

Planned — implementation in progress. See `docs/superpowers/specs/2026-05-19-pressure-conditioning-design.md` for the design and `/home/m22729ad/.claude/plans/add-pressure-conditioned-generation-to-frolicking-wigderson.md` for the implementation plan.
