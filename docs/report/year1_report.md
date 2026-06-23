---
title: "Diffusion-based inverse design of aerodynamic shapes, validated against an independent lattice-Boltzmann solver"
date: 2026-06-23
tags: [report, phd, methodology, results, diffusion, cfd, inverse-design]
---

# Diffusion-based inverse design of aerodynamic shapes

**Validated against an independent lattice-Boltzmann solver.**

## Abstract

We present a graph-diffusion generative model for aerodynamic inverse design: conditioned on a
target surface-pressure curve `Cp(x/c)`, it produces the 2D shape that should carry it. The model
denoises the radial field `r(θ)` of a closed boundary with a graph-network score function, treating
the boundary as a graph of surface nodes. The target pressure is encoded as a fixed-size discrete
cosine transform (DCT) spectrum and injected through a jointly-trained DeepSets head, so the
conditioning is order-invariant and decoupled from sampling resolution. The decisive contribution
is *independent physical validation*: generated shapes are run through a from-scratch 2D OpenLB
lattice-Boltzmann solve, and the measured `Cp` is compared to the conditioning target — a check
that never touches the model's own pressure head. The loop reproduces dataset pressure on known
ellipses at correlation 0.998, establishing the baseline gate, and grades generated shapes at
correlation ≈ 0.95 in-distribution. This couples a generative inverse-design model to an
independent physics ground truth rather than a self-graded surrogate.

## 1. Introduction

<!-- STUB: draft in Year-2 paper pass -->

_To cover:_

- Motivation: aerodynamic inverse design is a one-to-many problem (many shapes ~ one target pressure), which suits a generative model rather than a deterministic regressor.
- Why diffusion models on graphs `[@ho2020ddpm]` and why a graph-network representation of the boundary `[@lino2022msgnn]`.
- The gap this work addresses: generative aerodynamic models are rarely validated against independent physics; pressure heads grade themselves.
- Contribution statement: a conditional GN-DDPM inverse-design generator + an independent lattice-Boltzmann validation loop; key results (ρ=0.998 baseline gate, ρ≈0.95 in-distribution inverse design); scope (2D laminar elliptical-cylinder regime).

## 2. Related Work

<!-- STUB -->

_To cover:_

- Denoising diffusion models and guidance `[@ho2020ddpm]`, `[@nichol2021improved]`, `[@ho2022cfg]`, `[@chung2023dps]`.
- Graph-network surrogates for CFD / mesh-based simulation `[@battaglia2018relational]`, `[@lino2022msgnn]`.
- Inverse design in aerodynamics (generative and optimisation-based approaches).
- Set representations for conditioning `[@zaheer2017deepsets]`.

## 3. Methodology

<!-- DRAFTED IN A LATER TASK (3.1-3.9) -->

## 4. Results

<!-- DRAFTED IN A LATER TASK (4.1-4.6) -->

## 5. Discussion

<!-- STUB -->

_To cover:_

- In-distribution trustworthiness vs out-of-distribution extrapolation failure.
- Cp magnitude creep as a model-side calibration issue (not a solver artefact).
- Why posterior guidance `[@chung2023dps]` is the wrong knob out-of-distribution (it overfits the self-trained head).
- Threats to validity, including that the validation framework is still being verified.

## 6. Conclusions

<!-- STUB -->

_To cover:_

- Phase 1 (independent physical validation) complete.
- Phase 2: CFD-in-the-loop optimisation in the model's latent/gradient space with OpenLB as ground truth.

## References

References are managed in `latex/refs.bib`; cited inline above as `[@key]`.
