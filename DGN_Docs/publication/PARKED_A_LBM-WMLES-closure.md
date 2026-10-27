---
title: "Parked idea A — data-driven closure for wall-modelled LBM LES"
date: 2026-09-17
tags: [programme, parked, building-blocks, lbm, les, wall-model, openlb]
status: parked
---

# Parked idea A — data-driven closure for wall-modelled LBM LES

> **Status: parked 2026-09-17, deliberately.** The thesis continues on Programme B
> ([[NEXT_PHASE_BRAINSTORM]]). This file exists so the idea, and the reasoning
> about when to pick it up, are not lost.

## What it is

A Building-block Flow Model style closure — wall shear stress plus near-wall and
outer eddy viscosity — for **3D wall-modelled LES on the lattice Boltzmann
method**, implemented in OpenLB.

In BFM ([[arranzBuildingblockflowComputationalModel2024]]) the solver resolves the
large scales. Three small feed-forward networks supply the unresolved physics
from local, frame-invariant inputs: velocity-gradient invariants plus
wall-parallel velocity, non-dimensionalised by `ν` and the local grid size `Δ`.
They are trained only on canonical flows (channel, turbulent Poiseuille–Couette:
ZPG, APG, FPG, separation), and wings are excluded on purpose. The networks
transfer to the NASA CRM high-lift aircraft at 0.9–1.1× the cost of a
conventional WMLES. **The gain is accuracy and generality, not speed.**

## Why it is interesting on LBM specifically

- BFM was built in charLES, a finite-volume code. An LBM wall model has to act
  through the populations — bounce-back with slip velocity, or reconstructed
  near-wall distributions — which is a different numerical coupling.
  ([[malaspinasWallModelLargeeddy2014]], [[ponsinEnhancedWallBoundary2026]])
- BFM's central lesson is **numerical consistency**: closures trained on filtered
  DNS misbehave a posteriori, so the training data is produced by an
  "exact-for-the-mean" WMLES (E-WMLES) in the *same solver and grid*. Whether
  that transfers to LBM's grid-aligned, bounce-back walls is an open question.
  *(Literature check needed before claiming it is unstudied.)*
- It is the natural **high-Re 3D grader** for Programme B. The current
  `ellipse2d` case is laminar at Re ≈ 750 and cannot grade turbulent lifting
  flows.

## Why it is parked

- It is a CFD research programme in its own right, and it is most of what the
  Lozano-Durán group produces.
- It needs sustained multi-GPU compute: the CSF, not the 8 GB laptop.
- The thesis line and the existing body of work sit on the generative / ML side.

## Triggers to un-park

| Trigger | Why it would force the question |
|---|---|
| Programme B needs an independent high-Re 3D grader and no external dataset or RANS route suffices | A then becomes the grader, the same role `ellipse2d` played for Paper 1 |
| Programme B extends to **bluff bodies** | The global model must then live in the volume (the wake), and B's structure converges on A's: resolved large scales plus building-block closures ([[NEXT_PHASE_BRAINSTORM]] §5) |
| A collaboration or visit with a closure-modelling group | It changes the cost of the CFD half |

## Relationship to Programme B

- **Shared:** the building-block flow database and the local, invariant,
  locally-scaled input philosophy.
- **Not shared:** A consumes *resolved LES velocity gradients* at the
  wall-adjacent cell. B's local model consumes an *outer pressure / edge-velocity
  distribution*. Same physics source, different networks.

## First steps, if un-parked

1. Turbulent channel flow at `Re_τ ≈ 1000–2000` in OpenLB-cuda, with an existing
   algebraic wall function, on a coarse WMLES grid. Establish the baseline error.
2. An E-WMLES-style controller: adjust the eddy viscosity to reproduce DNS mean
   profiles, and log the required closure values as numerically consistent
   training data.
3. Train a wall-shear network on channel data only, and test a posteriori at an
   unseen `Re_τ` and grid.
4. Only then add pressure-gradient blocks (TPC flows).

The `openlb-*` subagents (physics, numerics, geometry, build-run, validation) are
configured for exactly this kind of work.

## References to add to the library

- Lozano-Durán & Bae (2023), the ML building-block-flow wall model in JFM that BFM
  builds on. **Not yet in `MyLibrary.bib`.**
- Channel-flow DNS databases used for the BFM training data (BFM refs 90, 91).
