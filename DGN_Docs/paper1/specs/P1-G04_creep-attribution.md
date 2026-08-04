---
id: P1-G04
group: G — Generalisation & failure modes
title: "Magnitude-creep attribution"
tier: core
status: planned
depends_on: [P1-F04, P1-C03]
blocks: []
paper_section: "4.6"
contribution: C4
cost: {train_runs: 0, solver_runs: 24, effort: "~2 days"}
---

# P1-G04 — Magnitude-creep attribution

## Question

Generated shapes grade at an affine slope `a ≈ 2.5–3.5` while solver-validated
dataset shapes grade at `a ≈ 1.74`. Is that discrepancy model-side, solver-side, or
grading-side?

## Hypothesis *(pre-registered)*

**Model-side, driven by classifier-free guidance.** Generated shapes are
systematically thinner and more sharply curved than the target implies, which the
solver correctly reads as a larger pressure magnitude. The proximate cause is the
guidance weight — see C03, which predicts `a` monotone in `w`.

Predicted: the creep survives every solver-side control (resolution, node density,
averaging window), and `w = 0` samples show `a` substantially closer to 1.74 than
`w = 3` samples do.

## Falsifier

The creep shrinks below `2σ` under any solver-side control — for instance if it
disappears when generated shapes are resampled to the dataset's node density. Then
it is a discretisation artefact of how generated shapes are exported to the solver,
not a property of the model, and the Year-1 finding is withdrawn.

## Method

A factorial design, because the Year-1 version varied factors one at a time and
could not exclude interactions.

1. **Factors** (2 × 2 × 2 × 3 shapes = 24 runs):

   | Factor | Levels | Isolates |
   |---|---|---|
   | Shape origin | dataset / generated | The model's contribution |
   | Node treatment | as-produced / resampled to matched density | Discretisation artefacts |
   | Solver resolution | rule / 2× rule | Numerical convergence |

2. Three shapes at matched (Re, H) so the comparison is within-condition and does
   not depend on F03's cross-condition stability.
3. **Add the guidance axis from C03**: `w = 0` and `w = 3` generated shapes, same
   targets. This is the mechanism test.
4. Compare every cell against the **F04 noise floor**. The creep is only real if it
   exceeds the instrument's own repeatability, which was never established when the
   finding was first reported.
5. Report the **geometric route**: thickness, curvature extrema and enclosed area of
   generated vs dataset shapes at matched target. If the model produces thinner
   bodies, `a` should follow, and the finding acquires a physical explanation rather
   than remaining a discrepancy.

## Measurements

| Quantity | Purpose |
|---|---|
| `a` in each of the 24 cells, vs the noise floor | The attribution table |
| **`Cd` in each cell, generated vs dataset at matched target** | **The independent physical axis — see below** |
| Main effects and interactions of the three factors | Which factor carries the creep |
| `a` at `w = 0` vs `w = 3`; `Cd` at `w = 0` vs `w = 3` | The guidance mechanism, on both axes |
| Thickness, curvature, area: generated vs dataset | The geometric route |

### The `Cd` cross-check

`a` is a **fitted parameter of a calibration model**, not a physical quantity. `Cd`
is an integrated force the solver computes directly, and it is already in
`tmp/ellipse2d_force.csv` for every run at zero additional cost.

The hypothesis makes a joint prediction: if generated shapes are genuinely thinner
and sharper, then `a` inflates **and** `Cd` shifts, in the direction thinner
geometry implies. Three outcomes, and they are diagnostic:

| `a` inflated | `Cd` shifted consistently | Reading |
|:--:|:--:|---|
| yes | yes | **Real geometric effect.** The strongest possible attribution — a fitted calibration parameter and an independent integrated force agreeing |
| yes | no | The creep is in the **Cp normalisation or the grading**, not the geometry. Withdraw the model-side claim |
| no | yes | The affine model is masking a real physical difference — revisit §2.1's residual-structure check |

Without `Cd`, the second row is indistinguishable from the first, and the paper
would attribute to the model something that may belong to the grading pipeline.

## Decision rule

| Outcome | Verdict |
|---|---|
| Creep persists across all solver-side cells, tracks `w` | **Model-side, mechanism identified.** The strongest C4 result available |
| Creep vanishes under node resampling | Export artefact — fix the export, withdraw the claim, and note it |
| Creep varies with solver resolution | Numerical — F03's qualification was insufficient; revisit |

## Consequence

This is C4's strongest candidate finding, and attribution is what makes it
publishable. "Generated shapes grade differently" is an observation that a reviewer
will attribute to a pipeline bug. "The discrepancy survives resolution doubling and
node resampling, scales with guidance weight, and is accompanied by a measurable
thinning of the geometry" is a result with a mechanism.

It also matters for **Paper 2**, where R2.2 targets `a → 1.74` through physical
conditioning. That work needs a known cause to act on; without G04 it is guesswork.
See [[../publication/RESEARCH_PROGRAMME]] §3.

## Notes

Paper 1 **measures and attributes** the creep. It does not fix it — closing the
calibration gap is registered as Paper 2 work and is explicitly out of scope here
([EXPERIMENT_REGISTER §5](../EXPERIMENT_REGISTER.md)).

## Results

_Not yet run._
