---
id: P1-D03
group: D — Discretisation independence
title: "Mixed-resolution training"
tier: conditional
status: planned
depends_on: [P1-D02]
blocks: []
paper_section: "4.4"
contribution: C2
cost: {train_runs: 3, solver_runs: 20, effort: "~3 days"}
---

# P1-D03 — Mixed-resolution training

> **Conditional experiment.** Run only if **D02 fails** — that is, if the 96-node
> model does not transfer across node counts. If D02 passes, this becomes a Paper 3
> item and is not run for Paper 1. Registered in advance so that the fall-back is
> pre-planned rather than improvised after a bad result (viva Part B3).

## Question

If training at a single node count does not yield transfer, does exposure to
varying node counts during training produce it?

## Hypothesis *(pre-registered)*

Yes, at a cost. Resampling each training shape to a randomly drawn `N` per epoch
should force the model to learn a prior in **arc-length** space rather than node-index
space, restoring transfer. Predicted: transfer across `N ∈ [48, 192]` recovers to
within `2σ` of the 96-node reference, at the price of a small but measurable loss
of fidelity **at** 96 nodes — the usual cost of a broader invariance.

## Falsifier

Mixed-resolution training does not restore transfer. Then the limitation is
architectural rather than a matter of training exposure — most plausibly the fixed
`[cos θ, sin θ]` positional context or the fixed `k_neighbors` receptive field
measured in nodes rather than arc length. That diagnosis is itself the useful
output, and it hands Paper 2 a well-posed problem.

## Method

1. Augment the dataset pipeline: resample each training shape's `r(θ)` and its
   `[cos θ, sin θ]` context to `N ~ U{48, …, 192}`, redrawn per epoch.
   Edge features (`ComputeAngularEdgeFeatures`, `ComputeArcLengthEdgeFeatures`)
   recompute automatically from `pos`, which is the property that makes this
   feasible without touching the model.
2. Everything else identical to P1-REF. Seeds 0, 1, 2.
3. Re-run the **D02 protocol verbatim** on the resulting model so the two are
   directly comparable — same targets, same node counts, same grading.
4. Also report fidelity at `N = 96` against P1-REF, to quantify the cost of the
   invariance.

## Measurements

| Quantity | Purpose |
|---|---|
| Validity and ρ vs `N`, overlaid on D02's curve | Whether transfer is restored |
| ρ at `N = 96` vs P1-REF | The cost of the broader invariance |
| Power spectrum of `r(θ)` vs `N` | Whether the learned prior moved from index space to arc-length space |
| `k_neighbors` effective arc-length span vs `N` | Diagnostic if transfer is still absent |

## Decision rule

- Transfer restored, cost at 96 within `2σ` → **the better recipe**, and C2 is
  earned by construction rather than by luck. Report both models; state that the
  reference was trained single-resolution.
- Transfer restored at material cost → report the trade-off as a finding. A
  measured invariance–accuracy trade-off is a more interesting result than either
  endpoint alone.
- Transfer not restored → report the diagnosis. Name the suspected cause
  (index-space positional context, node-count receptive field) and hand it to
  Paper 2 as a defined problem.

## Consequence

This is a **registered fall-back**, which is its main value. The viva panel asked
specifically for *"reflection on risks and discussion of fall-back options"*
(Part B3). Having the mitigation for D02's failure written and costed before D02
runs is the concrete form of that reflection — and it means a negative D02 result
produces a next step rather than a gap.

## Results

_Not yet run — conditional on D02._
