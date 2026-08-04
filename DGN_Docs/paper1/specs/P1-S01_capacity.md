---
id: P1-S01
group: S — Scale & data efficiency
title: "Capacity sweep"
tier: support
status: planned
depends_on: [P1-F05, P1-G01]
blocks: []
paper_section: "4.7"
contribution: risk
cost: {train_runs: 9, solver_runs: 12, effort: "~2 days compute"}
---

# P1-S01 — Capacity sweep

## Question

Is 147k parameters the right model size, and where does performance saturate?

## Hypothesis *(pre-registered)*

Performance **saturates at or before the incumbent** configuration
(`node_dim = 32`, `n_layers = 4`). The task is a two-parameter shape family
conditioned on eight numbers; there is not much to fit. Predicted: no detectable
improvement beyond `node_dim = 32`, and depth beyond 4 layers neutral or slightly
worse through optimisation difficulty.

The saturation itself is the informative part. A model that saturates at 147k
parameters on 5,701 samples is telling you the *task* is low-complexity — which
supports B03's finding and belongs in §5's honest limitations, not only in §4.7.

## Falsifier

Performance still improving at `node_dim = 64`, `n_layers = 8`. Then the reference
model is under-sized and every reported number understates the method.

## Method

1. Grid: `node_dim ∈ {16, 32, 64}` × `n_layers ∈ {2, 4, 8}` = 9 runs, single seed
   each, with the incumbent `(32, 4)` cell reusing P1-REF's seeds.
2. Everything else fixed at the P1-REF recipe.
3. Grade a 4-target subset at the corner cells and the incumbent (12 solver runs);
   rank the interior on validation loss.
4. Report **parameters and wall-clock per cell** alongside performance. The
   efficiency argument — competitive results at 147k parameters — is worth making
   explicitly, particularly against the transformer-scale models this field
   increasingly reaches for.
5. Report the **receptive field** implied by each `n_layers` (with `k = 6`) so this
   connects to A04 rather than reading as an unrelated sweep. Depth changes
   receptive field *and* capacity, and those should not be conflated.

## Measurements

| Quantity | Purpose |
|---|---|
| Val loss across the 3×3 grid | The saturation surface |
| Held-out ρ at corners and incumbent | Confirms val loss tracks solver grading |
| Parameters and wall-clock per cell | The efficiency argument |
| Receptive field per depth | Disentangles depth from capacity, links to A04 |

## Decision rule

- Saturation at or before `(32, 4)` → the incumbent is justified. Report the
  surface and the efficiency point.
- Still improving at the largest cell → report that the reference is under-sized
  and quantify the cost. Do not retrofit ([PROTOCOL §1](../PROTOCOL.md)).

## Consequence

Discharges the scale half of the user's own viva note. Cheap — nine runs at
85 minutes each, parallelisable.

Its most useful output may be the negative one: if a 147k-parameter model saturates
this task, that is quantitative evidence that pOnEllipse is too simple to
distinguish methods sharply — which is the same conclusion B03 reaches
analytically and B02 reaches generatively. Three independent routes to the same
statement about the dataset make a much more credible case for the Paper 3 move to
richer families than any one of them alone.

## Results

_Not yet run._
