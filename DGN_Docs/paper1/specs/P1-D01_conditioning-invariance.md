---
id: P1-D01
group: D — Discretisation independence
title: "Conditioning invariance under resampling"
tier: core
status: planned
depends_on: [P1-F01]
blocks: [P1-D02]
paper_section: "4.4"
contribution: C2
cost: {train_runs: 0, solver_runs: 0, effort: "~half a day"}
---

# P1-D01 — Conditioning invariance under resampling

## Question

Is the DCT conditioning genuinely independent of node count, as claimed by
construction?

## Hypothesis *(pre-registered)*

Yes, to a good approximation. Resampling the same Cp curve to 48, 64, 80, 96, 128
or 192 points changes the K=8 DCT vector by **less than 1% in relative L2**. The
residual is interpolation error concentrated in the highest retained mode, and it
grows as node count falls.

## Falsifier

Relative deviation above 5% at any tested node count, or a systematic drift in the
low-order modes. Then the conditioning is *not* node-count independent, C2 is
overclaimed, and D02's node-count transfer results would be confounded by a moving
conditioning signal rather than showing a property of the generative model.

## Method

Analytic — no training, no solver. Half a day, and it underwrites half of C2.

1. Take 200 Cp curves spanning the training partition.
2. Resample each to `N ∈ {48, 64, 80, 96, 128, 192}` points and compute the K=8
   DCT-II conditioning vector with `dct_ii`
   (`src/graph_diffusion/data/pOnEllipseConditional.py:58`).
3. Report relative L2 deviation from the 96-point reference, **per mode** and
   overall.
4. Repeat for `K ∈ {3, 8, 16, 32}` — invariance should degrade as K approaches the
   Nyquist limit of the coarsest sampling. Establishing where it breaks bounds C02's
   sweep and gives C2 a stated validity range rather than an unqualified claim.
5. Include the **AoA10** curves (80 nodes, a genuinely different discretisation
   from a different simulation campaign) rather than only synthetic resamplings of
   96-node data. This is the only place in the suite where the invariance claim
   meets real data at a different node count.

## Measurements

| Quantity | Purpose |
|---|---|
| Relative L2 deviation of the conditioning vector vs `N`, per mode | The invariance number |
| Deviation vs `K` | Where the claim stops holding — the stated validity range |
| Same, for real AoA10 curves | Real-data confirmation |
| Nyquist bound: max valid `K` for each `N` | A stated design rule for §3.3 |

## Decision rule

- Deviation `< 1%` for `K = 8` across `N ∈ [48, 192]` → C2's conditioning-side
  claim is demonstrated, and §3.3 can state the invariance with a measured bound
  rather than asserting it from construction.
- Deviation `> 5%` anywhere in the operating range → rescope C2 to the range where
  it holds, and re-examine D02's design.

## Consequence

C2 has two halves: the **conditioning** is node-count independent (this experiment)
and the **model** transfers across node counts (D02). Only the first is true by
construction, and even that has never been checked. Doing so costs half a day and
removes an assertion from the paper.

If D02 later shows the model does *not* transfer, D01 is what proves the failure is
model-side rather than a conditioning artefact — which turns an awkward negative
result into a precisely located one.

## Results

_Not yet run._
