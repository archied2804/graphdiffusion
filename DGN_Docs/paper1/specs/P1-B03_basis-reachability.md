---
id: P1-B03
group: B — Basis necessity
title: "Basis reachability analysis"
tier: core
status: planned
depends_on: [P1-F01, P1-F04]
blocks: [AUGUST-GATE, P1-B02, P1-G03]
paper_section: "4.3"
contribution: C1
cost: {train_runs: 0, solver_runs: 8, effort: "~2 days"}
---

# P1-B03 — Basis reachability analysis

## Question

How many low-dimensional basis modes are needed to reconstruct this dataset's
shapes to within the solver's own measurement uncertainty?

## Hypothesis *(pre-registered)*

**Very few — around three to five.** pOnEllipse is a two-parameter family
(essentially Re and semi-minor axis `b`), so its shape manifold is nearly
two-dimensional. Predicted: 3 PCA modes reconstruct held-out `r(θ)` to a geometric
error whose Cp consequence is below `2σ_ρ`, making it *undetectable by the
instrument*.

## Falsifier

Reconstruction error stays above the noise floor beyond `M = 20`. That would mean
the shape manifold is genuinely high-dimensional, C1's premise binds even on this
dataset, and B02 should show a clear node-native advantage.

## Method

No training. The point is to answer C1's premise **analytically**, independent of
any model's optimisation behaviour.

1. Fit shape-PCA on the training partition; compute reconstruction error of
   **held-out** `r(θ)` versus retained modes `M ∈ {1, 2, 3, 5, 10, 20, 50}`.
2. Report explained variance, but do not stop there — explained variance is not a
   physically meaningful tolerance.
3. **Convert geometric error into a physical tolerance.** Take 4 held-out shapes,
   reconstruct each at `M = 3` and at `M = 10`, and push original and reconstructed
   through the solver (8 runs). Measure the change in Cp. That gives the mode count
   at which reconstruction error becomes invisible to the grading instrument.
4. Repeat the reachability curve for the **DCT** basis so it can be compared
   directly against A03's truncation ladder — same question, two bases.
5. Extend the analysis to the **AoA10** shapes: how many modes does a
   *training-fitted* basis need to reconstruct a shape family it never saw? This
   is the quantitative form of "a basis bounds the reachable design space" and it
   feeds directly into G03.

## Measurements

| Quantity | Purpose |
|---|---|
| Held-out reconstruction error vs `M`, both bases | The reachability curve |
| Δρ from reconstruction at `M = 3`, `M = 10` vs the F04 noise floor | The physically meaningful tolerance |
| Reconstruction error of AoA10 shapes under the train-fitted basis | Evidence that a basis bounds reach |
| Effective dimensionality (modes to 99% variance) | One-number summary for §4.3 |

## Decision rule

- If ~3 modes suffice within the noise floor → state plainly in §4.3 and §5:
  **on this dataset a low-dimensional basis is nearly sufficient.** The paper then
  argues C1 on reach (AoA10, G03) rather than on in-distribution accuracy.
- If the AoA10 shapes need many more modes than in-distribution shapes → that
  contrast **is** the C1 argument, made without training a single model. It is the
  most compact evidence available for the parameterisation-bottleneck premise.

## Consequence

The cheapest and most honest test of C1's premise. It is also the experiment that
sets up the paper's central limitation *before* the results section reaches it, so
§5 confirms what §4.3 already showed rather than conceding something new.

Two days of work, no training, eight solver runs — and it determines how §1 must be
worded.

## Results

_Not yet run._
