---
id: P1-G02
group: G — Generalisation & failure modes
title: "AoA10 physical out-of-distribution"
tier: core
status: planned
depends_on: [P1-F05, P1-D02]
blocks: []
paper_section: "4.6"
contribution: C4
cost: {train_runs: 0, solver_runs: 24, effort: "~2 days"}
---

# P1-G02 — AoA10 physical out-of-distribution

## Question

Does the model transfer to a genuinely different physical regime — 10° incidence,
asymmetric, lifting — that no training sample resembles?

## Hypothesis *(pre-registered)*

**Substantial degradation.** The training set is entirely symmetric ellipses at
zero incidence, so the asymmetric upper/lower Cp distributions of the AoA10 set lie
outside anything the model has represented. Predicted: ρ falls well below G01's
held-out value, and the characteristic failure is that the model produces a
*symmetric* shape approximating the mean of the asymmetric target — the conditional
mean of a distribution whose support excludes the target.

Predicted: mean ρ < 0.75, and generated shapes measurably more symmetric than the
targets require.

## Falsifier

ρ comparable to G01's held-out value. That would be a strong and surprising
generalisation result — and would need checking against the possibility that the
DCT conditioning is discarding the asymmetry, so that the model is solving an
easier problem than it appears to.

## Method

1. All **24** sims in `data/ellipse/raw/pOnEllipseAoA10.h5` as targets
   (`variant="aoa10"`, already supported). Re 500–1000, H = 5.5, **80 nodes**.
2. Grade at each sample's own Re and H, through the standard pipeline. 24 runs.
3. **Declare and disentangle the confound.** The AoA10 shapes have 80 nodes, not
   96, so this is simultaneously a physical *and* a discretisation extrapolation.
   **D02 must run first**: it measures the pure node-count effect at `N = 80`
   in-distribution, so G02's degradation can be decomposed into
   `Δ_physical = Δ_total − Δ_node-count`. Report the decomposition, not just the
   total.
4. **Measure the asymmetry failure mode directly.** Compute the upper/lower Cp
   asymmetry of each target and of each generated shape. If the hypothesis holds,
   generated asymmetry should be near zero regardless of target asymmetry — a
   sharper and more legible statement than a degraded ρ.
5. Plot ρ against target asymmetry. If degradation scales with asymmetry, the
   mechanism is established.

## Measurements

| Quantity | Purpose |
|---|---|
| Mean ρ, `a` over 24 targets ± CI | The OOD result |
| **`Cl` generated vs target, per case** | **The primary metric here — see below** |
| `Cd` generated vs target | Secondary; drag is less discriminating on a lifting case |
| ρ vs target asymmetry | The mechanism |
| Generated vs target asymmetry | The predicted failure mode, measured directly |
| Node-count-corrected degradation (using D02) | Separates physics from discretisation |
| Validity rate | Whether OOD also breaks geometry |

### `Cl` is the right metric for this experiment

These are **lifting** cases at 10° incidence. `Cl_mean` is written to
`tmp/ellipse2d_force.csv` by every run and is currently read by nothing in the
repository — this experiment is where that changes.

The pre-registered failure mode is that the model produces a symmetric shape
approximating the mean of an asymmetric target. If that is what happens, then
**`Cl ≈ 0` on every generated shape while targets carry finite lift** — a single,
unambiguous number that states the failure far more sharply than a degraded
correlation coefficient. Report generated `Cl` against target `Cl` as a scatter; a
horizontal line at zero is the entire finding in one panel.

Grading a lifting case on pressure-coefficient correlation while never reporting
lift would be a conspicuous omission to any fluids reviewer, and the data is
already being produced.

## Decision rule

- Degradation `> 3σ` beyond the node-count effect → a genuine physical OOD failure,
  reported with its mechanism. This is C4 evidence.
- Degradation entirely explained by the node-count effect → the finding is about
  discretisation, not physics. Report accordingly and note that the physical OOD
  question remains open pending a matched-node-count OOD set.

## Consequence

This is the answer to the viva's *"reconsider the construction of out-of-distribution
samples"*. Real CFD data from a different simulation campaign, asymmetric and
lifting, entirely outside the training manifold — it cannot attract the objection
that the OOD set was constructed by the authors to be beatable, which the synthetic
PCA-perturbation ladder invites.

It has been sitting unused in `data/ellipse/raw/` since May. Twenty-four solver runs.

## Notes

Honest limitation to state: 24 samples is a small OOD set, and all at a single
incidence and channel height. It establishes *that* extrapolation fails and
plausibly *how*; it does not map the failure boundary. G03's ladder does that, in
the conditioning space where distance is measurable.

## Results

_Not yet run._
