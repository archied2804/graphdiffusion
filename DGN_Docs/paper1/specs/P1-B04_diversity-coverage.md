---
id: P1-B04
group: B — Basis necessity
title: "Diversity and design-space coverage"
tier: core
status: planned
depends_on: [P1-F05, P1-B01]
blocks: []
paper_section: "4.3"
contribution: C1
cost: {train_runs: 0, solver_runs: 40, effort: "~3 days"}
---

# P1-B04 — Diversity and design-space coverage

## Question

Is this design-space *exploration*, or one-to-one regression with noise?

## Hypothesis *(pre-registered)*

At a fixed Cp target with different sampling seeds, the model produces a
**non-degenerate family**: geometrically distinguishable shapes whose solver-graded
Cp curves are near-identical. This is the one-to-many fibre the method section
argues for — several designs meeting the same performance specification.

Predicted: pairwise geometric distance among samples at a fixed target is at least
5× the geometric resolution of the discretisation, while the spread in
solver-measured ρ is within `2σ_ρ`.

## Falsifier

Samples collapse — pairwise geometric spread indistinguishable from numerical
noise. Then conditioning has made the model effectively deterministic, "exploration"
must be removed from the title and the abstract, and the paper describes a
conditional regressor with a diffusion training objective.

## Method

1. 8 held-out targets. For each, draw **50 samples** from P1-REF varying only the
   sampling seed.
2. Geometric diversity, per target:
   - mean pairwise L2 distance between `r(θ)` vectors, normalised by mean radius
   - spread in shape-PCA coefficient space (B03's basis) — shows *which* modes the
     model explores and which it holds fixed
   - the envelope plot: all 50 boundaries overlaid
3. **Performance diversity.** Grade 5 of the 50 samples per target (40 solver runs).
   The claim being tested is that geometric spread does **not** come with
   performance spread — many designs, one performance.
4. **Coverage.** Over all held-out targets, compare the region of shape-PCA space
   the model reaches against the region occupied by the training set, and against
   what B01's retrieval can reach (zero diversity: one target, one shape).
5. Report diversity **as a function of CFG weight `w`** using the C03 runs.
   Guidance trades diversity for fidelity, and that trade-off curve is a genuine
   result rather than a caveat.

## Measurements

| Quantity | Purpose |
|---|---|
| Mean pairwise geometric distance per target, with CI | The diversity number |
| **Spread of solver `Cd` across samples at fixed target** | The claim in engineering terms: many geometries, one drag |
| Spread of solver ρ across samples at fixed target | Tests "same performance, different shape" |
| **Precision / recall and MMD vs the real shape distribution** | Standard generative-model evaluation — see below |
| **Unconditional sample quality** (conditioning dropped) | Did the model learn the distribution at all? |
| PCA-space coverage: model vs training data vs retrieval | The coverage figure |
| Diversity vs `w` (from C03) | The fidelity–diversity trade-off |

### Distributional evaluation

Everything else in this suite measures **per-sample conditional fidelity** — which
evaluates a regressor. This is a generative model going to an ML venue, and
reviewers there will expect the output *distribution* to be evaluated, using the
established vocabulary: precision/recall for generative models
(Sajjadi et al. 2018; Kynkäänniemi et al. 2019), or MMD. An ad-hoc pairwise-distance
diversity measure will read as improvised.

**Unconditional sample quality is currently unmeasured on the aerodynamic data
entirely.** The circle series used a radius KS statistic; there is no aero
analogue. Sampling with conditioning dropped (`p_uncond` already supports this)
and comparing against the real shape distribution separates two claims the paper
currently conflates: *the model learned the shape distribution*, and *conditioning
steers it correctly*. Only the second is tested anywhere else in the suite.

This is also the honest way to describe **coverage**: recall against the real data
manifold is a defined quantity, whereas "the model explores the design space" is
currently a figure of speech.

## Decision rule

- **Geometric spread large, performance spread `< 2σ`** → "exploration" is earned
  and quantified. This is the result that justifies the word in the title.
- Geometric spread large **and** performance spread large → the model is not
  reliably hitting the target; that is a fidelity problem, and it reframes G01's
  mean ρ as concealing high variance. Check this before writing §4.6.
- Spread indistinguishable from noise → drop "exploration" from the title and say
  why in §5.

## Consequence

The paper's title and framing promise design-space exploration; nothing in the
Year-1 work measures it. This closes reviewer question 10 and is the only
experiment in the suite that tests the *one-to-many* property the method section
claims as a motivation for using a generative model at all.

## Notes

Scheduled in Stage 5 because it depends on C03 for the diversity–guidance
trade-off, but the geometric half needs no solver time and can be run as soon as
P1-REF exists. Doing that early is worth it: if samples collapse, several
downstream framings change.

## Results

_Not yet run._
