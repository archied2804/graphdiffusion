---
id: P1-G01
group: G — Generalisation & failure modes
title: "Held-out inverse design"
tier: core
status: planned
depends_on: [P1-F05, P1-F04, P1-F02]
blocks: [P1-A04, P1-C01, P1-C02, P1-C03, P1-S01, P1-S02, P1-S03]
paper_section: "4.6"
contribution: C1, C4
cost: {train_runs: 0, solver_runs: 100, effort: "~4 days"}
---

# P1-G01 — Held-out inverse design

> **The paper's headline result.** It replaces the Year-1 in-distribution claim,
> which was measured on training samples.

## Question

How accurately does the model produce shapes matching Cp targets it has never seen?

## Hypothesis *(pre-registered)*

**ρ ∈ [0.88, 0.94]**, mean over 20 held-out targets — lower than the ρ ≈ 0.96
reported in Year 1 on training targets, because that number benefited from
memorisation and from farthest-first target selection. Magnitude creep persists at
`a ≈ 2.5–3.5`.

## Falsifier

Mean held-out ρ below 0.80. That would mean the Year-1 result was substantially
memorisation, and the in-distribution contribution must be rebuilt rather than
re-measured.

## Method

1. **20 targets drawn uniformly at random** from the test partition, with a
   recorded selection seed. Farthest-first selection is forbidden here — it samples
   the extremes of the design space and does not estimate typical performance
   ([PROTOCOL §1](../PROTOCOL.md)).
2. 5 model seeds per target → 100 solver runs. Grade at each target's own Re and H.
3. Report mean ± 95% CI, **and the full distribution** — a histogram of per-target
   ρ. The mean alone conceals whether failures are rare and severe or uniform and
   mild, and those imply different things about the method.
4. **Report the Year-1 comparison honestly.** Additionally grade the two Year-1
   targets (#4849, #4066) with this model and label them explicitly as training
   samples. Placing the leaked and clean numbers side by side is the clearest
   possible statement that the issue was found and corrected.
5. Record each target's nearest-training-neighbour distance from B01, so ρ can be
   plotted against target difficulty rather than reported as a bare average.

## Measurements

| Quantity | Purpose |
|---|---|
| Mean held-out ρ, `a`, ε_aff ± 95% CI, n = 20 × 5 | **The headline numbers** |
| **`Cd` generated vs target, per target, with relative error** | **The engineering quantity.** A designer asks for drag, not pressure correlation. Free — already in `force.csv` |
| Chordwise-resolved residual: LE / suction peak / aft | Where the error lives; a single ρ conflates a nose error with a wake error |
| Suction-peak magnitude and location, target vs measured | The two numbers a designer actually reads |
| Per-target distribution of ρ | Failure structure |
| ρ vs NN distance to training set | Difficulty-resolved performance — the more informative figure |
| Train-target ρ (labelled) vs held-out ρ | The generalisation gap, stated plainly |
| Validity rate, roughness | Geometric quality |

## Decision rule

- ρ in the predicted range → the pre-registered prediction is confirmed and §4.6
  reports a defensible in-distribution result with a real CI.
- ρ above 0.94 → better than predicted; check the partition audit (F01) before
  believing it. An unexpectedly small generalisation gap on a two-parameter family
  is more likely a leak than a triumph.
- ρ below 0.80 → the memorisation falsifier. Escalate: the framing and the August
  gate decision both change.

## Consequence

- Closes the top submission blocker and viva point B1.3.
- Fixes the **evaluation target set** used by A04, C01, C02, C03, S01–S03, so every
  later comparison is against the same targets — which is why so much depends on
  this experiment.
- The train-vs-held-out gap is a genuine result in its own right. On a
  two-parameter family with 5,701 samples the gap should be small; if it is large,
  that is informative about the sample efficiency question S02 asks.

## Notes

100 solver runs, roughly 3–4 hours of compute. Run it as one batch through
`run_from_manifest.sh` with a manifest named `P1-G01_holdout`.

## Results

_Not yet run._
