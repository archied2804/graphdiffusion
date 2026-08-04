---
id: P1-S02
group: S — Scale & data efficiency
title: "Data-fraction curve"
tier: core
status: planned
depends_on: [P1-F05, P1-G01]
blocks: []
paper_section: "4.7"
contribution: risk, C4
cost: {train_runs: 6, solver_runs: 24, effort: "~2 days compute"}
---

# P1-S02 — Data-fraction curve

## Question

How much training data does this method actually need, and how does the answer
scale with shape complexity?

## Hypothesis *(pre-registered)*

Performance **saturates well before 100%** of the training partition. The design
space is effectively two-dimensional with 5,131 training samples, so the sampling
density is extremely high. Predicted: 25% of the data reaches within `2σ` of the
full-data result, and even 10% is close.

## Falsifier

Performance still improving between 50% and 100%. That would mean the method is
data-hungry even on a two-parameter family, which is a serious warning for
Paper 3's richer families and would demand a change of plan.

## Method

1. Train on `{2, 5, 10, 25, 50, 100}%` of the training partition. Subsets are
   **nested** (each contains the smaller) and **stratified over the design space**,
   not random — a random 2% of a two-parameter family may miss whole regions, which
   would confound sample count with coverage.
2. Val and test partitions are unchanged throughout. Six training runs, one seed
   each except 100% which reuses P1-REF's five.
3. Grade 4 held-out targets per fraction (24 solver runs).
4. **Report against two axes**: absolute sample count, and samples per unit volume
   of design space. The second is the quantity that transfers to a different shape
   family, and therefore the one Paper 3 needs.
5. Fit a saturation curve and report the **knee** — the fraction beyond which
   improvement is below the noise floor. That single number is the deliverable.

## Measurements

| Quantity | Purpose |
|---|---|
| Held-out ρ vs data fraction, with the noise-floor band | **The §4.7 headline figure** |
| Val loss vs data fraction | The cheaper proxy, and whether it agrees |
| Knee location, in samples and in samples per unit design volume | The transferable number |
| Diversity vs data fraction (from B04's metric) | Whether small-data models collapse as well as degrade |

## Decision rule

- Knee below 50% → the method is data-efficient on this family. State the number
  and use it as the basis for a Paper 3 extrapolation.
- Still improving at 100% → **flag as a programme risk.** Paper 3's richer families
  occupy a much larger design space; if 5,131 samples are insufficient for a
  two-parameter family, the data requirement for aerofoil families may be
  prohibitive, and the risk register in [[../publication/RESEARCH_PROGRAMME]] §5
  needs updating.

## Consequence

This experiment discharges **viva Part B3** directly. The panel asked for
*"reflection on risks (e.g. anticipated dependency between shape complexity and
training data size) and discussion of fall-back options."* This turns that
anticipated dependency into a measured curve with a knee, and the extrapolation to
richer families becomes an argument from data rather than an assertion.

It is the strongest single item in the suite for the *programme* rather than the
paper: the entire three-paper arc assumes that scaling to complex shape families is
feasible, and this is the only experiment that tests that assumption at Paper 1's
cost.

## Notes

The stratification detail matters more than it looks. A naive random 2% subset of a
two-parameter family will leave gaps in (Re, `b`), and the resulting degradation
would be attributed to sample count when it was really coverage. Stratify, and
state that you did.

## Results

_Not yet run._
