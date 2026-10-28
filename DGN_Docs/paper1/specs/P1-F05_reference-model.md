---
id: P1-F05
group: F — Foundation & protocol
title: "Reference model under the clean protocol"
tier: gate
status: planned
depends_on: [P1-F01, P1-F02]
blocks: [P1-A01, P1-A02, P1-A03, P1-A04, P1-B02, P1-B04, P1-C01, P1-C02, P1-C03, P1-C04, P1-D02, P1-G01, P1-G02, P1-G03, P1-S01, P1-S02, P1-S03]
paper_section: "3, 4.1"
contribution: protocol
cost: {train_runs: 5, solver_runs: 0, effort: "~1 day compute, 5 × 85 min"}
---

# P1-F05 — Reference model under the clean protocol

## Question

Does the EXP-022 recipe reproduce when retrained on a clean partition, and what is
its true held-out performance?

## Hypothesis *(pre-registered)*

Validation loss reproduces the EXP-022 result within seed noise. Held-out
inverse-design fidelity is **lower** than the ρ ≈ 0.96 reported in Year 1 on
training targets — the pre-registered prediction is **ρ ∈ [0.88, 0.94]**, and
magnitude creep persists at `a ≈ 2.5–3.5`.

Stating the predicted drop in advance is deliberate. If ρ comes in at 0.93 it is
then a confirmed expectation rather than a number rationalised after the fact.

## Falsifier

Held-out ρ below 0.80. That would mean the Year-1 result was largely memorisation
of training targets, and the in-distribution contribution has to be rebuilt rather
than re-measured.

## Method

1. Config `configs/paper1/P1-F05_reference.yaml`, forked verbatim from
   `configs/EXP-022_T1000.yaml` with only these changes:
   - partition loaded from `data/splits/paper1_partition.json` (F01)
   - explicit `train_seed` / `sample_seed`
   - `run_meta.json` emission per [PROTOCOL §7](../PROTOCOL.md)
2. Train **5 seeds** (0–4), 1000 epochs, cosine annealing, early stopping
   patience 100. ~85 min each.
3. Freeze the resulting checkpoints as **`P1-REF`**. Every A/B/C/D/G/S experiment
   is a delta from this and must reuse these seeds.
4. Record the parameter count (expected 147,585) — it is the budget every matched
   baseline must hit within ±5%.
5. Grading is **not** done here. G01 grades P1-REF on held-out targets; keeping
   that separate stops the reference experiment from quietly becoming the headline
   result before the baselines have reported.

## Measurements

| Quantity | Purpose |
|---|---|
| Train/val loss curves, 5 seeds | Reproduction check vs EXP-022 |
| Best val loss, mean ± CI | The seed-noise scale for all model comparisons |
| Parameter count | The matched-baseline budget |
| Sample validity rate, boundary roughness | Geometric sanity before any grading |
| Wall-clock per run | Calibrates the suite's cost estimate |

## Decision rule

- **Pass:** val loss within the seed CI of EXP-022's, validity rate ≥ 95% →
  P1-REF is adopted as the reference.
- **Fail:** if the clean partition materially changes val loss, investigate before
  proceeding — it would suggest the previous split was doing something unintended.
  Do not proceed to Stage 2 with an unexplained reference model.

## Consequence

- **Supported →** the paper has one reference model, five seeds, defined before
  any comparison. Everything downstream is a difference against it with a real CI.
- The **seed spread measured here is the model-side noise scale**, complementing
  F04's instrument-side noise floor. Report both: a difference must clear both to
  count.

## Notes

Resist the temptation to tune. Any hyperparameter change made after seeing a test
number contaminates the experiment ([PROTOCOL §1](../PROTOCOL.md)). Tuning happens
on val, or in the C and S groups where it is the registered subject.

## Results

_Not yet run._
