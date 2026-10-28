---
id: P1-F04
group: F — Foundation & protocol
title: "Grading noise floor"
tier: gate
status: planned
depends_on: [P1-F02, P1-F03]
blocks: [P1-B03, P1-G01, P1-G04, "significance rule for all experiments"]
paper_section: "3.5, 4.1"
contribution: protocol
cost: {train_runs: 0, solver_runs: 18, effort: "~2 days"}
---

# P1-F04 — Grading noise floor

## Question

What is the measurement uncertainty of ρ and `a` attributable to the grading
apparatus alone, with the shape held fixed?

## Hypothesis *(pre-registered)*

Solver-side variability is an order of magnitude below the model-side effects the
paper reports: `σ_ρ < 0.005` and `σ_a < 0.05`. Under that, the Year-1 OOD decay
(ρ 0.92 → 0.86) is real and the DPS negative result is real.

## Falsifier

`σ_ρ` is comparable to the effects being claimed — concretely, `2σ_ρ > 0.03`,
which would swallow the OOD decay. In that case **no OOD claim survives** without
many more repeats, and G03 must be rescoped or dropped.

## Method

Hold the geometry fixed; vary only nuisance parameters that ought not to matter.

1. Choose **3 shapes** spanning the envelope: low-Re thin, mid, high-Re thick.
   Use dataset shapes, not generated ones — this measures the instrument, not the
   model.
2. For each shape, vary one nuisance factor at a time, ≥6 runs total per shape:

   | Factor | Levels | Why it should not matter |
   |---|---|---|
   | `RESOLUTION` | rule, rule ±1 step | Above the stability rule, results should be converged |
   | Node resampling phase | 3 rotations of the same polygon | Same body, different discretisation offset |
   | `AVG_WINDOW_CHORDS` | 15, 25, 35 | Time-averaging window |
   | `MAX_PHYS_T` | 40, 50, 60 | Statistical convergence |

3. Compute ρ and `a` for every run through the F02 module against the same target.
4. Pool into `σ_ρ` and `σ_a`. Report per-factor contributions so the dominant
   source is known.

## Measurements

| Quantity | Purpose |
|---|---|
| `σ_ρ`, `σ_a` per shape and pooled | **The numbers the whole paper is calibrated against** |
| **`σ_Cd`, `σ_Cl`** pooled | Noise floor for the integrated coefficients, which are now primary metrics ([PROTOCOL §2.3](../PROTOCOL.md)) |
| **`Cd_std` from `force.csv`** | A *free* within-run unsteadiness estimate, already time-averaged over a stated window. Compare it against the between-run `σ_Cd`: if within-run ≫ between-run, the averaging window is too short and `MAX_PHYS_T` must rise |
| Variance decomposition by factor | Which nuisance parameter dominates |
| The resulting `2σ` and `3σ` thresholds, per metric | Written into [PROTOCOL.md §6](../PROTOCOL.md) |

## Decision rule

The pooled `σ` values become the significance thresholds for every subsequent
experiment. From this point:

| Observed difference | Reported as |
|---|---|
| `< 2σ` | no detectable difference |
| `2σ – 3σ` | marginal |
| `> 3σ` | a result |

**Apply symmetrically.** This rule must be as able to erase a favourable result as
an unfavourable one — including erasing the graph's advantage in A01/A02.

## Companion analysis — the chance level of ρ

The noise floor gives the smallest *detectable* difference. It does not say what a
ρ value **means**, and for this dataset that is a live risk: all ellipse Cp curves
look broadly alike, so ρ may be near-saturated for any plausible shape.

Compute, with no additional solver runs:

| Quantity | From |
|---|---|
| Distribution of ρ between **random pairs of dataset samples** | Existing `cp.csv` and dataset curves |
| ρ between each target and a **random valid ellipse** | Same |
| ρ between each target and the **dataset mean** Cp curve | Same |

If random pairs correlate at ρ ≈ 0.85, then the meaningful axis is 0.85 → 1.0, not
0 → 1, and **every reported ρ in the paper must be positioned on that axis.** A
headline of 0.92 reads very differently against a chance level of 0.5 than against
one of 0.85.

**Do this before committing to ρ as the headline metric.** It is pure analysis, it
costs an afternoon, and it is the sharpest available attack on the paper's central
number — see [METRICS.md §2.5](../METRICS.md).

## Consequence

- **Supported →** every figure in the paper can carry a shaded noise-floor band,
  and §3.5 states the significance rule before any result is shown. This directly
  answers reviewer question 9 and viva point B1.3.
- **Falsified →** the OOD and creep claims shrink to what survives. Better to
  discover this in August with 18 solver runs than in review.
- **Protocol freezes when this reports.**

## Notes

This experiment does not exist in the Year-1 work and is the single cheapest thing
in the suite that raises the paper's defensibility. Eighteen solver runs, under an
hour of compute, and it converts every "noisier than monotonic" hedge into a
number.

## Results

_Not yet run._
