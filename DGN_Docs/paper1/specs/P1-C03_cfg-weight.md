---
id: P1-C03
group: C — Conditioning
title: "Classifier-free guidance weight w"
tier: core
status: planned
depends_on: [P1-F05, P1-G01]
blocks: [P1-G04, P1-B04]
paper_section: "4.5"
contribution: C2
cost: {train_runs: 0, solver_runs: 24, effort: "~1 day"}
---

# P1-C03 — Classifier-free guidance weight w

## Question

Is `w = 3.0` justified — and does guidance *cause* the magnitude creep?

## Hypothesis *(pre-registered)*

Two predictions, and the second is the interesting one:

1. **Fidelity** ρ rises with `w` and plateaus around `w ≈ 3`, the standard
   guidance trade-off.
2. **Magnitude** — the affine slope `a` grows **monotonically** with `w`. Guidance
   sharpens the conditional distribution by pushing samples away from the
   unconditional mean, which for a shape field means thinner, more extreme
   geometry, which the solver reads as an inflated Cp magnitude.

If both hold, **classifier-free guidance is a mechanism for the magnitude creep**
reported in Year 1, and G04 has a candidate cause to test rather than an open
question.

## Falsifier

`a` is flat in `w`. Then guidance is not the creep mechanism, and G04 must look
elsewhere — the noise schedule, the clamp range, or a systematic bias in the
training objective.

## Method

Cheapest experiment in the suite that touches a headline finding: **no retraining**.
`guidance_scale` is a sampling-time parameter.

1. Sweep `sampling.guidance_scale ∈ {0, 1, 2, 3, 5, 7}` on the frozen P1-REF
   checkpoints.
2. 4 held-out targets × 6 weights = 24 solver runs.
3. Vary **sampling seed only** — the model is fixed. Record this explicitly in
   `run_meta.json` as a sampling-only repeat; it measures a smaller variance than a
   full seed repeat and must not be presented as one
   ([PROTOCOL §4](../PROTOCOL.md)).
4. Record **sample diversity** at each `w` for B04 — the fidelity–diversity
   trade-off is a genuine result, and this sweep is where it comes from.
5. Record geometric statistics at each `w`: mean radius, thickness, curvature
   extremes. If the creep mechanism is real, these should move monotonically with
   `w` in the direction that explains the `a` trend.

## Measurements

| Quantity | Purpose |
|---|---|
| ρ vs `w`, with CI | The fidelity trade-off; justifies or refutes `w = 3.0` |
| **`a` vs `w`** | **The creep hypothesis** |
| Sample diversity vs `w` | Feeds B04 |
| Thickness and curvature statistics vs `w` | The proposed geometric mechanism |

## Decision rule

- ρ plateaus at `w ≈ 3` → the incumbent value is justified; report the curve.
- `a` monotone increasing in `w` with `> 3σ` between endpoints → **guidance is
  implicated in the creep.** G04 then tests it directly by comparing `w = 0` and
  `w = 3` generated shapes against dataset shapes through the same solver.
- ρ best at a different `w` → report; do not retrofit into P1-REF.

## Consequence

- Justifies a hyperparameter that is currently unmotivated (viva B1.1).
- More importantly, this is where the Year-1 magnitude-creep finding may acquire a
  **mechanism**. "Generated shapes grade at `a ≈ 2.5–3.5` against a dataset value
  of 1.74" is an observation; "guidance weight drives magnitude, monotonically,
  and here is the geometric route" is a result. The difference is roughly a day of
  compute.

## Notes

Because no retraining is involved, this can be run the moment P1-REF exists — ahead
of its scheduled stage if the creep question becomes pressing.

## Results

_Not yet run._
