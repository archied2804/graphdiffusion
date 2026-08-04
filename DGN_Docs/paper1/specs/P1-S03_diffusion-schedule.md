---
id: P1-S03
group: S — Scale & data efficiency
title: "Diffusion schedule justification"
tier: support
status: planned
depends_on: [P1-F05, P1-G01]
blocks: []
paper_section: "4.7, 3.2"
contribution: risk
cost: {train_runs: 6, solver_runs: 24, effort: "~2 days compute"}
---

# P1-S03 — Diffusion schedule justification

## Question

Are `T = 1000` and the cosine noise schedule justified, or inherited from EXP-022
and from Ho et al.?

## Hypothesis *(pre-registered)*

Cosine at `T = 1000` is best, but the margin over `T = 200` is **small and largely
cosmetic** — consistent with the Year-1 finding that the boundary wobble at
`T = 200`, which motivated the move to `T = 1000` in the first place, proved
inconsequential under independent solver grading.

Predicted: Δρ between `T = 200` and `T = 1000` within `2σ`, with a measurable
difference only in boundary roughness — a geometric property, not a physical one,
at 5× the sampling cost.

## Falsifier

`T = 1000` beats `T = 200` by `> 3σ` on solver-graded ρ. Then the finer schedule
buys real physical fidelity rather than visual smoothness, and the 5× sampling cost
is justified on performance grounds instead of aesthetic ones.

## Method

1. Grid: `T ∈ {200, 500, 1000}` × `schedule_type ∈ {linear, cosine}` = 6 runs,
   single seed each, with `(1000, cosine)` reusing P1-REF.
2. Grade 4 held-out targets per cell (24 solver runs).
3. Report **sampling wall-clock** per cell. The cost axis is half the argument:
   whether the schedule is justified depends on what it buys per unit of compute.
4. Report boundary roughness separately from ρ. The Year-1 motivation for `T = 1000`
   was visual smoothness; separating the geometric metric from the physical one is
   what makes the finding legible.
5. Also record `clamp_range` behaviour — how often samples hit the `[0.5, 2.0]`
   bounds at each `T`. A clamp that activates frequently is doing unacknowledged
   work in the generative process, and that is worth knowing about a setting the
   paper currently presents without motivation.

## Measurements

| Quantity | Purpose |
|---|---|
| Held-out ρ, `a` across the 6 cells | The comparison |
| Boundary roughness across cells | Where the schedule actually shows up |
| Sampling wall-clock per cell | The cost side of the trade-off |
| Clamp activation rate per cell | An unmotivated setting, made visible |

## Decision rule

- Within `2σ` between `T = 200` and `T = 1000` → **report that the finer schedule
  buys smoothness, not fidelity.** This is a small but genuinely useful result: it
  says the community's default `T = 1000` may be over-specified for smooth
  low-dimensional shape fields, and it explains the Year-1 §4.2 finding rather than
  simply cutting it.
- `T = 1000` better by `> 3σ` → the choice is justified on performance; report and
  move on.

## Consequence

Discharges the remaining hyperparameters flagged in
[[../publication/PUBLICATION_PLAN]] §5.7 as needing *"a sentence of motivation or a
citation"*. It also rescues something from the Year-1 roughness/T sweep, which is
otherwise slated for cutting — the conclusion survives as a paragraph, now
supported by held-out solver grading rather than by a single-seed radius-KS metric
on circles.

## Notes

The clamp-activation measurement is a small addition with disproportionate value.
`clamp_range: [0.5, 2.0]` currently appears in every config with no justification
anywhere. If it never activates it is harmless and can be described as a safety
bound; if it activates often it is shaping the output distribution and must be
described as part of the method. Either answer is a sentence in §3.2 that is
currently missing.

## Results

_Not yet run._
