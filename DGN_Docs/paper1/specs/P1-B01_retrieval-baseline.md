---
id: P1-B01
group: B — Basis necessity
title: "Nearest-neighbour retrieval baseline"
tier: gate
status: planned
depends_on: [P1-F01, P1-F02]
blocks: [AUGUST-GATE, P1-B04]
paper_section: "4.3"
contribution: C1
cost: {train_runs: 0, solver_runs: 35, effort: "~2 days"}
---

# P1-B01 — Nearest-neighbour retrieval baseline

## Question

Does generation beat returning the training shape whose Cp is closest to the
target?

## Hypothesis *(pre-registered)*

Retrieval is **strong in-distribution and fails out-of-distribution.** The design
space is effectively two-dimensional (Re, `b`) with 5,701 training samples, so for
any in-distribution target a near-identical training shape exists. Predicted:
retrieval within `2σ` of the model on held-out targets, and clearly worse on the
G03 OOD ladder — the generative advantage is off-manifold, not on it.

## Falsifier

Retrieval matches or beats the model **on the OOD ladder as well**. Then the model
offers no practical advantage over a lookup table anywhere in the tested regime,
and the paper's case reduces to a representation-learning argument with no
demonstrated utility. That is a serious finding and must be reported as such.

## Method

No training. This is the cheapest decisive experiment in the suite — run it in
week one.

1. Index every **training** shape by its K=8 DCT-Cp vector — the exact conditioning
   vector the model receives, so the comparison is like-for-like.
2. For each evaluation target (held-out from G01, and OOD from G02/G03), return
   `argmin` L2 distance in that space.
3. Push the retrieved shape through the **same** solver pipeline at the **target's**
   Re and H — not the retrieved sample's. This is the honest comparison: the
   retrieved geometry is being asked to perform at a condition it was not drawn
   from, exactly as the generated geometry is.
4. Report a **second, stronger variant**: retrieve the *k* = 5 nearest and return
   the one with the best solver-measured ρ. This is an oracle-ish upper bound on
   retrieval and makes the baseline hard to dismiss.
5. Also report the **distance to the nearest training neighbour** for every target.
   That number is the single best summary of how hard each target actually was.

## Measurements

| Quantity | Purpose |
|---|---|
| Retrieval ρ, `a` on held-out targets vs P1-REF | The in-distribution comparison |
| Retrieval ρ, `a` on OOD targets vs P1-REF | Where the advantage should appear |
| NN distance in DCT space per target | Difficulty axis for every later figure |
| ρ plotted against NN distance, model and retrieval overlaid | **The key figure of §4.3** |

The last row is the one that carries the argument. If the two curves converge at
small NN distance and diverge as it grows, the paper has shown exactly where
generation buys something — which is far more informative than a single win/lose
comparison.

## Decision rule

| Outcome | Verdict |
|---|---|
| Model > retrieval by `> 3σ` in-distribution | Strong, and honestly surprising given the sampling density — check for a bug before celebrating |
| Comparable in-distribution, model better OOD | **The expected and publishable result.** Frame the contribution as extrapolation, not interpolation |
| Retrieval ≥ model everywhere | Report it. Rescope the paper's claims to representation and evaluation methodology |

## Consequence

Whatever happens, **this figure belongs in the paper.** Including a baseline that
might beat you is the single clearest signal to a reviewer that the evaluation is
honest, and reviewer question 6 is asked in essentially every submission of this
kind. Answering it unprompted is worth more than the result itself.

## Results

_Not yet run._
