---
id: P1-C01
group: C — Conditioning
title: "λ_pressure — DeepSets head necessity"
tier: core
status: planned
depends_on: [P1-F05, P1-G01]
blocks: []
paper_section: "4.5"
contribution: C2
cost: {train_runs: 8, solver_runs: 24, effort: "~2 days compute"}
---

# P1-C01 — λ_pressure: DeepSets head necessity

## Question

Does the auxiliary DeepSets pressure head contribute anything, or is the DCT
conditioning vector sufficient on its own?

## Hypothesis *(pre-registered)*

The head helps. It acts as a consistency regulariser — forcing the node embeddings
to retain enough information to reconstruct the pressure field, which should
improve inverse-design fidelity beyond what conditioning alone achieves. Predicted:
`λ = 0` degrades held-out ρ by more than `3σ` relative to `λ = 0.1`.

## Falsifier

`λ = 0` matches `λ = 0.1` within `2σ`. Then the head is architectural weight with
no measured benefit, and it should be **removed** — from the model and from the
paper's contribution list.

This is the single most important control in the C group. `λ = 0` is not a sweep
point; it is the null hypothesis for a component the method section currently
presents as a design contribution.

## Method

1. Sweep `model.lambda_pressure ∈ {0, 0.01, 0.1, 1.0}`. Nothing else changes.
2. **Seeds 0, 1, 2 at `λ = 0` and `λ = 0.1`** — the two points the decision rests
   on. Single seed at 0.01 and 1.0. Eight training runs.
3. At `λ = 0` the `PressurePredictionHead` still exists but contributes no gradient.
   Confirm and record that the score network's parameter count is unchanged across
   the sweep, so this is a pure loss-weighting comparison rather than a capacity
   comparison.
4. Grade 4 held-out targets × 3 seeds at the two decision points; single seed
   elsewhere. 24 solver runs.
5. Also report the head's **own** accuracy — how well it predicts Cp from the
   generated shape — at each `λ`. If the head is inaccurate, any benefit is
   regularisation rather than the physics-consistency mechanism claimed in §3.3,
   and the paper should say which.

## Measurements

| Quantity | Purpose |
|---|---|
| Held-out ρ, `a` vs `λ`, CI at the decision points | The ablation result |
| Best val loss vs `λ` | Whether val agrees with solver grading |
| Auxiliary head's own Cp prediction error vs `λ` | Distinguishes regularisation from physics consistency |
| Parameter count across the sweep | Confirms it is a loss-weight comparison |

## Decision rule

| Outcome | Action |
|---|---|
| `λ = 0.1` better than `λ = 0` by `> 3σ` | The head is justified; §3.3 keeps it and cites this |
| Within `2σ` | **Report that the head shows no detectable benefit.** Consider removing it — a simpler model that performs identically is a result, not a setback |
| `λ = 1.0` best | The auxiliary task is under-weighted; report, but do not retrofit into P1-REF ([PROTOCOL §1](../PROTOCOL.md)) |

## Consequence

Directly discharges reviewer question 7 and the viva's *"ablation studies focusing
on [the key contributions] in particular"*. The DeepSets head is presented in the
method section as part of the conditioning design; presenting a component as a
contribution without a `λ = 0` control is exactly what the panel was warning about.

If it turns out the head does nothing, deleting it makes the paper shorter, the
architecture cleaner, and the remaining claims stronger.

## Results

_Not yet run._
