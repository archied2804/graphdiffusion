---
id: P1-A01
group: A — Architecture necessity
title: "No-edge control"
tier: gate
status: planned
depends_on: [P1-F05]
blocks: [AUGUST-GATE]
paper_section: "4.2"
contribution: C1
cost: {train_runs: 3, solver_runs: 12, effort: "~1 day"}
---

# P1-A01 — No-edge control

## Question

Does message passing contribute anything, or does a per-node MLP with the same
conditioning do just as well?

## Hypothesis *(pre-registered)*

The no-edge model is **worse**. Without neighbour information nothing couples
adjacent radii, so `r(θ)` should be visibly rougher and inverse-design fidelity
lower. Predicted: boundary roughness up by ≥2×, held-out ρ down by more than 3σ.

## Falsifier

No-edge matches the graph within `2σ` on ρ **and** produces comparable roughness.
That would mean message passing is decorative — the conditioning vector alone
carries the signal — and the architectural half of C1 is unsupported.

## Method

The cheapest of the three architecture baselines: `GraphNetworkBlock` already
makes each sub-model optional, so this is a config change plus a width rebalance.

1. Disable `EdgeModel` in every `GraphNetworkBlock`. `NodeModel` then sees only its
   own features and the global vector — a per-node MLP conditioned on `u`.
2. **Rebalance width** so parameter count lands within ±5% of P1-REF's 147,585.
   Removing the edge MLPs frees parameters; widen `hidden_dims` to spend them.
   Record both counts in the results file.
3. Identical data partition, conditioning, noise schedule, optimiser, epoch budget
   and seeds (0, 1, 2) as P1-REF.
4. Grade 4 held-out targets × 3 seeds through the F02/F03 pipeline.

## Measurements

| Quantity | Vs P1-REF |
|---|---|
| Parameter count | must match ±5% |
| Best val loss, mean ± CI | Δ |
| Held-out ρ and `a`, mean ± CI | Δ, tested against `2σ`/`3σ` |
| Boundary roughness | Δ — the mechanism, if the hypothesis holds |
| Sample validity rate | Δ |

## Decision rule

| Outcome | Verdict |
|---|---|
| Graph better by `> 3σ` on ρ | Message passing earns its place |
| Difference `< 2σ` | **No detectable benefit from message passing** — report as such |
| No-edge better by `> 3σ` | Message passing hurts; investigate before publishing either way |

## Consequence

- **Supported →** §4.2 has its first piece of evidence that graph structure
  matters. Necessary but not sufficient: A02 is the harder test, because a 1-D CNN
  also has neighbour coupling.
- **Falsified →** the architectural claim is in serious trouble before A02 even
  runs, and the paper reframes around the *representation* rather than the network.
  Cheap to discover: one day.

## Notes

Note the asymmetry between this and A02. A01 asks *"is any neighbour coupling
needed?"* — a low bar the graph should clear easily. A02 asks *"is a **graph** the
right way to do that coupling?"* — the bar that actually decides the framing. A01
passing means little on its own; A01 failing is decisive.

## Results

_Not yet run._
