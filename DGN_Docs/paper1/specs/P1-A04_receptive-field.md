---
id: P1-A04
group: A — Architecture necessity
title: "Receptive field on aerodynamic data"
tier: core
status: planned
depends_on: [P1-F05, P1-G01]
blocks: []
paper_section: "4.2"
contribution: C1
cost: {train_runs: 6, solver_runs: 12, effort: "~1 day compute"}
---

# P1-A04 — Receptive field on aerodynamic data

## Question

Does `k = 6`, chosen on the circle benchmark in EXP-002, transfer to the
aerodynamic dataset — and what does the optimum say about how the model uses the
ring?

## Hypothesis *(pre-registered)*

The optimum shifts **upward**. The circle task was local (recover a perturbed
radius field); inverse design from a Cp target is global, because a change anywhere
on the surface changes pressure everywhere. Predicted: performance improves to
`k ≈ 8–12` and then saturates, with `k = 6` measurably but not catastrophically
sub-optimal.

## Falsifier

`k = 6` is optimal, or performance is flat across `k ∈ [2, 12]`. Flatness would be
the more interesting outcome: it would say the receptive field barely matters,
which — combined with A01 and A02 — points to the conditioning vector doing nearly
all the work and the ring topology contributing little.

## Method

1. Sweep `k_neighbors ∈ {1, 2, 4, 6, 8, 12}` in the P1-REF config; nothing else
   changes. Note that `k` alters the edge count and therefore the parameter count
   only marginally (edge MLPs are shared), so matching is not an issue here —
   confirm and state this.
2. Seeds 0, 1, 2 at the endpoints (`k = 1`, `k = 12`) and at the incumbent
   (`k = 6`); single seed for the interior points.
3. Grade 4 held-out targets per setting.
4. Report the **effective receptive field** (`k × n_layers`, in nodes and as a
   fraction of the ring) alongside `k` — that is the quantity that transfers to
   A02's CNN comparison and to D02's node-count transfer.

## Measurements

| Quantity | Purpose |
|---|---|
| Held-out ρ and `a` vs `k`, with CI at the seeded points | The sweep result |
| Best val loss vs `k` | Whether val and solver grading agree on the optimum |
| Effective receptive field as a fraction of the ring | The transferable quantity |
| Wall-clock vs `k` | Cost of the larger neighbourhood |

## Decision rule

- If a `k` other than 6 wins by `> 3σ`, **do not retrofit it into P1-REF.** Report
  the sweep, state that the reference used `k = 6` inherited from the circle study,
  and quantify what that cost. Silently re-running everything at the new optimum
  after seeing test numbers would contaminate the entire suite
  ([PROTOCOL §1](../PROTOCOL.md)).
- If flat within `2σ` across the range, report flatness — and connect it to A01/A02
  in §4.2's discussion.

## Consequence

- Converts an inherited hyperparameter into a measured one — viva point B1.1,
  *"justification and motivation of methodological choices"*.
- Under the reframing this is no longer an incidental setting: receptive field is a
  claim about how much of the boundary the model must see at once to satisfy a
  global pressure target. That makes it contribution-relevant, which is why the
  Year-1 circle-only version (report §4.1) is being redone on aero data rather than
  simply cut.

## Notes

Scheduled after G01 so that the held-out grading pipeline is already exercised and
the target set is fixed. It is otherwise independent and could run any time after
F05.

## Results

_Not yet run._
