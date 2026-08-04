---
id: P1-D02
group: D — Discretisation independence
title: "Node-count transfer at sampling time"
tier: core
status: planned
depends_on: [P1-F05, P1-D01]
blocks: [P1-G02, P1-D03]
paper_section: "4.4"
contribution: C2
cost: {train_runs: 0, solver_runs: 60, effort: "~4 days"}
---

# P1-D02 — Node-count transfer at sampling time

> ⚠️ **Reframed 2026-08-03 by [[../results/P1-F01_data-partition]].** This spec was
> written on the belief that training used a fixed 96-node ring. It does not: the
> corpus spans **52–96 nodes across 12 distinct values** (the HDF5 array is padded
> to its 96-node maximum; only 37 of 5701 samples actually have 96). The model is
> therefore *already* trained across discretisations, so the question is no longer
> "does it transfer at all?" but **"does it extrapolate outside 52–96?"** — better
> posed, and harder. The hypothesis below is restated accordingly; the original is
> preserved in git history.
>
> Bonus: the `thick` evaluation set reaches **100 nodes**, above the training
> maximum, giving a free real-CFD extrapolation test.

## Question

The model is trained across 52–96 nodes. Does it (a) perform uniformly *within*
that range, and (b) extrapolate *outside* it — below 52 and above 96?

## Hypothesis *(pre-registered 2026-08-03, post-F01)*

**In range (52–96): flat.** Validity ≥ 95% and ρ within `2σ` of the reference
across the whole trained range. Anything else would mean the model is quietly
worse at node counts it has actually seen, which would itself be a finding.

**Out of range: asymmetric degradation.** Predicted acceptable at `N = 40`
(below range, coarser than trained) and worse at `N = 192` (far above). The
mechanism proposed is that the model has learned spatial frequency content
relative to *node spacing* rather than to arc length, so at high node counts the
same learned per-node variation appears as higher-frequency boundary roughness.

Predicted: roughness at `N = 192` at least 2× the in-range value; ρ degrading
monotonically above `N = 96`.

## Falsifier

**For the C2 claim:** ρ varying by more than `2σ` *within* 52–96. That would mean
"discretisation-native" fails even on the range the model was trained on, and the
claim must be dropped rather than merely bounded.

**For the extrapolation claim:** invalid shapes (self-intersection, collapse)
anywhere in 40–192. The claim then becomes a bounded one — "transfers across the
trained range" — with D03 as the registered fall-back for extending it.

Note the two are now separable, which the original single-hypothesis version could
not do.

## Method

1. Resample the template ring to `N ∈ {40, 52, 64, 80, 96, 128, 192}` — two rungs
   below the trained range, three inside it, two above. The angular
   parameterisation is uniform, so this is a change to the ring template and the
   `[cos θ, sin θ]` context channels only — no retraining, no architecture change.
2. **Report in-range and out-of-range separately.** They test different claims
   (§Hypothesis) and averaging them would hide the distinction.
3. 4 held-out targets × 7 node counts × 3 seeds, all graded.
4. **Free real-data check:** the `thick` evaluation set contains samples at 100
   nodes, above the training maximum. Grade those separately as a real-CFD
   extrapolation point that required no construction.
3. Grade at **matched physical solver resolution**, not matched node count — the
   solver's `RESOLUTION` follows the Re rule and must not co-vary with `N`, or the
   experiment confounds model behaviour with solver discretisation.
4. Report the geometry metrics *before* grading. If shapes are geometrically
   invalid, the Cp number is meaningless and should not be reported as though the
   comparison were like-for-like.

## Measurements

| Quantity | Purpose |
|---|---|
| Validity rate vs `N` | The first-order question — are these usable designs? |
| Held-out ρ, `a` vs `N`, with CI and the noise-floor band | **The §4.4 headline figure** |
| Boundary roughness vs `N` | The predicted mechanism |
| Power spectrum of `r(θ)` vs `N` | Tests the *mechanism* directly: is the learned frequency content fixed in node index or in arc length? |

The spectral measurement is what separates a finding from an anecdote. If the peak
frequency is constant in node-index units across `N`, the model has learned an
index-space prior — which is a precise, citable statement about what per-node
diffusion on a fixed ring actually learns, and it is the technical seed of Paper 3.

## Decision rule

| Outcome | Verdict |
|---|---|
| Validity ≥ 90% and ρ within `2σ` across `N ∈ [48, 192]` | **"Discretisation-native" is earned.** Strongest single result available to C2 |
| Valid but degrading fidelity | Report the working range explicitly — e.g. "transfers over 64–128 nodes"; a bounded claim is still a claim |
| Invalid shapes | The claim is dropped; run **D03** and report it as the mitigation |

## Consequence

Described in [[../publication/PUBLICATION_PLAN]] §5.1 as the **highest
value-per-hour item in the plan**, and that assessment holds: no training, four
days, and it converts C2 from an assertion into a measurement.

It is also a **dependency for Paper 3** — see [[../publication/RESEARCH_PROGRAMME]]
§2. If per-node generation does not survive a change of discretisation, "mesh-native"
is not reachable and the programme needs re-planning. Far better to know in
September 2026 than in 2028.

Finally, it is what makes **G02** interpretable: the AoA10 set has 80 nodes, so
without D02 any degradation there is confounded between physical extrapolation and
node-count extrapolation. Run this first.

## Results

_Not yet run._
