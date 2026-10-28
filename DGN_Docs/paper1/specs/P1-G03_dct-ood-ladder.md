---
id: P1-G03
group: G — Generalisation & failure modes
title: "DCT-space out-of-distribution ladder"
tier: core
status: planned
depends_on: [P1-F05, P1-B03, P1-F04]
blocks: []
paper_section: "4.6"
contribution: C4
cost: {train_runs: 0, solver_runs: 45, effort: "~3 days"}
---

# P1-G03 — DCT-space out-of-distribution ladder

> ⚠️ **Rebuilt 2026-08-03 by [[../results/P1-F01_data-partition]].** This spec
> planned to *synthesise* a DCT-distance ladder. The publisher provides **five real
> OOD sets** that already form one, measured in the conditioning space:
>
> ```
> indist 0.99×  →  highre 1.78×  →  lowre 2.39×  →  thin 10.8×  →  aoa10 12.9×  →  thick 33.3×
> ```
> *(median nearest-neighbour DCT distance to train, normalised by the within-train median)*
>
> Six rungs of real solved CFD over a 33× range. **The real ladder is now primary**
> and the synthetic one becomes an optional densification. Two consequences: the
> physical-realisability screen is unnecessary (every rung is a converged
> simulation), and the "you constructed your own OOD set" objection cannot be
> raised — these are the dataset publisher's sets, not ours.

## Question

How does performance decay with distance from the training distribution, measured
in the space the model is actually **conditioned** on?

## Hypothesis *(pre-registered, updated 2026-08-03 post-F01)*

Decay is **monotone in DCT distance across the six real rungs**, unlike the Year-1
PCA-σ ladder which was "noisier than monotonic". The reason is that PCA σ-units on
raw Cp curves are not isometric to the K=8 DCT conditioning space, so the Year-1
ladder was not a well-ordered distance axis at all — the non-monotonicity was an
artefact of the parameterisation, not a property of the model.

Second prediction, now testable with real data: **difficulty is anisotropic, and
geometry dominates.** F01 already shows thickness perturbations displace the
conditioning vector **5–14× further** than Reynolds perturbations do
(thin 10.8× / thick 33.3× versus lowre 2.39× / highre 1.78×). Predicted: ρ
degradation tracks DCT distance rather than the *physical* size of the
perturbation, so `thick` is the hardest rung and `highre` the easiest — and a
model conditioned only on Cp has no way to distinguish "far in Re" from "far in
thickness" except through that distance.

Third prediction: the **`lowre`/`highre` asymmetry** (2.39× vs 1.78×) means low
Reynolds is further out of distribution than high, despite both being one-sided
extensions of the same axis.

## Falsifier

Still non-monotone in DCT space. Then the non-monotonicity is a property of the
model or of the grading, not of the ladder's parameterisation — and the OOD decay
claim needs a different treatment or must be dropped.

## Method

### Primary — the real ladder

1. Six rungs, all real solved CFD: `indist`, `highre`, `lowre`, `thin`, `aoa10`,
   `thick`. Take **10 targets per rung** (all 24 for `aoa10`), 3 seeds each.
2. Grade every sample at its **own** Re and H through the standard pipeline.
3. Plot ρ and `a` against each rung's **measured** median DCT distance — a real
   distance axis, not a constructed one. Shade the F04 noise-floor band.
4. Report `Cd` per rung as well; degradation in an integrated force is the
   engineering statement of the same finding
   ([PROTOCOL §2.3](../PROTOCOL.md)).
5. **Report per-rung, never pooled.** The whole point is that the axes differ.
6. **Link to B03.** For each rung, report what fraction of the target is reachable
   by a 10-mode PCA basis fitted on train. This reframes the ladder as a test of
   C1 — *out of reach of a low-dimensional basis*, not just out of distribution —
   and F01 already suggests `thin`/`thick` will separate sharply from `lowre`/`highre`.

No realisability screen is needed: every rung is a converged simulation with a Cp
curve that a real body actually produced.

### Optional — synthetic densification

Six rungs is a coarse axis. If the real ladder proves monotone and more resolution
is wanted between `lowre` (2.4×) and `thin` (10.8×), interpolate synthetically in
DCT space along per-mode and random directions. **Only do this if the real ladder
justifies it** — synthetic targets reintroduce the realisability problem and the
"you built your own OOD set" objection, which is precisely what the real sets avoid.

## Measurements

| Quantity | Purpose |
|---|---|
| ρ, `a` vs `d`, per direction, with CI and the noise-floor band | **The §4.6 headline figure** |
| Monotonicity test against the noise floor | Whether the decay claim holds at all |
| Fraction of rungs passing the physical-realisability screen | Honesty about the synthetic ladder |
| PCA-basis reachability per rung | Links OOD to C1 |
| Decay rate by direction | The variation-in-difficulty finding |

## Decision rule

- Monotone in at least two of three directions, with endpoints separated by
  `> 3σ` → the OOD decay claim is established, now with a defensible distance axis.
- Direction-dependent decay rates → **report as a finding in its own right.** "OOD
  is not a scalar" is a more useful contribution than a single decay curve, and it
  is exactly what the panel asked to see examined.
- Non-monotone → report honestly; state that OOD behaviour in this conditioning
  space is not well-ordered by distance, and lean on G02's real physical OOD for the
  extrapolation claim instead.

## Consequence

Directly addresses two viva points at once — OOD construction with respect to the
conditioning space, and variation in difficulty across directions. It also
replaces the Year-1 PCA-σ ladder, whose axis was not commensurate with the space
the model sees.

G02 and G03 are complementary and both are needed: G02 is real but small and
uncontrolled; G03 is synthetic but controlled and dense. Neither alone supports the
extrapolation claim.

## Results

_Not yet run._
