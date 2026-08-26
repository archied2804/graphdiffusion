---
id: P1-B01
spec: "../specs/P1-B01_retrieval-baseline.md"
title: "Nearest-neighbour retrieval baseline"
date_run: 2026-08-26
date_reported: 2026-08-26
status: partial
verdict: no verdict — comparison awaits P1-G01
git_sha: 7b212c3
---

# P1-B01 — results

> **Spec:** [[../specs/P1-B01_retrieval-baseline]] · **Verdict:** ⏸ **none
> assigned.** The pre-registered rule is a *comparison* against the model, and
> [[../specs/P1-G01_holdout-inverse-design|G01]] has not run. This half measures
> the baseline and fixes the difficulty axis.
>
> It measures something decisive anyway: **retrieval sits at the solver's own
> self-grading ceiling in-distribution**, leaving no measurable room for the
> model to beat it there.

## 1. What was run

35 solver runs. For each evaluation target the nearest **training** shape in K=8
DCT-Cp space — the exact vector the model is conditioned on — was run at the
**target's** Re and H, and graded against the **target's** dataset reference
curve. Zero dead probes across all 35.

The evaluation target set was drawn here and frozen: 20 in-distribution plus 3
from each OOD set, seed 42, recorded in `data/splits/paper1_eval_targets.json`.
See §6 — this was G01's job and G01 has not run.

## 2. Headline — retrieval is at the ceiling

| | ρ | vs σ_ρ |
|---|--:|--:|
| Chance level (unrelated dataset shapes, F04 §4) | 0.99540 | — |
| **Retrieval, held-out targets (median, n = 20)** | **0.99905** | — |
| F04 correct-pairing median — *the solver grading its own shape* | 0.99896 | **+0.09σ** |
| A perfect score | 1.00000 | **0.88σ away** |

**Retrieval scores as well as the solver grading the correct shape against
itself**, and sits within 1σ of a perfect grade. Whatever G01 measures for the
model, it cannot beat this in-distribution by a detectable margin — there is
less than one σ of headroom in the metric.

Median `a` = 0.9234, consistent with the reference scale
[[P1-F07_normalisation-audit|F07]] measured (0.9065 ± 0.0352).

### Why: the design space is saturated

| | DCT-space distance |
|---|--:|
| Within-train median nearest-neighbour | 0.001262 |
| **Held-out target to nearest training shape (median)** | **0.001514** |
| Ratio | **1.20×** |

A held-out target's nearest training neighbour is **20% further away than a
training sample's own nearest neighbour**. In the space the model is
conditioned on, the held-out set is not meaningfully outside the training set.

This is [[P1-B03_basis-reachability|B03]]'s one-dimensional manifold seen from
the other side: 5,701 samples on an effectively 1-D manifold leave no gaps to
generalise into.

## 3. Retrieval survives four of the five OOD sets

| set | median ρ | vs chance | NN distance |
|---|--:|--:|--:|
| `highre` | 0.99959 | **+3.88σ** | 2.7× within-train |
| `thin` | 0.99858 | **+2.94σ** | 15.2× |
| `thick` | 0.99842 | **+2.79σ** | 23.5× |
| `lowre` | 0.99705 | **+1.53σ** | 2.4× |
| **`aoa10`** | **0.96962** | **−23.87σ** | 15.6× |

Retrieval remains above chance on `thick` at **23.5× the within-train
neighbour distance**. Distance in DCT space is not, on its own, enough to defeat
a lookup table on this corpus.

**Only `aoa10` breaks it**, and it breaks it decisively — below chance by 24σ.
But `aoa10` confounds three things at once and cannot carry the argument alone:

1. B03 §4 showed the AoA10 contrast is a **rotation**, not a reach limitation.
2. It has **80 nodes, not 96** — [[../EXPERIMENT_REGISTER]] §3 flags this, which
   is why D02 must precede G02.
3. It is the only set with a different angle of attack, so it is also the only
   set where the flow physics differ.

## 4. Discrimination

Cross-grading each retrieved run against all 20 in-distribution targets:

| | rate | chance |
|---|--:|--:|
| top-1 | **35.0%** | 5.0% |
| top-3 | **85.0%** | 15.0% |

Seven times chance. Retrieval carries real information about *which* target it
was asked for — it is not merely producing plausible ellipse curves.

## 5. The difficulty axis is weaker than the spec hoped

The spec expected nearest-neighbour distance to be *"the single best summary of
how hard each target actually was"* and the x-axis of §4.3's key figure.

**corr(NN distance, ρ) = −0.31** across all 35 runs. The sign is right and the
relationship is real, but it is weak — and it is weak because **ρ has almost no
range to vary over** ([[P1-F04_noise-floor]] §4), not because the axis is wrong.

The axis should be kept and plotted against `Cd` error instead, once G01
supplies model runs at the same targets.

## 6. Protocol deviations

Dated, per [[../PROTOCOL]] §9.

- **2026-08-26** — **The evaluation target set was drawn by this experiment, not
  by G01.** G01's method step 1 specifies the draw and states that it fixes the
  target set for A04, C01–C03 and S01–S03 too. B01's key figure is meaningless
  unless both use the same targets, so the draw was made here with a recorded
  seed (42) and written to `data/splits/paper1_eval_targets.json`. G01 consumes
  it rather than re-drawing. No spec was edited.
- **2026-08-26** — The draw **excludes the 8 in-distribution test rows F01 found
  duplicated in training** (and the 4 in AoA10). A target already present in
  training measures memorisation, and retrieval would score perfectly on it by
  construction — flattering the baseline, not the model.
- **2026-08-26** — The spec's step 4, the *k* = 5 oracle variant, is **not run**.
  It costs 4 further runs per target (80 in-distribution) and it is an upper
  bound on a baseline that is already at the measurement ceiling (§2). Deferred
  as low-value rather than dropped; revisit only if G01 shows the model
  competitive.
- **2026-08-26** — Metrics follow [[../PROTOCOL]] §2.1 as amended by F04. ρ is
  reported and is corroboration only; rank/top-k is reported alongside. `Cd`
  error against the target cannot be computed yet — it needs the target
  geometries themselves through the solver, which G01 will produce.

## 7. What this changes

**For the paper.** §4.3 can state, with a number, that on pOnEllipse a
nearest-neighbour lookup over 5,701 training shapes reaches the grading
instrument's ceiling in-distribution and survives four of five OOD rungs. That
is a strong baseline and it belongs in the paper regardless of what G01 finds.

**For C1.** [[P1-B03_basis-reachability|B03]] falsified the premise
analytically and physically; B01 now shows the *practical* consequence — there
is no in-distribution headroom for any method to demonstrate an advantage. C1's
surviving route, G03, is narrower than it looked: retrieval already handles
`thin` and `thick`, two of the ladder's rungs, at up to 23.5× the within-train
distance.

**For G01.** Its target set is fixed and its comparator is measured. When it
runs, the comparison is a lookup.

**Not concluded.** Whether the model beats retrieval. That is the spec's actual
question and it needs G01.

## 8. Files

| | |
|---|---|
| Index and difficulty axis | `scripts/paper1/p1_b01_retrieval_index.py` |
| Targets and bodies | `scripts/paper1/p1_b01_make_targets.py` |
| Runner | `scripts/paper1/run_p1_b01_solver.sh` |
| Grader | `scripts/paper1/p1_b01_grade.py` |
| Frozen target set | `data/splits/paper1_eval_targets.json` |
| Data | `outputs/paper1/P1-B01_retrieval-baseline/retrieval_grading.json` |
