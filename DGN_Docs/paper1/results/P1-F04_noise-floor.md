---
id: P1-F04
spec: "../specs/P1-F04_noise-floor.md"
title: "Grading noise floor"
date_run: 2026-08-08
date_reported: 2026-08-08
status: partial
verdict: falsified (companion analysis)
git_sha: 1957f10
---

# P1-F04 — results

> **Spec:** [[../specs/P1-F04_noise-floor]] · **Status:** 🟡 nuisance sweep
> running (27 runs). **The companion analysis is complete and is the most
> consequential result in the suite so far.**
>
> ❌ **ρ has almost no discriminative power on this dataset.** Grading a solver
> run against the **wrong** shape's target gives a median ρ of 0.9526 against
> 0.9569 for the **correct** target — a signal of **0.0043** — and 44.8% of
> wrong pairings beat the median correct pairing.

## 1. The companion analysis — chance level of ρ

The spec required this *"before committing to ρ as the headline metric"*,
calling it *"the sharpest available attack on the paper's central number"*. It
is.

### 1a. Dataset-versus-dataset

300 training curves, 4,000 random pairs, graded with the production
`cp_agreement`:

| reference | median ρ | p5 | p95 | min |
|---|--:|--:|--:|--:|
| random **pair** of dataset shapes | **0.9954** | 0.9805 | 0.9995 | 0.9684 |
| the corpus **mean** curve | 0.9969 | 0.9932 | 0.9993 | 0.9899 |
| the most **extreme** curve found | 0.9926 | 0.9701 | 0.9981 | 0.9673 |

**Two unrelated dataset shapes correlate at ρ = 0.995.** Every Cp curve in this
corpus looks alike to Pearson correlation. A predictor that ignored its input
entirely and always returned the corpus mean would score **0.9969**.

### 1b. Solver-versus-dataset — the matched comparison

§1a is not directly comparable to the paper's numbers: it compares two dataset
curves, whereas every graded result compares a *solver* curve against a dataset
curve and therefore also carries the solver's own discrepancy. Cross-grading the
15 F03 envelope runs puts the chance level in the paper's own units:

| pairing | median ρ | range |
|---|--:|---|
| solver *i* vs **its own** reference | **0.9569** | 0.8649 – 0.9996 |
| solver *i* vs **another shape's** reference | **0.9526** | 0.8538 – 0.9997 |

| | |
|---|---|
| Discriminative signal | **+0.0043** |
| Wrong pairings beating the median correct pairing | **44.8%** |

**The solver's own reproduction error (≈0.04 in ρ) is an order of magnitude
larger than the shape-discrimination signal (0.004).** The instrument's noise
exceeds the quantity being measured.

## 2. What this means

**ρ measures "is this a plausible ellipse Cp curve", not "is this the *correct*
ellipse Cp curve".** A reported ρ of 0.95 is therefore not evidence that a
generated shape hit its target.

This is the falsifier in [[../specs/P1-F04_noise-floor]] firing in its strongest
form. The spec anticipated `2σ_ρ > 0.03` swallowing the OOD decay; the
discrimination signal is 0.0043, far smaller.

**Scope of the claim.** The chance level is relative to *this corpus*. The 15
envelope shapes span `b` 0.26–0.37 and Re 500–1000, so a "wrong" pairing is
still a fairly similar ellipse. The correct statement is that **within the
design space this dataset spans**, ρ cannot distinguish targets — which is
exactly the regime every Paper 1 claim operates in. It is not a general claim
about ρ as a metric.

**This does not invalidate the solver work.** [[P1-F03_solver-qualification]]
and [[P1-F06_schafer-turek]] used ρ to assess *whether the solver reproduces a
known curve*, which is a fit-quality question, not a discrimination question.
Those results stand.

## 3. What it changes

**ρ cannot be the headline metric.** [[../PROTOCOL]] §2.1 and
[[../REPORT_OUTLINE]] both put it there; both need revising before any G-group
experiment reports.

Candidate replacements, in order of promise:

| Metric | Status |
|---|---|
| **Rank / top-k**: does a generated shape match its own target better than the other targets? | The natural fix — it *is* the cross-grading above, turned into a metric. Discriminative by construction |
| **Integrated `Cd`** | [[P1-F03_solver-qualification]] §7 found it the most numerically stable quantity, ~1% under refinement against 2–11% for `a` |
| **Chordwise band residuals** | Already in `cp_metrics`; localises error instead of averaging it away |
| Affine slope `a` | Grid-dependent (11–26%), so weak on its own |

**Immediate consequence for [[../specs/P1-B01_retrieval-baseline]]:** a retrieval
baseline that returns *any* plausible training shape will score ρ ≈ 0.95. B01
was already the paper's honesty check; it is now near-certain to match the model
on ρ, and the interesting comparison moves to whichever discriminative metric
replaces it.

## 4. Status of the main sweep

27 runs (3 shapes × 9 configurations) varying `RESOLUTION`, node phase,
`AVG_WINDOW_CHORDS` and `MAX_PHYS_T`. Running. `σ_ρ`, `σ_a`, `σ_Cd`, `σ_Cl` and
the per-factor variance decomposition are reported when it completes, and the
2σ/3σ thresholds go into [[../PROTOCOL]] §6.

## 5. Reproduction

| | |
|---|---|
| Companion analysis | `scripts/paper1/p1_f04_chance_level.py` |
| Cross-grading | 15 F03 envelope runs, `cp_metrics.cp_agreement(mode="collapsed")` |
| Output | `outputs/paper1/P1-F04_noise-floor/chance_level.json` |
| Nuisance sweep | `scripts/paper1/run_p1_f04.sh`, variants from `p1_f04_make_variants.py` |
