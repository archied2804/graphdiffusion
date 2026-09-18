---
id: P1-G01
spec: "../specs/P1-G01_holdout-inverse-design.md"
title: "Held-out inverse design"
date_run: 2026-08-28
date_reported: 2026-08-28
status: complete
verdict: above prediction — but the prediction was in the wrong units
git_sha: ab5a3ca
---

# P1-G01 — results

> **Spec:** [[../specs/P1-G01_holdout-inverse-design]] · **The paper's headline
> experiment**, replacing the Year-1 claim that was measured on training samples.
>
> **Registered verdict: ABOVE PREDICTION.** Mean held-out ρ = 0.99418 against a
> predicted 0.88–0.94. That is not a success. **The chance level is 0.9954**, so
> the same number sits *below* what an unrelated dataset shape scores — and
> against retrieval on identical targets the model loses at **4.3σ**.
>
> Four findings sit alongside it: the **conditioning demonstrably works**
> (§4a, shape-to-request correlation 0.97), **target drag is matched inside the
> noise floor** (§4b, 1.08σ), **no memorisation**, and the Year-1 magnitude creep
> is **gone**.

## 1. Verdict against the pre-registered rule

| | |
|---|---|
| Hypothesis | ρ ∈ [0.88, 0.94] |
| Falsifier | ρ < 0.80 |
| **Measured** | **0.99418 ± 0.00156** |
| Registered branch | *"ρ above 0.94 → better than predicted; check the partition audit (F01) before believing it."* |

The audit was done. [[P1-F01_data-partition|F01]]'s 8 leaked in-distribution rows
were **excluded from the draw** ([[P1-B01_retrieval-baseline]] §6), so this is not
a leak.

**The prediction was in the wrong units.** It was calibrated against a Year-1
ρ ≈ 0.96 produced by one of the four disagreeing graders
[[P1-F02_grading-metric|F02]] later consolidated, and written before
[[P1-F04_noise-floor|F04]] measured what ρ means on this corpus. The registered
*success* range, 0.88–0.94, sits far **below** the 0.9954 chance level. A rule
that returns "better than predicted" for a number worse than chance is not a
rule that can be believed, and this is the clearest available demonstration of
why F04 demoted ρ.

The rule is reported as written. It is not reinterpreted, and the spec is not
edited.

## 2. Headline numbers

n = 100 (20 held-out targets × 5 `P1-REF` seeds). 110/110 runs completed, **zero
invalid geometry**, zero dead probes.

| quantity | mean | 95% CI | median |
|---|--:|--:|--:|
| ρ | 0.99418 | ± 0.00156 | 0.99717 |
| `a` | 0.8810 | ± 0.0087 | 0.8865 |
| ε_affine | 0.0394 | ± 0.0057 | 0.0282 |
| **`Cd` relative error** | **−5.12%** | **± 1.08pp** | −5.43% |

**ρ against its chance level: −0.00122, or −1.1σ.** The mean is below chance.
**Six of the twenty targets** score below what an unrelated dataset shape would.

## 3. The gate question — retrieval wins

[[P1-B01_retrieval-baseline|B01]] ran nearest-neighbour retrieval on these exact
targets, so this is a **paired** design and must be tested as one.

| | |
|---|--:|
| Model, median ρ | 0.99717 |
| Retrieval, median ρ | 0.99905 |
| **Paired mean difference** | **−0.00462** |
| In noise-floor units | **−4.28σ** |
| Paired *t* | **−3.05** on 19 dof |
| Targets where the model wins | **2 / 20** |

> **Generation does not beat a lookup table on this corpus. It loses to one, on
> 18 of 20 held-out targets, by a margin that clears the noise floor.**

**The test matters.** Comparing medians unpaired gives −1.7σ, which the 2σ rule
would report as "no detectable difference". That is the wrong test for a matched
design and it flatters the model; the paired result is the one that stands. Both
are printed by the grader so the difference is on the record.

Against B01's decision table this is the third branch — *"Retrieval ≥ model →
report it, rescope the paper's claims to representation and evaluation
methodology"* — for the **in-distribution** half. The second branch,
*"comparable in-distribution, model better OOD"*, which the spec called "the
expected and publishable result", is now **foreclosed on its first clause**. The
model is not comparable in-distribution; it is worse. Only the OOD half remains,
and that is G02/G03.

## 4. Two findings that go the other way

### No memorisation — the generalisation gap is *negative*

Spec step 4 required the two Year-1 targets to be graded with this model and
labelled as **training samples**:

| | ρ |
|---|--:|
| Year-1 targets (#4849, #4066) — **TRAINING SAMPLES** | 0.98402 |
| Held-out targets | 0.99418 |
| **Gap** | **−0.01016** |

The model does **worse** on the two samples it trained on than on targets it has
never seen. Whatever is limiting this model, it is not memorisation — which was
the specific worry [[P1-F01_data-partition|F01]] raised and the falsifier this
spec was built around.

### The magnitude creep is gone

| | `a` |
|---|--:|
| Generated shapes (this experiment) | 0.8810 ± 0.0124 |
| Dataset shapes ([[P1-F03_solver-qualification|F03]] envelope) | 0.9065 ± 0.0331 |
| Ratio | **0.972** |
| Difference | −0.0255 = **−0.73σ_a** |

**A creep of −2.8%, well inside the noise floor.** Year-1 reported a magnitude
excess of ~1.4× ([[../specs/P1-G04_creep-attribution|G04]] §1). It is not
reproducible under this protocol.

This is direct evidence for the export-scale defect recorded in
[[../PROTOCOL]] §2.1b: every Year-1 generated body entered the channel 2.1–2.8×
oversized, and a confined oversized body deepens its own suction peak. **G04's
central quantity may not exist**, and its spec should be revisited before it runs.

## 4a. The conditioning works — added 2026-09-14

§2 notes that the model's median ρ sits only **0.26σ above a corpus-mean
predictor**, one that ignores its input entirely. Read alone that invites the
conclusion that the model is barely conditioned at all. **It is not true**, and
the test that settles it uses data this experiment already produced.

Every generated body was resampled to a common 128-point `r(θ)/mean(r)` profile,
then compared two ways:

| | |
|---|--:|
| within-target spread — same target, 5 seeds | 0.10825 |
| between-target spread — across 20 targets | 0.30736 |
| **ratio** | **2.84×** |

A model ignoring its conditioning would give a ratio near 1. And across all 190
target pairs:

> **corr(target-Cp distance, generated-shape distance) = +0.9665**

Ask for two different pressure curves and you get two correspondingly different
shapes, with a correlation of 0.97. Figure:
`outputs/paper1/figures/fig5_conditioning_response.{png,pdf}`. **The model is strongly and correctly steered
by its conditioning**, and C2's mechanism works end to end.

**The 0.26σ figure was an artefact of ρ's compression, not evidence of a
constant predictor.** It is precisely the failure mode
[[P1-F04_noise-floor]] §4 warned about: every ellipse Cp curve looks alike, so ρ
cannot see a difference the geometry shows at ρ = 0.97 in its own units. This is
the strongest single argument in the suite for reporting ρ only against its
chance level, and it is now a worked example rather than an assertion.

**What this does and does not establish.** The model responds to the request
correctly. It does **not** follow that the shape it returns is the most accurate
available — retrieval is closer on both ρ (§3) and `Cd` (§4b). Those are separate
claims and the paper must keep them separate.

## 4b. `Cd` — the same verdict in the metric F04 elevated — added 2026-09-14

ρ was demoted by F04, so the comparison in §3 must be repeated in `Cd`. It needs
the target geometries' own solver runs, which §6 added.

| | mean | RMS | vs σ_Cd |
|---|--:|--:|--:|
| model | −5.12% | 7.10% | 1.50σ |
| retrieval | +0.14% | 1.87% | 0.39σ |

Paired: the model's absolute error is **4.6pp larger**, *t* = **+4.74** on 19
dof, retrieval closer on **15 of 20** targets. **Switching to the better metric
does not change the answer** — it strengthens it (*t* = 4.74 against ρ's 3.05).

Two readings must be kept apart:

1. **The model hits target drag within the noise floor.** Mean error −5.12% is
   **1.08σ_Cd**, and the suite's rule — applied symmetrically — calls anything
   under 2σ no detectable difference. On its own terms, the architecture performs
   the task.
2. **Retrieval is more accurate.** But retrieval reaching 1.3% mean drag error is
   a statement about **pOnEllipse**, not about the architecture's ceiling: with
   5,701 samples on a one-dimensional manifold and held-out targets only 1.20×
   further from training data than training points are from each other
   ([[P1-B01_retrieval-baseline]] §2), a lookup is near-optimal by construction.

**A saturated corpus cannot discriminate between methods that both reach its
ceiling.** That is a limitation of the evaluation set, and it belongs in §5 as
such rather than being reported as a property of the model.

## 4c. Template bias — re-attribution from D02, added 2026-09-17

[[P1-D02_node-count-transfer]] §3 found that the model takes shape information
from the sampling template's node spacing. **Every sample in this experiment was
generated on `dataset[0]`'s template, whose spacing encodes an aspect ratio of
0.511** — a thin ellipse.

Across the twenty targets, generated aspect regressed on true aspect has slope
**0.620**: the model tracks the target (correlation 0.933) but cannot produce
anything thicker than 0.686 against a truth reaching 0.784.

- **The measured numbers in §§2–4b stand.** They are what this pipeline produced.
- **Their attribution changes.** The "median `b/a` 0.538, skews thin"
  observation is the template. The two worst targets, `t039` and `t008`, are both
  thick and both under-produced by ~0.10 in aspect. Part of the drag error and
  part of the loss to retrieval are template-induced, and fixable at sampling
  time.
- Retrieval (§3) is unaffected: it returns stored geometry and has no template.

A matched-template re-sample is proposed in D02 §7 to measure how much.

## 4d. Surface-pressure comparison with no fitted offset — added 2026-09-18

`outputs/paper1/figures/fig11_cp_holdout.{png,pdf}` (six targets, ranked by
|ΔCd| and taken at ranks 0, 4, 8, 11, 15, 19) and `fig12_cp_holdout_all` (all
twenty). Each panel carries three curves, and **no curve is scaled by a fitted
constant**:

- **target geometry, same solver** — the like-for-like reference for the model
- **generated** — mean of the five `P1-REF` seeds, min–max band
- **dataset** — stored `p/ρU²` × 2, the sourced unit conversion of
  [[P1-F07_normalisation-audit]], replacing the Year-1 fitted 1.74

What they show:

- **Leading edge and stagnation region match on every target.** Error lives in
  the suction peak (x/c ≈ 0.25–0.5) and the aft body.
- **The worst targets are the thick ones** (#040, #039, #028, #008, #022), where
  the model under-produces suction — the same thin bias [[P1-D02_node-count-transfer]]
  §3 traced to the sampling template.
- **Dataset against target-geometry is F07's 9.35% residual, now visible**: the
  dataset curve carries stronger suction than the same body through this solver on
  most targets. It is a solver-to-solver difference, not a model error, and it is
  why the model must be judged against the black curve, not the blue.
- The seed band widens at the leading-edge shoulder (x/c ≈ 0.1–0.15) on thick
  targets — the per-node jitter of D02 §5 reaching the pressure field.

## 5. Failure structure — rare and severe, not uniform and mild

The spec asks for this distinction explicitly, because the two imply different
things about the method.

| | per-target ρ |
|---|--:|
| min (`t039`) | 0.97478 |
| 25th percentile | 0.99482 |
| median | 0.99719 |
| max (`t020`) | 0.99868 |

Fourteen of twenty targets sit above 0.9948. The mean is dragged down by four
(`t039`, `t022`, `t008`, `t028`), all of which also carry the largest drag
errors (−9% to −13%) and the largest suction-peak errors (+0.14 to +0.20).

**Failures are rare and severe.** The suction peak is systematically *too weak*
on every target — peak error is positive throughout, 0.068 to 0.223 — consistent
with `a` < 1: the model under-produces pressure magnitude, and does so worst
where it does worst overall.

## 6. Protocol deviations

Dated, per [[../PROTOCOL]] §9.

- **2026-08-28** — 132 solver runs against the registered 100. The extra 22 are
  the **target geometries themselves**, needed for the spec's own headline
  measurement: it calls `Cd` generated-versus-target "the engineering quantity"
  and says it is free from `force.csv`, but that gives only the *generated*
  shape's drag and the dataset stores Cp, not forces. Running both sides through
  one solver also means F07's constant 9.35% offset cancels.
- **2026-08-28** — The two Year-1 targets were run at **5 seeds each** (10 runs).
  The spec requires them graded and labelled but does not state a seed count;
  five keeps them comparable to the held-out numbers.
- **2026-08-28** — The evaluation target set was drawn by B01, not by this
  experiment. See [[P1-B01_retrieval-baseline]] §6.
- **2026-08-28** — **An export defect was found and fixed before any run.**
  Sampled bodies were in mean-radius-normalised units, giving chord 2.3–2.7
  against the corpus constant of 1.0. Recorded as [[../PROTOCOL]] §2.1b. All
  numbers here are post-fix.
- **2026-08-28** — Model-versus-retrieval is tested **paired**, not by comparing
  medians. The spec does not specify the test; the paired one is correct for a
  matched design and the two disagree here (§3).

## 7. What this changes

**The August gate has its answer, and it is not the expected one.** The model
does not beat retrieval in-distribution; it loses at 4.3σ on 18 of 20 targets.
Combined with [[P1-B03_basis-reachability|B03]] — one PCA mode inside the noise
floor — the in-distribution case for C1 is closed.

**C1 now rests entirely on OOD reach**, and B01 already showed retrieval survives
four of five OOD rungs. G03 must show the model degrading *more gracefully* than
retrieval, on a ladder where retrieval is strong.

**What survives and is worth writing.** No memorisation. Zero invalid geometry
across 110 samples. Drag within 1.1σ_Cd of target. A magnitude creep that
disappears once the instrument is fixed. These are real, and they belong in §4.6
whatever happens to C1.

**G04 needs revisiting before it runs.** Its central quantity, a ~1.4× magnitude
excess, is not reproducible here.

## 8. Files

| | |
|---|---|
| Sampler | `scripts/paper1/p1_g01_sample.py` |
| Target geometries | `scripts/paper1/p1_g01_make_truth.py` |
| Runner | `scripts/paper1/run_p1_g01_solver.sh` |
| Grader | `scripts/paper1/p1_g01_grade.py` |
| Frozen targets | `data/splits/paper1_eval_targets.json` |
| Data | `outputs/paper1/P1-G01_holdout-inverse-design/grading.json` |
