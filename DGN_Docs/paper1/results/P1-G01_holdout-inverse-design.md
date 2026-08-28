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
> Two genuinely good findings sit alongside it: **no memorisation**, and the
> Year-1 magnitude creep is **gone**.

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
