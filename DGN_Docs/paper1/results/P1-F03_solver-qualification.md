---
id: P1-F03
spec: "../specs/P1-F03_solver-qualification.md"
title: "Solver instrument qualification"
date_run: 2026-08-05
date_reported: 2026-08-06
status: complete
verdict: falsified
git_sha: 69bdca8
---

# P1-F03 — results

> **Spec:** [[../specs/P1-F03_solver-qualification]] · **Verdict:** ❌
> **falsified.** The affine slope `a` drifts 11.3% across the Re envelope,
> above the ±10% falsifier. But the more consequential finding is underneath:
> **`a` moves more under a change of numerical resolution at fixed Re (21.5%)
> than it does across the entire Re span.** It is not a stable calibration
> constant.

## 1. Verdict against the pre-registered decision rule

| Criterion | Threshold | Measured | |
|---|---|---|:--:|
| mean ρ | ≥ 0.95 | **0.9504** | ✅ (by 0.0004) |
| systematic drift in `a` | < ±10% | **11.28%** with Re | ❌ |
| Δρ on doubling resolution | < 0.01 | +0.034, −0.024, −0.000 | ❌ 2 of 3 |

The decision rule's **fail** branch applies: *"drift over 10% → the solver
cannot support cross-condition magnitude claims. G04 becomes a within-condition
comparison only, and the paper says so."*

Two honesty notes on the margin, both of which cut against overstating this:

- Mean ρ clears its bar by **0.0004**. That is a pass by the letter of the rule
  and nothing more. The worst case is ρ = 0.865.
- The 11.28% drift has a bootstrap CI of **[2.94%, 18.11%]**, which straddles
  the 10% threshold. The point estimate falsifies; the interval does not
  resolve the question on its own. §4 is what makes the finding solid, not this
  number.

27 of 27 runs completed. No case diverged.

## 2. Part A — envelope gate, 15 cases

| case | Re | H | b | ρ | `a` | b_off | rms | \|z\| | Cd |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| re0501_h50 | 501 | 5.01 | 0.344 | 0.9463 | 1.606 | 0.096 | 0.137 | 8.4 | 1.023 |
| re0503_h55 | 503 | 5.50 | 0.336 | 0.9569 | 1.746 | 0.054 | 0.129 | 8.1 | 1.054 |
| re0502_h60 | 502 | 5.99 | 0.305 | 0.9641 | 1.499 | 0.027 | 0.092 | 9.7 | 0.916 |
| re0625_h50 | 625 | 5.00 | 0.332 | 0.9400 | 1.786 | 0.128 | 0.153 | 7.9 | 1.008 |
| re0627_h55 | 627 | 5.50 | 0.283 | 0.9154 | 1.739 | 0.087 | 0.159 | 7.1 | 0.916 |
| re0621_h60 | 621 | 6.00 | 0.264 | **0.8649** | 1.547 | 0.079 | 0.177 | 9.2 | 0.855 |
| **re0751_h50** | 751 | 5.01 | 0.369 | **0.9996** | **1.876** | −0.003 | 0.015 | 3.7 | 1.158 |
| re0750_h55 | 750 | 5.50 | 0.263 | 0.9273 | 1.706 | 0.063 | 0.135 | 9.7 | 0.859 |
| **re0745_h60** | 745 | 6.00 | 0.299 | **0.9985** | **1.882** | 0.012 | 0.021 | 7.7 | 0.899 |
| re0873_h50 | 873 | 5.00 | 0.272 | 0.9633 | 1.658 | −0.041 | 0.091 | 10.1 | 0.881 |
| re0875_h55 | 875 | 5.50 | 0.270 | 0.9261 | 1.714 | 0.000 | 0.139 | 7.9 | 0.887 |
| re0880_h60 | 880 | 6.00 | 0.372 | 0.9714 | 1.781 | −0.025 | 0.115 | 8.9 | 1.105 |
| **re0998_h50** | 998 | 5.01 | 0.275 | **0.9992** | **1.863** | 0.024 | 0.015 | 4.8 | 0.862 |
| **re0998_h55** | 998 | 5.49 | 0.284 | **0.9987** | **1.885** | 0.022 | 0.019 | 6.4 | 0.865 |
| re0997_h60 | 997 | 6.00 | 0.264 | 0.8849 | 1.798 | 0.057 | 0.176 | 9.4 | 0.823 |

`a` = 1.739 ± 0.121 (CV 6.93%), range 1.499–1.885. The spec predicted ±5%.

## 3. Finding 1 — the Re drift is confounded with resolution by construction

| predictor | slope | 95% CI | drift over span | CI excludes 0 |
|---|--:|---|--:|:--:|
| **Re** | 3.95e−4 | [1.03e−4, 6.34e−4] | **+11.28%** | **yes** |
| H | −5.66e−2 | [−2.31e−1, 1.16e−1] | −3.25% | no |
| b | 4.82e−1 | [−1.23e0, 1.68e0] | +3.04% | no |

Only Re shows a significant association. But **the protocol ties resolution to
Reynolds number**: `RES = max(100, 0.2·Re)`. Every Part A case therefore has its
resolution determined by its Re, so "`a` varies with Re" and "`a` varies with
resolution" are **perfectly confounded in Part A**. Part A alone cannot
attribute the drift to physics.

Part B is the only experiment that separates them, because it varies resolution
at fixed Re. It does — decisively, and against the physical reading.

## 4. Finding 2 — `a` is numerically unconverged, and that is the real result

Spread of `a` from **resolution alone**, at fixed Re, fixed geometry, fixed
everything else:

| case | `a` range over 0.5–2.0× rule | span | ρ range |
|---|---|--:|---|
| re0501_h50 (Re 501) | 1.607 – 1.989 | **21.5%** | 0.9463 – 0.9984 |
| re0751_h50 (Re 751) | 1.673 – 1.965 | **16.1%** | 0.9416 – 0.9996 |
| re0998_h50 (Re 998) | 1.452 – 1.902 | **25.9%** | 0.9299 – 0.9992 |

> **Resolution alone moves `a` by 21.5% on average — roughly twice the 11.28%
> attributed to the entire Re span.**

The across-Re trend is smaller than the within-case numerical scatter. The
most economical reading is that the apparent Re dependence is **largely or
wholly a resolution artefact**, inherited through the stability rule, not a
physical dependence of the pressure scale on Reynolds number.

Either way the consequence for the paper is the same, and it is stronger than
the falsifier alone: **`a` is not a constant that can be measured once and
applied**. It is a number that depends on the grid.

ρ is not converging either. It is **non-monotonic in resolution** in all three
cases, and for re0501 the *coarsest* grid gives the *best* agreement
(ρ = 0.9984 at 0.5× versus 0.9463 at the rule). Agreement is not improving under
refinement, which is what one would demand of an instrument.

## 5. Finding 3 — `a` is only meaningful where ρ is high, and there it is 1.877

Splitting the envelope on fit quality:

| group | n | `a` | CV | Re span | `a`~Re drift |
|---|--:|---|--:|---|--:|
| ρ ≥ 0.99 | 4 | **1.8765 ± 0.0098** | **0.52%** | 745–998 | −0.27% |
| ρ < 0.99 | 11 | 1.6891 ± 0.1001 | 5.93% | 501–998 | — |

`corr(a, ρ) = 0.529`.

Among the four cases where the solver and the dataset genuinely agree in
*shape*, the magnitude ratio is a **tight constant of 1.877 with a coefficient
of variation of 0.52% and no Re dependence at all** — across a 745–998 Re span
that covers half the envelope.

The scattered, lower values come exclusively from poorly-fitting cases, and they
are biased *downward*. That is the signature of **regression attenuation**: when
the predictor carries error, the fitted slope is pulled toward zero. So the
low-ρ cases do not measure a different physical scale; they measure the same
scale badly, and the bias has a known direction.

This reframes the headline number of the whole magnitude-creep story:

- The Year-1 value **`a` ≈ 1.74** is the mean over a mixture of well- and
  poorly-fitted cases. It is an attenuation-contaminated average.
- The value from cases where the affine model actually holds is **≈ 1.877**.
- Neither is stable to ±26% under grid refinement.

**This is not a licence to substitute 1.877 for 1.74 and carry on.** The
conditioning of that estimate on ρ ≥ 0.99 is a post-hoc split, chosen after
seeing the data, and it is reported here as an observation to be tested — not
as a calibration. [[../specs/P1-F07_normalisation-audit]] must treat "is `a` a
single constant?" as open, and F06's absolute benchmark is now the only clean
way to answer it, because it compares against published values rather than
against this dataset.

## 6. Finding 4 — the structured residual is confirmed, independently

Median |runs-test z| = **8.12** across the 15 envelope cases (max 10.11). Every
case exceeds |z| > 2.

[[P1-F02_grading-metric]] reported median |z| = 10.0 but on 22 correlated runs
from a single perturbation family at one operating point, and §2 of that file
explicitly bounded the claim to that sample. **This is the independent
confirmation it asked for**: 15 cases varying Re, H and b independently, and the
affine residual is structured in all of them.

The structure is weaker where the fit is good (|z| ≈ 3.7–7.7 for the ρ ≥ 0.99
group) but never absent. So it is not purely grid autocorrelation, and it is
not a property of one shape family. **The affine model is a misfit in general.**

That closes the question F02 left open, and it closes it the unfavourable way.

## 7. Integrated forces

`Cd` spans 0.823–1.158 across the envelope and is far better behaved than `a`
under refinement: Δ`Cd` from 1.0× to 2.0× is +0.041, −0.012, +0.011 — around 1%,
against 2–11% for `a`. `Cl` is ~0 throughout, as expected at zero incidence.

Worth noting for [[../PROTOCOL]] §2.3: the integrated forces are the most
numerically stable quantity this experiment produced. If a magnitude claim has
to rest on something, `Cd` is a better foundation than `a`.

## 8. What this changes

**In the paper**
- §4.1 must state that the grading instrument is qualified for **shape**
  agreement (ρ) and **not** for cross-condition magnitude.
- Appendix A carries the 15-case table with its ρ range, including the 0.865.
- §5 must state the resolution non-convergence plainly. It is the honest limit
  of an instrument framing, and a reviewer will find it if the paper does not.

**In the plan**
- [[../specs/P1-G04_creep-attribution]] is **restricted to within-condition
  comparisons.** Cross-condition magnitude attribution is not supported.
- [[../specs/P1-F07_normalisation-audit]] rises in importance again. Its Step 4
  regression of `a` on Re/H/b must control for resolution, or it will re-measure
  this confound.
- [[../specs/P1-F06_schafer-turek]] is now the **critical experiment for C4**. It
  is the only planned check against published absolute values rather than against
  this dataset, so it is the only thing that can decide whether ≈1.877 is
  physical or an artefact of the pipeline.
- [[../specs/P1-F04_noise-floor]] should sample across resolutions, not only
  seeds — this experiment shows the grid contributes more variance to `a` than
  anything else measured so far.

## 9. Protocol deviations

**Dated 2026-08-05/06. Two interruptions, neither affecting the results.**

1. **Power loss during Part A** (14 of 15 complete). The original runner was not
   restartable — it truncated its own log and cleared each case's outputs — so a
   resume script was written that judges completion from the run directories
   rather than the log. The log had under-counted by one: a completed run's
   final line was lost to the power cut.
2. **A latent bug in Part B, exposed by the restart.** The runner sourced
   geometry as `<name>.dat`, but the shape files are `sim_<idx>.dat`. Part B had
   never executed before the interruption, so the bug was unexercised; it would
   have failed all 12 runs. Fixed in both scripts.
3. **Six runs killed by the GPU driver watchdog** (`cudaErrorLaunchTimeout`), not
   by divergence — two had reached t = 46.5 and t = 49.0 of 50. The GPU drives a
   display, so long kernels are killed by X, intermittently and in proportion to
   desktop activity rather than problem size. **These were explicitly not
   recorded as solver failures**, which would have put a fabricated instability
   into an experiment whose purpose is to locate real ones. The runner now
   classifies the failure mode and retries watchdog kills up to three times; all
   six then completed. Logs retained under `watchdog_failures/`.

**No case diverged. The stability envelope is clean over the tested range**, at
all four resolution multipliers including 0.5× the rule.

## 10. Regrade — the factor of two, 2026-08-07

**The `a` values throughout this file were computed before the dataset's units
convention was identified.** [[../DATASET_PROVENANCE]] §4 established that the
stored reference is `p* = p/(rho U^2)` while the solver reports
`Cp = p/(0.5 rho U^2)`, so a factor of 2 was folded into every slope.

`cp_metrics.read_reference_cp` now applies it. Regraded values:

| quantity | as first reported | regraded |
|---|--:|--:|
| `a`, all 15 cases | 1.739 | **0.8695** |
| `a`, well-fitted (rho >= 0.99) | 1.8765 ± 0.0098 | **0.9383 ± 0.0049** |
| CV | 6.93% | 6.93% (unchanged) |
| drift with Re over span | 11.28% | 11.28% (unchanged) |
| rho (all) | 0.9504 | 0.9504 (unchanged) |

**ρ is unchanged to every digit**, as affine invariance requires — the pass on
mean ρ ≥ 0.95 and the whole shape-fidelity half of this experiment stand
untouched. The percentages are ratios, so the falsifier verdict is also
unchanged: `a` still drifts 11.28% with Re, and the resolution spread is still
11–26%.

**What changes is the interpretation.** There is no 74% magnitude discrepancy.
After removing the units convention the solver agrees with the dataset to
**6.2% ± 0.5pp** on the cases where the affine model actually fits. That is a
validation result, not a defect.

## 11. fp64 follow-up — precision is not the cause

[[P1-F06_schafer-turek]] found that single precision destroys the lift
coefficient under refinement, which made fp32 round-off the leading candidate
for §4's resolution spread. **It is not.** The identical sweep in double
precision:

| case | fp32 spread | fp64 spread |
|---|--:|--:|
| re0501 | 21.5% | **25.9%** |
| re0751 | 16.1% | **22.6%** |
| re0998 | 25.9% | **27.9%** |

The spread survives, slightly larger. §4's conclusion stands on its own:
**`a`'s resolution dependence is discretisation error, not round-off.**

### The direction is the informative part

fp64 `a` against the multiple of the stability rule:

| case | 0.5× | 1.0× | 1.5× | 2.0× |
|---|--:|--:|--:|--:|
| re0501 | **0.994** | 0.833 | 0.886 | 0.769 |
| re0751 | **0.982** | 0.939 | 0.857 | 0.781 |
| re0998 | **0.980** | 0.871 | 0.741 | 0.846 |

**The coarsest grid agrees best — to within 1–2% — and refinement degrades
agreement.** That is backwards for a convergence study, and it contradicts the
"our solver under-resolves the suction peak" reading offered in the first draft
of this file, which predicted the opposite.

**Working hypothesis, not a conclusion.** `a` measures agreement with *a
particular reference simulation*, not with truth. The released dataset is
described by [[../DATASET_PROVENANCE]] §1 as *"low-resolution versions … reduced
by approximately a factor of two"*. If the reference itself carries numerical
dissipation characteristic of a coarse grid, our best agreement should occur
where our dissipation matches theirs — and refining our grid moves us toward
the true solution and away from the reference.

That is consistent with every number above, but it is an inference. Testing it
requires a reference at known resolution, which this dataset does not provide.
**Stated as a limitation, not resolved.**

## 12. Reproduction

| | |
|---|---|
| Runs | `scripts/paper1/run_p1_f03_resume.sh` (27 runs) |
| Grading | `scripts/paper1/p1_f03_grade.py` |
| Output | `outputs/paper1/P1-F03_solver-qualification/grading.json` |
| Archived results | `.../envelope/`, `.../resolution/` |
| Metric | `cp_metrics.cp_agreement(mode="collapsed")`, per [[P1-F02_grading-metric]] |
| CI method | Bootstrap, 10,000 resamples, seed 0 |
