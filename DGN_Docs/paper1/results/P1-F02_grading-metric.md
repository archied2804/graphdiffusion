---
id: P1-F02
spec: "../specs/P1-F02_grading-metric.md"
title: "Canonical grading metric"
date_run: 2026-08-03
date_reported: 2026-08-03
status: complete
verdict: falsified
git_sha: 39afe0d
---

# P1-F02 — results

> **Spec:** [[../specs/P1-F02_grading-metric]] · **Verdict:** ❌ **falsified —
> the implementations do not agree, and one of them is broken**

## 1. Verdict against the pre-registered decision rule

The hypothesis was that the four copies *"agree to numerical precision"*, with any
difference coming from interpolation rather than the fit. **Both halves are wrong.**

The two distinct variants disagree by up to **Δρ = 1.19** and **Δa = 3.43** — six
and five orders of magnitude beyond the pre-registered tolerances of `1e-6` and
`1e-4`. This is not a rounding difference. One variant is producing meaningless
numbers.

The spec's fail branch applies: **document which implementation was correct, why,
and recompute anything carried forward.**

## 2. Headline numbers

| Quantity | Value | Tolerance | Verdict |
|---|--:|--:|---|
| max \|Δρ\| between historical variants | **1.19** | 1e-6 | ❌ |
| max \|Δa\| between historical variants | **3.43** | 1e-4 | ❌ |
| max \|Δa\| historical vs consolidated | 1.01 | — | material |
| median \|residual runs-test z\| | **10.02** (max 10.99) | \|z\|>2 ⇒ structured | ❌ |

n = 22 stored OpenLB runs with a matching reference curve.

## 3. Finding 1 — `compare_cp_baseline.py` interpolates onto a non-monotonic grid

The two variants differ in resampling direction:

| Variant | Used by | Method |
|---|---|---|
| `target_on_meas` | `make_exp04x_doc_figures`, `analyze_ood_sweep`, `compare_dps_sweep` | `np.interp(x_meas, sort(x_ref), cp_ref)` |
| `meas_on_ref` | **`compare_cp_baseline`** | `np.interp(x_ref, x_meas, cp_meas)` |

The first sorts the reference before interpolating. The second does not sort
`x_meas` — and `x_meas` **is not monotonic**: the solver writes `cp.csv` in
boundary-traversal order, so `x/c` runs 1 → 0 → 1 around the body (verified on
`sim_0`: `argmax` at index 30 of 60).

`np.interp` requires monotonically increasing `xp` and **returns silent garbage
otherwise**. That is exactly what happens: `compare_cp_baseline` reports ρ values
of −0.25, 0.42, 0.51 on runs the other variant grades at 0.93–0.98, and `a ≈ 0`
throughout.

**Scope of the damage is narrow but must be checked.** `compare_cp_baseline` is
the script behind the EXP-040 baseline gate. Its reported ρ ≈ 0.96 / `a` ≈ 1.74
are *not* reproduced by either variant here (`target_on_meas` gives ρ = 0.983,
`a` = 2.364 on `sim_0`), so the published numbers were produced by a code path
that no longer exists in the current file, or on different inputs. **Any EXP-040
number carried into the paper must be regenerated** — F03 does this by design.

## 4. Finding 2 — the affine model is a misfit

Median runs-test `|z| = 10.0` across all 22 runs, maximum 10.99. The residual of
`Cp_meas ≈ a·Cp_ref + b` is **strongly structured**, not noise.

This is the diagnostic added on the strength of [[../METRICS]] §2.1, and it fires
immediately. The consequence is stated in the spec and in
[[../specs/P1-F07_normalisation-audit]]: if the residual has systematic chordwise
structure then the affine calibration is the wrong model, and **both ρ and `a` are
contaminated by the misfit**.

`a` is the entire basis of the magnitude-creep finding. Its interpretation as a
clean unit-conversion constant is now in question **before** F06/F07 have run, and
those two experiments become more important rather than less. F07's Step 4
regression of `a` on Re, `H`, `b` should now be read alongside this: a structured
residual and an operating-point-dependent `a` would together mean the single-constant
story is wrong.

**Do not treat this as settled.** The runs test is sensitive to serial correlation,
and a curve sampled on a fine grid is autocorrelated by construction, which
inflates `|z|`. The finding is that the residual is *not* white; quantifying how
much of that is genuine misfit versus grid autocorrelation is F03's job, where the
same diagnostic runs across the 15-case envelope.

## 5. Finding 3 — reference curves are stored already collapsed

The dataset-derived `*_ref_cp.csv` files are **sorted by `x/c` with duplicated
stations** (`0.0, 0.006, 0.006, 0.023, …`), i.e. single-valued. The solver's
`cp.csv` is in traversal order and retains surface identity.

So the upper/lower distinction — the signal that *is* lift — is destroyed
**upstream, in the export**, not merely in the DCT conditioning. `split_branches`
detects an already-sorted input and degrades gracefully to collapsed rather than
fabricating a second surface.

Consequence for [[../specs/P1-G02_aoa10-ood]]: grading AoA10 on lift requires
re-exporting reference curves in traversal order. Until then, `Cl` from
`force.csv` is the only lift-sensitive quantity available — which is a further
argument for promoting it, per [[../PROTOCOL]] §2.3.

**A correction to my own first pass:** an initial run reported an
`a_collapsed / a_split` ratio of 2.76 and I nearly recorded it as a finding. It
was an artefact of `split_branches` degenerating on these already-sorted reference
curves. Fixed before it reached any conclusion; noted here because the same trap
will recur wherever collapsed and traversal-order curves are mixed.

## 6. What was built

`src/graph_diffusion/postprocessing/cp_metrics.py`, 19 tests, all passing:

| Feature | Why |
|---|---|
| Common uniform 128-point grid for **both** curves | Each historical copy fitted on one side's nodes, weighting the fit by that side's node density, which clusters at LE/TE. Also makes the metric node-count agnostic — required for a corpus spanning 52–96 nodes and a `thick` set reaching 100 |
| Surface-branch splitting | Correct grading of asymmetric bodies |
| Runs test on the affine residual | Finding 2 above |
| Chordwise band residuals (LE / peak / aft) | Localises error; per [[../PROTOCOL]] §2.2 |
| Suction-peak and stagnation locations | The two numbers a designer reads |
| `read_force_csv` → `Cd`, `Cl`, `Cd_std`, `Cl_std` | Promotes forces to primary metrics; `Cl` had never been read by anything |
| `polygon_validity` | Closure, self-intersection, node spacing |

Tests assert the analytic invariants directly: exact recovery of `(a, b)` under an
affine relation, ρ invariance to positive affine rescaling, sign flip → ρ = −1,
resampling invariance across 52/96/100 nodes, and cross-node-count grading.

## 7. Protocol deviations

The spec anticipated a fourth-implementation comparison; only **two distinct
variants** exist — three scripts share `target_on_meas` verbatim. The audit
compares those two plus both modes of the new module.

## 8. Reproduction

| | |
|---|---|
| Module | `src/graph_diffusion/postprocessing/cp_metrics.py` |
| Tests | `tests/test_cp_metrics.py` (19) |
| Audit | `scripts/paper1/p1_f02_metric_audit.py` → `outputs/paper1/P1-F02_grading-metric/audit.json` |
| Command | `python scripts/paper1/p1_f02_metric_audit.py` |

> ⚠️ **Environment note.** `.venv/bin/pytest` had a shebang pointing at a
> *different project's* interpreter (`DGN_Simple/.venv/bin/python3`), which broke
> single-file test runs. Full-suite runs survived via pytest's rootdir path
> insertion, so earlier gate results stand. Repaired locally, but the venv is not
> version-controlled and a reinstall will reintroduce it — **prefer
> `uv run python -m pytest`**, which is immune.

## 9. What this changes

**In the paper**
- §3.5 states one tested grading implementation, with the common-grid rationale.
- §4.1 must report that the affine residual is structured, or that F03 showed it
  was grid autocorrelation. Either way it is stated, not hidden.

**In the plan**
- [[../specs/P1-F03_solver-qualification]] — run the residual-structure diagnostic
  across all 15 envelope cases; this is now its most important output.
- [[../specs/P1-F07_normalisation-audit]] — **raised in importance.** A structured
  residual means `a` may not be a clean constant at all.
- [[../specs/P1-G02_aoa10-ood]] — reference curves need re-exporting in traversal
  order before lift can be graded from Cp; use `Cl` meanwhile.
- Any EXP-040 number entering the paper must be regenerated through this module.
