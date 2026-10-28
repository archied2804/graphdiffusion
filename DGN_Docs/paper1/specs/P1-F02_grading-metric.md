---
id: P1-F02
group: F — Foundation & protocol
title: "Canonical grading metric"
tier: gate
status: planned
depends_on: []
blocks: [P1-F03, P1-F04, P1-F05, P1-B01, "all graded experiments"]
paper_section: "3.5, 4.1"
contribution: protocol
cost: {train_runs: 0, solver_runs: 0, effort: "~2 days"}
---

# P1-F02 — Canonical grading metric

## Question

Do the four independent implementations of `(ρ, a)` currently in this repository
agree with each other — and what is the single definition the paper will use?

## Hypothesis *(pre-registered)*

They agree to numerical precision on the correlation and the affine fit. Any
disagreement will come from **Cp interpolation onto the chordwise grid**, not from
the fit itself — the four copies were written at different times against different
node counts.

## Falsifier

Any pair of implementations differs by more than `1e-6` in ρ or `1e-4` in `a` on
the same stored run. If so, some Year-1 numbers are not mutually comparable and
every number carried forward must be recomputed.

## Method

1. Locate every implementation. Currently four, all with their own copy of the
   `corrcoef` + `lstsq` logic:
   - `scripts/make_exp04x_doc_figures.py:72`
   - `scripts/analyze_ood_sweep.py:45`
   - `scripts/compare_dps_sweep.py:70`
   - `scripts/compare_cp_baseline.py`
2. Run all four over the same ~20 stored runs in `cfd/ellipse2d/runs/`. Tabulate
   per-run deviations.
3. Write `src/graph_diffusion/postprocessing/cp_metrics.py` with a single public
   function returning `{rho, a, b, affine_rms, n_points, grid}`, taking target and
   measured curves with their own abscissae and resampling **both** onto a fixed
   128-point chordwise grid. Node-count-agnostic by construction — D02 and G02
   grade shapes with 48–192 nodes through this same function.
4. Unit tests (`tests/test_cp_metrics.py`), seeded, with analytic cases:
   - exact affine relation → `ρ = 1`, `a` recovered exactly
   - ρ invariant under `Cp → αCp + β` for `α > 0`
   - sign flip → `ρ = −1`
   - resampling invariance: same curve at 48/96/192 points → same ρ to `1e-9`
   - degenerate constant input → documented, non-crashing behaviour
5. Add the **validity** metric here too: closed ∧ non-self-intersecting ∧
   `min_spacing > 0.1 × mean_spacing`. Needed from D02 onward, and it becomes a
   primary metric in Paper 2.
6. **Add a force reader.** `tmp/ellipse2d_force.csv` carries
   `Cd_mean, Cd_std, Cl_mean, Cl_std` for every run. Six analysis scripts each
   parse it separately; `Cl` is parsed by none of them. Fold it into the same
   module so one call returns Cp agreement *and* integrated coefficients — they are
   always wanted together ([PROTOCOL §2.3](../PROTOCOL.md)).
7. **Add the residual-structure diagnostic.** The pipeline fits
   `Cp_meas ≈ a·Cp_ref + b` and reports the residual magnitude, but never tests
   whether the residual is structureless. Return the residual as a function of
   `x/c` plus a runs test. If the residual has systematic chordwise structure, the
   affine calibration is the wrong model and both ρ and `a` are contaminated by the
   misfit — which would matter a great deal, since `a` is the basis of the
   magnitude-creep finding.
8. **Add chordwise band errors** (LE / suction peak / aft) and suction-peak
   magnitude and location, per [PROTOCOL §2.2](../PROTOCOL.md). Same data, no extra
   cost, and far more diagnostic than a whole-curve scalar.
6. Regenerate every Year-1 number that will appear in the paper through the new
   module and diff against the published values.

## Measurements

| Quantity | Purpose |
|---|---|
| Max abs deviation in ρ and `a` between implementations, per run | The pass/fail number |
| Diff between regenerated and published Year-1 values | What must be corrected |
| Unit-test pass under the strict gate (`ruff`, `black`, `mypy`, `pytest`) | Code quality per CLAUDE.md |

## Decision rule

- **Pass:** max deviation below tolerance → adopt the module, note in §3.5 that all
  numbers come from one tested implementation.
- **Fail:** document which implementation was correct and why, recompute
  everything, and record the correction in [[../publication/PUBLICATION_DIARY]].
  Any Year-1 number that changes materially is flagged in the paper.

## Consequence

- Either way the deliverable is the same: **one tested grading function**, cited
  in §3.5. Four private copies of the headline metric is the kind of detail that
  is invisible until a reviewer asks for code, and then it is expensive.
- Blocks F03/F04 because a noise floor measured with an inconsistent metric is
  meaningless.

## Results

_Not yet run._
