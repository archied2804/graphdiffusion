---
id: P1-F07
group: F — Foundation & protocol
title: "Cp normalisation audit — the source of a ≈ 1.74"
tier: gate
status: planned
depends_on: [P1-F02, P1-F06, P1-F04]
blocks: [P1-G04]
paper_section: "3.4, 4.1"
contribution: C3, C4
cost: {train_runs: 0, solver_runs: 6, effort: "~3 days"}
---

# P1-F07 — Cp normalisation audit: the source of `a ≈ 1.74`

## Question

Where does the affine magnitude factor `a ≈ 1.74–1.78` between solver-measured Cp
and dataset Cp come from?

## Why this is a blocker

`a` is the basis of C4's magnitude-creep finding. The paper currently reports a
calibration constant it cannot explain, applied to every graded number. A reviewer
will read an unexplained multiplicative factor of 1.78 as a fudge factor absorbing
a solver error, and it is not currently possible to answer them.

EXP-040 already noticed this and left it open:

> *"`a ≈ 1.78` … is strikingly close to the paper's note that the dataset velocity
> was 'scaled by 1.8'. This still needs to be pinned exactly from the dataset card
> before it is treated as confirmed."*
> — [[../../experiments/EXP-040_openlb_cp_baseline]], with an unticked action item
> *"Pin `a ≈ 1.78` analytically from the dataset card."*

That action was never completed and the number has been used in every downstream
result since.

## Hypothesis *(pre-registered)*

`a` is a **definitional constant in the dataset's non-dimensionalisation**, not a
solver error. Specifically: the dataset's reference dynamic pressure is ≈1.8× the
textbook `½ρU²` at the matched Reynolds number.

Two supporting predictions, both falsifiable:

1. **F06 passes.** The solver reproduces Schäfer–Turek Δp within bounds, so its
   absolute pressure scale is correct and `a` cannot originate there.
2. **`a` is constant.** Regressed on Re, `H` and `b`, no coefficient differs from
   zero by more than the F04 noise floor. A definitional constant is by definition
   independent of the operating point.

## Falsifier

Either supporting prediction failing:

- **F06 shows a Δp error near `a`** → the factor is a solver pressure-scale bug.
  C3 is damaged, G04's creep finding is measured against a mis-scaled reference,
  and part of the Year-1 report must be recomputed.
- **`a` depends significantly on Re, `H` or `b`** → it is not one constant but a
  constant plus a physics-dependent term. The affine calibration is then an
  approximation to something structured, and both ρ and `a` inherit that error.

Note EXP-040 already reports a spread — 1.78 on the cleanest case against a 1.74
mean. **Whether that ~2% spread is signal or noise is exactly the question**, and
F04's `σ_a` is what decides it. If the spread is real, the single-constant story is
already wrong.

## Method

### Step 1 — Eliminate the solver *(from F06)*

If Schäfer–Turek Δp is within bounds, the solver's absolute pressure scale is
correct and the search reduces to the dataset convention and the comparison code.
Do not proceed until F06 has reported.

### Step 2 — Internal self-consistency

Cp is dimensionless: at fixed Re it **must** be invariant to the choice of `U`.

Run the same shape twice at matched Re with `U` doubled and `ν` doubled. If the two
Cp curves differ by more than the noise floor, there is a normalisation error
**inside the solver** — most likely a mismatch between `PHYS_CHAR_DENSITY` /
`PHYS_CHAR_VELOCITY` as used in `qInf` (`results.h:380`) and the values the
converter actually applies to `SuperLatticePhysPressure2D`. Two runs, and it tests
the solver's normalisation without any external reference.

### Step 3 — Candidate mechanisms, with predicted factors

Each candidate predicts a *specific number*. This makes the audit a matching
exercise rather than a rationalisation, which is why the table is registered before
the work starts.

| # | Candidate mechanism | Predicted `a` | Signature |
|:--:|---|:--:|---|
| 1 | Dataset reference dynamic pressure = 1.8 × `½ρU²` | **1.80** | Constant; matches the dataset card note |
| 2 | Dataset `U_ref` = 4/3 × ours | **1.78** | Constant; no known origin — needs one |
| 3 | Missing factor of ½ somewhere | 2.00 | Constant |
| 4 | `U_max` vs `U_mean`, parabolic profile (3/2) | 2.25 or 0.44 | Constant |
| 5 | Dataset *velocity* scaled 1.8× (Cp ∝ U⁻²) | 3.24 or 0.31 | Constant — **the reading in EXP-040, and it does not fit** |
| 6 | Momentum-average of a parabolic profile (6/5) | 1.44 | Constant |
| 7 | Channel blockage at `H = 5.5` | ~1.25–1.50 | **Varies with `b`** |
| 8 | Compressibility, O(Ma²) | ≈1 | **Varies with Re** |
| 9 | `ρ` mismatch between converter and `qInf` | any constant | Caught by Step 2 |
| 10 | Averaging-window mismatch (dataset's 101 steps vs 25 chords) | ≈1 | Varies with Re |

**Two observations that do most of the work.** Candidates 7, 8 and 10 predict
dependence on the operating point, so the constancy test in Step 4 eliminates them
as the *primary* mechanism. And candidate 5 — the reading recorded in EXP-040 — is
arithmetically inconsistent: a velocity scaling of 1.8 gives a Cp factor of
`1.8² = 3.24`, not 1.78. The numerical coincidence between "1.8" and the observed
1.78 is real, but it only works if the 1.8 is applied to the **dynamic pressure**
(candidate 1), not to the velocity. Resolving that is the crux.

### Step 4 — Constancy regression

Fit `a` against Re, `H` and `b` across all 15 envelope cases from F03, with F04's
`σ_a` as the error bar. Report each coefficient with a CI. A pure convention
constant gives three coefficients indistinguishable from zero.

### Step 5 — Run down the dataset convention

The task EXP-040 deferred. Sources, in order of authority:

1. The HuggingFace dataset card for `mariolinov/Ellipse`
2. The publication the dataset accompanies — find the Cp definition, the reference
   velocity, the reference pressure location and any stated scaling
3. `TimeEllipseTrain.npy` (`DatasetUrl.TIME_TRAIN_NPY`, never downloaded) — the
   time axis may pin the averaging window and the velocity scale
4. Contact the dataset authors if the convention is not documented. Slow, so start
   it in week one and let it run in the background

Note the dataset stores **Cp**, not raw pressure (`pOnEllipse.py:31`), and the
conditioning uses the **time-averaged** curve — so temporal treatment is matched on
both sides and candidate 10 is a priori weak.

### Step 6 — Trace the comparison code

Confirm the factor is not introduced downstream: chordwise interpolation, node
ordering, the `x/c` convention (leading-edge origin vs centroid origin), or any
rescaling in the export path. F02's consolidation makes this a single-file review
rather than four.

## Measurements

| Quantity | Purpose |
|---|---|
| Step 2 Cp difference under `U` rescaling at fixed Re | Solver-internal normalisation check |
| Regression coefficients of `a` on Re, `H`, `b`, with CI | Constant vs structured |
| Observed `a` against each candidate's predicted value | The matching exercise |
| Documented dataset convention, with citation | The definitive answer |
| Residual `a` unexplained after the identified mechanism | Honest statement of what remains |

## Decision rule

| Outcome | What the paper says |
|---|---|
| Mechanism identified and matches the observed `a` within the noise floor | **State the convention explicitly in §3.4** and report Cp under a single stated definition. The affine fit becomes a documented unit conversion, not a fitted fudge factor |
| `a` constant but no mechanism found | Report it as an unexplained constant offset between conventions, bounded by F06's absolute benchmark so it is provably not a solver error. Weaker, but honest and defensible |
| `a` structured in Re/`H`/`b` | The affine model is inadequate. Escalate — this affects every graded number and G04 in particular |
| F06 shows a solver Δp error near `a` | **Stop.** Fix the solver, re-run F03, recompute all Year-1 numbers |

## Consequence

- **Best case:** the paper replaces a fitted constant with a stated unit
  conversion. `a` stops being a calibration parameter and becomes a definition,
  which makes the magnitude-creep finding much cleaner — creep is then a deviation
  from a *known* value rather than from a fitted one.
- **G04 depends on this.** Attributing creep to the model requires knowing what the
  reference scale is. Until F07 reports, "generated shapes grade at `a ≈ 2.5–3.5`
  against 1.74" compares one unexplained number to another.
- **Paper 2 depends on it more.** R2.2 targets `a → 1.74` through physical
  conditioning; a target that is partly a unit-conversion artefact is not a target.

## Notes

Six solver runs — two for Step 2 and four spare for candidate testing. The rest is
document archaeology and regression on data F03 already produces.

The most likely outcome is undramatic: a documented convention, one sentence in
§3.4, and an open action item from EXP-040 finally closed. That sentence is worth
several days of work, because without it every graded number in the paper carries
an unexplained factor.

## Results

_Not yet run._
