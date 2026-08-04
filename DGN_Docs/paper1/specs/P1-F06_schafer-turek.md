---
id: P1-F06
group: F — Foundation & protocol
title: "Schäfer–Turek absolute benchmark"
tier: gate
status: planned
depends_on: [P1-F02]
blocks: [P1-F07, P1-F03]
paper_section: "4.1, Appendix A"
contribution: C3
cost: {train_runs: 0, solver_runs: 12, effort: "~4 days"}
---

# P1-F06 — Schäfer–Turek absolute benchmark

## Question

Does the solver reproduce the Schäfer–Turek (1996) laminar cylinder benchmark to
within its published bounds — and therefore, is the affine magnitude factor
`a ≈ 1.74–1.78` a solver error or a definitional constant?

## Why this experiment exists

Every validation number in the Year-1 report is **relative**: the solver is graded
by correlation against the pOnEllipse dataset. Correlation cannot detect an error
the two share, and an affine fit *absorbs* exactly the class of error most likely
to be present — a constant scale. The 15-case envelope gate (F03) is therefore
strong evidence of *consistency* and no evidence of *absolute correctness*.

Schäfer–Turek is the standard fix. It is a closed benchmark with tight published
bounds on `Cd`, `Cl`, Δp and Strouhal number, so it tests the solver's **absolute
pressure and force scale** against something external to this project.

Its second function is diagnostic. If the solver reproduces the benchmark's
**pressure difference** Δp within bounds, the solver's absolute pressure scale is
correct, and `a ≈ 1.74–1.78` cannot originate in the solver — it must live in the
dataset's convention or in the comparison code. That single result eliminates most
of F07's search space.

## Hypothesis *(pre-registered)*

The solver reproduces both benchmark configurations within published bounds, or
within a stated and small excess attributable to the LBM boundary treatment.
Predicted: `Cd` within 2%, Δp within 3%, Strouhal within 2%.

Consequently `a` is **not** a solver error, and F07 will locate it in the dataset's
non-dimensionalisation.

## Falsifier

`Cd` or Δp outside 5% of the benchmark bounds. Then the solver has a genuine
absolute-accuracy problem, C3's instrument claim is weakened, **and `a ≈ 1.74` may
be partly or wholly a solver artefact** — which would mean the Year-1 magnitude-creep
finding is measured against a mis-scaled reference. That is the outcome worth
finding, and it is why this runs before G04.

## Method

Three levels. The differences between them are the diagnostic.

| Level | What runs | Isolates |
|:--:|---|---|
| **L0** | Stock `$OPENLB_ROOT/examples/laminar/cylinder2d`, unmodified | The OpenLB build, the library's own force/pressure functors, the machine |
| **L1** | Schäfer–Turek geometry through **your `ellipse2d` binary**, with a parabolic inlet | **Your** case setup: boundary conditions, Bouzidi treatment, sponge layer, and your `Cd`/`Cp` extraction in `results.h` |
| **L2** | Same geometry through `ellipse2d` with the **uniform** inlet it normally uses | What the uniform-inlet simplification itself costs |

L0 → L1 is the important comparison: it separates "OpenLB is fine, my case is
wrong" from "the library behaves differently than I assume". `lattice.h:15` records
that this case was derived from `cylinder2d` and deliberately replaced the
Poiseuille inlet with a uniform one — L2 quantifies the consequence of that choice
for the first time.

### Configurations

| Config | Regime | Reported quantities |
|---|---|---|
| **2D-1** | Steady, Re = 20 | `Cd`, `Cl`, Δp across the cylinder |
| **2D-2** | Unsteady periodic, Re = 100 | `Cd_max`, `Cl_max`, Strouhal, Δp(t) |

Geometry: 2.2 × 0.41 channel, cylinder diameter 0.1 centred at (0.2, 0.2) — the
deliberate vertical asymmetry is what triggers shedding in 2D-2 and must be
reproduced exactly. Parabolic inlet with the benchmark's `U_mean` convention.

> ⚠️ **Take the reference bounds from the source, not from here.** Schäfer & Turek
> (1996), *Benchmark computations of laminar flow around a cylinder*, DFG Priority
> Programme. Transcribe `Cd`, `Cl`, Δp and St bounds for 2D-1 and 2D-2 into the
> results file with the citation. The paper will quote them, so they must come from
> the source.

### Procedure

1. Run L0 at the stock resolution; confirm the published values. If this fails,
   stop — the problem is the build or the environment, not the case.
2. Implement a parabolic-inlet option in `lattice.h` (a flag, not a replacement —
   the uniform inlet must remain the default for every other experiment).
3. Run L1 at 3 resolutions to show the benchmark values are converged, not hit by
   luck at one grid. This is the one place a resolution study is genuinely needed.
4. Run L2 and report the uniform-inlet delta.
5. **Report Δp explicitly.** It is the pressure-scale check and therefore the part
   that matters most for F07; `Cd` alone would not settle the question.

## Measurements

| Quantity | Purpose |
|---|---|
| `Cd`, `Cl`, Δp (2D-1); `Cd_max`, `Cl_max`, St, Δp (2D-2) | Against published bounds |
| Percentage deviation per quantity, per level, per resolution | The absolute-accuracy statement |
| L0 → L1 delta | Library vs your case |
| L1 → L2 delta | Cost of the uniform inlet |
| **Δp deviation specifically** | **The pressure-scale verdict that feeds F07** |

## Decision rule

| Outcome | Verdict |
|---|---|
| All within bounds at L1 | **Solver qualified in absolute terms.** `a` is not a solver error → F07 searches the dataset convention and the comparison code only |
| L0 passes, L1 fails | The fault is in *your* case setup — boundary conditions or the extraction in `results.h`. Fix before any further grading; F03's envelope gate must then be re-run |
| L0 fails | Build or environment. Resolve first; everything downstream is unsafe |
| Δp off by a factor close to `a` | **The creep is a solver pressure-scale error.** Escalate — G04's finding and part of the Year-1 report change |

## Consequence

- Converts C3 from *"consistent with the dataset"* to *"absolutely correct against
  an external benchmark"*. That is a materially stronger instrument claim and it
  costs twelve runs.
- Narrows F07's search space to roughly two candidates.
- Answers the sharpest available attack on the whole evaluation: *"your affine fit
  is a fudge factor absorbing a solver error."*

## Notes

**This supersedes F03's cylinder benchmark**, which specified only `Cd` and
Strouhal at Re 100/200 without a pressure check or published bounds. F03 now
defers external validation to this experiment and retains the envelope gate and
resolution sensitivity.

Distinguish from **Turek–Hron**, the FSI benchmark deferred to Paper 2. This is the
rigid-cylinder laminar benchmark, and it is the right one here.

## Results

_Not yet run._
