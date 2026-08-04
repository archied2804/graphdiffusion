---
id: P1-F03
group: F — Foundation & protocol
title: "Solver instrument qualification"
tier: gate
status: planned
depends_on: [P1-F02]
blocks: [P1-F04]
paper_section: "4.1, Appendix A"
contribution: C3
cost: {train_runs: 0, solver_runs: 29, effort: "~3 days"}
---

# P1-F03 — Solver instrument qualification

## Question

Is the OpenLB case fit to *grade* generated shapes — accurate against known data,
stable across the operating envelope, and insensitive to numerical resolution
above the stability rule?

## Hypothesis *(pre-registered)*

Across the (Re, H) envelope the solver reproduces dataset Cp with `ρ ≥ 0.95` and an
affine slope `a` stable to within ±5% of 1.74; the cylinder benchmark falls inside
published `C_d` and Strouhal ranges; and ρ changes by less than 0.01 when
resolution is doubled above `RES = max(100, 0.2·Re)`.

## Falsifier

`a` varies **systematically** with Re or H by more than ±10%. That would mean the
grading scale itself depends on the operating point, which confounds every
cross-condition comparison in the paper and makes G04's creep attribution
impossible.

## Method

Framing matters here: the solver is an **instrument**, not a contribution. It must
be demonstrably fit for grading, which is a lower and different bar than a
publication-grade CFD study. Full grid convergence is explicitly out of scope.

1. **Envelope gate.** Re-run the 15-case `eval_grid` (`cfd/ellipse2d/shapes/eval_grid/`)
   under the frozen protocol settings, graded with the F02 module. This repeats
   Year-1 EXP-040 — it is expected to reproduce; the purpose is protocol
   consistency and a regenerated appendix table.
2. **External benchmark — moved to [P1-F06](P1-F06_schafer-turek.md).** This
   experiment originally specified a loose cylinder check at Re 100/200 on `C_d`
   and Strouhal against "literature". That is too weak: it has no published
   tolerance and, critically, **no pressure check**, so it could not settle whether
   `a ≈ 1.74` is a solver artefact. F06 replaces it with the Schäfer–Turek
   benchmark — tight published bounds on `Cd`, `Cl`, Δp *and* St. Run F06 first;
   this experiment now covers the envelope and resolution behaviour only.
3. **Resolution sensitivity.** Three cases spanning the envelope (low/mid/high Re),
   each at `RES ∈ {0.5, 1.0, 1.5, 2.0} ×` the stability rule. 12 runs.
4. Record every run's stability status; a diverged run is a data point about the
   envelope, not a failure to hide.

## Measurements

| Quantity | Purpose |
|---|---|
| Per-case ρ, `a`, `b`, ε_aff for all 15 | Appendix A table |
| Per-case `Cd`, `Cl` (already in `force.csv`) | Integrated-force agreement across the envelope |
| **Mean and spread of `a`; regression of `a` on Re, `H`, `b` with CI** | The falsifier test — **and F07's constancy diagnostic.** EXP-040 reports 1.78 on the cleanest case against a 1.74 mean; whether that ~2% spread exceeds `σ_a` decides whether `a` is one constant |
| Residual-structure diagnostic per case | Whether the affine model is valid across the envelope |
| Δρ, Δ`a` vs resolution | Confirms the stability rule; feeds F04 |
| Stability envelope (which (Re, H, RES) diverge) | Honest statement of instrument limits |

## Decision rule

- **Pass:** mean ρ ≥ 0.95, `|slope of a on Re|` not significantly different from
  zero, cylinder within literature range → the instrument is qualified, C3 stands.
- **Marginal:** systematic `a` drift under 10% → qualified *with* a stated
  correction, applied identically to real and generated shapes so G04 remains a
  valid comparison.
- **Fail:** drift over 10% → the solver cannot support cross-condition magnitude
  claims. G04 becomes a within-condition comparison only, and the paper says so.

## Consequence

- **Supported →** §4.1 and Appendix A qualify the instrument in about half a page,
  and every later ρ carries its authority.
- The **grid-convergence bar drops from mandatory to advisable** under the
  instrument framing, but only at an ML-for-engineering venue. If the paper falls
  back to Physics of Fluids, this experiment must be extended — note the coupling
  to the venue decision at the August gate.

## Results

_Not yet run._
