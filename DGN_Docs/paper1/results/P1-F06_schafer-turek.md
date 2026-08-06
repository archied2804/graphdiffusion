---
id: P1-F06
spec: "../specs/P1-F06_schafer-turek.md"
title: "Schäfer–Turek absolute benchmark"
date_run: 2026-08-06
date_reported: 2026-08-06
status: partial
verdict: pending
git_sha: 5ec4fe4
---

# P1-F06 — results

> **Spec:** [[../specs/P1-F06_schafer-turek]] · **Status:** 🟡 **partial.** L0
> (library and build) is complete in both precisions and produces a decisive
> finding. L1/L2 are running.
>
> **Headline so far:** the solver reproduces the benchmark drag to **0.02%**, so
> the build and the library are sound. But the production build is **single
> precision**, and in single precision the **lift coefficient stops converging
> and becomes noise as the grid is refined** — 29.5% scatter at N=80 against
> 0.1% in double. That is a candidate explanation for the resolution
> pathologies reported in [[P1-F03_solver-qualification]].

## 1. Status against the pre-registered procedure

| Step | State |
|---|---|
| 1. L0 at stock resolution, confirm published values | ✅ complete, both precisions |
| 2. Parabolic-inlet option in `lattice.h` | ✅ implemented, flag-gated |
| 3. L1 at 3 resolutions | 🔄 running |
| 4. L2, uniform-inlet delta | 🔄 running |
| 5. Report Δp explicitly | ✅ for L0; pending L1/L2 |

**No verdict is recorded yet.** The decision rule needs L1, and the reference
bounds need transcribing (§6).

## 2. L0 — the build is sound

Stock `cylinder2d`, unmodified, 2D-1 (Re = 20). The stock example defaults to
`RESOLUTION = 10`, a demo setting; a ladder is the only meaningful form of this
check.

| build | N | `Cd` | `Cl` | `Δp` |
|---|--:|--:|--:|--:|
| float | 10 | 5.82052 | 0.015429 | 0.122479 |
| float | 20 | 5.63182 | 0.012130 | 0.118343 |
| float | 40 | 5.59011 | 0.010136 | 0.118001 |
| float | 80 | **5.58131** | *0.003458* | 0.117586 |
| double | 10 | 5.82053 | 0.015272 | 0.122480 |
| double | 20 | 5.63210 | 0.012159 | 0.118358 |
| double | 40 | 5.59131 | 0.011056 | 0.117990 |
| double | 80 | **5.58247** | **0.010815** | 0.117638 |

`Cd` converges monotonically to **5.5813 (float) / 5.5825 (double)** against
OpenLB's own Schäfer–Turek reference of **5.58** (`case.h:313`) — **0.02% and
0.04%**. Δp converges to 0.1176.

**The build, the CUDA path and the machine are not the problem.** That closes
the spec's step-1 gate: work may proceed to L1.

## 3. Finding 1 — single precision destroys the lift coefficient under refinement

Scatter of the reported coefficient over the last 20 samples of a *steady*
(Re = 20) simulation, where the true scatter should be ~0:

| build | | N=10 | N=20 | N=40 | N=80 |
|---|---|--:|--:|--:|--:|
| float | `Cl` sd/mean | 1.0% | 3.1% | 6.8% | **29.5%** |
| double | `Cl` sd/mean | 1.0% | 0.1% | 0.0% | **0.1%** |
| float | `Cd` sd/mean | 0.28% | 0.03% | 0.02% | 0.04% |
| double | `Cd` sd/mean | 0.28% | 0.03% | 0.01% | 0.00% |

Consecutive `Cl` samples at N = 80 make it plain:

```
double :  0.0108109  0.0108124  0.0108137  0.0108147  0.0108153   <- converged
float  :  0.00400    0.00522    0.00598    0.00909    0.00346     <- noise
```

In single precision the lift coefficient at N = 80 **is not converging at all**.
The scatter is comparable to the value itself, and the single-precision error
**grows with refinement** — roughly doubling per resolution doubling — while
double precision converges monotonically to 0.010815.

### Why this happens

LBM recovers pressure from density fluctuations of order `Ma²`. At `Ma ≈ 0.07`,
`ρ − 1 ∼ 5×10⁻³`, so the physical signal already sits ~2–3 decimal digits below
1 before any subtraction. Lift is then a **near-cancelling sum** of surface
contributions: refining the grid increases the number of terms while shrinking
each, so round-off accumulates while the signal does not. Drag survives because
it does not cancel — the contributions share a sign.

This is a general property of the scheme, not a bug in the case, and it is
exactly the regime the production runs occupy.

## 4. Why this matters beyond F06

The production stability rule is `RES = max(100, 0.2·Re)`, i.e. **100–200 cells
per chord** — at or above the N = 80 where single-precision lift collapsed.

Three earlier results become newly explicable:

- [[P1-F03_solver-qualification]] found **ρ non-monotonic under refinement**,
  with the *coarsest* grid sometimes giving the best agreement, and `a` moving
  by up to 25.9% from resolution alone at fixed Re. Cp is a pressure-derived
  field with the same cancellation structure as lift. F03 §4 called this
  "numerically unconverged" — the mechanism may simply be float32.
- [[P1-F02_grading-metric]] found the affine residual **structured** in every
  case. Precision noise would not by itself produce structure, so this is not a
  complete explanation, but it plausibly contributes to the scatter.
- The **magnitude story** — `a ≈ 1.74` from Year-1, ≈ 1.877 among well-fitted
  cases — rests entirely on Cp magnitudes computed in single precision at
  exactly these resolutions.

**This is the first mechanistic candidate for the whole magnitude problem**,
which is what F06 was promoted to answer. It is not yet proven: the test is to
re-run F03's resolution sweep in double and see whether `a` stabilises.

## 5. What was built

| Item | Purpose |
|---|---|
| `INLET_PROFILE`, `WALL_NOSLIP`, `BODY_Y_OFFSET` case parameters | Flag-gated benchmark configuration; all default to existing behaviour |
| `scripts/paper1/p1_f06_make_cylinder.py` | Benchmark cylinder as a 200-gon (area deficit 0.0164%) |
| `scripts/paper1/run_p1_f06.sh` | L0/L1/L2 ladders, both configurations, restartable |
| `/home/m22729ad/Software/OpenLB-1.9-0-double/` | Parallel double-precision build; production tree untouched |
| [[../SOLVER_CASE]] | Case reference: configuration, options, known deviations |

**Default-path regression.** Re-running F03's `re0501_h50_idx0018` with the
modified case gives `Cd` 1.023419 against the recorded 1.023075 — **0.034%**.
The default path is logically unchanged (it adds 0.0 and takes an `else` branch
of the original code), so this is recompilation-level non-determinism. Recorded
because it sets a **reproducibility floor**: solver results are not
bit-reproducible across rebuilds, and quoting more than three significant
figures on `Cd` implies precision the pipeline lacks. [[../specs/P1-F04_noise-floor]]
should treat a rebuild as a noise source.

**Early L1 signal** (N = 20, not yet the final ladder): `Cd` = 5.6296 against
L0's 5.6318 — **0.04%**. The case setup reproduces the stock example when given
the same configuration. To be confirmed across the ladder.

## 6. ⚠️ Reference bounds are not yet verified

The spec requires the bounds to come from Schäfer & Turek (1996), not a
secondary source. **Only one value is confirmed from a primary artefact in this
repository**: OpenLB's `examples/laminar/cylinder2d/case.h:313` plots a drag
reference of **5.58** for 2D-1.

`Cl` and `Δp` bounds for 2D-1, and all 2D-2 bounds including Strouhal, **must be
transcribed from the paper** before any verdict is recorded or any number
enters the report. Comparisons in §2 against remembered ranges are deliberately
omitted for that reason.

## 7. Protocol deviations

**Deviation from the spec's step 1.** The spec says to run L0 "at the stock
resolution". The stock resolution is `N = 10`, a demo default at which no
benchmark agreement should be expected, and taking it at face value would have
produced a false failure (`Cd` 4.3% high). A resolution ladder was run instead.
Dated 2026-08-06.

**Addition not in the spec.** The precision comparison (§3) was not
pre-registered. It was prompted by noticing `FLOATING_POINT_TYPE := float` in
`config.mk` while setting up the build. It is reported as an **exploratory**
finding under [[../PROTOCOL]] §6.1, and the confirmatory test — whether double
precision stabilises `a` in F03's sweep — has not yet been run.

## 8. Open decision

**Should the production build move to double precision?**

| | |
|---|---|
| For | `Cl` and, by extension, Cp magnitudes are demonstrably unreliable in float at production resolutions; it is a candidate root cause for F03, and possibly for the whole `a` story |
| Against | It invalidates every solver number produced so far — F03's 27 runs, all Year-1 EXP-04x results — and costs a re-run of the F03 sweep |
| Cost | Library build ~2 min; F03 re-run ~4 h |

**Not decided here.** The cheap discriminating test is to re-run F03's
resolution sweep (3 cases × 4 resolutions, 12 runs) in double and see whether
the 25.9% spread in `a` collapses. That is ~1 h and settles it without
committing to anything.

## 9. Reproduction

| | |
|---|---|
| L0 | `cylinder2d --RESOLUTION <N> --REYNOLDS 20`, both trees |
| L1/L2 | `scripts/paper1/run_p1_f06.sh` |
| Geometry | `scripts/paper1/p1_f06_make_cylinder.py` |
| Case reference | [[../SOLVER_CASE]] |
| Double build | `/home/m22729ad/Software/OpenLB-1.9-0-double/` (see [[../SOLVER_CASE]] §5 — stale `.o` files must be deleted or the rebuild silently keeps float) |
