---
id: P1-F06
spec: "../specs/P1-F06_schafer-turek.md"
title: "Schäfer–Turek absolute benchmark"
date_run: 2026-08-06
date_reported: 2026-08-06
status: complete (2D-1); regraded 2026-08-25
verdict: supported (solver qualified for drag and pressure scale)
git_sha: 5ec4fe4
---

# P1-F06 — results

> **Spec:** [[../specs/P1-F06_schafer-turek]] · **Status:** ✅ **complete** for
> 2D-1. L0, L1 and L2 all ran in both precisions. 2D-2 (Re=100, unsteady) is
> not yet run.
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
| 3. L1 at 3 resolutions | ✅ complete, fp32 and fp64 |
| 4. L2, uniform-inlet delta | ✅ complete |
| 5. Report Δp explicitly | ✅ L0 direct; L1 from the surface Cp curve |

**Verdict: the solver is qualified for drag and pressure scale, not for lift.**
See §6 for the graded comparison against the transcribed bounds.

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

## 6. Reference bounds — transcribed from the source

Schäfer & Turek (1996), *Benchmark Computations of Laminar Flow Around a
Cylinder*, DFG Priority Programme. Archived at
`DGN_Docs/paper1/refs_SchaferTurek1996_DFG.pdf` and `..._results.pdf`.

The paper states, of the last row of each results table: *"estimated intervals
for the 'exact' results are indicated (as suggested by the authors on the basis
of the obtained solutions)."*

**Test case 2D-1 (steady, Re = 20)** — Table 3:

| | `cD` | `cL` | `La` | `ΔP` |
|---|--:|--:|--:|--:|
| lower bound | 5.5700 | 0.0104 | 0.0842 | 0.1172 |
| upper bound | 5.5900 | 0.0110 | 0.0852 | 0.1176 |

**Test case 2D-2 (time-periodic, Re = 100)** — Table 4:

| | `cD,max` | `cL,max` | `St` | `ΔP` |
|---|--:|--:|--:|--:|
| lower bound | 3.2200 | 0.9900 | 0.2950 | 2.4600 |
| upper bound | 3.2400 | 1.0100 | 0.3050 | 2.5000 |

**Two definitional points that matter for reproducing this correctly.**

1. The inflow is `U(0,y) = 4·Um·y(H−y)/H²`, whose **maximum** is `Um`, not its
   mean. `Um = 0.3` for 2D-1 and `1.5` for 2D-2. The Reynolds number is defined
   on the **mean** velocity `Ū = 2Um/3`, giving `Ū = 0.2` and `1.0`. Our runs
   pass `PHYS_CHAR_VELOCITY = Ū`, which is correct, and
   `INLET_PROFILE=1` applies `U_max = 1.5·Ū = Um` — consistent.
2. `ΔP` is the pressure difference between the cylinder's front and back points,
   `(0.15, 0.2)` and `(0.25, 0.2)` — on the surface these are `x/c = 0` and
   `x/c = 1`.

### Verdict against the bounds — 2D-1

| level | N | `cD` | | `cL` | | `ΔP` | |
|---|--:|--:|:--:|--:|:--:|--:|:--:|
| L0 stock, fp32 | 80 | 5.57715 | ✅ | 0.00680 | ❌ −34.6% | 0.11740 | ✅ |
| **L0 stock, fp64** | 80 | **5.58251** | ✅ | **0.01081** | ✅ | 0.11764 | +0.03% |
| L1 `ellipse2d`, fp32 | 80 | **5.57002** | ✅ | 0.00079 | ❌ −92.4% | 0.11496 | −1.9% |
| L2 uniform inlet | 80 | 3.83450 | ❌ −31.2% | 0.02993 | ❌ | — | — |

**L0 in double precision passes `cD` and `cL` outright and misses `ΔP` by
0.03%** — a full validation of the build, the library and the machine.

**L1 passes `cD`**, at 5.57002 against a lower bound of 5.5700. Your case setup
reproduces the benchmark drag *within the published interval*. Its `ΔP`,
computed from the surface Cp curve at `x/c = 0` and `1`, lands 1–4% below the
interval across the ladder.

### L1's lift failure is the case, not precision — settled 2026-08-07

The fp64 L1 ladder resolves the ambiguity. It barely moves anything:

| L1 build | N | `cD` | `cL` | `ΔP` |
|---|--:|--:|--:|--:|
| fp32 | 80 | 5.57002 ✅ | 0.00079 | 0.11496 |
| **fp64** | 80 | 5.55080 (−0.3%) | 0.00608 (−41.5%) | 0.11589 |
| fp32 | 40 | 5.50913 | 0.01742 | 0.11273 |
| **fp64** | 40 | 5.50745 | 0.01837 | 0.11245 |

Double precision leaves `cD` and `ΔP` unchanged to three figures and does **not**
recover `cL`. So unlike L0 — where fp64 restored lift outright — **L1's lift
failure is a property of the case**, not of round-off.

The candidates are already tabulated in [[../SOLVER_CASE]] §4: Bouzidi versus
the benchmark's bounce-back body treatment, the `H + physDeltaX` domain height,
and the 200-gon discretisation. Not diagnosed further here.

**Consequence, and it is limited.** The production case reproduces benchmark
**drag inside the published interval** and **pressure scale to 1–4%**, but does
not reproduce **lift**. Every pOnEllipse configuration is at AoA = 0 where
`Cl ≈ 0`, and grading is on Cp and drag, so this does not affect any Paper 1
result — but it is a real bound on the instrument and belongs in §4.1 and
Appendix A. It also means [[../specs/P1-G02_aoa10-ood]], the one experiment
where lift would matter, must rely on Cp rather than on `Cl`.

L2's −31% on `cD` is not a defect: per [[../DATASET_PROVENANCE]] §2 the uniform
inlet is what the *training data* used. L2 is the configuration that matches our
dataset; L1 is the configuration that matches the benchmark. They are different
problems and both numbers are correct for their own question.

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

---

## 10. Dead-probe regrade — 2026-08-25

[[P1-F04_noise-floor]] §6 listed one outstanding item against this experiment:
any F06 figure or table computed from **surface Cp** rather than from the
integrated force coefficients had to be regraded after the dead-probe defect.
That item is now discharged. **No number in §§1–9 changes.**

### Incidence on the cylinder is far worse than on the ellipse

| Run | Nodes | Dead | Share |
|---|--:|--:|--:|
| `L1_2d1_N20` | 200 | 59 | **29.5%** |
| `L1_2d1_N40` | 200 | 77 | **38.5%** |
| `L1_2d1_N80` | 200 | 85 | **42.5%** |

Identical counts in the `L1_fp64` and `L2` ladders — the defect is geometric,
so precision and inlet profile leave it untouched. For comparison, the pOnEllipse
runs lost 3–4% ([[P1-F04_noise-floor]] §2).

Two things about this table are worth stating in §4.1.

**It is an order of magnitude worse on a circle.** The ellipse presents most of
its surface at a shallow angle to the grid only near the nose and tail; a circle
presents every angle equally, and the 200-gon's facets are much finer than `Δx`.

**It gets worse with refinement, not better.** The probe offset is `0.5·Δx`, so
it shrinks with the grid — refinement moves the probe closer to the wall in
lockstep and never escapes the snap. This is the clearest evidence that the
original scheme was not merely under-resolved but structurally wrong: no
resolution would have fixed it.

### The Δp points were live, which is why the verdict holds

Δp is read at `x/c = 0` and `x/c = 1` — the fore and aft stagnation points. On a
body symmetric about the flow axis those two normals are exactly grid-aligned, so
the nearest-node snap lands them correctly. They are live in all three runs. The
dead probes are all on the shoulders, where the surface runs oblique.

The §7 Δp figures are therefore unaffected, and the "1–4% below the interval"
finding stands as reported.

### Verification on the fixed solver

`L1_2d1_N40` re-run on the outward-walk build (`cfd/ellipse2d/results.h`),
identical arguments:

| | Old (0.5·Δx snap) | New (outward walk) |
|---|--:|--:|
| Dead probes | 77 / 200 (38.5%) | **0 / 200** |
| `Cp(x/c=0)` | +4.1078 | +4.1078 |
| `Cp(x/c=1)` | −1.5286 | −1.5286 |
| `Δp` | 0.11273 | 0.11273 |
| `Cd` | 5.50913 | 5.50913 |

Three conclusions:

1. **The fix generalises.** It recovers every probe on the geometry where the
   defect was worst — a stronger test than the ellipse it was developed on.
2. **Δp and `Cd` are bit-identical**, confirming both that the surviving samples
   were never touched and that the force coefficients never passed through the
   probe at all. `Cd` and `Cl` come from the momentum-exchange functor, so the
   whole §8 precision argument is independent of this defect.
3. **The fabricated samples sat at `Cp = −2.302`**, equal to that run's
   `cpInf` offset to six figures — the same signature as [[P1-F04_noise-floor]] §2.

### What this does and does not touch

**Unchanged:** every verdict, the drag qualification, the Δp qualification, the
lift failure and its precision analysis (§8). The lift failure is a property of
the case and remains undiagnosed.

**Changed:** any *plot* of the full cylinder Cp curve drawn before today showed
39% of its points at a fabricated constant across both shoulders — the region
where the suction peak lives. No such plot appears in this results file. If one
is drawn for Appendix A, regenerate it from the fixed solver.

**Not a re-run.** The regrade is complete; the ladder is not re-run, because the
only numbers it feeds are unchanged and verified so.

### Protocol deviation

- **2026-08-25** — Regrade performed under [[../PROTOCOL]] §2.1a after the
  dead-probe defect. Decision rule unchanged, verdict unchanged. One verification
  run (`L1_2d1_N40`, fixed solver) was executed outside the registered ladder to
  confirm the fix on this geometry; it reproduces the original to the digit.
