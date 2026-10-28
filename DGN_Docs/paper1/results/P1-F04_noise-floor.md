---
id: P1-F04
title: "Grading noise floor — results"
spec: "[[../specs/P1-F04_noise-floor]]"
date: 2026-08-17
status: reported
verdict: supported
tags: [paper1, results, F04, noise-floor, gate]
---

# P1-F04 — grading noise floor

**Verdict: SUPPORTED**, but only after a solver defect found *by* this experiment
was fixed. The raw sweep was FALSIFIED. Both numbers are reported below; §2 is
the reason they differ and is the more important result.

Runner `scripts/paper1/run_p1_f04.sh` · grader `scripts/paper1/p1_f04_grade.py` ·
data `outputs/paper1/P1-F04_noise-floor/grading.json`

---

## 1. What was run

27 solver runs: 3 shapes spanning the envelope, each run 9 times with the
geometry held **bit-identical** and one nuisance factor varied at a time.

| Shape | Dataset idx | Re | H | nodes |
|---|--:|--:|--:|--:|
| `lowre_thin` | 31 | 502 | 5.99 | 64 |
| `mid` | 2824 | 750 | 5.50 | 76 |
| `highre_thick` | 4018 | 880 | 6.00 | 60 |

| Factor | Levels | Should not matter because |
|---|---|---|
| `RESOLUTION` | rule, rule ±25% | above the stability rule the result is converged |
| node phase | 3 rolls of the point list | same body, different node offset |
| `AVG_WINDOW_CHORDS` | 15, 25, 35 | time-averaging window |
| `MAX_PHYS_T` | 40, 50, 60 | statistical convergence |

The node-phase variants roll the polygon's point list, so the geometry is
identical to floating point while every node moves along the surface
(`p1_f04_make_variants.py` asserts `sort(rolled) == sort(original)`). Anything
that moves under a roll is pure discretisation noise.

---

## 2. What this experiment actually found — dead surface probes

The raw sweep gave `σ_ρ = 0.0181`, tripping the pre-registered falsifier
(`2σ_ρ = 0.036 > 0.03`). It also gave a **non-monotonic** resolution response:
for two of three shapes the ±25% resolution variants scored ρ ≈ 0.998 while the
baseline at the intermediate resolution scored 0.93–0.97. Resolution does not
work that way, so the runs were inspected rather than the verdict accepted.

**Cause.** `results.h` places each surface probe half a lattice spacing outside
the body along the node normal, then snaps it to the *nearest* lattice node
(`cuboidDecomposition.getLatticeR`). Where the surface runs at a shallow angle
to the grid, that snap lands the probe **back inside the body**. The
time-averaged pressure functor returns exactly `0` on a solid cell *and still
reports success*, so the writer emitted a sample of `−cpInf` — the negated
free-stream offset — identically at every affected node, with a time-RMS of
exactly zero.

Three independent confirmations:

1. Every affected point in a run carries the **same** `Cp`, equal to `−cpInf`
   from that run's log to six decimals (`mid_phase0`: offset `−0.0604146`,
   dead-point `Cp = 0.060415`).
2. `Cp_rms` is **exactly** `0.0`. A time-averaged probe in an unsteady flow
   cannot produce that; the accumulator never received a sample.
3. The affected points sit at `Cp ≈ 0` in the middle of a suction peak, giving
   a residual of ~1.0 in Cp — the single largest error on the curve.

**Incidence.** Not rare, and not confined to F04:

| Sweep | Runs affected | Points affected |
|---|--:|--:|
| P1-F04 | 23 / 27 | 71 / 1800 (3.9%) |
| P1-F03 | 62 / 82 | 250 / 8052 (3.1%) |

Worst individual runs reach 12% of surface nodes. **Every graded Cp number
produced by this project before 2026-08-17 is affected.**

### 2.1 The two fixes

**Metric side** — `cp_metrics.dead_probe_mask` rejects samples with an exactly
zero `Cp_rms` or a non-finite `Cp`; `read_cp_csv` now drops them by default and
the graders report the count per run. Applies to data already on disk.

**Solver side** — `results.h` now walks outward along the normal (up to 6 ×
0.5 Δx), accepting the first offset that reads the fluid, and leaves a node
NOT-FOUND rather than emitting a fabricated value. Sampling slightly further
from the wall is benign: `∂p/∂n ≈ 0` across a thin boundary layer, which is what
makes wall pressure recoverable from a near-wall cell at all.

### 2.2 The existing runs do not need re-running

`mid_phase0` was re-run from scratch with the fixed solver and compared against
the same case filtered on the metric side:

| | n | ρ | a | ε_aff | Cd | Cl |
|---|--:|--:|--:|--:|--:|--:|
| old run, filtered | 72 | 0.99896 | 0.8884 | 0.0159 | 0.8595 | +0.00107 |
| new run, guard fixed | 76 | 0.99894 | 0.8879 | 0.0160 | 0.8588 | +0.00074 |
| **difference** | | **0.00002** | **0.0005** | 0.0001 | **0.0007** | 0.00033 |

Every difference is below a fifth of the corresponding σ in §3. **The metric-side
filter is a validated stand-in for the solver fix**, so the ~85 existing F03/F04
runs are recoverable by regrading and only future runs need the rebuilt binary.
This is the difference between a day of regrading and a week of re-running.

---

## 3. The noise floor

Pooled within shape (the between-shape spread is physics and must not enter);
all 27 runs, dead probes filtered.

| Metric | mean | **σ** | CV | **2σ** | **3σ** |
|---|--:|--:|--:|--:|--:|
| **ρ** | 0.9985 | **0.00108** | 0.11% | **0.00215** | **0.00323** |
| **a** | 0.8707 | **0.03521** | 4.04% | **0.07042** | **0.10563** |
| **Cd** | 0.9542 | **0.04742** | 4.97% | **0.09483** | **0.14225** |
| **Cl** | −0.0011 | **0.01542** | — | **0.03084** | **0.04625** |

`Cl`'s CV is meaningless — the mean is ~0 for a symmetric body at zero
incidence, which is the correct physical answer. Use the absolute σ.

Against the pre-registered hypothesis (`σ_ρ < 0.005`, `σ_a < 0.05`): **both
hold.** The falsifier (`2σ_ρ > 0.03`) is not met — 0.00215.

### 3.1 Which knob dominates

| Metric | phase | resolution | avg window | phys time | dominant |
|---|--:|--:|--:|--:|---|
| ρ | 0.00002 | 0.00043 | 0.00089 | 0.00182 | phys time |
| a | 0.00029 | 0.02514 | 0.03558 | 0.05017 | phys time |
| Cd | 0.00039 | 0.02348 | 0.05114 | 0.07218 | phys time |
| Cl | 0.00027 | 0.01106 | 0.01592 | 0.01228 | avg window |

**Node phase contributes essentially nothing** (σ_ρ = 2×10⁻⁵, two orders below
every other factor). This is a genuine and quotable result for C2: the solver's
grading is insensitive to where the boundary nodes sit, so a node-count or
node-placement effect measured downstream is the *model's*, not the instrument's.

**Statistical convergence dominates everything else.** `MAX_PHYS_T` 40 → 60
moves Cd by more than resolution does. The averaging window is the cheapest
place to buy precision if any later experiment needs a tighter floor.

### 3.2 Within-run vs between-run unsteadiness

| Shape | σ_Cd (between runs) | mean Cd_std (within run) | ratio |
|---|--:|--:|--:|
| `lowre_thin` | 0.0436 | 0.0606 | 1.4 |
| `mid` | 0.0352 | 0.0565 | 1.6 |
| `highre_thick` | 0.0601 | 0.1092 | 1.8 |

The flow's own unsteadiness exceeds the run-to-run scatter by only 1.4–1.8×.
The spec's criterion was "within-run ≫ between-run"; 1.6 is not ≫. The time
average is working but the window is **not generously long** — consistent with
`MAX_PHYS_T` dominating §3.1. Any experiment whose primary metric is `Cd` or
`Cl` should use `MAX_PHYS_T ≥ 60`, and this is a live risk for **B04** and
**G01**.

---

## 4. Companion analysis — what a ρ value means

Chance-level distributions over 300 training curves, 4000 random pairs
(`p1_f04_chance_level.py`, `chance_level.json`). Dataset-vs-dataset, so
unaffected by the probe defect.

| reference | median | p5 | p95 | min |
|---|--:|--:|--:|--:|
| random pair of dataset shapes | **0.9954** | 0.9805 | 0.9995 | 0.9684 |
| dataset corpus mean as predictor | **0.9969** | 0.9932 | 0.9993 | 0.9899 |

**Two unrelated shapes correlate at ρ = 0.9954, and a predictor that ignores its
input entirely scores 0.9969.**

### 4.1 Matched cross-grading, in the paper's own units

F03's 15 envelope runs, each graded against every shape's target (post-fix):

| pairing | median ρ |
|---|--:|
| solver vs **its own** target | 0.99896 |
| solver vs **another shape's** target | 0.99669 |
| **signal** | **+0.00226** |

With `σ_ρ = 0.00108`, the discrimination signal is **2.1σ** — by this suite's own
rule, *marginal*. 10.5% of wrong pairings still beat the median correct pairing.

Ranking is the more honest read:

| | value | chance |
|---|--:|--:|
| top-1 (own target ranked first) | **33.3%** | 6.7% |
| top-3 | **66.7%** | 20% |
| median rank | 2 of 15 | 8 |

So ρ *does* carry information — 5× chance at top-1 — but its entire usable
dynamic range is **0.9967 → 0.9990**. A headline of "ρ = 0.95" would sit *below*
the chance level of 0.9954.

### 4.2 Consequence for the protocol

**ρ cannot be the headline metric.** Three changes follow, and all three are now
made:

1. **Report rank / top-k as the discriminative metric.** It is discriminative by
   construction, has a defined chance level, and needs no extra solver runs —
   the cross-grading matrix is free once the runs exist.
2. **Report ρ on a chance-referenced axis**, never 0 → 1. State the 0.9954 floor
   beside every ρ in the paper.
3. **`Cd` is the better scalar.** It has real dynamic range across the envelope
   (0.82–1.16, a 40% span against a 5% noise floor) where ρ has 0.2%.

[[../PROTOCOL]] §2.1 and [[../REPORT_OUTLINE]] both placed ρ as the headline;
both are amended.

---

## 5. Protocol deviations

Dated, per [[../PROTOCOL]] §9.

- **2026-08-17** — The spec says 18 solver runs; 27 were run (9 per shape rather
  than 6) to give each of the four factors a full three levels. More runs than
  registered, same design.
- **2026-08-17** — The dead-probe defect (§2) was found during this experiment
  and fixed mid-flight. The pre-registered decision rule was **not** changed; the
  same rule was applied to corrected data. The raw FALSIFIED verdict is reported
  above alongside the corrected one, and the diagnosis (§2) was reached from the
  non-monotonic resolution response, *before* it was known which way the verdict
  would move. The correction was not chosen to pass the rule.
- **2026-08-17** — The OpenLB build configuration used for every production run
  was not recorded anywhere in the repository. It is now checked in at
  `cfd/ellipse2d/config/production_gpu.mk` (GPU_CUDA, `PARALLEL_MODE := NONE`,
  fp32, `CUDA_ARCH 120`). Reconstructed from the run banners of the completed
  sweeps, not from memory.

---

## 6. What this unblocks and what it invalidates

**Unblocks.** The protocol freezes: [[../PROTOCOL]] §6 now carries real numbers.
**B03**'s solver half, **G01**, **G04** and **C04** all list F04 as a dependency
and are now clear on that count.

**Requires regrading** — every experiment already reported that consumed solver
Cp. Done for **F03** (see its results file; verdict moved FALSIFIED → MARGINAL).
Done for **F06** (2026-08-25, see its §10): incidence on the cylinder is 29–42%,
far worse than the ellipse, but the two Δp sample points are grid-aligned and were
live, so no F06 number changes. A verification run on the fixed solver recovers
**200/200** probes and reproduces Δp and `Cd` to the digit.

**Unaffected.** **F01** (partition), **F02** (metric definitions), **D01**
(conditioning, analytic), **B03** part 1 (PCA on shapes, no solver), **F05**
(training only).

---

## 7. Files

| | |
|---|---|
| Runner | `scripts/paper1/run_p1_f04.sh` |
| Variant generator | `scripts/paper1/p1_f04_make_variants.py` |
| Grader | `scripts/paper1/p1_f04_grade.py` |
| Chance level | `scripts/paper1/p1_f04_chance_level.py` |
| Guard (metric side) | `src/graph_diffusion/postprocessing/cp_metrics.py::dead_probe_mask` |
| Guard (solver side) | `cfd/ellipse2d/results.h`, surface-probe loop |
| Build config | `cfd/ellipse2d/config/production_gpu.mk` |
| Data | `outputs/paper1/P1-F04_noise-floor/{grading,chance_level}.json` |
