---
title: "The ellipse2d solver case — configuration, options and known deviations"
date: 2026-08-06
tags: [paper1, methods, cfd, openlb, solver]
status: reference
---

# The `ellipse2d` solver case

> **What this file is.** A reference for the CFD case used to grade every
> generated shape: how it is configured, what its options do, and where it
> knowingly departs from the benchmark it is validated against. It exists so
> the methods section and Appendix A can be written from a checked source.
>
> **What this file is not.** Not report prose. Numbers here are to be cited,
> not pasted.
>
> Results live in [[results/P1-F03_solver-qualification]] (envelope and
> resolution behaviour) and [[results/P1-F06_schafer-turek]] (absolute
> accuracy).

---

## 1. What the case is

`cfd/ellipse2d/` is a 2-D lattice-Boltzmann case built on **OpenLB 1.9.0**,
derived from the stock `examples/laminar/cylinder2d` but converted to an **open
domain**: a body in a free stream rather than in a duct.

| | |
|---|---|
| Scheme | D2Q9, BGK, single relaxation time τ = 0.545 |
| Body treatment | **Bouzidi** curved no-slip (stock `cylinder2d` uses bounce-back) |
| Geometry source | CCW polygon point list (`.dat`), so any shape loads through one path |
| Resolution rule | `RES = max(100, 0.2·Re)` — cells per **chord**, not per semi-minor axis |
| Precision | `float` (single) in the production build — see §5 |

### Material scheme

| # | Region | Default treatment |
|:--:|---|---|
| 1 | bulk fluid | BGK |
| 2 | top **and** bottom strips | free-stream velocity Dirichlet `(U,0)` |
| 3 | left strip | velocity inlet, uniform `(U,0)` |
| 4 | right strip | pressure outlet, `p = 0` |
| 5 | body polygon | Bouzidi no-slip |
| 6 | sponge zone | `SpongeLayerDynamics`, absorbing, upstream of the outlet |

The free-stream top/bottom (material 2) and the sponge (material 6) are the two
substantive departures from `cylinder2d`, and both follow from the open-domain
choice: there are no duct walls, and the outlet needs protection from reflected
vortices.

---

## 2. Benchmark options (added 2026-08-05 for P1-F06)

Three parameters let the same binary run a **duct** configuration without
disturbing any other experiment. **All three default to the open-domain
behaviour**, so every P1-F01…F05 run is unaffected.

| Parameter | Default | Alternative |
|---|:--:|---|
| `INLET_PROFILE` | `0` uniform `(U,0)` | `1` parabolic, `U_max = 1.5·U_mean` |
| `WALL_NOSLIP` | `0` free-stream material 2 | `1` bounce-back duct walls |
| `BODY_Y_OFFSET` | `0` on the centreline | physical displacement from `H/2` |

Two design points worth stating:

- **A parabolic inlet without walls throws.** A Poiseuille profile is only
  defined between walls, so `INLET_PROFILE=1` with `WALL_NOSLIP=0` is rejected
  in `prepareLattice` rather than producing a meaningless run.
- **The profile maths is borrowed, not reimplemented.** `Poiseuille2D` with
  `distance2Wall = 0.5·physDeltaX` is taken verbatim from the stock example, so
  the L0→L1 comparison isolates the *case*, not the inlet definition.

The sponge is disabled with the pre-existing `SPONGE_CHORDS = 0`.

### The uniform inlet is a modelling choice, not an oversight

`lattice.h:15` records that the Poiseuille inlet was deliberately replaced with
a uniform one when this case was derived. That is correct for an open domain —
a free stream has no wall-bounded profile — but it had never been quantified.
`L2` in [[specs/P1-F06_schafer-turek]] measures its cost for the first time by
running the benchmark geometry with the uniform inlet.

---

## 3. The three P1-F06 levels

| Level | Binary | Configuration | Isolates |
|:--:|---|---|---|
| **L0** | stock `cylinder2d` | unmodified | the OpenLB build, library functors, machine |
| **L1** | `ellipse2d` | parabolic inlet, no-slip walls, ST geometry | **this case setup** |
| **L2** | `ellipse2d` | ST geometry, **uniform** inlet | cost of the open-domain simplification |

`L0 → L1` separates *"OpenLB is fine, my case is wrong"* from *"the library
behaves differently than I assume"*. `L1 → L2` prices the inlet choice.

---

## 4. Known deviations from the Schäfer–Turek geometry

These are real and must be stated wherever L1 numbers are quoted. None is a
bug; each is a consequence of running the benchmark through the production
case rather than a bespoke one — which is the point of L1.

| Deviation | Size | Why it exists |
|---|---|---|
| Channel height is `H + physDeltaX`, not `H` | +0.24% at N=100, +0.98% at N=25 | Inherited `cylinder2d` trick keeping the body off the node line so the top/bottom strips stay symmetric. **Grid-dependent, so it does not vanish under refinement in the usual way** — it shrinks, but the blockage ratio changes with N |
| Cylinder is a 200-gon | area deficit 0.0164% | Geometry enters through the polygon loader, as every dataset shape does |
| Body treatment is Bouzidi | — | The production choice; stock `cylinder2d` uses bounce-back. A genuine L0↔L1 difference and a candidate explanation for any residual gap |
| `RESOLUTION` counts cells per **chord** | — | For a cylinder, chord = diameter, so it coincides with the benchmark's convention |

The first item deserves emphasis: because the domain is `H + physDeltaX` tall,
**the blockage ratio is a weak function of resolution**. Any resolution study on
this case is therefore not a pure grid-convergence study. That is a limitation
of the case, and it is a candidate explanation for the non-monotonic ρ reported
in [[results/P1-F03_solver-qualification]] §4.

---

## 5. Floating-point precision

The production build sets `FLOATING_POINT_TYPE := float` in `config.mk`.

This matters more here than in most CFD. In LBM the pressure field is recovered
from density fluctuations of order `Ma²`, i.e. from **small differences between
numbers close to 1**. At `Ma ≈ 0.07`, `ρ − 1 ∼ 5×10⁻³`, so single precision
retains roughly four significant figures of the pressure signal before the
subtraction, and fewer after.

A parallel double-precision build exists at
`/home/m22729ad/Software/OpenLB-1.9-0-double/` — a separate tree, so the
production float build is untouched. Build it with:

```bash
cd /home/m22729ad/Software/OpenLB-1.9-0-double/olb-release
find . -name '*.o' -delete && find . -name 'libolbcore.a' -delete
make -j6 && cd examples/laminar/cylinder2d && make -j4
```

> ⚠️ Copying a tree and running `make` is **not** enough: stale `.o` files are
> relinked and you get the float binary back with a double config. Verify with
> `grep -o "DEFAULT_FLOATING_POINT_TYPE=[a-z]*"` on the build log, and by
> checking the results actually differ.

Measured effect at N=10 (2D-1): drag and pressure drop agree to `2×10⁻⁶`
relative, **lift differs by 1.0%**. That is the expected pattern — lift is a
small near-cancelling quantity and is the first thing precision costs you.

---

## 6. Reproducibility floor

Re-running an identical case with a **recompiled** binary reproduces
`Cd` to **0.034%** and `Cl` to 0.43% (measured on `re0501_h50_idx0018`, against
the value recorded in [[results/P1-F03_solver-qualification]]).

The default code path was logically unchanged across that recompile, so this is
compiler/GPU-ordering non-determinism, not a behavioural difference. Two
consequences:

- Solver results are **not bit-reproducible** across rebuilds. Quoting more than
  three significant figures on `Cd` implies a precision the pipeline does not
  have.
- [[specs/P1-F04_noise-floor]] should include a recompile as one of its noise
  sources, alongside seeds and resolution.

---

## 7. Operating the case

```bash
# Production (open domain) — every P1-F01..F05 experiment
./ellipse2d shape.dat --REYNOLDS 754 --CHANNEL_HEIGHT 5.0 \
    --RESOLUTION 160 --MAX_PHYS_T 50 --AVG_WINDOW_CHORDS 25

# Schäfer–Turek 2D-1 (L1) — see scripts/paper1/run_p1_f06.sh
./ellipse2d cylinder_d0p1.dat --REYNOLDS 20 --PHYS_CHAR_VELOCITY 0.2 \
    --RESOLUTION 40 --MAX_PHYS_T 16 \
    --INLET_PROFILE 1 --WALL_NOSLIP 1 \
    --CHANNEL_HEIGHT 0.41 --BODY_Y_OFFSET -0.005 \
    --UPSTREAM_CHORDS 1.5 --DOWNSTREAM_CHORDS 19.5 --SPONGE_CHORDS 0
```

Geometry is generated by `scripts/paper1/p1_f06_make_cylinder.py`, which prints
the exact parameter set it implies.

| Script | Purpose |
|---|---|
| `scripts/paper1/p1_f06_make_cylinder.py` | Benchmark cylinder polygon |
| `scripts/paper1/run_p1_f06.sh` | L0/L1/L2 ladders, both configurations |
| `scripts/paper1/run_p1_f03_resume.sh` | Envelope + resolution sweep, restartable |

---

## 8. Reference values

> ⚠️ **To be transcribed from the source before publication.** Schäfer & Turek
> (1996), *Benchmark computations of laminar flow around a cylinder*, DFG
> Priority Programme results. The bounds must come from the paper, not from a
> secondary source or from memory.

The only value confirmed from a primary artefact in this repository is
OpenLB's own `examples/laminar/cylinder2d/case.h:313`, which plots a
Schäfer–Turek drag reference of **5.58** for 2D-1 (Re = 20). That is a
secondary source and is cited as such.

`Cl`, `Δp` and the 2D-2 Strouhal bounds are **not yet verified** and are marked
as such wherever they appear in [[results/P1-F06_schafer-turek]].
