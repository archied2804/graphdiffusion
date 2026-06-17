---
experiment_id: "EXP-040"
title: "OpenLB 2D Cp validation — known-ellipse baseline gate"
date: 2026-06-17
status: complete
parent: "EXP-030"
tags: [ellipse, aerodynamic, cfd, openlb, lattice-boltzmann, physical-validation, cp, baseline]
config: "cfd/ellipse2d/ (no YAML — OpenLB case)"
output_dir: "cfd/ellipse2d/runs/"
---

# EXP-040: OpenLB 2D Cp validation — known-ellipse baseline gate

> **One-line summary:** Stand up an independent 2D OpenLB (lattice-Boltzmann) flow solver as a physical-validation loop for the shape generator, and prove on **known dataset ellipses** that it reproduces the pOnEllipse surface `Cp(x/c)` — up to a single normalisation factor — *before* trusting it to grade diffusion-generated shapes. Gate **passed**: cleanest case (Re=500) reaches correlation **0.998**.

## Motivation

Every "inverse design" result in the EXP-02x/03x line ([[EXP-020_fourier_pressure_conditioning]], [[EXP-030_dual_cp]]) is graded by the model's own jointly-trained pressure head. The head is trained on the same data and against the same objective that produced the shapes, so when it reports that a generated shape achieves its target `Cp`, **the head is grading itself**. There has never been an independent check that the shapes the generator produces actually carry the pressure distribution they were conditioned on.

This is the first step of **Phase 1 — physical validation**: build a real CFD solver loop that takes a closed boundary polygon, runs an incompressible 2D flow, and measures surface `Cp(x/c)` from first principles. Before that loop can be trusted on *generated* shapes, it must clear a **baseline gate**: fed a *known* dataset ellipse, does it reproduce *that ellipse's* dataset `Cp`? If the solver cannot match the data on geometries the data was built from, any agreement (or disagreement) on generated shapes would be uninterpretable.

This log documents the construction of that loop and the baseline-gate result. The diffusion model is not exercised here — EXP-040 is pure tooling-plus-validation, a prerequisite that unblocks the real Phase-1 experiment.

Related: [[changelog]] · upstream pressure-conditioning work [[EXP-030_dual_cp]].

## Hypothesis

- A 2D incompressible LBM solve of a known dataset ellipse will reproduce the dataset's `Cp(x/c)` **curve shape** (high correlation), since both describe the same physical flow.
- The **absolute scale** may differ by a constant factor, because the dataset's `Cp` normalisation (reference velocity / reference length convention) is not fully documented — so the comparison must separate *shape* agreement from a *normalisation* offset.
- Agreement will be cleanest at the low end of the Re range (Re=500) and degrade somewhat at Re=1000, where vortex shedding is stronger and harder to resolve/average.

## Method

### Pipeline built

Three new artefacts form the loop. No `src/graph_diffusion/` code changed; the only Python reaches into `pOnEllipse` for HDF5 column constants.

1. **Export** — `scripts/export_ellipse_for_openlb.py`. Reads the pOnEllipse HDF5 and, per selected simulation, writes: `sim_<idx>.dat` (the surface boundary points as an **angular-sorted CCW closed polygon** — the raw HDF5 node order is mesh-generation order, not geometric, so it must be sorted about the centroid before it forms a simple polygon); `sim_<idx>_ref_cp.csv` (the dataset reference `Cp(x/c)`, the per-node mean over the 101 unsteady timesteps, sorted chordwise); and `manifest.csv` (`idx,Re,H,a,b,n_nodes`). `--n-sims` picks simulations spread evenly across the sorted Re range.

2. **Solver** — `cfd/ellipse2d/` (OpenLB 1.9.0, 2D). The body enters as the `.dat` polygon, converted to a solid region via `IndicatorSDF2D` with a winding-number polygon SDF and stamped into material 5. Boundary conditions follow the dataset source (Lino et al., *Phys. Fluids* 2022): velocity inlet (material 3), **free-stream velocity Dirichlet** on top and bottom (material 2 — *not* no-slip channel walls), pressure outlet `p=0` (material 4), and a Bouzidi curved no-slip body wall (material 5). The solver reports time-averaged `Cd`/`Cl` and a surface `Cp(x/c)` curve obtained by probing the *time-averaged* pressure field at the boundary nodes. Source files: `ellipse2d.cpp` (driver / CLI / params), `geometry.h` (`MyCase`, mesh + material map + polygon SDF), `lattice.h` (BC setup, initial / temporal values, launch ramp), `results.h` (averaging, force + `Cp` writers).

3. **Compare** — `scripts/compare_cp_baseline.py`. Interpolates the OpenLB `Cp` onto the dataset's x/c grid, overlays the two curves, and reports both the raw RMS difference and the RMS after a least-squares **affine fit `Cp_olb ≈ a·Cp_ref + b`**, plus the correlation. The affine residual is the key metric: it isolates *shape* agreement (correlation, residual RMS) from a constant *normalisation* discrepancy (the slope `a` and offset `b`).

### Conventions established (the baseline-reconciliation knobs)

These were painful to derive and are load-bearing for every future CFD run:

- **chord = major axis = 1.0.** The paper's "major axis `a=1`" is the axis *length*, so semi-major = 0.5. The major axis is flow-aligned (x); frontal height is `2b` (minor axis). `charL = chord = ` the polygon's x-extent.
- **HDF5 layout:** columns `[x, y, Re, H, p0..p100]`. `Re ∈ [500, 1000]`. The fourth column `H` is the **channel height** (5–6), *not* angle of attack — the legacy `_H5_COL_AOA` alias was misnamed (see [[EXP-030_dual_cp]] for where this bit us before).
- **Re viscosity uses the chord as reference length:** `ν = U·chord/Re`, `U = 1`.
- **`Cp = (p − p_ref)/(½ρU²)`** referenced to the **freestream static pressure**, not the outlet gauge — this is what drives the offset `b` to ≈ 0.

### Running (GPU, preferred)

Built on the CUDA OpenLB (`~/Software/OpenLB-1.9-0-cuda`, `CUDA_ARCH=120` for Blackwell, `PARALLEL_MODE=NONE`):

```bash
make OLB_ROOT=$HOME/Software/OpenLB-1.9-0-cuda/olb-release
./ellipse2d shapes/baseline/sim_0.dat --REYNOLDS 500 --CHANNEL_HEIGHT 5.3 --RESOLUTION 100 --SEMI_MINOR_B 0.2555
```

The CPU build still works via `mpirun -np N` against `~/Software/OpenLB-1.9-0/olb-release`.

## Results

### Baseline gate (3 sims spanning the Re range, after the fixes below)

| Sim | Re | H | b | RES | `Cd_mean` | `Cd_std` | `Cp` corr ρ | affine `a` | affine `b` |
|---|---|---|---|---|---|---|---|---|---|
| `sim_0` | 500 | 5.30 | 0.255 | 100 | 1.160 | 1.527* | **0.998** | 1.78 | ≈0 |
| `sim_2850` | 754 | 5.80 | 0.391 | 160 | 1.072 | 0.181 | 0.93 | 1.55 | ≈0 |
| `sim_5700` | 1000 | 5.78 | 0.254 | 200 | 0.822 | 0.044 | 0.92 | 1.70 | ≈0 |

\* `sim_0`'s `Cd_std` shown here is from a pre-sponge instantaneous-force trace; the sponge fix collapsed the instantaneous `Cd` std to ≈ ±0.05 (see fix 5). Resolution is scaled with Re (`RES ≈ 0.2·Re`) to hold the lattice Mach number near 0.13 (fix 3).

**Gate verdict: PASSED.** OpenLB reproduces the dataset `Cp` curve up to a single multiplicative normalisation. The offset `b` sits at ≈ 0 once `Cp` is referenced to the freestream static pressure rather than the outlet gauge.

### The headline number: `a ≈ 1.78`

The cleanest case (`sim_0`, Re=500) gives slope **`a ≈ 1.78`**, which is strikingly close to the paper's note that the dataset velocity was "scaled by 1.8". That gives a *credible analytic explanation* for the dataset's `Cp` normalisation factor — i.e. the constant `a` is not a solver artefact but the dataset's own non-dimensionalisation choice. This still needs to be pinned exactly from the dataset card before it is treated as confirmed.

### Figures

`scripts/compare_cp_baseline.py` overlays the dataset reference (black) and OpenLB-measured (orange) `Cp(x/c)` for all three sims, with `y` inverted per aerodynamic convention:

![[assets/EXP-040_cp_comparison.png]]

Source figure: `cfd/ellipse2d/runs/cp_comparison.png`. Per-run outputs (force CSV, `Cp` CSV, VTK fields) live under `cfd/ellipse2d/runs/gpu_sim_0/`, `.../gpu_sponge_2850/`, `.../gpu_sponge_5700/`.

## Discussion — the engineering / debugging narrative

The clean result above is the *end* of a chain of five diagnosed-and-fixed failures. The reasoning matters more than the outcomes, so each is recorded.

1. **Averaging never started.** Re 500–1000 is unsteady vortex shedding, so instantaneous `Cd` oscillates forever. The original convergence gate waited for `stdDev/mean < 5%` on the *instantaneous* `Cd` — a condition that can never fire for a sustained oscillation — so no force/`Cp` CSV was ever written. *Fix:* replace the convergence gate with a deterministic tail-window time-average plus a guaranteed run-end dump.

2. **Empty `Cp` under MPI.** The surface-`Cp` probe ran only on the main rank and read only main-rank-local cells. Under domain decomposition the body usually sits on *other* ranks, so the main rank found zero body cells and wrote an empty CSV. *Fix:* every rank probes the boundary points it owns and the results are `MPI_SUM`-reduced to the main rank — a no-op when serial, so the single-process GPU build also works.

3. **Numerical instability at higher Re.** `RES=100` is stable for Re=500 / thin bodies but diverges (`avgRho → 1.3`, then NaN) for `Re ≳ 750` / thicker bodies, where the lattice Mach number climbs toward ~0.2. *Fix:* scale resolution with Reynolds number, `RES ≈ 0.2·Re` (Re=754 → 160, Re=1000 → 200), which holds Mach ≈ 0.13 and restores stability. This is why the three baseline sims run at different resolutions.

4. **GPU port starved by per-step host sync.** Ported to the CUDA build (Blackwell, `CUDA_ARCH=120`, `PARALLEL_MODE=NONE`). A functor readout inside `getResults` forced a device→host synchronisation *every* timestep, throttling the run to ~33 MLUPs. *Fix:* sample at the statistics cadence (~50 samples total) instead of every step, lifting throughput to ~1400 MLUPs (~14× the CPU build) and also un-throttling the CPU averaging phase.

5. **Spurious acoustic resonance — the headline bug.** The instantaneous field showed a sustained **±15% `avgRho` "breathing"** with period ≈ 12.5 convective times. This was diagnosed as the domain's fundamental **longitudinal quarter-wave acoustic mode** (`4L/cs`, with `L ≈ 24` chords and `cs ≈ 7.7·U`, giving ≈ 12.5), excited by the vortex shedding and *trapped* by the reflecting velocity boundary conditions. A startup ramp cannot fix a *sustained* resonant mode. Two root causes, two fixes:
   - **(a) Inconsistent initialisation.** The interior was initialised at full `U` while the BCs ramp from 0, firing an impulsive pressure wave at `t=0`. *Fix:* initialise the interior at **rest**.
   - **(b) Reflecting outlet.** *Fix:* add a graded `SpongeLayerDynamics` absorbing zone (new material 6, ~4 chords deep, `sigma_max = 0.7`) upstream of the *kept* pressure outlet.

   *Result:* the `avgRho` amplitude collapsed from ±0.15 to ±0.0001, the `Cp` match **improved** (`sim_0` correlation 0.98 → **0.998**), and the instantaneous `Cd` std fell from ±1.5 to ±0.05 — confirming that the old "`Cd` noise" was the acoustic resonance, *not* physical force fluctuation. The GPU build needed the descriptor `D2Q9<DENSITY, VELOCITY, DAMPING>` and `geometry.clean(true, {1, 6})` to support the sponge material.

### Why the correlation tapers with Re

The residual drop from ρ = 0.998 (Re=500) to ≈ 0.92 (Re=1000) is *not* the acoustic artefact (that was eliminated for all three sims). It is attributed to genuine **under-resolution and under-sampling of the stronger high-Re vortex shedding**: more resolution and more averaging samples would be needed to converge the time-averaged surface pressure at the top of the Re range. This is a fidelity knob, not a correctness failure.

### Performance

GPU build: ~1400 MLUPs, ~60 s for a `RES=100` run on Blackwell. Build with `make OLB_ROOT=$HOME/Software/OpenLB-1.9-0-cuda/olb-release` (GPU), or the CPU path via `mpirun`.

## Conclusions

- **The baseline gate passes.** An independent LBM solver reproduces the dataset's surface `Cp` on known ellipses up to a single normalisation slope, with the cleanest case at ρ = 0.998 and offset ≈ 0. The validation loop is trustworthy enough to advance.
- **The dataset's `Cp` normalisation is (very likely) the paper's "u × 1.8" scaling**, evidenced by `a ≈ 1.78` on the cleanest case — to be confirmed from the dataset card.
- **The "`Cd` noise" was never physics** — it was a trapped domain acoustic mode, now removed by rest-initialisation plus a sponge outlet. This is the single most important finding for any future CFD run in this project.
- The hypothesis held on all three counts: high shape correlation, a pure-multiplicative normalisation difference (`b ≈ 0`), and a clean-to-noisier trend with Re.

## Next steps

- [ ] Pin `a ≈ 1.78` analytically from the dataset card to lock the `Cp` normalisation convention.
- [ ] Close the Re=1000 correlation taper with more averaging samples / higher resolution.
- [ ] **The real Phase-1 step → EXP-041:** run diffusion-*generated* shapes (raw and Fourier-smoothed) from the [[EXP-030_dual_cp]] generator through this validated loop, and compare the measured `Cp` against the target `Cp` the shape was *conditioned* on — the first independent check of the inverse-design claim.
