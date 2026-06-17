---
tags: [changelog, features]
---

# Feature Changelog

Tracks code-level additions and modifications that support experiments.
Each entry links to the experiment that motivated it.

## 2026-06-17 — Generated-shape export for OpenLB inverse-design validation (EXP-041)

Motivated by: [[EXP-041_openlb_generated_inverse_design]]

Drives the first independent check of the inverse-design claim: sample the conditional
generator for target `Cp` curves and run the resulting shapes through the EXP-040 OpenLB
loop. No `src/` and no `cfd/ellipse2d/` solver code changed — the validated EXP-040 build
is reused as-is, and `scripts/compare_cp_baseline.py` (from EXP-040) does the target-vs-measured comparison.

### New files

| File | Purpose |
|------|---------|
| `scripts/export_generated_for_openlb.py` | Sample the EXP-022 checkpoint for given target `Cp` curves; reconstruct each boundary polygon; export it as an `x y` `.dat` in **raw** and **Fourier-smoothed** (low-pass `r(θ)`, 6 modes) variants; also write the decoded target `Cp(x/c)` and a manifest (per-shape Re / H) |

### Reused

- `scripts/compare_cp_baseline.py` — OpenLB-measured vs target `Cp` correlation + affine fit.
- `cfd/ellipse2d/` — the EXP-040 GPU OpenLB case, unchanged.
- Generator: `outputs/EXP-022_T1000/checkpoint_best.pt` (K=8 DCT single-`Cp` conditioning, T=1000, CFG=3.0).

---

## 2026-06-17 — OpenLB 2D Cp validation loop (EXP-040)

Motivated by: [[EXP-040_openlb_cp_baseline]]

First Phase-1 physical-validation tooling: an independent 2D OpenLB (lattice-Boltzmann)
flow solver that measures surface `Cp(x/c)` from first principles, to grade the shape
generator's `Cp` against something other than its own pressure head. No `src/` code
changed (the only Python reads `pOnEllipse` HDF5 column constants).

### New files

| File | Purpose |
|------|---------|
| `scripts/export_ellipse_for_openlb.py` | pOnEllipse HDF5 → angular-sorted CCW boundary polygon (`sim_<idx>.dat`), reference `Cp(x/c)` CSV, and Re/H/a/b manifest |
| `scripts/compare_cp_baseline.py` | Overlay OpenLB vs dataset `Cp`; report correlation + affine fit (`Cp_olb ≈ a·Cp_ref + b`) to separate shape match from a normalisation offset |
| `cfd/ellipse2d/ellipse2d.cpp` | OpenLB 1.9.0 2D driver — param defaults, CLI, create→prepare→simulate |
| `cfd/ellipse2d/geometry.h` | `MyCase`, mesh + material map, `IndicatorSDF2D` winding-number polygon body (material 5) |
| `cfd/ellipse2d/lattice.h` | BC setup (velocity inlet, free-stream Dirichlet top/bottom, pressure outlet, Bouzidi body), rest-init, launch ramp, sponge zone |
| `cfd/ellipse2d/results.h` | Tail-window time-averaging, MPI-reduced surface-`Cp` probe, force + `Cp(x/c)` writers |
| `cfd/ellipse2d/README.md` | Physical setup, conventions, build/run instructions |

### Conventions established

- chord = major axis = 1.0 (semi-major = 0.5, flow-aligned); `charL = chord`; `ν = U·chord/Re`.
- HDF5 cols `[x, y, Re, H, p0..p100]`; `Re ∈ [500,1000]`; column 4 is channel height `H`, not AoA.
- `Cp` referenced to freestream static pressure (drives affine offset `b ≈ 0`).

### Key fixes (see experiment log for the reasoning)

- Convergence gate replaced by deterministic tail-window averaging + run-end dump (averaging never started on a shedding signal).
- Surface-`Cp` probe gathers across MPI ranks via `MPI_SUM` (empty CSV under domain decomposition).
- Resolution scaled with Re (`RES ≈ 0.2·Re`) to hold Mach ≈ 0.13 (instability at `Re ≳ 750`).
- GPU port: sample at stat cadence instead of every step (~33 → ~1400 MLUPs).
- Acoustic-resonance fix: rest-initialise interior + graded `SpongeLayerDynamics` outlet (material 6); `avgRho` breathing ±0.15 → ±0.0001, `Cp` corr 0.98 → 0.998.

---

## 2026-04-13 — Circle experiment infrastructure

Motivated by: [[EXP-001_circle_radial_baseline]]

### New classes

| Class | Module | Purpose |
|-------|--------|---------|
| `UnitCircleDataset` | `graph_diffusion.data.circledataset` | Ring graphs with Fourier-perturbed radial profiles |
| `ComputeAngularEdgeFeatures` | `graph_diffusion.data.transforms` | `[sin(Δθ), cos(Δθ)]` edge features from `data.pos` |

### Modified classes

| Class | Change | Backward-compatible? |
|-------|--------|---------------------|
| `ScoreNetwork` | Added `input_dim: int \| None` — optional input/output linear projection | ✅ Yes (`None` = no change) |
| `GraphDiffusionModel` | Added `clamp_range: tuple \| None` to `sample()` | ✅ Yes (`None` = no clamping) |

### New files

| File | Purpose |
|------|---------|
| `src/graph_diffusion/data/circledataset.py` | `UnitCircleDataset` |
| `configs/circle_radial.yaml` | Circle experiment config |
| `train_circle.py` | Training + visualisation script |
| `tests/test_circledataset.py` | 15 unit tests |
| `tests/test_angular_features.py` | 7 unit tests |
| `tests/test_circle_integration.py` | 5 integration tests |

### Test impact

- Before: 109 tests
- After: 139 tests (+30)
- All passing, zero regressions

---

## 2026-05-07 — Full training infrastructure (EXP-005)

Motivated by: [[EXP-005_circle_radial_full]]

### Modified classes

| Class | Change | Backward-compatible? |
|-------|--------|---------------------|
| `train_circle.py` | Added `CosineAnnealingLR` scheduler; early stopping with best-checkpoint saving | ✅ Yes (opt-in via config) |

---

## 2026-05-07 — Rich node features (EXP-006)

Motivated by: [[EXP-006_circle_radial_rich-features]]

### Modified classes

| Class | Change | Backward-compatible? |
|-------|--------|---------------------|
| `UnitCircleDataset` | Added `include_curvature`, `include_arc_length` params; `_compute_curvature` and `_compute_arc_length_fraction` methods | ✅ Yes (both default to `False`) |
| `train_circle.py` | Passes `include_curvature`, `include_arc_length` to `UnitCircleDataset` from config | ✅ Yes |
| `scripts/postprocess_circle.py` | Passes same flags; evaluation uses `x[:, 0]` (r only) for comparability | ✅ Yes |
