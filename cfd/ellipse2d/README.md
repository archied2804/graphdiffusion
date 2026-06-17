# ellipse2d — 2D OpenLB validation case

Physical-validation tool for the graph-diffusion shape generator. Runs a single
closed-polygon body (an elliptical cylinder, or a diffusion-generated shape) in a
2D incompressible free-stream flow and reports **time-averaged Cd/Cl** and a
**surface Cp(x/c) curve**.

Built against **OpenLB 1.9.0** (CPU). The hand-written case lives here, under
version control; it is symlinked into the OpenLB example tree to build.

## Physical setup (after Lino et al., Phys. Fluids 2022 — the dataset source)

2D incompressible flow around an elliptical cylinder, semi-major axis a=1 (major
axis aligned with the flow), semi-minor axis b, channel of height H. Re ∈
[500,1000], laminar vortex shedding. Boundary conditions:

| Material | Region        | BC                                         |
|----------|---------------|--------------------------------------------|
| 1        | bulk fluid    | BGK collision                              |
| 2        | top + bottom  | free-stream velocity Dirichlet (u=1, v=0)  |
| 3        | left          | velocity inlet (u=1, v=0)                  |
| 4        | right         | pressure outlet (p=0)                      |
| 5        | body wall     | Bouzidi curved no-slip                     |

Note: top/bottom are **free-stream Dirichlet**, NOT no-slip channel walls — this
differs from the stock `laminar/cylinder2d`.

## Geometry input

The body is an ordered closed list of boundary points, one `x y` per line (blank
lines and `#` comments ignored). It is turned into a solid via `IndicatorSDF2D`
with a winding-number polygon SDF and stamped into material 5. No STL, no HDF5.

## Conventions (the baseline-reconciliation knobs)

- **charL = chord = x-extent of the polygon** (filled in by `loadBodyPolygon`).
- **RESOLUTION = voxels per chord** (default 300).
- **ν = U·chord/Re**, U = `PHYS_CHAR_VELOCITY` = 1.
- **Cp = (p − p_outlet)/(½ρU²)**, p_outlet = 0; x/c measured from the leading edge.
- **Cd/Cl** from `SuperLatticePhysDrag2D` (already a coefficient, normalised on the
  projected frontal length from `SuperGeometryFaces2D`).

⚠️ The dataset's Re is defined on *some* reference length. This case uses the
chord. If the dataset uses a different length, ν is off by a constant factor and
the known-ellipse baseline will not reproduce — that is the intended validation
gate, not a bug. Reconcile here before trusting generated-shape results.

## Build

```bash
export OPENLB_ROOT=~/Software/OpenLB-1.9-0/olb-release
# one-time symlink (already created):
#   ln -s /home/m22729ad/Projects-Local/DGN_Sup/cfd/ellipse2d $OPENLB_ROOT/examples/ellipse2d
cd $OPENLB_ROOT/examples/ellipse2d
# OLB_ROOT must be passed explicitly: make resolves the symlink to the physical
# case dir, so the conventional relative `../../..` would miss OpenLB.
make OLB_ROOT=$OPENLB_ROOT
```

## Run

```bash
# positional arg = body point file; --NAME VALUE overrides any parameter
./ellipse2d shapes/smoke_ellipse_a1_b0p3.dat \
    --REYNOLDS 1000 --CHANNEL_HEIGHT 8.0 --SEMI_MINOR_B 0.3
```

Outputs (in `./tmp/`):

- `ellipse2d_force.csv` — `Cd_mean,Cd_std,Cl_mean,Cl_std,n_samples,t_start,t_end`
- `ellipse2d_cp.csv` — `x_over_c,Cp,Cp_rms` per boundary node
- `vtkData/` — ParaView series (geometry material map, velocity/pressure fields,
  time-averaged pressure)

## Key parameters (defaults in `ellipse2d.cpp`)

`RESOLUTION` 300 · `REYNOLDS` 1000 · `MAX_PHYS_T` 200 · `CHANNEL_HEIGHT` 8.0 ·
`UPSTREAM_CHORDS` 6 · `DOWNSTREAM_CHORDS` 17 · `WASHOUT_WINDOW_CHORDS` 10 ·
`AVG_WINDOW_CHORDS` 50 (mean/std accumulated only after transient washout, over
≥10 shedding periods).

## File layout

- `ellipse2d.cpp` — driver: param defaults, CLI, create→prepare→simulate ordering
- `geometry.h` — `MyCase`, params, mesh + material map, polygon SDF body
- `lattice.h` — BC setup, initial/temporal values, launch ramp
- `results.h` — washout detection, time-averaging, Cd/Cl + Cp(x/c) writers
- `shapes/` — body point files (`.dat`)
- `tmp/` — run outputs (gitignore)
