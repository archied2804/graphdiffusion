# Design: Systematic OOD Cp sampling, generation, validation & assessment (EXP-043)

Date: 2026-06-23
Depends on: the dataset-viz / OpenLB-domain / sampling work merged today
(`docs/superpowers/specs/2026-06-23-dataset-viz-physical-conditioning-design.md`)

## Motivation

EXP-041/042 probed out-of-distribution (OOD) inverse design with **five ad-hoc**
synthetic Cp targets (sin amplitude/sign/frequency/tilt). That was enough to show
fidelity declines with OOD distance and that a universal magnitude creep exists,
but it could not trace *where* and *how fast* inverse design breaks down, because
the OOD axis was not controlled.

This experiment builds a **systematic, principled OOD ladder**: perturb the
dataset-mean Cp curve along the dataset's **own principal Cp modes**, sweeping
amplitude in units of the dataset's std along each mode. That gives a controlled
in-distribution → border → OOD axis. Each target is generated into a shape by the
inverse-design model, validated through OpenLB, and assessed with a visualisation
suite that places the generated samples **within the dataset distribution** so the
validity of OOD sampling can be judged qualitatively and quantitatively. The same
visualisation suite is also run on the **eval_grid** in-distribution validation
(15 dataset shapes through OpenLB) completed earlier today.

## Pipeline (four stages, linear dependency)

```
Stage 1  systematic OOD Cp targets (PC-mode amplitude ladder)
   → Stage 2  model generation (inverse-design checkpoint, plain CFG)
      → Stage 3  OpenLB validation (run_from_manifest.sh, per-row RES scaling)
         → Stage 4  visualisation + written assessment (+ eval_grid)
```

## Established facts / conventions

- Inverse-design checkpoint: `outputs/EXP-022_T1000/checkpoint_best.pt` with
  `configs/EXP-022_T1000.yaml` (the checkpoint EXP-041/042 used; keeps new OOD
  results comparable). Sampling uses **plain classifier-free guidance, DPS = 0**
  (EXP-042: DPS guidance degrades OOD).
- Generation flow already exists in `scripts/export_ood_sweep.py` /
  `postprocess_exp020.py`: build target Cp on the `N_CP_GRID` grid → DCT-encode to
  the conditioning vector → sample a radial shape via CFG → decode radial→xy →
  export `.dat` + decoded `target_cp.csv` + manifest, with
  `ood_dist` = min-L2 distance to the training set in DCT-cond space.
- OOD runs use **fixed Re/H = 754 / 5.5** so the Cp-target morphology is the only
  variable.
- OpenLB run driver `cfd/ellipse2d/runs/run_from_manifest.sh` resolves manifest
  columns by header name, prefers `<name>.dat`, and auto-scales
  `RES = max(100, round(0.2·Re))` when no resolution is passed.
- Run outputs: measured Cp at `runs/<name>/tmp/ellipse2d_cp.csv`
  (`x_over_c,Cp,Cp_rms`); force at `ellipse2d_force.csv`
  (`Cd_mean,Cd_std,Cl_mean,Cl_std,n_samples,t_start,t_end`).
- The OpenLB surface Cp is **non-monotonic** in x/c (traces the full boundary), so
  the correct match metric interpolates the **monotonic target** onto the measured
  nodes — NOT the reverse (the legacy `compare_cp_baseline` direction was buggy for
  non-monotonic Cp; see validation memory / EXP-042).

## Stage 1 — Systematic OOD Cp-target sampler

**Shared primitive (new):** `src/graph_diffusion/data/cp_manifold.py`
- `dataset_cp_curves(raw, n_grid) -> np.ndarray` — every dataset sim's mean Cp(x/c)
  resampled onto a shared `n_grid` x/c grid, shape `(N_sims, n_grid)`.
- `cp_pca_basis(curves) -> CpPcaBasis` — returns the mean curve, the first few
  principal components (right singular vectors of the mean-centred curves), and the
  std of the dataset projections along each component (`proj_std`). Pure NumPy,
  depends only on `pOnEllipse` column constants (respects the data/ DAG).

This primitive is shared by Stage 1 (target construction), Stage 4 (manifold
context plot), and is the natural home for the PCA currently inlined in
`dataset_overview.plot_cp_pca` (refactor that to call it).

**Target family (16 targets):** for each principal mode `m ∈ {PC1, PC2}`, amplitude
`α ∈ {1, 2, 3, 4}` (× the dataset `proj_std[m]`), and sign `s ∈ {+, −}`:

```
target_cp(x) = mean_cp(x) + s · α · proj_std[m] · PC_m(x)
```

- α = 1 ≈ interior (in-distribution), α ≈ 2 ≈ border of the dataset spread,
  α ≥ 3 ≈ genuinely OOD. The amplitude in σ-units IS the controlled OOD axis;
  `ood_dist` (min-L2 in DCT-cond space) is also recorded as the model-space
  distance and should increase monotonically with α.
- **Naming:** `pc{m}_{sign}{α}sig`, e.g. `pc1_p2sig`, `pc2_m3sig` (sortable,
  unambiguous, encodes mode/sign/amplitude). Run-driver compatible manifest:
  columns `name,idx,Re,H,ood_dist,mode,sign,amp_sigma` (`idx` a sequential
  integer so the driver is happy; `Re`/`H` capitalised to match the driver).

This replaces the ad-hoc `perturbations()` family in a new sweep script (keep the
old `export_ood_sweep.py` intact for reproducibility of EXP-041/042).

## Stage 2 — Model generation

`scripts/export_ood_systematic.py` — mirrors `export_ood_sweep.py` but draws targets
from Stage 1's PC-mode ladder. For each target: DCT-encode, sample a shape with the
checkpoint (plain CFG, `--guidance` default matching EXP-041, `dps=0`), decode to a
polygon, write `<name>.dat` (raw, as-generated) + `<name>_target_cp.csv` + the
manifest. Output dir `cfd/ellipse2d/shapes/ood_systematic/`. Device CPU by default,
`--device cuda` optional.

## Stage 3 — OpenLB validation

Run `cfd/ellipse2d/runs/run_from_manifest.sh cfd/ellipse2d/shapes/ood_systematic`
(auto RES from Re=754 → RES≈151; fixed because Re is constant). Produces
`runs/<name>/tmp/ellipse2d_cp.csv` + force per target. ~16 serial GPU runs
(~30–45 min). The driver's `|| WARN` keeps the sweep alive if any single run fails.

## Stage 4 — Visualisation + assessment

**New plotting module:** `src/graph_diffusion/visualisation/sweep_results.py`
— filesystem-free, takes loaded arrays (respects data/→visualisation/ DAG). Reuses
`dataset_overview` for the dataset backdrop and `cp_manifold.cp_pca_basis`.

- `cp_match_metrics(target_x, target_cp, meas_x, meas_cp) -> dict` — the **corrected**
  metric: interpolate the monotonic target onto the measured nodes, then report
  Pearson `corr`, affine slope `a` and offset `b` (cpo ≈ a·target + b), and RMS.
- `plot_sample_gallery(samples, ncols)` — grid of panels, one per run: generated
  shape outline + measured Cp vs target/reference Cp overlaid, titled with ρ and `a`.
- `plot_design_space_context(raw, samples)` — sample (Re,H) points over the dataset
  cloud (coverage; for the fixed-Re/H OOD set these coincide, so this view is most
  informative for eval_grid).
- `plot_cp_manifold_context(raw, samples)` — project each sample's **target and
  measured** Cp into the dataset Cp-PCA plane; the dataset cloud is the backdrop, so
  OOD targets visibly sit outside it. Coloured by `ood_dist`/amplitude. This is the
  qualitative "how OOD" picture.
- `plot_cp_envelope_overlay(raw, samples)` — dataset mean±σ Cp band with sample
  measured (solid) and target (dashed) curves overlaid.
- `plot_results_summary(samples)` — per-sample ρ, affine `a`, and Cd vs `ood_dist`
  (reproduces the EXP-042 magnitude-creep trend from the live run outputs).

**CLI:** `scripts/plot_sweep_results.py --set {ood_systematic,eval_grid,...}` holds
the per-set filesystem loaders (manifest schema + run-dir naming + target/reference
Cp location differ per set), builds the `list[SampleRecord]`, and writes the five
figures (PNG+PDF) to `outputs/sweep_results/<set>/`. Loaders covered now:
`ood_systematic` (target Cp, `ood_dist`) and `eval_grid` (dataset reference Cp).

**Written assessment:** a short note in the EXP-043 experiment log summarising where
inverse-design fidelity holds vs breaks down across the OOD ladder (ρ and `a` vs α),
and what the manifold-context plot shows about the reachable Cp region.

## Testing

- `cp_manifold`: `cp_pca_basis` on synthetic curves — component orthonormality,
  `proj_std` ordering, mean-centring; deterministic (`np.random.default_rng(0)`).
- Stage 1 sampler: 16 targets produced; names unique/sortable; `ood_dist` increases
  with amplitude along a mode; target shape `(n_grid,)`.
- `cp_match_metrics`: ρ = 1 and a = 1 for identical curves; correct interp direction
  on a synthetic monotonic-target / non-monotonic-measured pair; affine `a` recovers
  a known scale.
- `sweep_results` plots: smoke tests (return a Figure, ≥1 axes) on a small synthetic
  `SampleRecord` list + fake `raw`.
- Stages 2–3 (model generation, OpenLB runs) are script-level, verified by execution
  (checkpoint + GPU), not unit tests — consistent with the existing CFD tooling.

## Out of scope (YAGNI)

- The Hron–Turek benchmark validation (explicitly a later, separate task).
- Re/H variation in the OOD set (fixed at 754/5.5 to isolate Cp-morphology OOD).
- Retraining or Part D physical conditioning (separate deferred spec).
- DPS guidance for OOD sampling (EXP-042 showed it hurts; plain CFG only).
- Smoothed-shape variants (raw, as-generated, per the chosen option).

## Execution order

1. Stage 1 + the `cp_manifold` primitive (+ refactor `plot_cp_pca` to use it).
2. Stage 2 generation script.
3. Stage 4 visualisation module + CLI (can be built/tested before the CFD finishes,
   against the existing `ood_sweep`/`eval_grid` sets).
4. Run Stage 2 → Stage 3 (generate + CFD sweep) → Stage 4 figures → write the
   EXP-043 assessment.
