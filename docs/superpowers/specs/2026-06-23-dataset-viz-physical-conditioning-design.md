# Design: Dataset visualisation, OpenLB domain correctness, systematic sampling, and Re-conditioning

Date: 2026-06-23
Branch context: `branch/exp04x-doc-figures` (report + conference prep)

## Motivation

Two orders of business ahead of the first-year report and conference presentation:

1. A way to **visualise the whole pOnEllipse dataset** (design-space → drag, the
   distribution of Cp curves, design-space coverage).
2. Make the **OpenLB validation framework physically faithful** to the dataset:
   the channel height must match the dataset per sample, the physical Reynolds
   number should condition the generator so its outputs are more physical, and
   model evaluation needs **systematic sampling** of the design space with a
   trackable naming scheme (the current evaluation touches only a handful of
   sims along a 1-D Re sweep).

The work decomposes into four parts. Parts A–C are tooling/validation (no
retrain). Part D is an architectural change requiring a retrain and is the long
pole; A–C are designed to feed it.

## Established facts (verified during brainstorming)

- HDF5 `data/ellipse/raw/pOnEllipseTrain.h5`, key `data`, shape `(5701, 96, 105)`.
  Columns: `[x, y, Re, H, p0..p100]`. **Verified empirically:** col 2 = Re ∈
  [500, 1000] (501 unique), col 3 = H = channel height ∈ [5, 6] (101 unique).
  The legacy `_H5_COL_AOA = 3` alias is a misnomer — the column is H.
- Convention: dataset table lists *full* axis lengths (major a = 1, minor
  b = 0.5–0.8); code stores *semi*-axes (semi-major 0.5, semi-minor ≈ 0.25–0.4),
  chord = major axis = 1.0, flow-aligned along x.
- The model's global attribute `u` is currently all-zeros (except `normalised`
  feature mode, which stores a, b). Re/H are **not** model conditioning today.
- The inverse-design model (EXP-020/030) conditions on the first K DCT-II modes
  of `⟨Cp⟩(x/c)`, stored as `Data.cond` and injected into `u`.
- OpenLB case `cfd/ellipse2d/` accepts `--REYNOLDS`, `--CHANNEL_HEIGHT`,
  `--SEMI_MINOR_B`, `--RESOLUTION`. The C++ `CHANNEL_HEIGHT` default is `8.0`
  (`ellipse2d.cpp:120`) — a silent-wrong-domain foot-gun if the flag is omitted.
  Baseline runs pass per-sim H (5.3/5.8/5.78); the OOD sweep hardcodes 5.5.
- Current evaluation sampler `scripts/export_ellipse_for_openlb.py::select_indices`
  is a 1-D linspace over Re only — ignores thickness and channel height.

## Part A — Dataset visualisation suite

**Goal:** publication-quality figures characterising the whole dataset.
Decision: **produce all figures including the extras**; final figure selection
for the report is deferred.

**Structure:**
- `src/graph_diffusion/visualisation/dataset_overview.py` — pure functions, each
  taking the loaded HDF5 array (or a thin dataset accessor) and a matplotlib
  `Axes`/`Figure`, returning the figure. No file I/O in the plotting functions.
- `scripts/plot_dataset_overview.py` — CLI that loads the HDF5, calls each
  plotting function, and writes PNGs (and PDFs for LaTeX) to an output dir.

**Figures (all produced):**
1. **Design-space → drag contour.** x = Re, y = semi-minor b; filled
   contour / hexbin of **pressure-drag coefficient**
   `Cd_p = ∮ Cp (n·x̂) ds` integrated from each sim's surface Cp + geometry.
   *Honesty constraint:* the dataset is pressure-only, so this is **pressure
   drag**, not total drag (no skin friction). Axis label and caption state this
   explicitly. Total Cd is available only for the OpenLB-validated sims.
2. **Cp envelope.** All (or stratified-sampled) `Cp(x/c)` curves overlaid on one
   axis, coloured by Re with a colourbar, plus a mean ± std band. A second
   variant coloured by b.
3. **Design-space coverage scatter.** (Re, b) points coloured by `Cd_p`, with a
   marker overlay showing where the Part-C evaluation samples land.

**Extras (also produced):**
4. Shape-outline gallery across b.
5. `Cd_p` vs Re line plot, binned by b.
6. 2-D DCT/PCA projection of the Cp curves coloured by Re/b (a learned-latent
   view, in case "latent space" meant an embedding rather than the physical
   design space).

**Pressure-drag helper:** a small reusable function computes `Cd_p` from a sim's
ordered boundary polygon and per-node Cp (trapezoidal surface integral of
`Cp * n_x` along the closed contour). Shared by figures 1, 3, 5 and reusable in
postprocessing.

## Part B — OpenLB domain correctness

Decision: **drop `--SEMI_MINOR_B`** from the per-sample driver — the body
geometry (and hence b) is fully determined by the generated/exported shape
polygon (`.dat` point file). (Implementation will confirm `SEMI_MINOR_B` is not
load-bearing for geometry construction; if it is used only for diagnostics it is
removed from the driver, and if it turns out to size anything it is flagged.)

1. **Remove the silent-wrong default.** Change `ellipse2d.cpp` so
   `CHANNEL_HEIGHT` has no usable default: if the flag is unset, abort with a
   clear error (or default to a sentinel that fails the geometry sanity check).
2. **Per-sample H/Re end-to-end.** Add a driver
   `cfd/ellipse2d/runs/run_from_manifest.sh` (or a small Python launcher) that
   reads a case `manifest.csv` and launches each case with its own
   `--REYNOLDS` and `--CHANNEL_HEIGHT` from the manifest row. Replaces hardcoded
   constants like the OOD sweep's `--CHANNEL_HEIGHT 5.5`.
3. **Verification.** The exporter asserts that the H (and Re) written to the
   manifest equals the HDF5 column value for that sim; the comparison script
   asserts the run used the manifest's H.

## Part C — Systematic design-space sampling + naming scheme

Decision: **do not track b** as a sampling axis — b is emergent from the
generated shape, not a controllable input. Sampling axes are **Re and H**.

- **Sampler:** replace the Re-only `select_indices` with a structured sampler
  over (Re, H). Grid (default) or Latin-hypercube; each target (Re, H) point is
  **snapped to its nearest real dataset sim** (nearest in normalised (Re, H)
  space) so we always validate against a ground-truth Cp. Includes the four
  corner/boundary points to test the model's bounds.
- **Naming scheme:** `re{RRRR}_h{HH}_idx{NNNN}`, e.g. `re0500_h53_idx0000`
  (Re zero-padded to 4 digits; H × 10, 2 digits; source HDF5 index 4 digits).
  Sortable and unambiguous. Manifest gains columns
  `name, idx, Re, H, a, b, n_nodes` (b retained as a *recorded property* of the
  chosen sim, just not a sampling axis).
- **Shared namespace:** the same sampler later selects generation targets for the
  retrained model, so evaluation cases and CFD cases share one naming namespace.

## Part D — Augment the Cp-conditioned model with Re (and optional H)

Decision: **augment** the existing Cp-DCT inverse-design model with the physical
Reynolds number; **produce both configs** — Re-only and Re+H. Column semantics
were verified (col 2 = Re, col 3 = H) so the dataset code samples the right data.

**Rationale (and the caveat):** the inverse map (target Cp → shape) is
Re-dependent — the same shape yields different Cp at different Re, so a target Cp
maps to different shapes across Re. The current model marginalises over a hidden
Re, making the inverse problem ill-posed; this is a plausible contributor to the
in-distribution magnitude-creep documented in EXP-042 (amplitude bias is
model-side). Conditioning on Re disambiguates the map → more physical outputs,
for the cost of one scalar in `u`. Caveat: Re is *partially* redundant with the
Cp magnitude, so expect an improvement, not a transformation. H is included as
requested but flagged low-expected-impact (narrow 5–6 range, small blockage
variation); its main value is domain-awareness flowing into OpenLB. H is a
separable toggle so Re-only vs Re+H can be ablated.

**Data changes (`pOnEllipseConditionalDataset`):**
- Read Re (col 2) and H (col 3) per sim. Normalise: `Re_norm = (Re-500)/500`,
  `H_norm = (H-5)/1`. Both land in [0, 1].
- Append to the conditioning vector: `cond = [DCT modes…, Re_norm, (H_norm)]`.
  A flag controls whether H is appended. The cache filename encodes the new
  conditioning width so caches don't collide.
- Existing classifier-free-guidance dropout applies to the whole `cond` vector,
  so the unconditional branch stays truly unconditional.

**Configs:**
- `configs/EXP-031_cp_re_cond.yaml` — Cp + Re.
- `configs/EXP-032_cp_re_h_cond.yaml` — Cp + Re + H.
- Both branch from the current Cp-only config; only the conditioning width and
  dataset flags differ.

**Inference → loop closure:** at sampling time the user specifies the Re (and H)
to generate for; that same value is written into the case manifest (Part C
naming), so OpenLB (Part B driver) and the eval comparison automatically use the
matching domain — closing the validation loop.

**Acceptance check:** does Re-conditioning move the in-distribution affine slope
`a` (magnitude-creep metric from EXP-042) toward the solver-validated ~1.78,
versus the Cp-only baseline? A concrete, report-ready result.

## Testing

- **Part A:** unit tests for the `Cd_p` integrator (closed circle → ~0 net drag
  along a symmetric reference; sign/orientation correctness). Smoke test that
  each plotting function returns a figure without error on a small slice.
- **Part B:** test the manifest-equals-HDF5 assertion; test that the driver fails
  loudly when CHANNEL_HEIGHT is unset.
- **Part C:** test the sampler returns the requested count, snaps to valid
  in-range sims, includes the four corners, and that names are unique and sortable.
- **Part D:** `torch.manual_seed(0)`; assert `cond` width = K (+1 Re, +1 H) and
  shape after `forward()`; assert normalisation maps Re/H endpoints to {0, 1};
  round-trip that an unconditional pass still works (CFG dropout).

## Out of scope (YAGNI)

- Conditioning on b/thickness (emergent, not an input).
- A physics-only (no-Cp) conditioning track — not requested; augmentation chosen.
- Total-drag (skin-friction) reconstruction from the pressure-only dataset.
- Changing the DCT-mode count or the feature_mode representations.

## Suggested execution order

1. **Part A** (independent, immediate report value, read-only).
2. **Part B** (small, de-risks every CFD run).
3. **Part C** (depends on B's manifest; feeds D's inference).
4. **Part D** (long pole: dataset change → both configs → retrain → ablation).
