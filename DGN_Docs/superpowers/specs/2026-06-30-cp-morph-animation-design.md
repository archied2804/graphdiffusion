# Cp-morph generation animation — design

**Date:** 2026-06-30
**Status:** approved (pending spec review)

## Goal

Produce a gif/mp4 that reuses the interactive-notebook sampling framework
([`CpCurveEditor`](../../../src/graph_diffusion/visualisation/cp_editor.py) +
[`sample_shapes_from_cond`](../../../src/graph_diffusion/postprocessing/inference.py)),
but *non-interactively*: the Cp spline programmatically morphs through four target
Cp curves — three in-distribution and one out-of-distribution — and the conditional
diffusion model regenerates one boundary shape at each target. This shows
"real-time" conditioned shape generation as a self-running clip for the report/talk.

"Drag bars" in the original request = the draggable spline **handles** of the editor;
there is no live drag-coefficient (Cd) panel (Cd only comes from OpenLB offline).

## Inputs

- **Model:** `outputs/EXP-021_long_train/checkpoint.pt` (canonical single-Cp model),
  config `configs/EXP-021_long_train.yaml`.
- **In-distribution targets:** three `*_ref_cp.csv` from
  `cfd/ellipse2d/shapes/eval_grid/` (columns `x_over_c,cp_ref`), auto-selected to
  span the dataset range — sort the `eval_grid/manifest.csv` cases by `Re` and take
  the lowest, middle, and highest so the clip visibly traverses the distribution.
- **OOD target:** `cfd/ellipse2d/shapes/ood_systematic/pc1_p3sig_target_cp.csv`
  (columns `x_over_c,cp_target`; `ood_dist = 2.33`, 3σ along PC1+), used as the
  climax frame and flagged as OOD.

Each CSV is resampled onto the editor's dense `x/c ∈ [0,1]` grid via `np.interp`.

## Motion & sampling

Chosen behaviour: **smooth morph, resample at targets**.

- Pre-sample all four shapes **once**, up front: `sample_shapes_from_cond(model,
  template, cond_vec, n_samples=1, seed=fixed)` per target. Single sample = fast;
  four reverse-diffusion runs total, none per-frame.
- Frame sequence: hold target 0 → ease-in-out tween 0→1 → hold 1 → tween 1→2 →
  hold 2 → tween 2→3 → hold 3. The right-panel shape shows the most-recently-reached
  target's pre-sampled shape and swaps on arrival.
- Tween interpolates the **handle y-values** (not the dense curve) with an
  ease-in-out (smoothstep) profile, so the handles slide the way a hand would drag
  them. The spline and DCT-8 curves are recomputed from the handles each frame.
- Defaults: `n_tween = 25` (1 s), `n_hold = 20` (0.8 s), `fps = 25` →
  ~155 frames ≈ 6 s.

## Layout (per frame)

- **Left** — editor view: grey cubic spline through the handles, blue DCT-8
  reconstruction (what the model sees), red handle dots. Axis limits fixed across
  the clip from the union of all target curves.
- **Right** — generated boundary shape as a closed polygon. Axis limits fixed from
  all four pre-sampled shapes (no jitter). Title shows the current target name; when
  on the OOD target the title is red and carries an "OOD" badge.

## Components

### `src/graph_diffusion/visualisation/cp_morph.py` (new)

Filesystem-free; no imports from `model/`. May import the DCT helpers from
`data.pOnEllipseConditional` (as `cp_editor` already does).

- `cp_spline_and_dct(handle_x, handle_y, x_grid, k_modes) -> tuple[spline, dct_curve]`
  — pure helper computing the natural cubic spline through the handles and its
  DCT-k reconstruction. **Extracted from `CpCurveEditor`**, which is refactored to
  call it, so both render identical curves (removes duplication).
- `build_handle_keyframes(targets, handle_x, x_grid) -> np.ndarray` — sample each
  dense target Cp `(T_targets, N_grid)` at the handle x-positions → `(T_targets,
  n_handles)` keyframes. Pure.
- `tween_handles(keyframes, n_tween, n_hold) -> list[FrameSpec]` — build the
  per-frame handle-y arrays (smoothstep between consecutive keyframes + hold
  frames). Each `FrameSpec` carries `handle_y` and `target_idx` (the index of the
  most-recently-reached target, used to pick the right-panel shape and OOD flag).
- `write_cp_morph_animation(targets, shapes, labels, ood_flags, handle_x, x_grid,
  k_modes, out_path_mp4, out_path_gif, n_tween, n_hold, fps) -> None` — renders via
  `matplotlib.animation.FuncAnimation`; writes MP4 (ffmpeg if available, warn-and-
  skip otherwise) and GIF (pillow), mirroring
  [`write_trajectory_animation`](../../../src/graph_diffusion/visualisation/plotting.py).
  Takes **pre-sampled `shapes`** (list of `(N_nodes, 2)` arrays) so it is fully
  testable without the model.

### `scripts/make_cp_morph_animation.py` (new)

The wiring (data loaders live under `scripts/`, per repo convention):

1. Load EXP-021 checkpoint + config; build the dataset to get a graph template and
   its `theta` grid (`template_thetas`).
2. Read the three `eval_grid` `*_ref_cp.csv` (Re-spanning) and the
   `pc1_p3sig` OOD `*_target_cp.csv`; resample each onto the dense grid → `targets`.
3. For each target: DCT-encode (`dct_ii`) → `cond_vec`; `sample_shapes_from_cond(...,
   n_samples=1, seed=0)`; `radial_to_xy` → `(N_nodes, 2)` polygon.
4. Call `write_cp_morph_animation` → `outputs/EXP-021_long_train/cp_morph.{mp4,gif}`.

CLI flags: `--checkpoint`, `--config`, `--ood-case` (default `pc1_p3sig`),
`--n-indist` (default 3), `--out-dir`, `--fps`, `--device`.

## Testing

`tests/test_cp_morph.py`, `torch.manual_seed(0)` at top:

- `cp_spline_and_dct`: output shapes `(N_grid,)`, DCT curve finite; parity with the
  old `CpCurveEditor` internals on a fixed handle set.
- `build_handle_keyframes`: returns `(T_targets, n_handles)`; values match
  `np.interp` of the targets at handle x.
- `tween_handles`: total frame count = `T*n_hold + (T-1)*n_tween`; first/last frame
  of each tween equal the bracketing keyframes; `target_idx` is non-decreasing and
  ends at `T-1`.
- `write_cp_morph_animation` smoke test: 2 fake targets + 2 stub `(N,2)` shapes,
  GIF only, into `tmp_path`; assert the file exists and is non-empty.

Existing `tests/test_visualisation.py` must still pass after the `CpCurveEditor`
refactor.

## Out of scope

- No live Cd / drag-coefficient panel (no real-time CFD).
- No human interactivity (this is a scripted clip, not a widget).
- No retraining; reuses the EXP-021 model as-is.

## Acceptance

- `outputs/EXP-021_long_train/cp_morph.gif` (and `.mp4` if ffmpeg present) shows the
  spline morphing through three in-distribution Cp targets and one 3σ OOD target,
  with the generated shape swapping at each, the OOD frame visibly flagged.
- Quality gate passes: `pytest`, `ruff`, `black --check`, `mypy`.
