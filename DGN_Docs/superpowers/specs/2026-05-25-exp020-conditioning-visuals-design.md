---
date: 2026-05-25
topic: EXP-020 conditioning experiments + visualisations
status: draft
parent_experiment: EXP-020
---

# EXP-020 Conditioning Experiments + Visualisations — Design

## Context

EXP-020 (Fourier-DCT pressure conditioning with classifier-free guidance and a
joint pressure-prediction head) is implemented and smoke-trained for 5 epochs
on CUDA. All 150 unit tests pass; the quality gate is green. The next step is
to (a) train the model in full and (b) produce the visualisations required to
demonstrate the inverse-design claim and the diffusion process itself.

This spec covers both: the training run plan, and the visualisation tooling
that turns a trained EXP-020 checkpoint into a publishable set of figures.

## Goals

1. Full 200-epoch training of EXP-020 with the existing config, producing a
   converged checkpoint.
2. **Figure A — conditioning grid (4 × 6)**: prove that distinct target Cp
   curves produce distinct shapes, including one out-of-distribution
   (synthetic asymmetric) target.
3. **Figure B — forward + reverse trajectory**: static filmstrip plus an MP4
   and GIF of the reverse diffusion process for one chosen example, showing
   what the diffusion is doing under the hood.
4. **Figure C — CFG sweep (1 × 3)**: same synthetic target, sampled at
   `w ∈ {1.0, 3.0, 7.0}`, head-predicted Cp overlaid on the target.
5. A reusable `graph_diffusion.visualisation` module so future experiments
   (EXP-021+) can pull the same plotting functions without duplicating code.

## Non-goals

- Aerofoil dataset loader.
- Cp-match scalar metric in `evaluation_report.json` (follow-up).
- DPS smoke (`dps_guidance_weight = 0.5`) — defer until CFG sweep results
  are visible.
- Comparison against EXP-010/011/012 metrics in the same script.

## Architecture

### New module: `src/graph_diffusion/visualisation/`

A small reusable plotting library with the dependency rule of the existing
`data/` package: no imports from `model/` except via the public model API
(it takes a `GraphDiffusionModel` instance, never reaches inside).

```
src/graph_diffusion/visualisation/
├── __init__.py
├── trajectory.py     # collect_forward, collect_reverse
└── plotting.py       # plot_conditioning_grid, plot_trajectory_filmstrip,
                      # write_trajectory_animation
```

**Public API:**

```python
# trajectory.py
def collect_forward(
    model: GraphDiffusionModel,
    template: Data,
    snapshots: list[int],          # 1-indexed timesteps to record
    seed: int = 0,
) -> list[torch.Tensor]:
    """Run forward diffusion on template.x and snapshot x_t at each t in
    snapshots. Returns a list of (N, C) tensors, one per snapshot."""

def collect_reverse(
    model: GraphDiffusionModel,
    template: Data,
    cond: torch.Tensor,
    snapshots: list[int],          # 1-indexed timesteps to record
    guidance_scale: float = 1.0,
    seed: int = 0,
) -> list[torch.Tensor]:
    """Run reverse diffusion conditioned on cond, snapshot x_t at each t in
    snapshots. Returns a list of (N, C) tensors, one per snapshot. The
    standard model.sample() loop is duplicated here so we can capture
    intermediate states without modifying the model API."""

# plotting.py
def plot_conditioning_grid(
    target_cps: list[np.ndarray],       # length R; each (N_cp_grid,)
    sample_shapes: list[list[np.ndarray]],  # R × S; each (N_nodes, 2) (x,y)
    head_pred_cps: list[np.ndarray],    # length R; mean head prediction
    head_pred_stds: list[np.ndarray],   # length R; std across S samples
    row_labels: list[str],              # length R
    figsize: tuple[float, float] = (16, 10),
) -> matplotlib.figure.Figure:
    """4 × (2 + S) grid: |target Cp | predicted Cp ±σ | sample 1 … sample S|"""

def plot_trajectory_filmstrip(
    forward_snapshots: list[np.ndarray],   # length F; each (N, 2) (x,y)
    reverse_snapshots: list[np.ndarray],   # length F; each (N, 2) (x,y)
    timesteps: list[int],                  # length F
    target_cp: np.ndarray,                 # (N_cp_grid,)
    figsize: tuple[float, float] = (14, 5),
) -> matplotlib.figure.Figure:
    """2 × F shape panels + side Cp panel."""

def write_trajectory_animation(
    reverse_snapshots: list[np.ndarray],   # all reverse steps, length T
    target_cp: np.ndarray,
    out_path_mp4: Path,
    out_path_gif: Path | None = None,
    fps: int = 25,
) -> None:
    """Encode reverse-diffusion animation via matplotlib.animation.FuncAnimation.
    MP4 via ffmpeg writer; GIF via Pillow writer."""
```

### New script: `scripts/postprocess_exp020.py`

```bash
python scripts/postprocess_exp020.py \
    --experiment-dir outputs/EXP-020_fourier_pressure_conditioning \
    --config configs/EXP-020_fourier_pressure_conditioning.yaml \
    --device cuda \
    --n-samples 4 \
    --n-targets 3 \
    --target-seed 0
```

**Responsibilities:**
1. Load checkpoint + config; rebuild dataset, model, head, schedule.
2. Pick targets:
   - **`n_targets` training Cp targets** (default 3) via greedy
     farthest-first selection over `dataset.cond` tensors: start from a
     deterministic seed index, repeatedly pick the dataset member whose
     minimum L2 distance to the already-picked set is largest. Guarantees
     well-separated targets with no extra dependency (pure numpy).
   - **1 synthetic asymmetric Cp** (always appended, outside `--n-targets`):
     `cp_mean + 0.3 * np.sin(np.pi * x_over_c)`, then DCT-encode to K=8
     modes to obtain `cond_synth`.
3. For each target, sample 4 shapes with deterministic per-sample seeds; record
   the head's Cp prediction for each sample.
4. Pick the synthetic asymmetric target as the trajectory subject; run
   `collect_forward` and `collect_reverse` at 6 log-spaced timesteps
   `[1, 4, 16, 50, 100, 200]` for the filmstrip; run full 200-step
   `collect_reverse` for the animation.
5. Run a CFG sweep on the synthetic target at `w ∈ {1.0, 3.0, 7.0}`,
   one sample per `w`.
6. Call the visualisation library to render Figure A/B/C; write
   `figure_a_conditioning_grid.png`, `figure_b_trajectory.png`,
   `figure_b_reverse.mp4`, `figure_b_reverse.gif`, `figure_c_cfg_sweep.png`
   into `--experiment-dir`.

### Training run

```bash
python train.py \
    --config configs/EXP-020_fourier_pressure_conditioning.yaml \
    --epochs 200 \
    --device cuda \
    --output outputs/EXP-020_fourier_pressure_conditioning/generated_shapes.png
```

Run in the background. A Monitor watches the log for `val_loss=`,
`Saved best checkpoint`, `NaN`, `Error`, `Traceback`, and `Killed` so a
divergence stops me early. Early stopping (`patience=20` in the YAML) caps
the run length if val loss plateaus.

Final artefacts: `checkpoint.pt`, `checkpoint_best.pt`, `loss_log.json`,
`generated_shapes.png` (training-time gallery), `tensorboard/*`.

## Data flow

```
configs/EXP-020_*.yaml ─┐
                        ├─→ train.py ─→ outputs/EXP-020_*/checkpoint_best.pt
data/ellipse/*.h5 ──────┘                      │
                                               ▼
                            postprocess_exp020.py
                                ├─ targets via K-means + synth
                                ├─ samples via model.sample()
                                ├─ trajectory via visualisation.trajectory
                                └─ figures via visualisation.plotting
                                       │
                                       ▼
                    outputs/EXP-020_*/figure_{a,b,c}_*.{png,mp4,gif}
```

## Testing

### Unit tests in `tests/test_visualisation.py`

- `test_collect_forward_shapes`: snapshots are returned in input order, each
  has the expected `(N, C)` shape, dtype is float32, dataset seed makes the
  output deterministic.
- `test_collect_reverse_shapes`: same contract for the reverse pass; assert
  the last snapshot (t=1) matches `model.sample()` output to floating-point
  tolerance when seeds match (sanity-check that the loop wasn't broken by
  the snapshotting wrapper).
- `test_plot_conditioning_grid_axes_count`: returned `Figure` has
  `R × (2 + S)` axes.
- `test_plot_trajectory_filmstrip_axes_count`: returned `Figure` has
  `2F + 1` axes (forward row + reverse row + Cp panel).
- `test_write_trajectory_animation_nonempty`: writes an MP4 to `tmp_path`
  whose file size > 1 kB. Skip if `ffmpeg` is not on `PATH`.

All tests use a fixed seed, a tiny synthetic graph (8 nodes), and CPU; full
suite should still run in < 10 s.

### Manual verification

- Visually confirm Figure A row 4 (synthetic asymmetric) shows broken
  symmetry vs the three ring-like training-Cp rows.
- Visually confirm Figure C: shapes at `w=7` track the head-predicted Cp
  closer to the target than `w=1`.
- Visually confirm Figure B reverse filmstrip transitions from noise to
  recognisable ring.

## Quality gate

Before declaring complete, all of:

- `uv run pytest tests/ -q` — 0 failures (155+ tests after additions)
- `uv run ruff check src/ tests/` — clean
- `uv run black --check src/ tests/` — clean
- `uv run mypy src/` — 0 issues
- Visual files written and non-empty (size > 1 kB each)

## Commit plan

1. `feat(visualisation): add trajectory snapshotting module`
2. `feat(visualisation): add plotting and animation helpers`
3. `test(visualisation): add test_visualisation.py`
4. `feat(train): add postprocess_exp020.py inverse-design script`
5. (After training run completes) `docs(experiments): update EXP-020 with results`

## Risks & mitigations

- **CFG produces degenerate shapes at high w.** If `w=7` collapses to a
  point, document it; the figure still tells a story (regime where guidance
  breaks down).
- **Greedy farthest-first picks numerical outliers rather than informative
  exemplars.** Mitigation: assert each picked target's nearest-neighbour
  distance to the rest of the dataset is below the dataset's 90th-percentile
  pairwise distance; if violated, log a warning so the operator can re-seed.
- **Synthetic asymmetric target is too aggressive and produces noise.**
  Mitigation: enable DPS gradient guidance (`dps_guidance_weight=0.5`)
  only if the un-DPS run fails; document either way.
- **MP4 encoder missing on the host.** Mitigation: GIF fallback is already
  in scope; the script logs a warning and skips MP4 if ffmpeg is absent.
- **Long-running training crashes mid-run.** Mitigation: the Monitor
  watches for `Traceback`, `NaN`, `Killed`; checkpoints are saved per-epoch
  so we can resume from the best.
