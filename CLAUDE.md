# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## What this project is

`graph_diffusion` implements **denoising diffusion probabilistic models (DDPM, Ho et al. 2020) on graph node features** using PyTorch and PyTorch Geometric. The primary deliverable is a radial shape generator — diffusing the radius field `r(θ)` of Fourier-perturbed unit circles — with the long-term goal of applying the same pipeline to aerodynamic mesh data (CFD).

---

## Commands

```bash
# Install (use uv)
uv pip install -e ".[dev]"

# Run all tests with coverage
pytest tests/ --cov=graph_diffusion --cov-report=term-missing

# Run a single test file
pytest tests/test_mlp.py -v

# Run a single test by name
pytest tests/test_graph_diffusion_model.py::test_compute_loss -v

# Lint
ruff check src/ tests/

# Format
black src/ tests/ && ruff check --fix src/ tests/

# Type check (strict)
mypy src/graph_diffusion

# Train circle experiment
python train_circle.py --config configs/circle_radial.yaml --epochs 100 --device cuda --output outputs/run/generated_shapes.png

# Post-process / evaluate a checkpoint
python scripts/postprocess_circle.py --checkpoint outputs/<run>/checkpoint.pt --config configs/circle_radial.yaml

# Multi-GPU training
torchrun --standalone --nproc_per_node=4 train_ddp.py --config configs/default.yaml --epochs 100 --amp
```

Pre-commit quality gate (all must pass before committing):
```bash
uv run pytest tests/ -q && uv run ruff check src/ tests/ && uv run black --check src/ tests/ && uv run mypy src/
```

---

## Architecture

Three sub-packages with a **strict one-way dependency DAG**:

```
data/          ← no imports from model/ or building_blocks/
building_blocks/ ← no imports from model/ or data/
model/         ← imports from building_blocks/ ONLY, never from data/
```

### `building_blocks/`
- **`MLP`** — fully-connected MLP with optional LayerNorm, residual connection (active only when `in_dim == out_dim`), and configurable activation (`silu` | `relu` | `gelu`).
- **`SinusoidalTimeEmbedding`** — maps timestep `t` to sinusoidal features then projects via MLP; used to inject temporal conditioning into the global attribute `u`.
- **`NoiseSchedule`** — pre-computes all DDPM schedule buffers (`betas`, `alphas_cumprod`, etc.) registered via `nn.Module.register_buffer` so they move with `.to(device)` and are saved in checkpoints. Supports `"linear"` and `"cosine"` schedules.
- **`GraphNetworkBlock`** — full Battaglia et al. (2018) GN block: `EdgeModel (φ^e) → NodeModel (φ^v) → GlobalModel (φ^u)`. Each sub-model is optional. Uses `torch_scatter.scatter` directly for aggregations.

### `model/`
- **`ScoreNetwork`** — takes a noisy graph `(G_t, t)` and predicts the noise `ε`. Time embedding is injected into the global attribute `u` (not concatenated to every node), which then propagates through the `GlobalModel → NodeModel` pathway. Stacks `n_layers` of `GraphNetworkBlock`.
- **`GraphDiffusionModel`** — top-level DDPM: owns `forward_diffusion()` (samples `x_t`), `compute_loss()` (MSE on noise prediction), and `sample()` (reverse diffusion from `x_T ~ N(0,I)`). The `sampler` argument is the extension point for DDIM/SDE.

### `data/`
- **`BaseGraphDataset`** (`InMemoryDataset`) — abstract base; subclasses implement `_build_graphs() -> list[Data]`.
- **`UnitCircleDataset`** — the primary benchmark: `n_nodes` points uniformly on `[0, 2π)`, radii `r(θ) = 1 + Σ Fourier modes`, bidirectional ring edges to `k_neighbors` nearest neighbours. Node feature `x = [r]`. Global feature vector `u` of dimension `global_dim`.
- **`SyntheticGraphDataset`** — random k-NN graphs with Gaussian node features; used for generic demos.
- **`GraphDataLoader`** — wraps PyG `DataLoader`; provides `train_loader()` / `val_loader()` with reproducible `random_split`.
- **Transforms** — `ComputeAngularEdgeFeatures`, `NormalizeNodeFeatures`, `AddSelfLoops`, `KNNGraph`, `Compose`. Applied as `pre_transform` in dataset construction.

### Data flow (circle experiment)
```
UnitCircleDataset → ComputeAngularEdgeFeatures (pre_transform)
  → GraphDataLoader (batched PyG Data with batch vector)
  → GraphDiffusionModel.compute_loss(batch)
      ├─ sample t ~ Uniform(1…T) per graph
      ├─ forward_diffusion → x_t, ε
      ├─ ScoreNetwork(noisy_data, t) → ε_pred
      └─ MSE(ε_pred, ε)
  → GraphDiffusionModel.sample(template, clamp_range=(0.5, 1.5))
      └─ x_T ~ N(0,I) → reverse DDPM steps → x_0
```

---

## Key conventions

**Import style** — always fully-qualified; no `from module import *`.

**OOP rules** — `nn.ModuleList` for variable-length module collections; `register_buffer` for non-learned tensors; `abc.ABC` + `@abc.abstractmethod` (never `raise NotImplementedError`); no business logic in `__init__`, only attribute/sub-module assignment.

**Types** — strict mypy; `torch.Tensor` (not bare `Tensor`); `list[int]` / `dict[str, Any]` (Python ≥ 3.9 lowercase generics); validate constructor args at `__init__` time with `ValueError`.

**Config contract** — no hardcoded hyperparameters inside class bodies; all come from `__init__` args sourced from YAML. Config keys must not be renamed.

**Tests** — `torch.manual_seed(0)` at the top of every test with random tensors; shape assertions after every `forward()` call; test each class in isolation before integration tests.

**Docstrings** — Google style on all public methods; module-level docstring in every `.py` file.

---

## Git workflow

Branches: `branch/<scope>` (e.g. `branch/circle-experiment`). Merge to `main` via `--no-ff`.

Conventional Commits format — every commit:
```
<type>(<scope>): <short summary>
```

Types: `feat` | `test` | `fix` | `refactor` | `docs` | `chore`  
Scopes: `data` | `building-blocks` | `model` | `config` | `train` | `integration` | `docs`

Atomic commits — one logical change per commit (one class, its tests in a separate commit, config in its own commit).

Merge commits: `merge: <description> into main`

---

## Experiments & outputs

Experiments are tracked in `docs/experiments/` (Obsidian vault). Each experiment has:
- A YAML config in `configs/EXP-NNN_*.yaml`
- Outputs in `outputs/<experiment-name>/` (checkpoint.pt, loss_log.json, evaluation_report.json, plots)
- A log in `docs/experiments/EXP-NNN_*.md` (copy `docs/experiments/_template.md`)

Current roadmap: EXP-00x circle ablation series complete (EXP-001 → EXP-006) → EXP-01x aerodynamic series (pOnEllipse dataset, HuggingFace `mariolinov/Ellipse`).

SLURM job script: `scripts/run_experiment.slurm`. Submit via:
```bash
sbatch --job-name=EXP-NNN --export=CONFIG=configs/...,EPOCHS=100,DEVICE=cuda scripts/run_experiment.slurm
```
