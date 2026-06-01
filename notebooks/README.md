# notebooks/

Exploratory notebooks for the `graph_diffusion` project. **Not part of CI** — they require an interactive Jupyter kernel and live state.

## Setup

Use the `[dev]` extras for jupyter + ipympl, or install them separately:

```bash
uv pip install jupyterlab ipympl
```

Launch JupyterLab from the repo root so paths resolve correctly:

```bash
uv run jupyter lab
```

## Notebooks

| File | What it does |
|---|---|
| `EXP-020_interactive_cp.ipynb` | Live pressure-conditioning explorer. Drag points on a single Cp(x/c) curve and watch the EXP-022 model (T=1000 retrain, current best) regenerate 4 candidate boundary shapes for the new 8-mode conditioning vector. CFG scale and seed are exposed as widgets. |
| `EXP-030_dual_cp_interactive.ipynb` | Dual-Cp variant. Two stacked Cp(x/c) editors — top controls the upper surface, bottom controls the lower surface — together forming a 16-mode conditioning vector for the EXP-030 model. Trained on the same AoA=0 data as EXP-021/022; validates the dual-cond architecture before any future wider-AoA work. |
