---
tags: [index, home]
---

# graph_diffusion — Research Vault

Welcome to the Obsidian documentation vault for the `graph_diffusion` project.

## Navigation

### Architecture
- [[architecture|Architecture Overview]] — package dependency graph, full class diagram, data flow

### Experiments
- [[experiments/README|Experiment Index]] — naming convention, config pairing rule, experiment list
- [[experiments/EXPERIMENT_PLAN|Experimentation Plan]] — roadmap + HPC/SLURM instructions
- [[experiments/_template|Experiment Template]] — copy this to start a new experiment log

### Experiment logs — EXP-00x circle series ✅
- [[experiments/EXP-001_circle_radial_baseline|EXP-001]] — Radial diffusion baseline on unit circle ✅
- [[experiments/EXP-002_circle_radial_k-neighbors|EXP-002]] — k-neighbors ablation (k=6 best) ✅
- [[experiments/EXP-003_circle_radial_amplitude|EXP-003]] — Amplitude-scale ablation (0.15 best) ✅
- [[experiments/EXP-004_circle_radial_logit|EXP-004]] — Logit-transform bounded diffusion (not adopted) ✅
- [[experiments/EXP-005_circle_radial_full|EXP-005]] — Full training, val_loss=0.0303, KS=0.1049 ✅
- [[experiments/EXP-006_circle_radial_rich-features|EXP-006]] — Rich node features [r,κ,s/L] ✅
- [[experiments/EXP-00x_series_summary|EXP-00x Series Summary]] — Circle series wrap-up & recommendations ✅

### Experiment logs — EXP-01x aerodynamic series 🔜
- [[experiments/EXP-010_ellipse_radial_baseline|EXP-010]] — pOnEllipse data pipeline + pressure baseline 🔜
- [[experiments/EXP-011_ellipse_shape_method_A|EXP-011]] — Shape Method A: unit ellipse r(θ) 🔜
- [[experiments/EXP-012_ellipse_shape_method_B|EXP-012]] — Shape Method B: Cartesian (x, y) direct 🔜
- [[experiments/EXP-013_ellipse_shape_method_E|EXP-013]] — Shape Method E: Fourier coefficient MLP ablation 🔜
- [[experiments/EXP-014_ellipse_shape_method_D|EXP-014]] — Shape Method D: aspect-ratio normalisation 🔜
- [[experiments/EXP-015_ellipse_conditional_global|EXP-015]] — Conditional inverse design: global pressure summary 🔜
- [[experiments/EXP-016_ellipse_conditional_node|EXP-016]] — Conditional inverse design: node-level pressure 🔜

### Reference
- [[experiments/changelog|Feature Changelog]] — code changes linked to experiments

## Quick links

| Resource | Path |
|----------|------|
| Configs | `configs/` |
| Source | `src/graph_diffusion/` |
| Tests | `tests/` |
| Training script | `train_circle.py` |
| SLURM job script | `scripts/run_experiment.slurm` |
| Series summary | `docs/experiments/EXP-00x_series_summary.md` |
