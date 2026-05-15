# Experiment Documentation

This directory contains structured experiment logs for the `graph_diffusion`
project.  Every file is plain Markdown with YAML frontmatter — compatible
with both GitHub rendering and **Obsidian** (link, graph-view, tag search).

## Quick start

1. Copy `_template.md` → `EXP-NNN_short-name.md`
2. Fill in the frontmatter and sections
3. Update the experiment index in this README
4. Commit alongside the matching config in `configs/`

## Naming convention

```
EXP-{NNN}_{geometry}_{mechanism}_{variant}
```

| Segment      | Purpose                           | Examples                          |
|-------------|-----------------------------------|-----------------------------------|
| `EXP-NNN`  | Sequential experiment ID (zero-padded 3 digits) | `EXP-001`, `EXP-042` |
| `geometry`  | Mesh / domain shape               | `circle`, `naca`, `cylinder`      |
| `mechanism` | Core diffusion / model change     | `radial`, `isotropic`, `sde`      |
| `variant`   | What distinguishes this run       | `baseline`, `k6`, `logit`, `deep` |

**Examples:**

- `EXP-001_circle_radial_baseline`
- `EXP-002_circle_radial_k6`
- `EXP-003_circle_radial_logit-transform`

The same slug is used for:

- The experiment doc: `docs/experiments/EXP-001_circle_radial_baseline.md`
- The config file:    `configs/EXP-001_circle_radial_baseline.yaml`
- The output dir:     `outputs/EXP-001_circle_radial_baseline/`
- Obsidian links:     `[[EXP-001_circle_radial_baseline]]`

## Config pairing rule

Every experiment **must** have a corresponding YAML config in `configs/`.
The config filename matches the experiment slug.  Shared base configs
(e.g. `default.yaml`, `circle_radial.yaml`) remain as-is; experiment
configs inherit from them or copy + modify.

## Experiment index

| ID | Slug | Status | One-liner |
|----|------|--------|-----------|
| 001 | [[EXP-001_circle_radial_baseline]] | ✅ complete | First radial diffusion on unit circle — validates pipeline |
| 002 | [[EXP-002_circle_radial_k-neighbors]] | ✅ complete | k-neighbors ablation — k=6 wins on distribution fidelity |
| 003 | [[EXP-003_circle_radial_amplitude]] | ✅ complete | Amplitude ablation — 0.15 is the right complexity |
| 004 | [[EXP-004_circle_radial_logit]] | ✅ complete | Logit-transform bounded diffusion — not adopted |
| 005 | [[EXP-005_circle_radial_full]] | ✅ complete | Full training — reference result val_loss=0.0303 |
| 006 | [[EXP-006_circle_radial_rich-features]] | ✅ complete | Rich features [r,κ,s/L] — marginal gain at 3× higher loss |
| — | [[EXP-00x_series_summary]] | ✅ complete | EXP-00x circle series summary & recommendations |
| 010 | [[EXP-010_ellipse_radial_baseline]] | 🔜 planned | pOnEllipse data pipeline + pressure field baseline |
| 011 | [[EXP-011_ellipse_shape_method_A]] | 🔜 planned | Shape Method A — unit ellipse r(θ) radial |
| 012 | [[EXP-012_ellipse_shape_method_B]] | 🔜 planned | Shape Method B — Cartesian (x, y) direct |
| 013 | [[EXP-013_ellipse_smoothness_reg]] | ✅ complete | Smoothness reg (λ=1e-4, SNR-weighted) — negative result, roughness +18% |
| 013b | [[EXP-013b_ellipse_smoothness_reg_strong]] | ✅ complete | Smoothness reg λ=1e-3 — monotone degradation; approach abandoned |
| 014 | [[EXP-014_ellipse_shape_method_D]] | 🔜 planned | Shape Method D — aspect-ratio normalisation + (a,b) conditioning |
| 015 | [[EXP-015_ellipse_conditional_global]] | 🔜 planned | Conditional inverse design — global pressure summary → u |
| 016 | [[EXP-016_ellipse_conditional_node]] | 🔜 planned | Conditional inverse design — node-level pressure concatenation |

## Obsidian integration

To use these docs in Obsidian:

1. Open Obsidian → "Open folder as vault" → select `docs/`
2. All `[[wikilinks]]` between experiment docs, the index, and feature
   changelogs will resolve automatically
3. Use graph view to visualise experiment lineage
4. Tags in frontmatter (`#circle`, `#radial`, `#baseline`) are searchable

> The `docs/` folder is version-controlled so experiment history travels
> with the code.  For private research notes that should not be committed,
> create a sibling vault outside the repo and use Obsidian's
> "Open another vault" feature.
