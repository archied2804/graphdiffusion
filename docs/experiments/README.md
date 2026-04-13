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
