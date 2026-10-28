# First-year PhD report

Diffusion-based inverse design of aerodynamic shapes, validated against an independent
lattice-Boltzmann solver. Methodology and Results are drafted in full; the remaining
sections are guided stubs.

## Two formats, one source of truth

- `year1_report.md` — **the master.** Edit prose here. Obsidian-readable.
- `latex/` — a section-for-section mirror for Year-2 paper reuse. Regenerate from the
  master when prose changes.

## Building the LaTeX

No local LaTeX toolchain. Compile on Overleaf or with `latexmk -pdf main.tex` once a
toolchain is installed. Figures resolve via `\graphicspath{{../../assets/}}` in `main.tex`.

## Provisional content

The CFD-validation solver description and all OpenLB-derived numbers are provisional
pending re-verification of the validation framework. They are isolated in tables and
flagged with `<!-- provisional -->` comments.
