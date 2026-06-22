---
title: "First-year PhD report — Methodology & Results — design spec"
date: 2026-06-22
status: approved-for-planning
tags: [report, phd, methodology, results, paper, spec]
---

# First-year PhD report (Methodology & Results) — design spec

## Purpose

Produce the **Methodology** and **Results** sections of a first-year PhD report on
diffusion-based inverse design of aerodynamic shapes, written so the same prose can be
lifted into a Year-2 paper. The report adopts a publication-style structure; the two
named sections are drafted in full, the remaining sections are scaffolded as guided
stubs.

This spec is the agreed structure. It is the source of decisions for the subsequent
writing plan.

## Decisions (locked)

| Question | Decision |
|---|---|
| Output format | **Both** — Markdown master (Obsidian) + mirrored LaTeX skeleton |
| Scope to draft now | **Methodology + Results in full**; Intro / Related Work / Discussion / Conclusions as guided stubs |
| Scientific scope | **Core inverse-design + CFD-validation story**, with the foundational ablations (circle EXP-00x, shape-method EXP-01x, surface-quality EXP-02x) folded in *condensed* |
| Venue/style | **Applied ML-for-science** (e.g. JCP-ML, Nature Machine Intelligence, ICLR ML4Science) — method novelty balanced with independent physical validation as the headline |
| Department template | Assumed: no rigid template / page limit. Structure is publication-style and trimmable. **Revisit if a fixed template emerges.** |

## Caveats / known-provisional areas

- **The CFD validation framework may contain minor errors still to be fixed.** The
  Methodology subsection describing the OpenLB solver and the Results subsections that
  depend on its measured `Cp` (EXP-040/041/042 numbers) are therefore written to be
  **easily revisable**: numeric claims live in tables (not baked into prose), and the
  solver description is modular. Treat these as provisional until the framework is
  re-verified. Flag any number that changes.

## File layout

```
docs/report/
  README.md                  # how the two formats relate + LaTeX build instructions
  year1_report.md            # Markdown master — source of truth, Obsidian-readable
  latex/
    main.tex                 # documentclass, packages, \input of section files
    sections/
      00_abstract.tex
      01_introduction.tex    # STUB (guided outline)
      02_related_work.tex    # STUB (guided outline)
      03_methodology.tex     # DRAFTED
      04_results.tex         # DRAFTED
      05_discussion.tex      # STUB
      06_conclusions.tex     # STUB
    refs.bib                 # seeded from validation_report.md (10 refs) + ablation refs
    figures/                 # pointer to ../../assets (no duplication)
```

The Markdown master holds the full drafted prose. The LaTeX mirrors it section-for-section.
Figures remain in `docs/assets/`; LaTeX references them via the `figures/` pointer.

## Report skeleton

Abstract · 1 Introduction *(stub)* · 2 Related Work *(stub)* · **3 Methodology (full)** ·
**4 Results (full)** · 5 Discussion *(stub)* · 6 Conclusions *(stub)* · References.

Stubs = section heading + bulleted "what goes here" guidance + the key claims and
citations to hit, so they can be written later without re-deriving the argument.

## Methodology (drafted) — subsection outline

1. **Problem formulation** — inverse design: target `Cp(x/c)` → boundary shape; why
   generative (one-to-many) rather than regression.
2. **Shape representation** — radial field `r(θ)` on a boundary graph with bidirectional
   ring edges; *condensed* justification from the shape-method studies
   (EXP-011/012/013) for choosing radial over Cartesian / Fourier-coefficient.
3. **Denoising diffusion model** — DDPM forward/reverse process, cosine `ᾱ_t` schedule,
   simple ε-prediction loss. `NoiseSchedule` buffer ownership noted.
4. **Graph-network score model** — Battaglia GN block
   (`EdgeModel → NodeModel → GlobalModel`); sinusoidal time embedding and conditioning
   injected **once** into the global attribute `u`, propagating to nodes.
5. **Pressure conditioning** — first `K = 8` type-II DCT coefficients of `Cp`;
   jointly-trained DeepSets pressure head as the geometry↔pressure coupling.
6. **Inference guidance** — classifier-free guidance (`w = 3`); Diffusion Posterior
   Sampling as the *tested alternative* (result lands in §4).
7. **Independent CFD validation solver** *(provisional)* — OpenLB 2D incompressible LBM;
   winding-number solid indicator, curved Bouzidi no-slip wall, velocity inlet,
   free-stream Dirichlet top/bottom, pressure outlet, upstream sponge layer; measures
   time-averaged surface `Cp` from first principles, never sees the model.
8. **Dataset** — pOnEllipse: 5701 training simulations, Re 500–1000, laminar
   vortex-shedding regime; steady-mean `Cp` collapse; HDF5 layout; the `H` =
   channel-height (not AoA) correction.
9. **Evaluation metrics** — `Cp` Pearson correlation (morphology verdict), affine slope
   `a` and offset `b` (amplitude), drag coefficient `Cd`, second-difference roughness of
   `r(θ)` (surface quality).

## Results (drafted) — subsection outline + figure mapping

| # | Subsection | Source experiments | Figures (existing) |
|---|---|---|---|
| 1 | Preliminary: representation studies | EXP-00x circle series (condensed) | `EXP-001_generated_shapes.png` |
| 2 | Surface-quality methodology study | EXP-02x (T=1000 winner; negative results Min-SNR/EMA/v-pred) | *(table; candidate new fig)* |
| 3 | CFD solver validation gate (ρ=0.998) | EXP-040 | `EXP-040_cp_comparison.png`, `EXP-040_domain_schematic.png`, `EXP-040_stability_trace.png` |
| 4 | Inverse-design validation (in-dist ρ≈0.95) | EXP-041 | `report_generated_shapes.png`, `EXP-041_cp_inverse_design.png`, `EXP-041_ood_spread.png`, `EXP-041_ood_cp_overlays.png` |
| 5 | OOD limits & magnitude creep; DPS negative | EXP-041 DPS sweep, EXP-042 | `EXP-041_dps_sweep.png`, `EXP-041_correlation_bars.png`, `EXP-042_creep_ladder.png`, `EXP-042_node_sweep.png`, `EXP-042_roughness_resolution.png` |
| 6 | Qualitative flow fields | EXP-041 ParaView | `report_paraview_velocity_wake.png`, `report_paraview_pressure.png` |

Numeric claims are kept in tables (see provisional caveat) so the validation numbers can
be updated in one place if the framework is corrected.

## Figures to produce (to discuss before generating)

These are gaps where a new diagram would materially help the methodology read like
published ML-for-science work. None are blocking; we agree each before producing it.

1. **Score-network / architecture diagram** — GN block stack with time + conditioning
   injection into the global attribute `u`. *(No architecture figure currently exists;
   highest priority.)*
2. **Conditioning pipeline diagram** — `Cp(x/c)` → DCT(K=8) → DeepSets head → context
   vector. Could merge with (1).
3. **Forward/reverse diffusion schematic on `r(θ)`** — a clean shape corrupted to noise
   and denoised back, to ground the DDPM description visually.
4. **DCT encoding illustration** — a `Cp` curve and its K=8 reconstruction, motivating
   the fixed-size conditioning.
5. **Roughness-metric illustration** *(optional)* — smooth vs wobbly `r(θ)` boundary with
   second-difference annotation, supporting Results §2.

Existing `report_pipeline_schematic.png` and `report_dataset_example.png` already cover
the end-to-end loop and dataset example.

## Style / citations

- Applied ML-for-science register: define notation once, lead each subsection with the
  claim, keep equations to the load-bearing ones (already in `validation_report.md`).
- `refs.bib` seeded from the 10 references in `validation_report.md`, extended with the
  ablation-method citations (Min-SNR-γ, EMA, v-prediction) used in EXP-02x.
- Cross-link experiment logs from the Markdown master via Obsidian `[[...]]`; the LaTeX
  uses plain `\cite`/`\ref`.

## Out of scope

- Writing the Introduction / Related Work / Discussion / Conclusions prose (stubs only).
- Re-running any experiment or regenerating existing figures.
- Building/fixing the OpenLB validation framework (tracked separately; report stays
  revisable around it).

## Next step

After user review of this spec → invoke `writing-plans` to produce the implementation
plan for drafting the two sections and scaffolding the rest.
