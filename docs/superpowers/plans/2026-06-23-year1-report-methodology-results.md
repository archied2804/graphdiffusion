# First-Year Report — Methodology & Results Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the Methodology and Results sections of a first-year PhD report (drafted in full) plus a scaffolded publication-style skeleton, in both Markdown (source of truth) and a mirrored LaTeX project, reusing existing figures and experiment records.

**Architecture:** A Markdown master (`docs/report/year1_report.md`) is the single source of truth; a LaTeX project under `docs/report/latex/` mirrors it section-for-section for paper reuse. Methodology and Results are written in full; Introduction, Related Work, Discussion, Conclusions are guided stubs. Foundational ablations are folded in by **logical role, not by EXP-NNN name**. CFD-validation numbers live in tables so they are cheap to revise.

**Tech Stack:** Markdown (Obsidian-flavoured), LaTeX (compiled later on Overleaf — no local toolchain), existing PNG figures in `docs/assets/`.

**Spec:** `docs/superpowers/specs/2026-06-22-year1-report-design.md`

**Authoritative source material (read before drafting):**
- `docs/reports/validation_report.md` — the master narrative, equations, numbers, references [1]–[10]
- `docs/experiments/EXP-02x_methodology_summary.md` — surface-quality study, roughness table
- `docs/experiments/EXP-00x_series_summary.md` — circle representation studies (preliminary)
- `docs/experiments/EXP-011_ellipse_shape_method_A.md`, `EXP-012_ellipse_shape_method_B.md`, `EXP-013_ellipse_smoothness_reg.md` — shape-representation justification
- `docs/experiments/EXP-040_openlb_cp_baseline.md`, `EXP-041_openlb_generated_inverse_design.md`, `EXP-042_ood_fidelity_magnitude_creep.md` — validation detail
- `docs/architecture.md` — package/class structure for the model description
- `CLAUDE.md` — architecture summary, naming

**Global constraints (apply to every drafting task):**
- Refer to foundational ablations **by logical role** ("an earlier representation study", "a surface-quality study"), **never** by `EXP-NNN` in report prose. Experiment IDs stay in the vault logs only.
- All CFD-derived numbers go in **tables**, not inline prose, so a framework fix updates one place. Mark the solver subsection and validation tables as provisional with a brief HTML comment `<!-- provisional: pending validation-framework re-verification -->`.
- Applied ML-for-science register: lead each subsection with its claim; define notation once; keep only load-bearing equations.
- Embed figures with Obsidian syntax in Markdown (`![[assets/NAME.png]]`) and `\includegraphics` in LaTeX.
- Do not invent numbers. Every quantitative claim must trace to one of the source docs above.

---

## Task 1: Scaffold the report directory and README

**Files:**
- Create: `docs/report/README.md`
- Create: `docs/report/latex/figures/.gitkeep`

- [ ] **Step 1: Create the figures pointer placeholder**

Create `docs/report/latex/figures/.gitkeep` (empty file). LaTeX will reference figures via a relative path to `../../assets`; this directory exists for any report-specific generated figures.

- [ ] **Step 2: Write the README**

Create `docs/report/README.md` with this content:

```markdown
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
```

- [ ] **Step 3: Commit**

```bash
git add docs/report/README.md docs/report/latex/figures/.gitkeep
git commit -m "docs(docs): scaffold first-year report directory and README

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Create the LaTeX skeleton (main.tex + empty section files)

**Files:**
- Create: `docs/report/latex/main.tex`
- Create: `docs/report/latex/sections/00_abstract.tex` … `06_conclusions.tex` (7 files, empty for now)

- [ ] **Step 1: Create the seven empty section files**

```bash
mkdir -p docs/report/latex/sections
for f in 00_abstract 01_introduction 02_related_work 03_methodology 04_results 05_discussion 06_conclusions; do
  printf '%% %s — filled in a later task\n' "$f" > "docs/report/latex/sections/$f.tex"
done
```

- [ ] **Step 2: Write main.tex**

Create `docs/report/latex/main.tex`:

```latex
\documentclass[11pt]{article}
\usepackage[a4paper,margin=1in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{siunitx}
\usepackage[numbers,sort&compress]{natbib}
\usepackage{hyperref}
\graphicspath{{../../assets/}}

\title{Diffusion-based inverse design of aerodynamic shapes,\\
validated against an independent lattice-Boltzmann solver}
\author{Archie Dobson \\ University of Manchester}
\date{\today}

\begin{document}
\maketitle

\input{sections/00_abstract}
\input{sections/01_introduction}
\input{sections/02_related_work}
\input{sections/03_methodology}
\input{sections/04_results}
\input{sections/05_discussion}
\input{sections/06_conclusions}

\bibliographystyle{plainnat}
\bibliography{refs}

\end{document}
```

- [ ] **Step 3: Verify \input names match files**

Run: `ls docs/report/latex/sections/ && grep '\\input' docs/report/latex/main.tex`
Expected: every `\input{sections/NN_name}` has a matching `NN_name.tex`.

- [ ] **Step 4: Commit**

```bash
git add docs/report/latex/main.tex docs/report/latex/sections/
git commit -m "docs(docs): add LaTeX skeleton for first-year report

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Seed refs.bib

**Files:**
- Create: `docs/report/latex/refs.bib`

- [ ] **Step 1: Write refs.bib**

Create `docs/report/latex/refs.bib` with BibTeX entries for the ten references already
listed in `docs/reports/validation_report.md` (Ho 2020 DDPM; Nichol & Dhariwal 2021 cosine
schedule; Battaglia 2018 graph networks; Zaheer 2017 Deep Sets; Ho & Salimans 2022 CFG;
Chung 2023 DPS; Krüger 2017 LBM book; Latt 2021 OpenLB; Xu & Sagaut 2013 sponge layer;
Lino 2022 pOnEllipse / MS-GNN), plus three ablation-method references used in the
surface-quality study: Hang et al. 2023 "Efficient Diffusion Training via Min-SNR Weighting
Strategy"; Salimans & Ho 2022 "Progressive Distillation for Fast Sampling of Diffusion
Models" (v-prediction); and the EMA usage as cited in Nichol & Dhariwal / Song & Ermon —
cite Song & Ermon 2020 "Improved Techniques for Training Score-Based Generative Models" for
EMA. Use citation keys: `ho2020ddpm`, `nichol2021improved`, `battaglia2018relational`,
`zaheer2017deepsets`, `ho2022cfg`, `chung2023dps`, `kruger2017lbm`, `latt2021openlb`,
`xu2013sponge`, `lino2022msgnn`, `hang2023minsnr`, `salimans2022progressive`,
`song2020improved`.

Pull exact author lists / venues from `validation_report.md` for the first ten. For the
three ablation refs, use standard published metadata. Every key above must appear in the
.bib file.

- [ ] **Step 2: Verify all keys present**

Run: `grep -c '@' docs/report/latex/refs.bib`
Expected: 13 (one `@inproceedings`/`@article`/`@book` per reference).

- [ ] **Step 3: Commit**

```bash
git add docs/report/latex/refs.bib
git commit -m "docs(docs): seed refs.bib for first-year report

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Markdown master — front matter, skeleton, Abstract, and the four guided stubs

**Files:**
- Create: `docs/report/year1_report.md`

- [ ] **Step 1: Write the master skeleton with Abstract and stubs**

Create `docs/report/year1_report.md` containing, in order:

1. YAML front matter: `title`, `date: 2026-06-23`, `tags: [report, phd, methodology, results]`.
2. `# Diffusion-based inverse design of aerodynamic shapes` + subtitle line.
3. `## Abstract` — adapt the abstract from `validation_report.md` lines 13–17 into ~150
   words; keep the headline (independent CFD validation, ρ=0.998 baseline, ρ≈0.95
   in-distribution inverse design). No EXP-NNN names.
4. `## 1. Introduction` — **stub**: heading + bullet list of what to cover (motivation:
   aerodynamic inverse design is one-to-many; generative diffusion as the tool; the gap:
   generative aero models are rarely validated by independent physics; contribution
   statement) + the citations to hit (`ho2020ddpm`, `lino2022msgnn`). Mark `<!-- STUB -->`.
5. `## 2. Related Work` — **stub**: bullets for (diffusion models, graph/GN surrogates for
   CFD, inverse design in aerodynamics, guidance methods CFG/DPS) with the matching
   citation keys. Mark `<!-- STUB -->`.
6. `## 3. Methodology` — heading only (filled in Tasks 5–6).
7. `## 4. Results` — heading only (filled in Tasks 7–8).
8. `## 5. Discussion` — **stub**: bullets (in-distribution trust vs OOD extrapolation
   failure; magnitude creep as a model-side calibration issue; why posterior guidance is
   the wrong knob; threats to validity incl. provisional solver). Mark `<!-- STUB -->`.
9. `## 6. Conclusions` — **stub**: bullets (Phase 1 validation complete; Phase 2
   CFD-in-the-loop optimisation). Mark `<!-- STUB -->`.
10. `## References` — note: "Managed in `latex/refs.bib`; see keys inline as `[@key]`."

- [ ] **Step 2: Verify structure**

Run: `grep '^##' docs/report/year1_report.md`
Expected: Abstract, 1 Introduction, 2 Related Work, 3 Methodology, 4 Results,
5 Discussion, 6 Conclusions, References — in that order.

- [ ] **Step 3: Commit**

```bash
git add docs/report/year1_report.md
git commit -m "docs(docs): add report master skeleton, abstract and section stubs

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Methodology subsections 3.1–3.4 (formulation, representation, diffusion, score model)

**Files:**
- Modify: `docs/report/year1_report.md` (under `## 3. Methodology`)

- [ ] **Step 1: Draft 3.1 Problem formulation**

Write ~120 words: define inverse design (target `Cp(x/c)` → boundary shape); argue the map
is one-to-many so a generative model is appropriate rather than a regressor. Introduce
notation: boundary as radial field `r(θ)`, conditioning context `c`.

- [ ] **Step 2: Draft 3.2 Shape representation**

Write ~150 words: radial field `r(θ)` sampled at surface nodes; boundary graph with
bidirectional ring edges. Fold in the representation justification **by logical role** —
"an earlier study compared radial, direct-Cartesian and Fourier-coefficient
parameterisations and found the radial field gave the most stable diffusion target" — cite
the source via the vault link only in a footnote comment, not in prose. Source:
`EXP-011/012/013` logs (read for the actual comparison outcome).

- [ ] **Step 3: Draft 3.3 Denoising diffusion model**

Write ~180 words + 2 equations. Use the forward marginal and the simple loss exactly as in
`validation_report.md` lines 28 and 37:
- `q(x_t | x_0) = N(x_t; √(ᾱ_t) x_0, (1−ᾱ_t) I)`
- `L = E[‖ε − ε_θ(x_t, t, c)‖²]`
Mention the cosine `ᾱ_t` schedule `[@nichol2021improved]` and that schedule buffers are
owned by `NoiseSchedule`. Cite `[@ho2020ddpm]`.

- [ ] **Step 4: Draft 3.4 Graph-network score model**

Write ~160 words: `ScoreNetwork` stacks Battaglia GN blocks
`[@battaglia2018relational]`, each `EdgeModel → NodeModel → GlobalModel`; messages along
ring edges. Key point: sinusoidal time embedding and conditioning are injected **once**
into the per-graph global attribute `u` (not per node), then propagate via
`GlobalModel → NodeModel`. Embed the architecture figure (now produced):
`![[assets/report_architecture.png]]` with a caption describing the time/conditioning
injection into `u` and the N× GN block stack. The auxiliary DeepSets pressure-head branch
shown in the figure is described in §3.5.

- [ ] **Step 5: Commit**

```bash
git add docs/report/year1_report.md
git commit -m "docs(docs): draft methodology 3.1-3.4 (formulation to score model)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Methodology subsections 3.5–3.9 (conditioning, guidance, solver, dataset, metrics)

**Files:**
- Modify: `docs/report/year1_report.md`

- [ ] **Step 1: Draft 3.5 Pressure conditioning**

~150 words: target `Cp(x/c)` encoded as first `K=8` type-II DCT coefficients (fixed-size,
node-count-independent); jointly-trained DeepSets pressure head `[@zaheer2017deepsets]`
predicts the DCT spectrum of the generated shape's pressure, supplying the geometry↔pressure
coupling gradient. Reference figure placeholders for the conditioning pipeline and DCT
illustration with `<!-- figure pending -->` comments.

- [ ] **Step 2: Draft 3.6 Inference guidance**

~120 words + 1 equation. CFG `[@ho2022cfg]`: `ε = (1+w)·ε_cond − w·ε_uncond`, `w=3`.
State DPS `[@chung2023dps]` as the tested alternative whose result appears in Results.

- [ ] **Step 3: Draft 3.7 Independent CFD validation solver (provisional)**

~180 words, opened with `<!-- provisional: pending validation-framework re-verification -->`.
2D incompressible LBM `[@kruger2017lbm]` on OpenLB `[@latt2021openlb]`: winding-number
signed-distance solid indicator, curved Bouzidi no-slip wall, velocity inlet, free-stream
Dirichlet top/bottom, pressure outlet, upstream graded sponge layer `[@xu2013sponge]`.
Emphasise: measures time-averaged surface `Cp` from first principles and **never sees the
model** — this independence is what makes the validation meaningful. Reference
`![[assets/EXP-040_domain_schematic.png]]`.

- [ ] **Step 4: Draft 3.8 Dataset**

~140 words: pOnEllipse `[@lino2022msgnn]` — 5701 training simulations, elliptical cylinder,
Re 500–1000 laminar vortex-shedding; 101 unsteady pressure timesteps collapsed to
steady-mean `Cp`; HDF5 per-node layout `[x, y, Re, H, p0..p100]`; note `H` is channel
height, not angle of attack. Embed `![[assets/report_dataset_example.png]]`.

- [ ] **Step 5: Draft 3.9 Evaluation metrics**

~120 words: `Cp` Pearson correlation (morphology — the inverse-design verdict); affine
slope `a` and offset `b` (amplitude calibration); drag coefficient `Cd`; second-difference
roughness of `r(θ)` (surface quality). State explicitly that correlation, not amplitude, is
the headline verdict.

- [ ] **Step 6: Commit**

```bash
git add docs/report/year1_report.md
git commit -m "docs(docs): draft methodology 3.5-3.9 (conditioning to metrics)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Results subsections 4.1–4.3 (preliminary, surface-quality, solver gate)

**Files:**
- Modify: `docs/report/year1_report.md`

- [ ] **Step 1: Draft 4.1 Preliminary representation studies**

~120 words, **no EXP-NNN names**. Summarise the circle-series finding (radial diffusion on
canonical shapes established the representation, feature set `[r, κ, s/L]`, and
ring-neighbour connectivity used downstream). Embed `![[assets/EXP-001_generated_shapes.png]]`.
Source: `EXP-00x_series_summary.md`.

- [ ] **Step 2: Draft 4.2 Surface-quality methodology study**

~200 words + the roughness table. Lead with the claim: roughness is a **sampling-time**
property, not a training-loss property. Reproduce the comparison table from
`EXP-02x_methodology_summary.md` (baseline → longer training → T=1000 winner at 0.01523,
−55%; and the validated **negative results**: Min-SNR-γ, EMA, v-prediction all net-worse).
Refer to configurations by description, not EXP-NNN. Table columns: Lever | T | Overall
roughness | Δ vs baseline | Best val loss.

- [ ] **Step 3: Draft 4.3 CFD solver validation gate**

~160 words + the baseline table `<!-- provisional -->`. Claim: the loop reproduces dataset
`Cp` on known ellipses at correlation **0.998**, differing only by a single multiplicative
normalisation `a ≈ 1.78` (matches the dataset's documented `u/1.8` scaling). Table: Sim | Re
| Cp corr | slope a | offset b (from `validation_report.md` lines 112–116). Mention the
sponge-layer fix collapsing Cd noise ±1.5 → ±0.05 and improving the match. Embed
`![[assets/EXP-040_cp_comparison.png]]` and `![[assets/EXP-040_stability_trace.png]]`.

- [ ] **Step 4: Commit**

```bash
git add docs/report/year1_report.md
git commit -m "docs(docs): draft results 4.1-4.3 (preliminary to solver gate)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: Results subsections 4.4–4.6 (inverse-design, OOD/DPS, flow fields)

**Files:**
- Modify: `docs/report/year1_report.md`

- [ ] **Step 1: Draft 4.4 Inverse-design validation**

~200 words + table `<!-- provisional -->`. Claim: generated shapes hit their target `Cp` at
correlation **≈0.95 in-distribution**, verified by the independent solver. Table: Target |
Type | Re | Cp corr raw | Cp corr smoothed (from `validation_report.md` lines 136–140).
Note roughness is cosmetic (raw vs smoothed Δcorr ~0.01). Add the OOD spread result
(0.70–0.93 declining with OOD distance). Embed `![[assets/report_generated_shapes.png]]`,
`![[assets/EXP-041_cp_inverse_design.png]]`, `![[assets/EXP-041_ood_spread.png]]`,
`![[assets/EXP-041_ood_cp_overlays.png]]`.

- [ ] **Step 2: Draft 4.5 OOD limits, magnitude creep, and the posterior-guidance negative**

~200 words + DPS sweep table `<!-- provisional -->`. Claim: posterior guidance is an
unambiguous **negative** — every metric degrades monotonically with DPS weight (corr falls,
slope `a` 3.5→6.5, Cd nearly doubles), because DPS optimises the self-trained head's error
where it is unreliable OOD. Plain CFG (`dps=0`) is the correct OOD configuration. Then:
magnitude creep is **model-side** — the affine slope is unmoved by CFD/discretisation levers
and is already ~2.8–3.0 in-distribution vs the solver's validated 1.78. Table from
`validation_report.md` lines 176–181. Embed `![[assets/EXP-041_dps_sweep.png]]` and
`![[assets/EXP-042_creep_ladder.png]]`.

- [ ] **Step 2b: Add the amplitude-vs-morphology caveat box**

Add a short callout: treat **correlation** (morphology) as the inverse-design verdict; the
model captures `Cp` shape, not absolute amplitude (known limitation).

- [ ] **Step 3: Draft 4.6 Qualitative flow fields**

~140 words: instantaneous OpenLB fields around a generated in-distribution shape — laminar
separation rolling into a von Kármán vortex street; stagnation high pressure on the nose,
low-pressure vortex cores in the wake; the surface trace of this field *is* the `Cp` curve
the loop grades. Embed `![[assets/report_paraview_velocity_wake.png]]` and
`![[assets/report_paraview_pressure.png]]`.

- [ ] **Step 4: Commit**

```bash
git add docs/report/year1_report.md
git commit -m "docs(docs): draft results 4.4-4.6 (inverse design, OOD/DPS, flow fields)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: Mirror Methodology into LaTeX

**Files:**
- Modify: `docs/report/latex/sections/03_methodology.tex`

- [ ] **Step 1: Convert section 3 of the master to LaTeX**

Translate `## 3. Methodology` (3.1–3.9) from `year1_report.md` into `03_methodology.tex`:
`\section{Methodology}` + `\subsection{...}` per subsection. Convert Obsidian `![[assets/X.png]]`
to `\begin{figure}[t]\centering\includegraphics[width=0.8\linewidth]{X.png}\caption{...}\label{fig:...}\end{figure}`.
Convert `[@key]` to `\cite{key}`. Convert block equations to `\begin{equation}...\end{equation}`.
Preserve the `<!-- provisional -->` note as a LaTeX comment `% provisional`.

- [ ] **Step 2: Verify citations and figures resolve to known keys/files**

Run: `grep -o '\\cite{[a-z0-9]*}' docs/report/latex/sections/03_methodology.tex | sort -u`
then check each key exists: `grep '@' docs/report/latex/refs.bib`.
Run: `grep -o 'includegraphics.*{[^}]*}' docs/report/latex/sections/03_methodology.tex` and
confirm each PNG exists in `docs/assets/` (pending-figure placeholders may be commented out).
Expected: every cite key is in refs.bib; every uncommented figure file exists.

- [ ] **Step 3: Commit**

```bash
git add docs/report/latex/sections/03_methodology.tex
git commit -m "docs(docs): mirror methodology into LaTeX

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 10: Mirror Results into LaTeX

**Files:**
- Modify: `docs/report/latex/sections/04_results.tex`

- [ ] **Step 1: Convert section 4 of the master to LaTeX**

Translate `## 4. Results` (4.1–4.6) into `04_results.tex` using the same conventions as
Task 9. Convert Markdown tables to `\begin{table}[t]\centering\begin{tabular}{...}\toprule
... \bottomrule\end{tabular}\caption{...}\label{tab:...}\end{table}` using `booktabs`. Keep
`% provisional` comments above the validation tables.

- [ ] **Step 2: Verify tables, citations, figures**

Run: `grep -c 'begin{table}' docs/report/latex/sections/04_results.tex`
Expected: ≥ 4 (surface-quality, solver gate, inverse-design, DPS sweep).
Repeat the cite-key and figure-existence checks from Task 9 Step 2.

- [ ] **Step 3: Commit**

```bash
git add docs/report/latex/sections/04_results.tex
git commit -m "docs(docs): mirror results into LaTeX

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 11: Mirror Abstract and stubs into LaTeX

**Files:**
- Modify: `docs/report/latex/sections/00_abstract.tex`, `01_introduction.tex`,
  `02_related_work.tex`, `05_discussion.tex`, `06_conclusions.tex`

- [ ] **Step 1: Fill the abstract**

Put the master Abstract prose into `00_abstract.tex` as `\begin{abstract}...\end{abstract}`.

- [ ] **Step 2: Fill the four stubs**

For each of `01_introduction`, `02_related_work`, `05_discussion`, `06_conclusions`: write
`\section{...}` followed by the guided-outline bullets from the master as LaTeX comments
(`% - bullet`) plus a one-line `% STUB — draft in Year-2 paper pass`, so the structure
compiles but the prose is intentionally deferred. Convert any `[@key]` mentioned in stubs to
`\cite{key}`.

- [ ] **Step 3: Verify all section files non-trivial**

Run: `wc -l docs/report/latex/sections/*.tex`
Expected: every file has content (no leftover single-comment placeholders from Task 2).

- [ ] **Step 4: Commit**

```bash
git add docs/report/latex/sections/
git commit -m "docs(docs): mirror abstract and section stubs into LaTeX

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 12: Link the report from the vault index and final consistency pass

**Files:**
- Modify: `docs/index.md`

- [ ] **Step 1: Add a Reports-section link**

In `docs/index.md` under `### Reports`, add:
`- [[report/year1_report|First-Year Report]] — methodology & results, publication-style (draft)`

- [ ] **Step 2: Consistency pass on the master**

Run: `grep -nE 'EXP-[0-9]' docs/report/year1_report.md`
Expected: **no matches** in prose (foundational ablations referred to by logical role).
If any appear, rewrite them by logical role. (HTML-comment source pointers are acceptable;
prose is not.)

Run: `grep -c '<!-- provisional' docs/report/year1_report.md`
Expected: ≥ 2 (solver subsection + validation tables flagged).

- [ ] **Step 3: Verify every embedded figure exists**

Run:
```bash
grep -oE '!\[\[assets/[^]]*\]\]' docs/report/year1_report.md | sed 's/!\[\[//;s/\]\]//' | while read f; do test -f "docs/$f" && echo "OK $f" || echo "MISSING $f"; done
```
Expected: no `MISSING` for figures that exist today; the four/five pending new figures
(architecture, conditioning pipeline, diffusion schematic, DCT, roughness) are referenced
only inside `<!-- figure pending -->` comments and so won't appear here.

- [ ] **Step 4: Commit**

```bash
git add docs/index.md docs/report/year1_report.md
git commit -m "docs(docs): link first-year report from vault index

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Deferred (not in this plan — by design)

- **New figures** (architecture, conditioning pipeline, diffusion schematic, DCT,
  roughness): referenced as `<!-- figure pending -->` placeholders; produced in a separate
  agreed pass per the spec's "Figures to produce" list.
- **LaTeX compilation**: no local toolchain; compile on Overleaf. Plan verifies structure
  (matching `\input`, cite keys, figure files) instead.
- **Intro / Related Work / Discussion / Conclusions prose**: stubs only, for the Year-2
  paper pass.
- **Validation-framework fixes**: tracked separately; report kept revisable around them.

---

## Self-review notes

- **Spec coverage:** file layout (Tasks 1–3), Methodology 1–9 (Tasks 5–6), Results 1–6
  (Tasks 7–8), Both-formats mirror (Tasks 9–11), stubs (Tasks 4, 11), index link (Task 12),
  figures-to-produce deferred-and-flagged, provisional caveat enforced (Task 12 Step 2),
  logical-not-by-name constraint enforced (Task 12 Step 2). All spec sections map to a task.
- **Placeholder scan:** intentional `<!-- STUB -->` / `<!-- figure pending -->` markers are
  deliberate deferrals, documented above — not plan placeholders. No "TBD/implement later"
  in task bodies.
- **Consistency:** citation keys defined in Task 3 are reused verbatim in Tasks 5–11; figure
  filenames match `docs/assets/` listing; table sources cite exact `validation_report.md`
  line ranges.
