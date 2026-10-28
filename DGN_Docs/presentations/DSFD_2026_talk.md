---
title: "Using Lattice Boltzmann to Inform Graph Diffusion Networks for Aerodynamic Design Exploration"
author: "Archie Dobson"
event: "DSFD 2026 — Discrete Simulation of Fluid Dynamics"
date: 2026-06-25
tags: [presentation, dsfd, diffusion, lattice-boltzmann, inverse-design]
cssclasses: []
---

<!--
HOW TO PRESENT THIS DECK
  - Obsidian: open this note, run command palette → "Slides: Start presentation".
    (The core "Slides" plugin has been enabled in this vault.)
  - Slides are separated by `---`. Speaker notes live in HTML comments and do NOT
    render on the slide — read them in edit mode while you rehearse.
  - To move to PowerPoint: present full-screen and screenshot each slide, or
    rebuild slides from the bullets + the embedded figure on each one.
  - Target: ~10 minutes oral. ~50s per slide. Don't read the bullets aloud verbatim.
-->

# Using Lattice Boltzmann to Inform
# Graph Diffusion Networks
### for Aerodynamic Design Exploration

**Archie Dobson**
University of Manchester

*DSFD 2026 — Discrete Simulation of Fluid Dynamics*

<small>Very early-stage PhD work · &lt; 6 months in · preliminary results</small>

<!--
SPEAKER NOTES — Slide 1 (Title, ~30s)
- "Thanks — I'm Archie, a first-year PhD student at Manchester."
- One-line hook: "I want to show you an idea: using a lattice-Boltzmann solver not
  as the end product, but as a *teacher and an examiner* for a generative AI model
  that designs aerodynamic shapes."
- Flag honesty up front: "This is genuinely early work — I'll be clear about what's
  promising and what's still broken."
-->

---

## Two things to set expectations

- I'm **under 6 months into the PhD** — this is an *exploration*, not a finished method.
- Everything here is **preliminary**: a 2D laminar test-bed, deliberately simple, chosen so I can probe whether the idea holds at all.
- My background is fluids, not machine learning — so I'll keep the ML at **altitude** and lean on analogies.

> The question I'm poking at:
> **can a discrete fluid solver *supervise* a generative model that proposes shapes?**

<!--
SPEAKER NOTES — Slide 2 (Framing, ~45s)
- Manage expectations honestly — the audience is fluids experts, not ML people.
- "I'm not going to claim state of the art. I'm going to share an idea and some
  first evidence that it's worth pursuing."
- Emphasise the 2D laminar elliptical-cylinder regime is a sandbox by choice.
- Set up the recurring character: lattice Boltzmann (LBM) as the independent
  physics that keeps the AI honest.
-->

---

## The problem: inverse design is one-to-many

**Forward:** shape → solver → pressure.  Routine.
**Inverse:** *I want this pressure — what shape gives it to me?*

- Many different shapes produce **almost the same** surface-pressure curve.
- A normal regressor, forced to pick one answer, **averages them into mush**.
- A *generative* model instead learns the **whole family** of valid shapes and can **sample** from it.

![[report_generated_shapes.png|520]]

<small>Two shapes our model produced for two different target-pressure curves.</small>

<!--
SPEAKER NOTES — Slide 3 (The problem, ~60s)
- This is the "why generative at all" slide. Make the one-to-many point concrete.
- Analogy: "Lots of room shapes give the same echo. If you only ever predict one,
  you blur all of them together and get something that's nobody's answer."
- Generative = learns a distribution p(shape | target), samples valid candidates.
- That sampling ability is exactly what we'll exploit later for design exploration
  and gradient-free optimisation.
-->

---

## Where lattice Boltzmann comes in

LBM is the **independent ground truth** — it never sees the model's internals.

![[report_paraview_velocity_wake.png|640]]

- 2D flow past the body, from-scratch **OpenLB** solve (Bouzidi no-slip, sponge outlet).
- Every shape the AI proposes is **re-simulated** here, and the measured `Cp` is checked against what was asked for.

![[EXP-043_domain_schematic.png|560]]

<!--
SPEAKER NOTES — Slide 4 (LBM ground truth, ~60s)
- This is the home-crowd slide — show the von Kármán street, they'll like it.
- The key methodological point: the validator is a SEPARATE physics engine.
  The model has its own internal pressure estimate; we deliberately DON'T trust it.
  We bounce every generated shape through OpenLB and measure Cp on the surface nodes.
- "It's the difference between a student grading their own homework versus an
  external examiner. LBM is the external examiner."
- Domain: ~24 chords long, velocity inlet, pressure outlet, sponge zone to kill
  reflections. Laminar regime, Re a few hundred to ~1000.
-->

---

## LBM setup — key parameters (as run)

![[EXP-043_lbm_setup_table.png|960]]

<small>Every validation run scales resolution with `Re` to hold **Mach ≈ 0.13** fixed — numbers taken straight from the OpenLB `UnitConverter` / `Timer` output.</small>

<!--
SPEAKER NOTES — Slide 4b (LBM parameter table, ~45s — DSFD home crowd)
- This is the "show your working" slide for a discrete-methods audience.
- Hit the three points they care about most:
  1. Mach is HELD FIXED at ~0.13 by scaling N with Re (N ~ 0.2*Re). At fixed tau,
     u_lat = Re*(tau-0.5)/3N, so without that scaling Ma runs away with Re.
  2. tau = 0.545 sits only 0.045 above the 0.5 floor — that's the stability margin;
     omega = 1.835 < 2, no over-relaxation. Knudsen ~3e-4 keeps us in the continuum.
  3. Bouzidi curved wall (sub-grid distance) is what makes arbitrary generated
     polygons usable without staircasing.
- Cost line is the pitch for online supervision: minutes per shape on one GPU.
- Source-file DEFAULT is N=300/Ma~0.087, but the production runs used the Re-scaled
  N at Ma~0.13 — the table reflects what actually generated the Cp curves.
-->

---

## Diffusion models, in one picture

Train: take a clean shape, **add noise step by step** until it's static.
Learn the reverse: **remove a little noise at a time**.

![[report_diffusion_process.png|760]]

- To generate: **start from pure noise and denoise** into a shape.
- Analogy: a **sculptor releasing a form from rough stone** — or a sandstorm run backwards until a shape settles out.

<!--
SPEAKER NOTES — Slide 5 (Diffusion analogy, ~60s)
- Keep it intuitive, no equations needed.
- "We corrupt a known shape with noise in many small steps — top row, left to right.
  The network's only job is to undo ONE step of that. Do that well, and you can
  start from complete static and walk backwards into a brand-new valid shape."
- Sculptor / sandstorm-in-reverse analogy — pick whichever lands.
- The shape here is a radius-vs-angle curve r(θ) of a closed boundary, so "denoising
  a shape" = denoising that curve. We'll need to STEER this — next two slides.
-->

---

## Steering it (1/2): encode the target pressure

We don't feed the model the raw pressure curve — we **compress it**.

![[report_dct_encoding.png|720]]

- A **discrete cosine transform (DCT)** keeps the first **8 coefficients** — like keeping only the main "notes" of a tune, or JPEG for a curve.
- Result: a tidy **8-number summary** of the target, the same size **no matter how many surface nodes** the shape has.

<!--
SPEAKER NOTES — Slide 6 (Conditioning part 1 — encoding, ~70s)
- THIS is the key technical slide pair — slow down here.
- Left plot: grey is the real target Cp around the boundary; orange is rebuilt from
  just 8 DCT coefficients — visually it's nearly perfect. Dashed (K=3) is too coarse,
  so 8 is the sweet spot. Right plot: those 8 numbers ARE the context vector.
- Why DCT and not raw values: (a) resolution-independence — 40 nodes or 400, the
  conditioning is still 8 numbers, so sampling resolution is decoupled from training;
  (b) it's a smooth low-dimensional handle on "the shape of the pressure curve".
- Analogy: "Like describing a melody by a few dominant notes instead of every
  vibration — enough to recognise the tune, small enough to work with."
-->

---

## Steering it (2/2): close the loop so it can't cheat

![[report_conditioning_pipeline.png|820]]

- The 8-number target is injected into the **score model** (graph network over the surface nodes).
- A **DeepSets head reads the generated shape back** and re-predicts its pressure code `ĉ`.
- An **auxiliary loss `‖ĉ − c‖²`** forces *the shape it draws* to actually carry *the pressure you asked for* — the conditioning can't be ignored.

<!--
SPEAKER NOTES — Slide 7 (Conditioning part 2 — closing the loop, ~70s)
- Walk the pipeline left→right: target Cp → DCT (8 numbers) → score model → shape.
- The crucial bit is the bottom return path: a DeepSets head looks at the GENERATED
  shape's geometry and tries to recover the pressure code. If the shape doesn't
  actually realise the requested pressure, that auxiliary loss punishes it.
- "Without this, a conditional generator can learn to quietly ignore the condition
  and just make plausible shapes. The loop ties the output back to the request."
- DeepSets = order-invariant: it doesn't care which node is 'first', matching the
  fact that a boundary has no canonical starting point.
- One line on the score model: it's a graph network — surface nodes are graph nodes,
  ring edges connect neighbours, time-step and target ride on a global token. Keep
  it at that altitude; details in the appendix slide if asked.
-->

---

## Does it work? Ask, generate, then *simulate*

Give a target `Cp` → model draws a shape → **run that shape in OpenLB** → compare.

![[EXP-041_cp_inverse_design.png|640]]

- Lines = the pressure we **asked for**; dots = what **OpenLB measured** on the generated shape.
- In-distribution: correlation **ρ ≈ 0.96** — the generated shape really does carry the requested pressure, confirmed by *independent* physics.

<!--
SPEAKER NOTES — Slide 8 (Hero result — inverse design + LBM, ~70s)
- THE money slide. Two co-headlines of the talk meet here: generative inverse design
  AND lattice-Boltzmann validation.
- "We asked for two different pressure distributions. The model drew two shapes.
  We threw those shapes into OpenLB — which has never seen the model — and measured
  the pressure. The dots land on the lines."
- ρ≈0.96 in-distribution. Stress: this is NOT the model grading itself. The validator
  is a separate solver.
- Be honest: there's scatter, especially near the rear; this is preliminary and 2D.
- Q&A ready: ρ is a *shape* metric (Pearson correlation — invariant to magnitude
  scale + offset). So ρ≈0.96 means the pressure *pattern* matches; absolute magnitude
  is tracked separately by the affine slope `a` (see OOD slide / magnitude creep).
  If asked "do the pressures match?": the distributions match in shape to ρ≈0.96; the
  level carries a separate scale factor.
-->

---

## The validation gate: does the loop even close?

Before trusting generated shapes, prove the **pipeline reproduces known physics**.

![[EXP-043_cp_validation.png|620]]

- Take **known ellipses**, run them through the same OpenLB loop across a grid of `Re`.
- Measured `Cp` matches the dataset reference at **ρ up to 0.999** (mean ≈ 0.96).
- That's the **baseline gate**: if this failed, no generated result would mean anything.

<!--
SPEAKER NOTES — Slide 9 (Validation gate, ~55s)
- Logic: you can't trust the examiner until you've calibrated the examiner.
- We run shapes whose pressure we ALREADY know (dataset ellipses) through our own
  from-scratch OpenLB pipeline and confirm it reproduces them — ρ up to 0.999.
- Only once that gate passes does the ρ≈0.96 on GENERATED shapes carry weight.
- This is the scientific spine of the project: an independent physics ground truth,
  calibrated, then used to grade the generative model.
-->

---

## Being honest: it breaks out-of-distribution

Push the target **far from the training data** and fidelity degrades.

![[EXP-043_ood_results_summary.png|860]]

- As out-of-distribution distance grows: **correlation drops**, pressure **magnitude creeps**, and **drag drifts up**.
- So the method is **trustworthy near the data, speculative beyond it** — exactly what you'd want to know before optimising with it.

<!--
SPEAKER NOTES — Slide 10 (Honest limits / OOD, ~55s)
- Don't hide the failure mode — a fluids audience will respect the candour.
- Left: correlation falls off with OOD distance. Middle: a systematic magnitude
  bias ("creep") — the model over/under-shoots pressure amplitude. Right: drag
  coefficient drifts as you extrapolate.
- This matters for the application: gradient-free optimisation will WANT to push
  into unexplored regions, so knowing where the model stops being reliable is
  essential. Quantifying and extending that trustworthy envelope is ongoing work.
-->

---

## Why this could matter

**1 · Gradient-free design optimisation**
- The model *samples* shapes for a target — no adjoint, no mesh-differentiation, no solver-in-the-loop gradients.
- Propose candidates → screen the promising ones with LBM → iterate. A **generator + cheap physics filter** loop.

**2 · Beyond aerodynamics: cardiac electrophysiology (AF)**
- The machinery is generic: a **diffusion model over a graph/mesh**, conditioned on a target *field*.
- Speculative transfer: generate **atrial geometries / activation patterns** conditioned on electrophysiology, to study **atrial fibrillation**.

<small>(2) is a direction, not a result — flagged as future work.</small>

<!--
SPEAKER NOTES — Slide 11 (Applications, ~70s)
- (1) Gradient-free optimisation: the big practical pull. Classic aero shape
  optimisation needs gradients (adjoint solvers, differentiable meshing — painful).
  A conditional generator sidesteps that: ask for a performance target, sample
  candidate geometries, validate the good ones in LBM. Especially useful for
  non-differentiable or multi-modal objectives.
- (2) Cardiac / AF: be explicit this is aspirational. The SAME pattern — diffusion
  on a graph/mesh conditioned on a target field — isn't specific to aerodynamics.
  Atrial fibrillation involves chaotic electrical activation over atrial tissue
  geometry; a conditional generative model over that mesh could propose anatomies
  or activation maps consistent with observed electrophysiology. Cross-disciplinary,
  long horizon, flagged as a direction.
- Tie back: in both cases LBM-style independent physics is the validator that keeps
  the generator honest.
-->

---

## Summary & what's next

**The idea:** a discrete fluid solver as **teacher and examiner** for a generative shape model.

- ✅ Generative **inverse design** works in-distribution (ρ ≈ 0.96, *independently* validated)
- ✅ A calibrated **LBM validation gate** (ρ up to 0.999) anchors every claim
- ⚠️ **Out-of-distribution** behaviour degrades — the current frontier

**Next:** extend the trustworthy envelope · move toward 3D / higher Re · close the optimisation loop.

**Thank you — questions welcome.**
<small>archie.dobson@postgrad.manchester.ac.uk · University of Manchester</small>

<!--
SPEAKER NOTES — Slide 12 (Summary + thanks, ~45s)
- Re-land the one-sentence thesis: LBM as teacher + examiner for a generative model.
- Recap the three beats: it works in-distribution, it's independently validated,
  it breaks OOD and that's the honest frontier.
- "All very early — under six months in — but the loop closes, and that's the
  signal that says keep going." Invite questions.
-->

---

<!-- ============================ APPENDIX ============================ -->

## Appendix — under the hood (for Q&A)

![[report_architecture.png|820]]

- Noisy boundary `xₜ = r(θ)` on a ring graph → stack of **graph-network blocks** (edge → node → global) predict the noise.
- Time-step and the 8-D target condition enter via a **global token** (`u`), not per-node — so conditioning propagates through the global→node pathway.
- Trained with classifier-free guidance dropout; DeepSets pressure head supplies the auxiliary `Cp` loss.

<!--
SPEAKER NOTES — Appendix (only if asked)
- Battaglia-style GN block: EdgeModel φᵉ → NodeModel φᵛ → GlobalModel φᵘ, N layers.
- DDPM (Ho et al. 2020) noise-prediction objective; cosine schedule.
- Why global-token conditioning beat per-node conditioning: cleaner, order-invariant,
  decoupled from node count. Mention if pushed on architecture choices.
-->
