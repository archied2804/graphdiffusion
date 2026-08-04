---
title: "Rolling publication diary"
date: 2026-07-29
tags: [publication, diary, planning]
status: active
---

# Rolling publication diary

Running log of publication activity across the whole series. Newest entries at the top.
Paper 1 assessment and schedule: [[PUBLICATION_PLAN]] · Three-paper arc and future research: [[RESEARCH_PROGRAMME]].

**How to use this:** one entry per meaningful event — a result that changes the story, a
submission, a decision, a review. Keep entries short. The value is in the *dated trail*:
when it comes to writing the thesis narrative or a fellowship application, the reasoning
behind each pivot is the part that is otherwise impossible to reconstruct.

---

## Live status

| Paper | Stage | Target venue | Next action | Due |
|---|---|---|---|---|
| **1. Node-native design** | Results gathering | MLST (alt: SMO; fallback PoF) | APS DFD abstract | **31 Jul 2026** |
| 2. Free node movement + closing the loop | Not started | TBD | — | Q1 2027 |
| 3. Genuinely mesh-native | Not started | TBD | — | Q3 2027 |

### Open blockers — Paper 1

Reordered 2026-07-29 for the node-native reframing: architecture and parameterisation
baselines are now ahead of the holdout work, because they can invalidate the *headline*.

| #   | Blocker                                                                                  | Owner | Status  |
| --- | ---------------------------------------------------------------------------------------- | ----- | ------- |
| 1   | **Architecture baselines** — DCT-MLP / 1-D CNN / no-edge. Does the graph earn its place? | AD    | 🔴 Open |
| 2   | **Low-dim parameterisation baseline** — shape-PCA at matched DOF                         | AD    | 🔴 Open |
| 3   | Held-out test set — headline result is on training data                                  | AD    | 🔴 Open |
| 4   | Nearest-neighbour retrieval baseline                                                     | AD    | 🔴 Open |
| 5   | Contribution-targeted ablations (λ_pressure, K, w, k_neighbors on aero)                  | AD    | 🔴 Open |
| 6   | Node-count transfer (train 96 → sample 48/192) — underwrites C1/C2                       | AD    | 🟡 Open |
| 7   | Seed repeats / error bars                                                                | AD    | 🟡 Open |
| 8   | DCT-space OOD ladder + AoA10 physical OOD                                                | AD    | 🟡 Open |
| 9   | Diversity / coverage metrics — "exploration" is unmeasured                               | AD    | 🟡 Open |
| 10  | Cylinder benchmark (instrument fitness; Hron–Turek deferred)                             | AD    | 🟡 Open |
| 11  | §1/§2/§6 written; refs.bib expanded toward geometric DL                                  | AD    | 🟡 Open |

### Deadline board

| Date | Event | Type |
|---|---|---|
| **31 Jul 2026** | APS DFD 2026 abstract (5pm ET) | Conference |
| 8–10 Sep 2026 | UK Fluids Conference, Leeds — poster | Conference |
| ~Sep 2026 | NeurIPS ML4PS workshop (unconfirmed) | Workshop |
| Nov 2026 | APS DFD 2026, Orlando | Conference |
| **1 Sep 2026** | **AUGUST GATE decision** — does the graph earn its place? | Milestone |
| Nov–Dec 2026 | **Paper 1 → MLST (alt SMO; fallback PoF) + arXiv** | Submission |
| ~May 2027 | AIAA SciTech 2028 abstract | Conference |
| Q1 2027 | P2 split decision (does R2.3 need its own paper?) | Milestone |
| Q3 2027 | P2 submission target | Submission |

---

## Entries

### 2026-08-03 — Tranche 1 launched; F01 reported and **falsified**, favourably

**Running now:** P1-F05 reference model, 5 seeds × 1000 epochs, ~90 min each
(~7.5 h total). `scripts/paper1/run_p1_f05.sh`, logs under
`outputs/paper1/P1-F05_reference/seed*/`. Status board:
`python scripts/paper1/p1_status.py --runs`.

**[[../paper1/results/P1-F01_data-partition|P1-F01]] verdict: falsified** — in a way
the spec did not anticipate, and to the project's benefit. Three findings:

1. **Node counts are not fixed at 96.** They span **52–96 across 12 distinct
   values**; the HDF5 array is *padded* to its maximum and reading `data.shape[1]`
   as the node count is the error. Only **37 of 5701** samples have 96 nodes; the
   mode is 60. [[PUBLICATION_PLAN]] §2.2 and §5.1 claimed the opposite and are now
   corrected. The model has been trained across discretisations from the start —
   direct support for C2, available all along. Knock-ons: D02 is reframed from
   "does it transfer?" to "does it extrapolate outside 52–96?"; A02's fixed-length
   1-D CNN is no longer a drop-in comparator and must declare a padding scheme
   (*an argument for the graph*); G02's node-count confound dissolves, since AoA10's
   68–80 nodes are inside the training range.
2. **`pOnEllipseTest.h5` does not exist** — 404, never fetched, so the failure was
   invisible. The publisher instead provides an evaluation suite nobody had
   downloaded: `InDist` (50) plus four single-axis OOD sets — `LowRe`, `HighRe`,
   `Thin`, `Thick` (50 each). Measured in DCT conditioning space they form a real
   ladder: **indist 0.99× → highre 1.78× → lowre 2.39× → thin 10.8× → aoa10 12.9×
   → thick 33.3×** (median NN distance to train, normalised by within-train).
   Six rungs of real CFD over a 33× range. G03 is rebuilt around these; the
   synthetic ladder becomes optional and its realisability screen unnecessary. This
   answers *"reconsider the construction of OOD samples"* completely — they are the
   publisher's sets, not ours. Difficulty is strongly anisotropic: **geometry
   perturbations displace the conditioning vector 5–14× further than Reynolds
   ones.** Also found: `uvpAroundEllipse*.h5`, full velocity/pressure fields
   (~266 MB each) — logged as a Paper 2/3 asset, explicitly *not* scoped in.
3. **`indist` is interpolation, not extrapolation** — 1.10× within-train NN
   distance in design space, 0.99× in DCT space, i.e. indistinguishable from
   another training sample. Must be stated; the OOD sets carry every generalisation
   claim. This makes B01's retrieval baseline more important, not less.

Year-1 targets #4849/#4066 re-confirmed as **TRAIN**.

**Code:** `DatasetUrl.TEST_H5` removed, five evaluation URLs added, variants
extended; `GraphDataLoader` accepts an explicit frozen partition; `train.py` gains
`--seed`, `--partition`, `--exp-id` and emits `run_meta.json` per PROTOCOL §7.
Partition frozen at `data/splits/paper1_partition.json` (5131/570/50 + 224 OOD).
Quality gate: 231 tests pass, ruff/black clean, mypy clean except one pre-existing
error in `postprocessing/loaders.py:16`.

**⚠️ Action required:** the working tree is uncommitted, so `run_meta.json` will
record `git_dirty: true` and PROTOCOL §7 forbids using such a run for a reported
number. Committing before the runs finish fixes it (git state is captured at run
end).

**Next:** F02 `cp_metrics` module — the last thing blocking any graded result.

---

### 2026-08-02 — Metric audit; Schäfer–Turek added to close the `a ≈ 1.74` question

**What happened:** Audited the metric set and added [[../paper1/METRICS|METRICS.md]]
— every metric assessed for what claim it carries, what it cannot do, its cost, and
how a reviewer attacks it, with recommended minimum / full / stretch sets. Added two
experiments, taking the suite to **29**.

**Metric gaps found — all fixable at zero solver cost:**
- **`Cd` and `Cl` were absent from the protocol.** `force.csv` carries both for
  every run; `Cd` is read by six existing scripts, **`Cl` by none**. ρ is
  affine-invariant and therefore *structurally* unable to see magnitude, and `a` is
  only a fitted calibration parameter — so the paper had no physical force metric
  at all. Now primary: `Cd` gives G04 an independent attribution axis, `Cl ≈ 0` on
  lifting targets is G02's failure mode in one number.
- **The affine model was assumed, never tested.** No check that the residual of
  `Cp ≈ a·Cp_ref + b` is structureless. `a` underpins C4, so a residual-structure
  test is now mandatory in F02.
- **No distributional metrics** — everything measured per-sample conditional
  fidelity, which evaluates a regressor, not a generative model. Precision/recall
  and MMD added to B04; unconditional sample quality was unmeasured on aero data
  entirely.
- **ρ has no established scale.** All ellipse Cp curves look alike, so ρ may be
  near-saturated for any plausible shape. **Chance-level ρ must be computed before
  ρ is used as a headline** — added to F04. Highest-risk item in the metric set.
- Also added: chordwise band errors, suction-peak magnitude/location, `Cp_rms`
  (already in `cp.csv`, unused), achievability bound from B01 decomposing error
  into target infeasibility vs model shortfall, sampling cost, and a
  multiple-comparisons rule (~80 comparisons at 3σ ⇒ ~1 false positive).
- **Roughness de-weighted to a diagnostic.** Year 1 justified `T = 1000` by boundary
  wobble; independent grading then showed the wobble was cosmetic.

**New: [[../paper1/specs/P1-F06_schafer-turek|F06]] — Schäfer–Turek benchmark.**
The reason this matters more than it first appears: **every solver number in the
Year-1 report is *relative***, graded by correlation against the dataset.
Correlation cannot detect an error the two share, and an affine fit absorbs exactly
the class of error most likely present — a constant scale. F06 is the first
*absolute* check, with published bounds on `Cd`, `Cl`, **Δp** and St. Three levels:
stock `cylinder2d` (validates the build), ST geometry through our `ellipse2d`
binary (validates our case and extraction), and the same with our uniform inlet
(prices the simplification `lattice.h:15` records but never quantified).

**New: [[../paper1/specs/P1-F07_normalisation-audit|F07]] — the source of `a`.**
EXP-040 already spotted this and left it open: *"Pin `a ≈ 1.78` analytically from
the dataset card"* — an unticked action item, while the number has been used in
every downstream result since. F07 registers ten candidate mechanisms each with a
**predicted factor**, so the audit is a matching exercise rather than a
rationalisation.

**One thing that needs correcting.** EXP-040 reads `a ≈ 1.78` as the dataset's
velocity being "scaled by 1.8". That is arithmetically inconsistent: Cp ∝ U⁻², so a
velocity scaling of 1.8 gives a Cp factor of **3.24**, not 1.78. The numerical
coincidence is real but only works if the 1.8 applies to the **dynamic pressure**,
not the velocity. Resolving that is F07's crux. Separately, √1.78 ≈ 4/3 almost
exactly, which is the other live candidate and currently has no known origin.

**Also:** EXP-040 reports 1.78 on the cleanest case against a 1.74 mean. Whether
that ~2% spread exceeds `σ_a` decides whether `a` is one constant at all — F04
answers it, F03 now runs the regression of `a` on Re, `H` and `b`.

**Consequence for G04:** creep attribution now depends on F07. "Generated shapes
grade at `a ≈ 2.5–3.5` against 1.74" compares one unexplained number to another
until the reference scale is known. Paper 2's R2.2 depends on it more, since it
targets `a → 1.74` — a target that may be partly a unit-conversion artefact.

---

### 2026-08-02 — Paper 1 experiment suite pre-registered

**What happened:** Built [[../paper1/README|the Paper 1 experiment suite]] — **27
experiments** across seven groups (F A B C D G S), each written as a
**pre-registration**: hypothesis, falsifier and pass/fail decision rule committed
*before* running, results appended and never back-fitted. Supporting documents:
[[../paper1/PROTOCOL]] (splits, seeds, metrics, grading pipeline, significance rule),
[[../paper1/EXPERIMENT_REGISTER]] (master table, dependency graph, coverage map),
[[../paper1/REPORT_OUTLINE]] (paper skeleton and writing guide — prose to be written
by hand, no generated text).

**Design decisions:**
- Naming `P1-<GROUP><NN>_<slug>`, group letter mapping one-to-one onto a results
  subsection, so any experiment ID states where its result lands.
- All ten unanswerable reviewer questions from [[PUBLICATION_PLAN]] §13 are now
  assigned to experiments; all twelve viva feedback items are discharged.
- Three experiments carry pre-registered hypotheses that **contradict the paper's
  own framing** — A02 (the 1-D CNN is predicted to match the graph), B02 and B03
  (a low-dimensional basis is predicted to suffice on ellipses). Registering the
  uncomfortable prediction in advance is what stops the result being spun later.

**Two new experiments that did not exist in any earlier plan:**
- **F04 — grading noise floor.** Measures the solver's own repeatability so every
  reported difference can be tested against `2σ`/`3σ`. Eighteen solver runs, and it
  converts every "noisier than monotonic" hedge in the Year-1 report into a number.
  Answers reviewer question 9 outright.
- **F02 — canonical grading metric.** The headline `(ρ, a)` metric currently exists
  as **four independent copies** across `scripts/`. Consolidating to one tested
  module, and checking the four agree, is a prerequisite for every graded number.

**Cost correction — the earlier estimate was wrong in both directions.**
[[PUBLICATION_PLAN]] §10 called CFD grading "the critical path" at ~100 runs.
Measured from this repository: a training run is **~85 min** (161 batches/epoch at
~33 it/s, `outputs/EXP-022_T1000.log`); a solver run is **7 s – 14 min**, median
~1–3 min, across 71 timed runs. The suite needs ~640 solver runs (6× the estimate)
but only ~21 h of solver time, against ~64 training runs and ~91 h. **Training is
~4× the solver cost.** The bottleneck is bookkeeping across ~64 training runs, not
CFD queue time — the throughput fix is a training sweep driver, not a CFD batch
driver.

**Next:** F01–F04 (protocol, no model needed) → F05 reference model → Stage 2
story-breaking experiments → August gate.

---

### 2026-07-29 — Three-paper programme adopted

**What happened:** Formalised the series into a three-paper arc with an explicit
one-word-per-paper discipline. Produced [[RESEARCH_PROGRAMME]] — arc diagram, dependency
graph, P2/P3 research content, risk register with fall-backs, programme Gantt, review
checkpoints. [[PUBLICATION_PLAN]] gained a P1 Gantt, a work-type breakdown, and an
execution-ordered todo.

**Claim discipline:** P1 earns *"discretisation-native"*, P2 earns *"direct node movement"*
(needs Cartesian or local-frame diffusion), P3 earns *"mesh-native"* (needs variable topology).
Each paper's claims are sized to the evidence gathered for that claim alone.

**P2 fleshed out:** R2.1 free-form node movement, with a **mitigation ladder** whose first rung
is **normal-offset diffusion** — diffuse a scalar offset along the local outward normal of a
template. Still 1 DOF but *locally* defined, so it escapes radial's star-convexity limit without
the rigid-motion ambiguity of full Cartesian. Strict generalisation of P1's representation and the
natural bridge. New metric needed: **validity rate** (simple, closed, non-self-intersecting,
well-spaced), which radial parameterisation made automatic and free movement gives up.
R2.2 physical (Re,H) conditioning against P1's `a → 1.74` target. R2.3 CFD-in-the-loop
optimisation — the solver promoted from examiner to oracle, directly motivated by P1's DPS
negative (replace the failed surrogate gradient with the real solver). **Flagged P2 as probably
two papers**, split decision at Q1 2027.

**P3 fleshed out:** variable topology (de-risked by P1's node-count transfer), real aerofoil
families + multi-objective, higher Re / 3D, and transfer learning as the answer to data cost.
Noted that **P3.2 is the strategic answer to P1's sharpest criticism** — the representational
advantage should appear in a genuinely high-dimensional design space, not a 2-parameter ellipse
family. P1 should *name* this rather than overclaim on ellipses.

**Two dependency edges worth remembering:** P1's node-count transfer study de-risks all of P3.1
for one week's work; P1's creep quantification is what makes R2.2 falsifiable at all.

**Worth-publishing verdict:** yes — as a foundations-and-protocol paper, not a performance paper.
One correction logged: *"lays foundations for further research"* is a valid programme rationale
but **not a contribution a reviewer credits** — reviewers assess the paper standalone. It should
shape what gets done, never appear as why the paper matters.

**Effort reality check:** P1 needs little novel *research* — the baselines are standard controls,
and all genuinely novel work is in P2/P3. But the critical path is **~100 CFD runs**, roughly
doubling the project's solver work to date (86 runs across EXP-040→043). Action: build a batch
run-manifest driver in week 0 and keep the GPU build fed continuously; front-load the
solver-independent work so the August gate does not wait on the queue.

**Next:** APS DFD abstract by Friday; batch CFD driver; then architecture baselines.

---

### 2026-07-29 — Reframed to node-native design; LBM demoted to instrument

**What happened:** Pivoted Paper 1's framing. New contribution hierarchy — **C1** node-native
generative shape design (no hand-designed low-dim parameterisation), **C2** resolution-independent
DCT/DeepSets conditioning, **C3** the LBM examiner as *instrument*, **C4** the failure taxonomy
as payoff. Demoting LBM from contribution to instrument is correct and adopted.

**Two claims rescoped as overclaims** (verified against the code, not inferred):

- *"Direct node movement"* — **all eleven configs** use `feature_mode: radial_norm` with
  `output_dim: 1`. The node feature is a single scalar `[r/r̄]`; `[cos θ, sin θ]` is fixed
  context, never diffused. Nodes move **radially only, along fixed rays, on a frozen angular
  grid** — 1 DOF, not 2. `feature_mode="cartesian"` (2 DOF, genuine node movement) exists in
  the dataset class and **has never been trained**; EXP-012 "Method B — Cartesian" is actually
  radial_norm + node-Cp concat. Radial-from-centroid is itself a parameterisation and cannot
  express non-star-convex boundaries. → say **"per-node diffused shape field"**.
- *"Mesh-native"* — the graph is a **96-node boundary ring**, fixed topology, no interior nodes,
  and **every training sample has exactly 96 nodes**, so the model has never seen another node
  count. → say **"boundary-discretisation-native"**.

**The consequence that reordered the plan:** a fixed, ordered, 96-node ring with one scalar per
node means `r(θ)` **is a 1-D periodic signal of length 96**, and ring message-passing over it is
functionally a 1-D convolution. So *"does the graph earn its place?"* becomes the paper's
load-bearing question — and it was **planned and dropped**: `EXPERIMENT_PLAN.md` EXP-013
"Method E — Fourier coefficient MLP: is the GN architecture necessary?" became smoothness
regularisation instead. The experiment that tests the new headline does not exist.

**What it changes:**
- Blockers reordered — architecture baselines (#1) and low-dim parameterisation baseline (#2)
  now precede the holdout (#3) and retrieval (#4) work.
- **Venue shifts**: MLST primary, SMO alternate, Physics of Fluids demoted to fallback. Under a
  representational contribution the fluids are the test-bed, and a fluids journal would ask why
  the flow physics is so simple. arXiv primary category flips to `cs.LG`.
- Grid-convergence bar **drops** from mandatory to advisable — an instrument needs to be fit for
  purpose, not publication-grade CFD. Cylinder benchmark keeps; Hron–Turek → Paper 2.
- Report §4.1 (circle k-sweep) changes from *cut* to *redo on aero data* — receptive field is now
  contribution-relevant.
- Series arc improved: each paper earns one word — "discretisation-native" (P1), "direct node
  movement" (P2, needs Cartesian), "mesh-native" (P3, needs a mesh).

**August gate:** does the graph beat a 1-D CNN, and does node-native beat ~10 PCA modes? **If no
to either, revert to the validation-harness framing** — which remains fully evidenced and is the
insurance policy. Keep that material and the PoF option live.

**Next:** APS DFD abstract by Friday (pitch C3/C4 — fluids audience); then §4.1 architecture
baselines in the first fortnight.

---

### 2026-07-29 — Publication readiness assessment

First formal assessment against the Year-1 viva feedback. Produced [[PUBLICATION_PLAN]].

**Decision: reframe Paper 1** from *"we demonstrate validated inverse design"* to
*"an independent LBM examiner, and the failure modes it exposes."* The evidence supports
the second framing strongly and the first one weakly.

**Findings that drove the decision:**

- **Verified: the headline in-distribution result (§4.4, ρ≈0.95) is measured on training
  data.** Reproduced the `GraphDataLoader` split (seed 42, val_split 0.1, 5701 → 5131/570)
  and confirmed both targets — sim #4849 and sim #4066 — are in the **train** subset.
  Cause: `export_generated_for_openlb.py` picks targets across the full dataset via
  `pick_targets_farthest_first()` and never sees the split. n=2, farthest-first selected.
  This is the project's most serious methodological issue.
- `pOnEllipseTest.h5` exists upstream (`DatasetUrl.TEST_H5`) and has **never been
  downloaded**. `split="test"` is already plumbed through. Near-free fix.
- `pOnEllipseAoA10.h5` (24 sims, AoA 10°, Re 500–1000) is **already local and unused** —
  a real, physically-genuine OOD set. Best value-per-hour item available.
- **No baselines exist.** Flagged the nearest-neighbour retrieval baseline as potentially
  fatal: the design space is effectively 2D (Re, b) with 5701 training sims, so retrieval
  may match the model in-distribution. Must be run before any writing.
- Zero ablations target the claimed contributions. `λ_pressure=0` — the control that tests
  the DeepSets head — does not exist.
- `ScoreNetwork` is **147,585 parameters**. Ablations and the scale study are cheap; the
  binding constraint is CFD queue time.

**Verdict:** not submittable now; ~3 months of focused work away, which fits the window.

**Conference deadlines checked:** APS DFD 2026 abstract closes **31 Jul, 5pm ET** — two
days out, abstract-only, meeting in November. UK Fluids 2026 (Leeds, 8–10 Sep) oral
deadline passed, **posters still open**. AIAA SciTech 2027 (deadline 21 May 2026) missed;
target SciTech 2028.

**Next:** APS DFD abstract by Friday; then held-out evaluation and the retrieval baseline
in August, in that order.

---

### 2026-06-25 — DSFD 2026, Delft *(retrospective)*

Presented *"Using Lattice Boltzmann to Inform Graph Diffusion Networks for Aerodynamic
Design Exploration"*. Framed honestly as early-stage. Deck: `DGN_Docs/presentations/`.

Worth noting in hindsight: the talk's framing — **LBM as teacher and examiner** — is closer
to the correct Paper 1 framing than the Year-1 report is. Reuse the deck's narrative
structure when drafting §1.

---

## Entry template

```markdown
### YYYY-MM-DD — <short title>

**What happened:**

**What it changes:** (story / claim / target venue / timeline)

**Evidence:** (experiment ID, output path, figure)

**Next:**
```
