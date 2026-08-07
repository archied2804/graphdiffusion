---
title: "What to write now — drafting order, format and style"
date: 2026-08-07
tags: [paper1, writing, guide]
status: active
---

# What to write now

> Companion to [[REPORT_OUTLINE]], which fixes the structure and the vocabulary
> rules. This file answers a narrower question: **given 8 of 29 experiments
> reported, which sections are safe to draft today, which are not, and why.**
>
> No prose here is for the paper. Everything below is scaffolding, ordering and
> critique — the writing is yours.

---

## 1. The rule for deciding what is safe to draft

A section is safe when **no pending experiment can change its claims**, only add
to them. By that test:

| Status | Meaning |
|:--:|---|
| 🟢 **Draft now** | Evidence complete. Later results add detail, not reversals |
| 🟡 **Draft the skeleton** | Structure and method are fixed; numbers pending |
| 🔴 **Do not start** | A pending experiment could invert the claim |

Writing a 🔴 section early is how a paper acquires a thesis it then defends
rather than tests. That is precisely the failure the pre-registration exists to
prevent, and it applies to the prose as much as to the analysis.

---

## 2. Section-by-section

### 🟢 §3 Method — **start here, today**

This is the largest block of genuinely finished work and it is the most
under-appreciated. Everything it describes is now sourced or measured.

| §3.x | Draft from | Why it is safe |
|---|---|---|
| Dataset & provenance | [[DATASET_PROVENANCE]] | Traced to Lino 2022/2025, Nektar++, parameter table cited from source |
| Partition & splits | [[results/P1-F01_data-partition]] | Frozen, hashed, leakage audited |
| Conditioning descriptor | [[CONDITIONING_MATHS]] | Derivation, validity bound, numerical validation all complete |
| Solver & grading | [[SOLVER_CASE]], [[results/P1-F02_grading-metric]] | Case documented, metric consolidated and tested |
| Reference model | [[results/P1-F05_reference-model-attempt3]] | P1-REF frozen, 5 seeds |

**Write §3.3 (conditioning) first.** It is the section with the cleanest
narrative — a claim that was asserted, a measurement that falsified it, a
derivation, and a fix with a stated validity range. You have the maths, the
table and the bound. Nothing pending touches it.

Three things §3 must now state that earlier drafts would not have:

1. The training data is **Nektar++ spectral/hp element**, released at ~half its
   original resolution. Your grading solver is **lattice Boltzmann** — a
   different method, not a different configuration.
2. The descriptor's invariance holds to ~1% for `N ≥ 64` and degrades below
   that. State the bound; do not assert invariance.
3. The boundary conditions of the production case **match the source exactly**
   (uniform inlet, free-stream top/bottom, `p=0` outlet, no-slip body).

### 🟢 §4.1 Instrument qualification — **draft now**

Complete, and the strongest methodological content in the paper.

Draft from [[results/P1-F03_solver-qualification]] and
[[results/P1-F06_schafer-turek]]. The spine:

- The solver reproduces Schäfer–Turek drag to **0.02%** (fp64, N=80) — cite
  [[SOLVER_CASE]] §8 for the reference-value provenance caveat.
- After the units convention is removed, it agrees with the dataset to
  **6.2% ± 0.5pp** where the affine model fits.
- `a` carries **11–26% sensitivity to grid resolution**, which survives fp64 —
  so magnitude claims are **within-condition only**.
- The affine residual is **structured in all 15 cases** (median |z| = 8.12).

⚠️ Held back pending your transcription of the Schäfer–Turek bounds. Draft
around it; leave the comparison table's reference column empty.

### 🟢 §5 Limitations — **draft now, and draft it early**

Counter-intuitive but right: **most of your limitations are already known and
measured**, which is unusual and is a strength. Writing them now stops §4 from
over-claiming, because you will already have written down what you cannot say.

Ready to draft:

- The shape manifold is effectively **one-dimensional** (1 PCA mode = 99.60%
  variance) — this dataset **cannot** demonstrate C1's parameterisation
  bottleneck. [[results/P1-B03_basis-reachability]]
- `r(θ)` is **orientation-locked**; a 10° rotation moves a shape 759× outside a
  train-fitted basis. Disclose it.
- `a` is grid-dependent; ρ is non-monotonic under refinement.
- Model-side seed CV is **49.3%** (19.4% excluding one outlier).
- Single precision destroys lift at production resolutions.

### 🟡 §1 Introduction, §2 Related work — **skeleton only**

§2 is safe to draft in full *if* you write it as a survey rather than as
positioning. The moment it argues "and therefore our method is needed", it
depends on §4.

§1 must wait for the August gate. Its framing depends on whether A01–A03 show
the graph earns its place.

### 🟡 §3.6 Baselines — skeleton now

Describe the baselines and matching protocol (±5% parameters). B01's retrieval
index exists; only the solver half is pending.

### 🔴 §4.2 Architecture necessity — **do not start**

A01–A04 have not run. The entire claim could invert.

### 🔴 §4.3 Basis necessity — **do not start the argument**

B03 has already shown the premise fails on this dataset. B01 and B02 will
likely confirm retrieval is competitive in-distribution. You may draft the
*setup*, but the conclusion is genuinely open, and it is the section most at
risk of being written to a predetermined answer.

### 🔴 §4.4–§4.7, §6 — not started

D02, C0x, G0x, S0x all pending. §6 is written last, always.

---

## 3. Drafting order

Not the reading order. This order minimises rework:

1. **§3.3 conditioning** — cleanest, fully evidenced, warms you up on the
   maths-heavy style
2. **§3.1–3.2 dataset and partition** — factual, fast
3. **§3.4–3.5 solver and grading** — feeds §4.1 directly
4. **§5 limitations** — while the negative results are fresh, and before §4
   tempts you to overstate
5. **§4.1 instrument qualification** — the finished results section
6. **§2 related work** — as a survey; a good low-energy task
7. *(gate)* — then §4.2/§4.3, then §1, then §6

**Write §5 before §4.** It is the single highest-leverage ordering decision
here. Every over-claim in a first draft comes from writing results before
writing limits.

---

## 4. Format and style

### Venue and format

Target is **Machine Learning: Science and Technology** ([[REPORT_OUTLINE]] §1) —
IOP, LaTeX, `iopart` class, ~8–10k words, no hard limit, numbered sections,
Harvard or numeric references (IOP accepts both; pick one and be consistent).

**Draft in Markdown, convert late.** You are already writing `.md` in the
vault, the results files are Markdown, and the cross-links work. Convert to
LaTeX once, at the end, with Pandoc. Do not draft in LaTeX — it makes
restructuring expensive and you will restructure.

### Style rules that fit this paper specifically

**Past tense for what you did, present for what is true.** *"The descriptor was
measured at 41% deviation"* / *"The descriptor is invariant to ~1% for N ≥ 64."*

**Lead each subsection with the finding, then the evidence.** Reviewers of an
ML-for-science paper skim for claims. Do not build to a reveal.

**Every number gets a unit, an uncertainty, and an n.** You have the seeds and
the CIs — use them. `0.938 ± 0.005 (n=4)` not `≈0.94`. This is the single
clearest signal that the work is careful, and it is nearly free for you because
the results files already carry it.

**Three significant figures maximum on solver output.** [[SOLVER_CASE]] §6
measured a 0.034% reproducibility floor across recompiles. Quoting four figures
implies precision the pipeline does not have.

**Name the falsifier in the text, not just the appendix.** *"This would have
been falsified by X; it was not"* is the sentence that distinguishes a
pre-registered study from a narrated one. It is your main methodological
differentiator — use it in §4 repeatedly.

**Report the negative results in the main text.** Six of eight experiments
falsified their hypothesis. Burying that loses the paper's actual contribution.

### What to avoid

| Avoid | Because |
|---|---|
| "direct node movement", "mesh-native" | Not earned — see [[REPORT_OUTLINE]] §3 |
| "significantly" without a test | You have σ and CIs; say which |
| Passive voice hiding agency | *"a factor of two was applied"* — by whom, when, why |
| Citing `a ≈ 1.74` | Superseded. It was a units convention |
| "state of the art" | You have no SOTA comparison and do not need one |

### Figures

Every figure should carry its own falsifier. Priority order:

1. **ρ vs nearest-neighbour distance**, model and retrieval overlaid — the §4.3
   argument in one panel. Index already built.
2. **The OOD ladder**, 44× dynamic range, six rungs.
3. **Resolution sensitivity of `a`**, fp32 vs fp64 — makes the instrument
   limitation visible.
4. **The AoA10 rotation control** — a plausible result that did not survive
   checking. Rare and worth showing.

---

## 5. Where this leaves the plan

**8 of 29 reported.** The foundation group is nearly closed; the science has
barely started.

| Group | State |
|---|---|
| **F — foundation** | F01 ✅ F02 ✅ F03 ✅ F05 ✅ F06 🟡 · **F04, F07 remain** |
| **B — basis** | B03 🟡 · B01 half-built · B02, B04 pending |
| **D — discretisation** | D01 ✅ · D02, D03 pending |
| **A, C, G, S** | none started |

### Immediate next, in order

1. **F04 noise floor** — now the binding constraint. Nothing in A/B/C/G can be
   graded "different" until it exists, and F03 showed it must sample
   **resolution**, not only seeds. ~18 runs.
2. **B01 solver half** — ~35 runs, retrieval assignments already stored.
3. **A01–A03** — the August gate. These decide §1's framing and whether C1
   survives.

F07 has shrunk to a confirmation: the mechanism is identified and sourced, so
its Step 2 (dimensionless self-consistency, 2 runs) is all that remains of it.

### What the gate now decides

C1's premise is already weakened by B03. If A01–A03 also show the graph does
not earn its place, the honest paper becomes a **methodology and evaluation**
contribution rather than an architecture one — and on the evidence so far that
would be a *better* paper, because the instrument work is genuinely novel and
the architecture claim is not.

Worth deciding that consciously at the gate rather than discovering it in §4.
