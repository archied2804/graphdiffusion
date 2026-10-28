---
title: "Paper 1 — report outline and writing guide"
date: 2026-08-02
updated: 2026-08-26
tags: [paper1, writing, outline, structure]
status: active
---

# Paper 1 — report outline and writing guide

**You write the prose.** This document is scaffolding only: what each section has
to *achieve*, which experiment supplies its evidence, roughly how long it should
be, and the traps specific to this paper. There is no drafted text here to paste,
deliberately — the argument only becomes yours by writing it.

Target: **Machine Learning: Science and Technology** (~8–10k words, no hard limit).
Alternates: SMO, then Physics of Fluids. See [[../publication/PUBLICATION_PLAN]] §8.

---

## 1. The one-sentence test

Before writing anything, finish this sentence in **twenty words or fewer**, on
paper, and pin it above the desk:

> *This paper shows that ______, measured by ______, and that this matters because ______.*

Every section then earns its place by serving that sentence. If a subsection does
not, it goes to the thesis instead. Re-do this test after the August gate — the
sentence will change depending on what A02 and B01 report, and the whole draft
should follow it.

---

## 2. Structure

Word budgets are guides, not targets. Total ≈ 9,000.

### §1 Introduction — ~900 words

**Achieves:** the reader knows what problem exists, why existing approaches leave
it open, and what this paper adds — before the end of page one.

Structure it as four paragraphs: the design problem → the parameterisation
bottleneck → what generative models on discretisations could do instead → what
this paper actually establishes. End with an explicit numbered contribution list.
Reviewers look for that list; make it findable.

Traps:
- Do not open with "Machine learning has revolutionised…". Open with the
  parameterisation bottleneck, because that is the paper's actual subject.
- The contribution list must match the results sections **one to one and in the
  same order**. This is viva point B1.2 and it is checked easily.

### §2 Related work — ~1,100 words

**Achieves:** the reader believes you know the field and that the gap is real.

Four threads, roughly a paragraph each: shape parameterisation for optimisation
(CST/Bézier/PARSEC/PCA); generative models for aerodynamic design; geometric deep
learning and mesh-based generative models; evaluation practice in ML-for-CFD.

The **fourth thread is the one that pays** — it sets up C3. Most work in this area
grades with a surrogate trained on the same data. Establish that as the norm, with
citations, and the independent solver becomes a contribution rather than an
implementation detail.

`refs.bib` currently has **13 entries** and needs ~35–40. The gap is in geometric
deep learning, mesh-based generative models, and shape parameterisation for
optimisation — not in CFD surrogates, where you are already covered.

### §3 Method — ~1,800 words

| § | Content | Notes |
|---|---|---|
| 3.1 | Problem statement and the boundary graph | Define `r(θ)` honestly as a 1-DOF-per-node field on a fixed ring |
| 3.2 | Graph diffusion model | Compress; cite Ho et al. and Battaglia et al. and move on |
| 3.3 | Conditioning: DCT + DeepSets | **Motivate node-count independence as a design goal here**, then D01/D02 discharge it |
| 3.4 | The LBM examiner | Instrument framing. Setup table → appendix |
| 3.5 | Experimental protocol | **Cite [PROTOCOL.md](PROTOCOL.md) content in full.** Splits, seeds, noise floor, matched-baseline rule |

§3.5 is unusual in a paper of this kind and it is your strongest single move.
Stating the significance rule *before* any result — including that a difference
under 2σ counts as no difference — pre-empts the "n=2, no error bars" attack
entirely and signals to a reviewer that the evaluation was designed, not assembled.

Every hyperparameter needs a sentence of motivation or a citation. `T=1000`,
cosine schedule, 4 layers, `k=6`, `clamp_range`, `w=3.0` — S03, A04, C02 and C03
exist to turn each of those into a measured choice rather than an inherited one.

### §4 Results — ~3,200 words

**Order the subsections C1 → C2 → C3 → C4**, matching §1's contribution list.

| § | Title | Experiments | Headline artefact |
|---|---|---|---|
| 4.1 | Protocol and instrument qualification | F01 F02 F03 F04 **F06 F07** | Table: partition sizes, frozen σ table, absolute benchmark, reference scale |
| 4.2 | Does the graph earn its place? | A01 A02 A03 A04 | Figure: four architectures, matched params, **`Cd` error and top-k** with CI vs the noise floor |
| 4.3 | Does node-native earn its place? | B01 B02 B03 B04 | Figure: performance vs basis dimension, with retrieval and node-native as horizontal lines |
| 4.4 | Discretisation independence | D01 D02 (D03) | Figure: validity and **`Cd` error** vs node count, 48→192 |
| 4.5 | What the conditioning contributes | C01 C02 C03 | Table: ablation grid, **Δ`Cd`** and Δ`a` vs reference, each against 2σ |
| 4.6 | Where it fails | G01 G02 G03 G04 C04 | Figure: **`Cd` error** and `a` vs DCT distance, noise-floor band shaded |
| 4.7 | Scale and data efficiency | S01 S02 S03 | Figure: data-fraction curve |

#### Which metric carries which claim — revised after F04, 2026-08-17

> [!warning] ρ cannot carry a discrimination claim on this corpus.
> Two **unrelated** dataset shapes correlate at **ρ = 0.9954**, and a predictor
> that ignores its input and returns the corpus mean scores **0.9969**
> ([[results/P1-F04_noise-floor]] §4). Across B03's 24 runs ρ spans 2.0σ in
> total while `Cd` spans 7.8σ. An earlier draft of this outline made ρ the
> headline everywhere; that is now wrong.

| Claim | Metric | Why |
|---|---|---|
| **Does A match B?** | **rank / top-k** | Discriminative by construction, with a defined chance level (top-1 33.3% vs 6.7%) |
| **How large is the error?** | **`Cd` relative error** | 40% dynamic range across the envelope against a 5% floor; and it is the quantity a designer actually asks for |
| **Is the shape right?** | `a`, ε_aff | `a` carries magnitude creep — the thing ρ hides |
| Corroboration only | ρ | **Always on a chance-referenced axis**, never 0→1, and always with the 0.9954 floor stated beside it |

Report ρ — do not hide it, a reviewer will compute it — but never as the
headline, and never without its chance level. Present `a` beside every number:
ρ alone hides magnitude creep, which is how it nearly went unnoticed in Year 1.

### §5 Discussion — ~1,300 words

Three things, in this order:

1. **What the evidence supports** — restated at exactly the strength measured, no
   further.
2. **Limitations, stated before a reviewer states them.** The honest list, updated
   after B03 and F07 reported:
   - **The shape manifold is one-dimensional.** One PCA mode explains 99.60% of
     variance, and a one-mode reconstruction is *inside the solver's noise floor*
     ([[results/P1-B03_basis-reachability]] §10). This is not "a low-dimensional
     basis may be nearly sufficient" — on pOnEllipse it is demonstrably
     sufficient, analytically and physically. **C1's premise does not hold on this
     dataset.**
   - **A constant 9.35% ± 0.91pp offset** between our solver and the dataset
     remains unexplained after the sourced factor of two
     ([[results/P1-F07_normalisation-audit]]). It is constant, so it cancels in
     every comparison the paper makes, but it must be stated with its CI.
   - Lift is not qualified ([[results/P1-F06_schafer-turek]] §7).
   - Nodes move radially only; the ring is fixed-length and fixed-topology; the
     solver is 2-D and moderate-Re.
3. **Where the approach is expected to pay, and why it is not visible here** —
   richer, sparsely-sampled families where a low-dim basis is genuinely limiting
   and retrieval genuinely fails. This is the honest motivation for Paper 3.2 and
   it converts limitation 2 into a research direction.

Do not hedge item 1 and do not apologise for item 2. A limitation stated crisply
reads as command of the work; the same limitation extracted by a reviewer reads as
a hole.

### §6 Conclusion — ~350 words

What was established, one sentence per contribution, plus the next step. No new
claims. No "future work" wish list beyond one or two concrete sentences.

### Appendices

A: full solver qualification table (F03) · B: architecture and matched-parameter
budgets (A group) · C: full ablation tables · D: partition indices and
reproducibility record.

---

## 3. Vocabulary — what Paper 1 may and may not say

Paper 1 earns the word **discretisation-native**. Nothing more. Keep this table
next to the draft.

| ❌ Do not write | ✅ Write instead | Why |
|---|---|---|
| "direct node movement" | "a per-node diffused shape field" | Nodes move radially on fixed rays — 1 DOF. Paper 2 earns this phrase |
| "mesh-native" | "boundary-discretisation-native" | A 96-node boundary ring, no interior nodes |
| "free-form geometry" | "generation without a hand-designed low-dimensional basis" | Radial-from-centroid *is* a parameterisation and cannot express non-star-convex shapes |
| "resolution-independent" (unqualified) | "conditioning that is independent of node count, and generation that transfers across 48–192 nodes *(D02)*" | Only true to the extent D02 measures |
| "outperforms" | "…by Δ, which is Nσ above the measurement floor" | Bare comparatives invite the error-bar question |
| "validated" | "graded by an independent solver, itself qualified against …" | "Validated" claims more than F03 establishes |
| "the model learns the physics" | "the model reproduces target Cp curves to ρ = …" | Unsupported and easy to attack |

If A02 shows the 1-D CNN matches the graph, **the representational claim survives
intact** — a CNN over `r(θ)` is also node-native. What dies is the *architectural*
claim. Rewrite §4.2 as an architecture ablation supporting C1 rather than as a
contribution of its own, and say plainly that on a fixed ordered ring the graph
confers no measurable advantage. That is a clean, publishable finding and it costs
the paper nothing. Plan for it now so the result cannot tempt you into spin.

---

## 4. Writing tactics

**Draft in this order:** figures → captions → results → method → discussion →
introduction → abstract → title. The introduction is the hardest section to write
and the easiest to write *wrong* before you know what the results say. Writing it
last takes about a third of the time.

**Make each caption self-contained.** A reviewer skims figures first. If the
caption states the setup, the n, and the finding, they arrive at your text already
agreeing.

**One claim per paragraph, evidence in the next sentence.** If a paragraph makes
two claims, split it. If it makes none, cut it.

**Write the results as verdicts on pre-registered hypotheses.** "We predicted X;
we observed Y" is stronger than "we observed Y", and you have the pre-registration
to back it — that is what [specs/](specs/) is for.

**Negative results get the same tone as positive ones.** No "unfortunately", no
"only", no "failed to". State the finding, state the mechanism, state what follows.
DPS being counterproductive and the graph showing no advantage are both *results*.

**Keep a running `refs.bib` from day one.** Retrofitting citations across 9,000
words costs a full day and is where errors get in.

**Number every claim in §1's contribution list**, then grep the draft for each
number before submitting. Anything unreferenced in §4 is a claim without evidence.

---

## 5. Rehearsals

| Venue | When | What it forces |
|---|---|---|
| **UK Fluids, Leeds** | 8–10 Sep 2026 (poster) | The whole story into one visual. Do this before drafting §1 |
| **APS DFD, Orlando** | Nov 2026 | Pitch C3/C4 — it is a fluids audience |
| Internal / group seminar | Oct 2026 | First full-draft dry run |

A poster is the cheapest available test of whether the argument holds together. If
you cannot lay the story out in one panel, §1 is not ready.

---

## 6. Drafting location

Draft at `DGN_Docs/paper1/draft/paper1.md`, one file per section if that suits your
workflow. Keep figures at `DGN_Docs/paper1/figures/` generated by committed
scripts, never by hand — a figure you cannot regenerate is a figure you cannot
correct at proof stage.
