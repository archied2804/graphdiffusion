---
title: "Supervision brief — August 2026"
date: 2026-08-17
updated: 2026-08-26
tags: [paper1, supervision, status, plan]
status: active
---

# Supervision brief — August 2026

> Web version (shareable): <https://claude.ai/code/artifact/5d92827c-1fb0-48ad-bda4-8d39e7108bf5>
>
> **The one-line state of play.** The audit phase is finished and the measurement
> protocol is frozen. It found six defects that invalidate parts of the Year-1
> report — and it has now also falsified the premise of the paper's central
> contribution.

*Updated 26 August: B03's solver half, F06's regrade and F07 have reported since
the 17 August version. §3 is substantially rewritten; the questions in §5 have
changed.*

---

## 1. Where the suite stands

**9 of 29 reported.** Stage 0 (protocol) and Stage 1 (reference) are complete.

| ID | Asked | Outcome | Consequence |
|---|---|---|---|
| [[../results/P1-F01_data-partition\|F01]] | Was there a clean split? | ❌ falsified | Year-1 headline was measured on **training samples** |
| [[../results/P1-F02_grading-metric\|F02]] | Do the four graders agree? | ❌ falsified | They differed; one canonical module now |
| [[../results/P1-F03_solver-qualification\|F03]] | Is the solver fit to grade? | ⚠️ marginal | ρ = 0.999 across the envelope; small real Re drift |
| [[../results/P1-F04_noise-floor\|F04]] | What is the measurement uncertainty? | ✅ supported | **Protocol frozen.** Found the probe defect |
| [[../results/P1-F05_reference-model\|F05]] | Does the recipe reproduce cleanly? | ✅ supported | `P1-REF`, 5 seeds — but seed CV is 49% |
| [[../results/P1-F06_schafer-turek\|F06]] | Does it hit published bounds? | ⚠️ partial | Qualified for drag and pressure, **not lift** |
| [[../results/P1-F07_normalisation-audit\|F07]] | Where does `a ≈ 1.74` come from? | ⚠️ partial | Factor of two **sourced**; 9.35% residual remains |
| [[../results/P1-D01_conditioning-invariance\|D01]] | Node-count independent conditioning? | ❌ falsified | Deviated 41.2%; **fixed to 0.49%** |
| [[../results/P1-B03_basis-reachability\|B03]] | How many modes does the data need? | ❌ falsified | **One mode, and it is inside the noise floor** |
| B01 | Does it beat a lookup table? | 🔵 running | 35 solver runs, launched 26 Aug |

Of the nine, four were falsified outright and three more failed part of their
pre-registered rule. **Eight of those nine were about instruments, not the
model** — the split, the metric, the conditioning descriptor, the solver, the
reference scale. The ninth is the problem.

---

## 2. What has been fixed since 17 August

**F06 regraded.** The dead-probe defect hit the Schäfer–Turek cylinder far harder
than the ellipse — **29–42%** of surface points against 3–4% — and got *worse*
with refinement, because the probe offset scales with `Δx`. No F06 number
changed: Δp is read at the fore and aft stagnation points, whose normals are
grid-aligned, so those probes were always live. A verification run on the fixed
solver recovers **200/200** probes and reproduces Δp and `Cd` to the digit.

**F07 closed the `a ≈ 1.74` action item**, open since EXP-040.

| | |
|---|--:|
| Observed factor, raw | **1.81298** [1.7773, 1.8486] |
| After the sourced factor of two, `a` | 0.90649 |
| **Residual** | **9.35% ± 0.91pp** (10.3 SEM) |

The residual is real but it is **constant** — χ²/dof = 0.885 for a constant
model, so the entire envelope spread sits inside the measurement floor. Because
it is constant it cancels in every comparison the paper makes. It is a
presentation problem, not a validity one. It is provably not a solver pressure
error (F06), not a de-dimensionalisation error (a 4× range in `U` and 2× in `ρ`
move `a` by ≤0.28σ), and not introduced downstream.

**A note on how F07 was reported.** The two candidate mechanisms that land
*inside* the confidence interval (1.80, 1.78) are the two struck before the run
for resting on a scaling stated for a **different dataset** — and the one
candidate derived from first principles predicts 2.00 and sits outside it. The
pre-registered candidate table was written to stop exactly this, so the
coincidence is recorded and **not acted on**.

---

## 3. The problem: C1 has no supported argument

This is the item for the meeting.

**C1** claims generative design at the level of the boundary discretisation,
reaching designs a hand-designed low-dimensional parameterisation cannot.

**B03 falsified its premise, twice over.**

1. **Analytically** — one PCA mode explains **99.60%** of shape variance in
   pOnEllipse; two explain 99.998%.
2. **Physically**, as of 26 August — four held-out shapes on a 2×2 (thickness,
   Reynolds) factorial, reconstructed and pushed through the solver. **Every
   truncation level is invisible to the instrument, `M = 1` included**: worst
   shape 1.10σ on ρ and 1.26σ on `Cd`, against a 2σ rule.

A one-mode reconstruction is **inside the solver's measurement uncertainty**. On
this dataset a low-dimensional basis is not "nearly sufficient" — it is
sufficient.

**And the registered fallback is gone.** B03's spec anticipated this outcome and
named the escape route: *"argue C1 on reach (AoA10, G03) rather than on
in-distribution accuracy."* B03 §4 then found the AoA10 contrast is a
**rotation, not a reach limitation** — `r(θ)` is orientation-locked, so a 10°
rotation moves a shape 759× outside a train-fitted basis for reasons that have
nothing to do with reach.

> **C1 has exactly one untested route left: G03, the DCT-space OOD ladder.**
> It has been promoted to the gate tier. 0 training runs, ~45 solver runs, and
> the ladder is already regenerated.

A second casualty: **B02 is near-pointless as scoped.** It was designed as a
10-mode PCA straw man, and `M = 10` is indistinguishable from the original body.

---

## 4. Does this ruin the results, and should we still publish?

**It does not ruin them, and it does not prohibit publication.** It changes what
the paper is, and that decision is better made now than in November.

**Fixed instrument defects help you.** F02's disagreeing graders, the dead
probes, F07's factor of two. Every one would have been fatal if a reviewer had
found it first. Found internally and fixed, they are evidence the evaluation is
honest.

**Bounded limitations are statable.** F06's unqualified lift (every pOnEllipse
configuration is AoA = 0, where `Cl ≈ 0`). F07's constant residual. F03's Re
drift, which is confounded with resolution and the same size as the resolution
factor F04 measured directly.

**What is actually supported today**

| | Status |
|---|---|
| **C2** — conditioning independent of the discretisation | ✅ D01 fixed to 0.49%; F04 showed the instrument is blind to node phase (σ_ρ = 2×10⁻⁵), which is what makes the claim measurable at all |
| **C3** — grading by an independent first-principles solver | ✅ Qualified absolutely against Schäfer–Turek |
| **C1** — node-native beats a low-dimensional basis | ❌ Premise falsified on this dataset; one route left |
| **C4** — failure taxonomy | ⬜ Not yet measured |

**An unplanned contribution.** The audit produced results useful regardless of
whether the model works: the chance level of ρ on an aerofoil-pressure corpus, a
noise floor with per-factor decomposition, a discretisation-invariant
conditioning descriptor, and a surface-probe defect any LBM code could share.

**My read.** You currently have a strong **evaluation-methodology** paper and a
weak **generative-modelling** paper. Both are publishable — pre-registered
falsifications are a legitimate contribution and increasingly valued — but they
are different papers at different venues. The limitation is the **dataset**, not
the method: a two-parameter ellipse family cannot demonstrate a
parameterisation-bottleneck claim, and that is a scoping decision made before
Paper 1 started.

---

## 5. Five decisions I need

### Q1 — Given B03, does C1 survive at all?

The paper's identity depends on this and everything else follows from it. Three
routes:

1. **Report and reframe.** Lead on C2 and C3, which the data supports; state
   C1's premise as a measured limitation in §5. Safe, honest, less ambitious.
2. **Wait for G03.** ~45 solver runs, no training. If performance degrades
   sharply along the DCT ladder where retrieval cannot follow, C1 has an
   extrapolation argument.
3. **Enrich the dataset** beyond the ellipse family. Real scope growth — Paper 2.

**Recommendation:** run G03 first (it is cheap and it is the last route), then
decide between 1 and 2 with the number in hand.

### Q2 — Methodology paper or modelling paper?

This follows from Q1 and it determines the venue. [[../REPORT_OUTLINE]] currently
targets *Machine Learning: Science and Technology*. If the answer to Q1 is route
1, the instrument work becomes the lead contribution and the venue question
reopens.

### Q3 — Seeds versus compute

`P1-REF` has **49% seed CV**, a 3.2× spread across five seeds. With three seeds,
A01–A03 will very likely return "no detectable difference" whether or not a real
effect exists — **and that is not evidence of no effect.** Either budget more
seeds, multiplying an already-tight training cost, or state the minimum
detectable effect size up front and accept it.

Twenty experiments remain: ~64 training runs (~91 h) and ~658 solver runs
(~22 h), serial, on the laptop I also work on. The GPU display watchdog killed
runs in both of the last two sweeps.

**What I need:** CSF/Condor access, or a decision to cut to gate + core now while
it is a plan rather than a retreat. The SLURM script already exists.

### Q4 — The metric change, now a reporting decision rather than a blocker

ρ cannot carry a discrimination claim here: unrelated shapes correlate at
**0.9954**, and a corpus-mean predictor scores **0.9969**. Independently
confirmed on B03's 24 runs, where ρ spans 2.0σ in total and `Cd` spans 7.8σ.

Five unrun specs state their rule in ρ. They are **not blocked**: run them, apply
the registered rule, report its verdict, and report rank/top-k and `Cd`
alongside. That adds information without editing a spec.

**Recommendation:** approve that, and make the chance-level analysis a *figure*
rather than a footnote. Far better from us than from a reviewer.

### Q5 — Does the instrument work belong in the paper as a contribution?

Currently supporting apparatus in §3 and §4.1. If Q1 lands on route 1, this is
the part of the work that is unambiguously solid, and it should be promoted.

---

## 6. Plan to December

**Before the gate** — ~116 solver runs and 12 training runs, roughly two weeks
part-time.

| Order | Cost | Why here |
|---|---|---|
| 🔵 **B01** retrieval *(running)* | 0 train · 35 solver | Zero training. B03 says the manifold is 1-D with 5,701 training samples, so retrieval may match the model outright — **know that before spending 91 h training** |
| **G03** OOD ladder | 0 train · ~45 solver | C1's last route. Promoted from Stage 5 |
| **A01** no-edge | 3 train · ~12 solver | Cheapest architecture control |
| **A02 · A03** | 6 train · ~24 solver | Completes the trio |

Then the gate: framing and venue, per Q1 and Q2.

| Window | Work | Decides |
|---|---|---|
| Sep | G01, D02 | The headline number, honestly measured for the first time |
| Oct | C01–C04, S02, B02 | Whether the design choices are justified or inherited |
| Nov | G02, G04, B04 | §4.6 and limitations |

### Safe to write now

Per [[../WRITING_ORDER]] — a section is safe when no pending experiment can
reverse its claims.

- 🟢 **§3 Method** — the largest block of finished work.
- 🟢 **§4.1 Instrument qualification** — F02, F03, F04, F06, F07. Complete.
- 🟢 **§5 Limitations** — deliberately early, while the falsifications are fresh.
- 🔴 **§4.2, §4.3** — do not start. B03 makes §4.3 the section most at risk of
  being written to a predetermined answer.

⚠️ **[[../WRITING_ORDER]] §4.1 was rewritten on 26 August.** Its earlier drafting
spine quoted four numbers that the dead-probe fix has since withdrawn. If any
prose was drafted from it before that date, check every figure against the
withdrawn list.

---

## 7. Vocabulary — worth a minute in the meeting

Paper 1 earns **"discretisation-native"** and nothing more. Not "direct node
movement" (nodes move radially, 1 DOF, fixed ring — Paper 2). Not "mesh-native"
(a 96-node boundary ring, no interior nodes — Paper 3). Two of the ten
anticipated reviewer questions are about exactly this wording. Table in
[[../REPORT_OUTLINE]].
