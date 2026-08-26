---
title: "Supervision brief — August 2026"
date: 2026-08-17
tags: [paper1, supervision, status, plan]
status: active
---

# Supervision brief — 17 August 2026

> Web version (shareable): <https://claude.ai/code/artifact/5d92827c-1fb0-48ad-bda4-8d39e7108bf5>
>
> **The one-line state of play.** The audit phase is finished and the measurement
> protocol froze today. It found six defects that invalidate parts of the Year-1
> report, and it has not yet tested the model's central claim once.

---

## 1. Where the suite stands

Eight of 29 reported; three more became runnable today.

| ID | Asked | Outcome | Consequence |
|---|---|---|---|
| [[../results/P1-F01_data-partition\|F01]] | Was there a clean split? | ❌ falsified | Year-1 headline was measured on **training samples** |
| [[../results/P1-F02_grading-metric\|F02]] | Do the four graders agree? | ❌ falsified | They differed; one canonical module now |
| [[../results/P1-F03_solver-qualification\|F03]] | Is the solver fit to grade? | ⚠️ marginal | ρ = 0.999 across the envelope; small real Re drift |
| [[../results/P1-F04_noise-floor\|F04]] | What is the measurement uncertainty? | ✅ supported | **Protocol frozen.** Found the probe defect |
| [[../results/P1-F05_reference-model\|F05]] | Does the recipe reproduce cleanly? | ✅ supported | `P1-REF`, 5 seeds — but seed CV is 49% |
| [[../results/P1-F06_schafer-turek\|F06]] | Does it hit published bounds? | ⚠️ partial | Qualified for drag and pressure, **not lift** |
| [[../results/P1-D01_conditioning-invariance\|D01]] | Node-count independent conditioning? | ❌ falsified | Deviated 41.2%; fixed to 0.49% |
| [[../results/P1-B03_basis-reachability\|B03]] | How many modes does the data need? | ❌ falsified | **One mode = 99.60%.** Against our own interest |
| B01 · D02 · G03 | retrieval, node transfer, OOD ladder | ⬜ ready | Unblocked as of today |

**The pattern is the point.** Of the eight reported, four were falsified outright
and two more failed part of their pre-registered rule — and not one of those
failures was about the model. They were about the split, the metric, the
conditioning descriptor, the solver and the dataset's normalisation.

---

## 2. What today's experiment found

### The frozen thresholds

| Metric | σ | 2σ | 3σ |
|---|--:|--:|--:|
| ρ | 0.00108 | 0.00215 | 0.00323 |
| `a` | 0.03521 | 0.07042 | 0.10563 |
| `Cd` | 0.04742 | 0.09483 | 0.14225 |
| `Cl` | 0.01542 | 0.03084 | 0.04625 |

### A solver defect that touched every graded number

The raw sweep failed its own falsifier, and failed in a way that made no physical
sense — a ±25% resolution change moved ρ by 0.07, non-monotonically. Chasing that
rather than accepting the verdict found the cause.

The solver samples surface pressure at a probe half a lattice spacing outside the
body, then snaps it to the *nearest* lattice node. Where the surface runs at a
shallow angle to the grid, the snap lands the probe **inside the body**. The
pressure functor returns exactly zero on a solid cell and still reports success,
so the file records a fabricated sample — sitting at `Cp ≈ 0` in the middle of a
suction peak, the largest error on the curve.

| Affected | Runs | Points | Worst run |
|---|--:|--:|--:|
| F04 | 23/27 | 3.9% | 6.7% |
| F03 | 62/82 | 3.1% | 12% |

**Every graded surface-pressure number produced before today is affected.** Both
fixes are in — a filter in the grading module, an outward probe walk in the
solver.

**The good news:** a case re-run on the fixed solver agrees with the same case
filtered on the metric side to within a fifth of σ on every metric. The ~85
existing runs are recoverable by regrading, not re-running.

### What regrading changed

| F03 criterion | Before | After |
|---|--:|--:|
| mean ρ | 0.9504 | **0.9990** |
| drift in `a` with Re | 11.28% | **6.54%** |
| Δρ on doubling resolution | ±0.034 | **±0.0004** |

F03 moves falsified → marginal, and its headline finding — `a` moving 21.5% under
refinement, which I reported as evidence of numerical non-convergence — **is
withdrawn**. That was the probe defect; the true figure is 2–4%.

### ρ cannot be the headline metric

Two **unrelated** dataset shapes correlate at **ρ = 0.9954**. A predictor that
ignores its input and returns the corpus mean scores **0.9969**.

| Grading each run against every target | median ρ |
|---|--:|
| against its own target | 0.99896 |
| against another shape's target | 0.99669 |
| **discrimination signal** | **+0.00226 (2.1σ)** |

Marginal by this suite's own rule. ρ answers "is this a plausible ellipse Cp
curve", not "is this the correct one". Replacement: **rank / top-k** (top-1
33.3% against 6.7% chance) and **Cd** (40% dynamic range against a 5% floor).

### One result that helps

Node phase — rolling the node list so the body is bit-identical but every node
moves — contributes **σ_ρ = 2×10⁻⁵**, two orders below every other factor. The
instrument is blind to where the nodes sit, which is what makes the
discretisation claim measurable at all.

---

## 3. Five decisions I need

### Q1 — Accept the metric change from ρ to rank/top-k plus Cd?

ρ underpins the Year-1 report, the outline and every draft figure, and it cannot
carry a discrimination claim on this corpus. Risk in changing: rank metrics are
unfamiliar to a fluids audience. Risk in not changing: a reviewer computes the
chance level themselves.

**Recommendation:** change it, and make the chance-level analysis a *figure*
rather than a footnote. Far better coming from us than from a reviewer.

### Q2 — B03 says the manifold is 1-D. Does C1 survive?

One PCA mode explains 99.60% of shape variance; two explain 99.998%. C1 argues
node-native generation reaches designs a low-dimensional basis cannot. **On this
dataset that premise is unsupported.** Routes:

1. Report it and reframe C1 around discretisation-independence (C2), which the
   data does support; state the limitation in §5.
2. Enrich the dataset beyond the ellipse family — real scope growth, Paper 2.
3. Argue C1 on *reach* via the OOD ladder rather than in-distribution accuracy —
   what the B03 spec's own fail branch directs.

**Recommendation:** 1 combined with 3. But this is a framing decision about the
paper's central contribution, so it is yours.

### Q3 — Is the full suite realistic by December on one laptop GPU?

Twenty remain: ~64 training runs (~91 h) and ~658 solver runs (~22 h), serial, on
the machine I also work on. Training is the bottleneck. Four months, and writing
has not started. The suite is tiered so cutting is pre-planned; the support tier
is S01, S03, A04.

**What I need:** either a decision to cut to gate + core now, while it is a plan
rather than a retreat, or CSF/Condor access for the training sweeps. The SLURM
script already exists.

### Q4 — How hard do we chase the remaining 9.3% magnitude gap?

After the factor-of-two correction (dataset stores `p/(ρU²)`, solver reports
`p/(½ρU²)` — the origin of the long-standing `a ≈ 1.74`), solver and dataset
agree to `0.907 ± 0.033`. The residual 9.3% is real and unexplained. F07 is
scoped to close it (~6 runs).

**Recommendation:** run F07 as scoped, then stop. If unresolved, report it as a
stated systematic with a bound — every claim is a comparison between conditions
sharing the offset, so it cancels. It is a presentation problem, not a validity
one.

### Q5 — Does the instrument work belong in the paper as a contribution?

The audit produced things useful independently of whether the model works: the
chance level of ρ on an aerofoil-pressure corpus, a quantified noise floor with
per-factor decomposition, a discretisation-invariant conditioning descriptor, and
a solver defect any LBM surface-sampling code could share. Currently supporting
apparatus in §3/§4.1; could be a stated contribution.

**My view:** worth raising. If the model results come back weak, this is the part
of the work that is unambiguously solid. But it changes the paper's identity and
possibly its venue — better asked now than realised in November.

---

## 4. Route to December

| Window | Work | Cost | Decides |
|---|---|--:|---|
| Now → early Sep | B01, A01, A02, A03, B03 solver half | 12 train, ~79 solver | **The August gate** — does it beat a lookup table, and does message passing contribute? |
| Sep | F07, G01, D02 | ~166 solver | The headline number, honestly measured for the first time |
| Oct | C01–C04, S02, B02 | ~23 train | Whether the design choices are justified or inherited |
| Nov | G02, G03, G04, B04 | ~133 solver | §4.6 and the limitations section |

Nothing after the gate is worth running until the framing survives it.

### Safe to write now

Per [[../WRITING_ORDER]] — a section is safe when no pending experiment can reverse
its claims, only add to them.

- 🟢 **§3 Method** — the largest block of finished work, and now the most defensible.
- 🟢 **§4.1 Instrument qualification** — F02, F03, F04, F06. Complete as of today.
- 🟢 **§5 Limitations** — deliberately early, while the falsifications are fresh.
- 🔴 **§4.2, §4.3** — do not start. A pending experiment can invert either.

### Live risks

1. **The gate may not survive.** If A01 shows message passing adds nothing within
   2σ, the honest report is "no detectable benefit" — publishable, but it changes
   the paper.
2. **Seed variance is high.** `P1-REF` best-val has CV 49% across five seeds, a
   3.2× spread. Model comparisons need enough seeds to see past that.
3. **Lift is not qualified.** F06 cleared drag and pressure but not lift, and that
   survives fp64 — a property of the case, not round-off.
4. **Compute.** See Q3.

---

## 5. Vocabulary — worth a minute in the meeting

Paper 1 earns **"discretisation-native"** and nothing more. Not "direct node
movement" (nodes move radially, 1 DOF, fixed ring — Paper 2). Not "mesh-native"
(a 96-node boundary ring, no interior nodes — Paper 3). Two of the ten
anticipated reviewer questions are about exactly this wording, so it is worth
agreeing before the prose accumulates. Table in [[../REPORT_OUTLINE]].
