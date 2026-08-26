---
title: "Paper 1 — experiment suite"
date: 2026-08-02
updated: 2026-08-26
tags: [paper1, experiments, preregistration, protocol]
status: active
---

# Paper 1 — experiment suite

This folder is the **complete experimental design for Paper 1** — 29 experiments,
written as a pre-registration: every experiment states its hypothesis, its
falsifier, and its pass/fail decision rule **before** it is run. Results are
appended, never back-fitted.

Everything here supersedes the exploratory `EXP-0xx` series in
[[../experiments/EXPERIMENT_PLAN]]. That work is not deleted — it is cited as the
pilot phase that motivated these designs — but **no Year-1 number is carried into
the paper without being regenerated under this protocol.**

## Where things stand — 26 August 2026

**9 of 29 reported.** The audit tier (F) is finished and the measurement protocol
is frozen. The model's central claims have not been tested yet.

| ID | Verdict | One line |
|---|:--:|---|
| [F01](results/P1-F01_data-partition.md) | ❌ falsified | No clean split; the Year-1 headline was measured on training samples |
| [F02](results/P1-F02_grading-metric.md) | ❌ falsified | Four graders disagreed; one canonical module now |
| [F03](results/P1-F03_solver-qualification.md) | ⚠️ marginal | ρ = 0.999 across the envelope — **read §13**, §§1–12 are superseded |
| [F04](results/P1-F04_noise-floor.md) | ✅ supported | **Protocol frozen.** Found the dead-probe defect |
| [F05](results/P1-F05_reference-model.md) | ✅ supported | `P1-REF`, 5 seeds — but seed CV is 49% |
| [F06](results/P1-F06_schafer-turek.md) | ⚠️ partial | Qualified for drag and pressure, **not lift** |
| [F07](results/P1-F07_normalisation-audit.md) | ⚠️ partial | Factor of two sourced; **9.35% ± 0.91pp residual** remains |
| [D01](results/P1-D01_conditioning-invariance.md) | ❌ falsified | Conditioning deviated 41.2%; fixed to 0.49% |
| [B03](results/P1-B03_basis-reachability.md) | ❌ falsified | **One PCA mode is inside the noise floor** — against C1's interest |

🔵 **B01 is running** (2026-08-26) — 35 solver runs against the frozen evaluation
target set, `data/splits/paper1_eval_targets.json`.

**On the metric question.** B01, A01–A03, D02 and G03 all have decision rules
written in ρ, which F04 demoted. They are *not* blocked: run them, apply the
registered ρ rule, report its verdict, and **additionally** report rank/top-k and
`Cd`. That adds information without editing a rule. See
[REPORT_OUTLINE §4](REPORT_OUTLINE.md) for which metric carries which claim.

**C1 has one route left.** B03 falsified its premise in-distribution *and*
its registered AoA10 fallback. **G03 is the only surviving argument** and has
been promoted to the gate tier.

Current status board: [EXPERIMENT_REGISTER.md](EXPERIMENT_REGISTER.md).

---

## Map

**Start here**

| Document | What it is |
|---|---|
| [PROTOCOL.md](PROTOCOL.md) | The rules every experiment obeys — splits, seeds, metrics, grading, significance. §6 carries the frozen σ. **Read this first.** |
| [EXPERIMENT_REGISTER.md](EXPERIMENT_REGISTER.md) | Master table of all 29, dependency graph, execution order, status |
| [specs/](specs/) | One file per experiment — the pre-registration. Read before touching an experiment |
| [results/](results/) | One file per experiment — written after running |

**Reference**

| Document | What it is |
|---|---|
| [METRICS.md](METRICS.md) | **Menu, not mandate.** Every metric: what claim it carries, what it cannot do, how a reviewer attacks it |
| [DATASET_PROVENANCE.md](DATASET_PROVENANCE.md) | Where pOnEllipse comes from and what its numbers mean. §§4–5 are F07's archaeology |
| [SOLVER_CASE.md](SOLVER_CASE.md) | The OpenLB case — geometry, materials, boundary treatment, build |
| [CONDITIONING_MATHS.md](CONDITIONING_MATHS.md) | The DCT conditioning descriptor, derived |
| [refs/](refs/) | Source papers — Lino 2022/2025, Schäfer–Turek 1996 |

**Writing**

| Document | What it is |
|---|---|
| [REPORT_OUTLINE.md](REPORT_OUTLINE.md) | Paper skeleton, which experiment feeds which figure, permitted vocabulary |
| [WRITING_ORDER.md](WRITING_ORDER.md) | Which sections are safe to write now and which a pending experiment can still invert |
| [meetings/](meetings/) | Supervision briefs |

**Housekeeping.** `results/superseded/` holds results files replaced by a later
attempt; they are kept, not deleted, because the record of what failed is part of
the audit trail. `results/_template.md` is the skeleton for a new results file.

Planning context: [[../publication/PUBLICATION_PLAN]] · [[../publication/RESEARCH_PROGRAMME]] · [[../publication/PUBLICATION_DIARY]]

---


## 1. Naming scheme

```
P1-<GROUP><NN>_<slug>
 │    │     │     └── short kebab-case description
 │    │     └──────── two-digit index within the group
 │    └────────────── group letter (below)
 └─────────────────── paper number
```

The group letter maps one-to-one onto a results subsection of the paper, so any
experiment ID tells you where its result lands.

| Code | Group | The question it answers | Paper § | Contribution |
|:--:|---|---|:--:|:--:|
| **F** | Foundation & protocol | What is the clean, defensible baseline everything is measured against? | §3, §4.1 | protocol / C3 |
| **A** | Architecture necessity | Does the *graph* earn its place? | §4.2 | C1 |
| **B** | Basis necessity | Does *node-native* earn its place against a low-dimensional basis? | §4.3 | C1 |
| **D** | Discretisation independence | Is "discretisation-native" actually true? | §4.4 | C2 |
| **C** | Conditioning | Does each conditioning component contribute? | §4.5 | C2 |
| **G** | Generalisation & failure | Where does it break, and why? | §4.6 | C4 |
| **S** | Scale & data efficiency | How does it depend on capacity and data? | §4.7 | risk / C4 |

Artefacts follow the ID everywhere:

```
configs/paper1/P1-F05_reference.yaml
outputs/paper1/P1-F05_reference/seed0/{checkpoint.pt,loss_log.json,run_meta.json}
cfd/ellipse2d/shapes/P1-G01_holdout/{manifest.csv,*.dat}
cfd/ellipse2d/runs/P1-G01_holdout/<case>/
DGN_Docs/paper1/specs/P1-F05_reference.md
DGN_Docs/paper1/results/P1-F05_reference.md
```

## 2. Contribution hierarchy this suite is built to defend

| # | Claim | Earned by |
|:--:|---|---|
| **C1** | Generative design at the level of the boundary discretisation, with no hand-designed low-dimensional parameterisation | **A** + **B** groups |
| **C2** | Conditioning that is independent of the discretisation | **D** + **C** groups |
| **C3** | Grading by an independent first-principles solver, not a self-trained surrogate | **F** group |
| **C4** | A failure taxonomy that only an independent grader can expose | **G** + **S** groups |

Claim discipline for the whole paper — from [[../publication/RESEARCH_PROGRAMME]]:
**Paper 1 earns the word *discretisation-native* and nothing more.** "Direct node
movement" is Paper 2. "Mesh-native" is Paper 3. See
[REPORT_OUTLINE.md §Vocabulary](REPORT_OUTLINE.md) for the permitted wording.

## 3. Execution order

Run in stages. Do not start a stage until the previous one has reported, because
later stages use earlier ones as their comparator.

| Stage | Experiments | Why here |
|:--:|---|---|
| ✅ **0 — Protocol** | F01 F02 **F06** F03 F04 **F07** | Nothing is measurable until the split, the metric, the instrument, the noise floor and **the reference scale** are frozen |
| ✅ **1 — Reference** | F05 | The single model every later result is a delta from |
| ◐ **2 — Story-breaking** | ~~B03~~ · **B01** · G03 A01 A02 A03 | These can invalidate the framing. Cheap. Run them before writing anything. G03 promoted here 08-26 — it is C1's last route |
| 🚦 | **← WE ARE HERE · AUGUST GATE** | Decide the paper's framing and venue. Log in [[../publication/PUBLICATION_DIARY]] |
| **3 — Headline** | G01 D01 D02 B02 | The results the paper is actually about |
| **4 — Ablations** | C01 C02 C03 A04 S01 S02 S03 | Contribution-targeted ablations (viva point 2) |
| **5 — Failure modes** | G02 G04 C04 B04 | The C4 payoff |
| **6 — Conditional** | D03 | Only if D02 fails |

Full dependency graph in [EXPERIMENT_REGISTER.md](EXPERIMENT_REGISTER.md).

## 4. Cost, measured not guessed

Calibrated from this repository on 2026-08-02:

| | Measured | Source |
|---|---|---|
| One training run (1000 epochs, 147k params) | **~85 min** | `outputs/EXP-022_T1000.log` — 161 batches/epoch at ~33 it/s |
| One solver run | **7 s – 14 min**, median ~1–3 min | 71 runs in `cfd/ellipse2d/runs/` with timing |
| Whole suite, training | ~64 runs ≈ **~91 h** serial, parallelises freely | — |
| Whole suite, solver | ~658 runs ≈ **~22 h** serial | — |

> **Correction to the earlier plan.** [[../publication/PUBLICATION_PLAN]] §10 called
> CFD grading "the critical path" and put it at ~100 runs. Both numbers were wrong.
> The suite needs ~6× more solver runs than that estimate, and even so the solver is
> roughly **four times cheaper than training** at these resolutions. The real
> critical path is **training runs × seeds**, and the real risk is disorganised
> bookkeeping across ~64 training runs, not queue time. The throughput fix is a
> training sweep driver, not a CFD batch driver.

## 5. Rules

1. **Write the spec before running.** Hypothesis, falsifier and decision rule are
   committed in `specs/` and never edited after the first run. Corrections go in
   the results file as a dated note.
2. **Every claim gets ≥3 seeds**, headline claims get 5. See PROTOCOL §4.
3. **Every difference is compared against the F04 noise floor.** A difference
   smaller than 2× the pooled measurement σ is reported as "no detectable
   difference", not as a win. See PROTOCOL §6.
4. **Baselines are matched on parameters and on degrees of freedom.** An unmatched
   baseline is not a baseline.
5. **A falsified hypothesis is a result.** Report it. Several of these experiments
   are predicted to falsify their own headline — that is deliberate, and it is what
   makes the paper defensible.
