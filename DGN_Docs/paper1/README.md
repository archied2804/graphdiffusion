---
title: "Paper 1 — experiment suite"
date: 2026-08-02
updated: 2026-08-02
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

| Document | What it is |
|---|---|
| [PROTOCOL.md](PROTOCOL.md) | The rules every experiment obeys — splits, seeds, metrics, grading pipeline, significance test. **Read this first.** |
| [METRICS.md](METRICS.md) | **Menu, not mandate.** Every metric assessed: what claim it carries, what it cannot do, what it costs, how a reviewer attacks it. Recommended minimum / full / stretch sets |
| [EXPERIMENT_REGISTER.md](EXPERIMENT_REGISTER.md) | Master table of all 29 experiments, dependency graph, execution order, status board |
| [REPORT_OUTLINE.md](REPORT_OUTLINE.md) | Paper skeleton, which experiment feeds which figure, and writing guidance |
| [specs/](specs/) | One file per experiment — the pre-registration |
| [results/](results/) | One file per experiment — filled in after running |

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
| **0 — Protocol** | F01 F02 **F06** F03 F04 **F07** | Nothing is measurable until the split, the metric, the instrument, the noise floor and **the reference scale** are frozen |
| **1 — Reference** | F05 | The single model every later result is a delta from |
| **2 — Story-breaking** | B01 B03 A01 A02 A03 | These can invalidate the framing. Cheap. Run them before writing anything |
| 🚦 | **AUGUST GATE** | Decide the paper's framing and venue. Log in [[../publication/PUBLICATION_DIARY]] |
| **3 — Headline** | G01 D01 D02 B02 | The results the paper is actually about |
| **4 — Ablations** | C01 C02 C03 A04 S01 S02 S03 | Contribution-targeted ablations (viva point 2) |
| **5 — Failure modes** | G02 G03 G04 C04 B04 | The C4 payoff |
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
