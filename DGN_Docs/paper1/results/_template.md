---
id: P1-XNN
spec: "../specs/P1-XNN_slug.md"
title: ""
date_run:
date_reported:
status: complete | inconclusive | abandoned
verdict: supported | falsified | inconclusive
git_sha:
---

# P1-XNN — results

> **Spec:** [[../specs/P1-XNN_slug]] · **Verdict:** ⬜ supported / ⬜ falsified / ⬜ inconclusive

## 1. Verdict against the pre-registered decision rule

State this **first**, in one paragraph, before any numbers. Which branch of the
spec's decision rule was taken, and what the paper may now say as a result.

## 2. Headline numbers

| Quantity | Value | 95% CI | n | vs noise floor |
|---|---|---|---|---|
| | | | | |

Every number carries `n` and a CI. Differences are stated in units of the F04
noise floor (`< 2σ` = no detectable difference).

## 3. Hypothesis assessment

| Pre-registered prediction | Observed | Assessment |
|---|---|---|
| | | |

If the hypothesis was falsified, say so plainly here and follow the spec's stated
consequence. Do not reinterpret the hypothesis to fit the result.

## 4. Figures

| File | Shows | Paper § |
|---|---|---|
| | | |

Figures live in `DGN_Docs/paper1/figures/` and are produced by a committed script,
never by hand.

## 5. Unmatched factors and confounds

What could not be held constant, and the argument for why it does not explain the
result. For A/B-group experiments, state parameter counts and the matching
tolerance explicitly ([PROTOCOL §5](../PROTOCOL.md)).

## 6. Protocol deviations

Any departure from the spec, dated, with the reason. If there were none, say so.

_None._

## 7. Reproduction

| | |
|---|---|
| Config | `configs/paper1/P1-XNN_slug.yaml` |
| Outputs | `outputs/paper1/P1-XNN_slug/` |
| Solver runs | `cfd/ellipse2d/runs/P1-XNN_slug/` |
| Partition | `data/splits/paper1_partition.json` (sha256 …) |
| Seeds | |
| Command | |

## 8. What this changes

- **In the paper:** which section, which claim, which figure.
- **In the plan:** any downstream experiment whose design this affects.
- **In the diary:** log gate decisions and withdrawn claims in
  [[../../publication/PUBLICATION_DIARY]].
