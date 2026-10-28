---
id: P1-A02
group: A — Architecture necessity
title: "1-D circular CNN over the ring"
tier: gate
status: planned
depends_on: [P1-F05]
blocks: [AUGUST-GATE]
paper_section: "4.2"
contribution: C1
cost: {train_runs: 3, solver_runs: 12, effort: "~3 days"}
---

# P1-A02 — 1-D circular CNN over the ring

## Question

On a fixed, ordered, 96-node ring carrying one scalar per node, is graph message
passing anything more than a 1-D convolution?

## Hypothesis *(pre-registered)*

**The CNN matches or beats the graph.** With a fixed-length ordered ring, `r(θ)` is
a 1-D periodic signal and circular convolution is the exactly correct inductive
bias — the graph network adds machinery without adding structure it can exploit.
Predicted: difference within `2σ`, with the CNN cheaper per step.

This is the one experiment in the suite whose pre-registered hypothesis
**contradicts the paper's original framing.** That is the point of registering it.

## Falsifier

The graph beats the CNN by `> 3σ` on held-out ρ at matched parameters. That would
be a genuine and interesting result — it would mean the GN's edge features
(angular / arc-length geometry) carry information a uniform convolution kernel
cannot, and §4.2 becomes a contribution rather than an ablation.

## Method

1. Replace `ScoreNetwork` with a circular-padded 1-D CNN over `r(θ)` indexed by
   node order. Keep **everything else identical**: same `GraphDiffusionModel`
   wrapper, same noise schedule, same DCT conditioning vector, same DeepSets
   pressure head, same loss.
2. Inject conditioning and the time embedding by **FiLM** on each conv block —
   the closest analogue to the GN's `GlobalModel → NodeModel` pathway. Document
   this as the one unmatched design choice; it is not neutral and must be argued.
3. Depth chosen so the **receptive field matches** the graph's: `n_layers = 4`
   with `k_neighbors = 6` gives a ~24-node receptive field, so size the kernel and
   depth to match, and report both receptive fields explicitly.
4. Parameters within ±5% of 147,585. Seeds 0, 1, 2. Same 4 held-out targets.

## Measurements

| Quantity | Purpose |
|---|---|
| Parameter count and **receptive field** | Both must be matched, and both stated |
| Best val loss, held-out ρ and `a`, mean ± CI | The comparison |
| Wall-clock per epoch and sampling time | The efficiency half of the argument |
| Boundary roughness, validity | Whether the two fail differently |

## Decision rule

| Outcome | Verdict and action |
|---|---|
| CNN ≥ graph within `2σ` | **The architectural claim is dropped.** §4.2 becomes an ablation supporting C1, not a contribution. Rewrite §1's contribution list accordingly |
| Graph > CNN by `> 3σ` | The architectural claim is earned; identify *why* (edge features are the candidate) and test it |
| CNN > graph by `> 3σ` | Report it plainly. The representational contribution still stands — a CNN over `r(θ)` is equally node-native |

## Consequence

**None of these outcomes kills the paper.** This is worth internalising before the
result arrives, because the temptation to spin will be strongest here:

- C1 is a claim about the **representation** — generating a per-node field with no
  hand-designed low-dimensional basis. A 1-D CNN over `r(θ)` satisfies that too.
  So if the CNN wins, C1 survives and only the *architecture* story dies.
- What changes is the venue emphasis and §4.2's role. See
  [REPORT_OUTLINE.md §Vocabulary](../REPORT_OUTLINE.md) for the rewrite.
- The failure mode to avoid is discovering this from Reviewer 2 in February rather
  than from three training runs in August.

## Notes

This experiment was planned in Year 1 as part of EXP-013 ("Method E — is the GN
architecture necessary?") and silently became smoothness regularisation instead.
The viva panel's request for *"additional baselines where the methodology is more
complex than immediately needed for the 2D case"* is pointing directly at it.

## Results

_Not yet run._
