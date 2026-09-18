---
id: P1-D02
spec: "../specs/P1-D02_node-count-transfer.md"
title: "Node-count transfer at sampling time"
date_run: 2026-09-14
date_reported: 2026-09-17
status: complete
verdict: count-invariant (positive); spacing-sensitive (negative); registered rho rule fires on noise
git_sha: f82c02b
---

# P1-D02 — results

> **Spec:** [[../specs/P1-D02_node-count-transfer]] · **Verdict:** split, and the
> split is the finding.
>
> **Node count: invariant.** 100% valid geometry at every rung from 24 to 384
> nodes, a 16× range. The shape the model draws holds its aspect ratio to
> within 1.9% of the in-range mean across that whole range — about 1% at most
> rungs, 1.9% at the coarsest (24 nodes). Inside the trained range, node count has no
> detectable effect on drag.
>
> **Node spacing: not invariant.** The model reads shape from the *spacing* of
> the sampling template's nodes, and a template's spacing encodes an aspect
> ratio. This is new, the spec did not anticipate it, and it partly
> **re-attributes G01's headline error.**

## 1. Verdict against the pre-registered rule

| Branch | Fires? |
|---|:--:|
| Validity ≥ 90% **and** ρ within 2σ across N ∈ [48, 192] → *"discretisation-native is earned"* | ❌ on ρ |
| Valid but degrading fidelity → report the working range | partly |
| Invalid shapes → claim dropped, run D03 | ❌ |

**Validity: 100% at every rung.** The extrapolation falsifier — *"invalid shapes
anywhere in 40–192"* — does not fire, across 24–384.

**The ρ criterion fails on the letter of the rule.** In-range, per-rung mean ρ
spans **0.01293 = 11.97σ_ρ**. The rule is reported as written.

**But the span is indistinguishable from noise**, which was declared before the
data existed. The grader's module docstring, written before any run, states that
seed-to-seed sd on ρ is 5.0σ_ρ and the 2σ rule cannot be resolved against it. On
this data, with each target's own mean removed so that between-target spread
cannot pose as a node-count effect:

| | span across 52–96 | typical SEM | span / (2·SEM) |
|---|--:|--:|--:|
| ρ | 11.97σ | 5.91σ | **1.01** |
| **`Cd` error** | **0.51σ** | 0.46σ | **0.56** |

Neither clears the noise. **Inside the trained range, node count has no
detectable effect** — on `Cd` comfortably, on ρ only just. Per
[[../PROTOCOL]] §2.1, `Cd` carries the verdict where ρ cannot resolve the rule.

## 2. Node count — the positive result

### Validity

| N | 24 | 32 | 40 | 48 | **52** | **64** | **80** | **96** | 128 | 192 | 256 | 384 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| valid | 100% | 100% | 100% | 100% | **100%** | **100%** | **100%** | **100%** | 100% | 100% | 100% | 100% |

144 of 144 usable designs, from 0.46× to 4.0× the trained range.

### The model draws the same shape at every node count

Aspect ratio `b/a` of the generated bodies, mean over seeds:

| N | t008 | t020 | t039 | t046 |
|--:|--:|--:|--:|--:|
| 24 | 0.7964 | 0.6657 | 0.8163 | 0.6704 |
| 52 | 0.8031 | 0.6761 | 0.8267 | 0.6799 |
| 96 | 0.8065 | 0.6791 | 0.8293 | 0.6777 |
| 192 | 0.8117 | 0.6833 | 0.8288 | 0.6847 |
| 384 | 0.8111 | 0.6810 | 0.8331 | 0.6823 |

**Within 1.9% of the in-range mean across a 16× range of node counts** — about
1% at most rungs, with the worst case at 24 nodes. That slight rise at low N is
what a coarse polygon does when its vertices miss the true extremes.

Figure: `outputs/paper1/figures/fig4_node_count_invariance.{png,pdf}`.

This is a capability **no baseline in this suite has.** Nearest-neighbour
retrieval can only return a stored 52–96-node body. A PCA basis fitted at one
node count cannot be evaluated at another. Asked for a 24- or 384-node design,
both have nothing to return.

## 3. Node spacing — the negative result, and it reaches G01

The four targets split into two pairs, and D02 **reverses G01's ranking** of
them. A uniform offset would move all four together; this swaps them.

### Why

The sampling template supplies `pos` and `p_cond = [cos θ, sin θ]` from its own
node angles. **A template's angular spacing encodes an aspect ratio**: an
ellipse's nodes crowd toward its tips in polar angle, by an amount set by how
elongated it is. Uniform spacing is what a *circle* has.

| | template aspect | effect on generated shapes |
|---|--:|---|
| **G01** — `dataset[0]`'s real node angles | **0.511** (thin) | pulled **thin** |
| **D02** — uniform θ | **1.0** (circle) | pulled **thick** |

On the shared targets:

| target | true aspect | G01 generated | error | D02 generated | error |
|---|--:|--:|--:|--:|--:|
| t008 | 0.7700 | 0.6667 | **−0.103** | 0.8065 | +0.037 |
| t020 | 0.5120 | 0.5114 | −0.001 | 0.6783 | **+0.166** |
| t039 | 0.7840 | 0.6858 | **−0.098** | 0.8284 | +0.044 |
| t046 | 0.5510 | 0.5157 | −0.035 | 0.6804 | **+0.129** |

Each template gets right the targets that resemble it and gets wrong the ones
that don't. Figure: `outputs/paper1/figures/fig6_template_bias.{png,pdf}`. **The model is taking shape information from the template's node
positions as well as from the conditioning vector.**

### Across all twenty G01 targets

| | |
|---|--:|
| true aspect range | 0.503 – 0.784 |
| G01 generated range | 0.511 – **0.686** |
| regression slope, generated on true | **0.620** |
| correlation | +0.933 |

G01 tracks the target (correlation 0.93, consistent with
[[P1-G01_holdout-inverse-design]] §4a) but is **compressed toward its thin
template** — slope 0.62, and it cannot produce anything thicker than 0.686
against a truth that reaches 0.784.

### What this changes in G01

- [[P1-G01_holdout-inverse-design]] §2 noted the model "skews thin, median `b/a`
  0.538". **That is the template, not a model preference.**
- G01's drag error, its worst targets (`t039`, `t008`, both thick) and part of
  its loss to retrieval are **template-induced**. The measurement stands as
  measured; its attribution changes.
- Retrieval is unaffected — it returns stored geometry and has no template.

## 4. Discretisation-native is half-earned

| | invariant? | evidence |
|---|:--:|---|
| **Node count** | ✅ | aspect within 1.9% across 24–384; no in-range `Cd` effect (0.51σ) |
| **Node spacing** | ❌ | template spacing shifts aspect by up to **0.17** |

[[P1-D01_conditioning-invariance|D01]] made the **conditioning descriptor**
spacing-invariant (41.2% → 0.49%). D02 shows the **model's positional input** is
not. The two findings are the same defect at two different points in the
pipeline, and only the first has been fixed.

**The paper may claim node-count invariance.** It may not claim
discretisation-invariance without qualification.

## 5. Mechanism — the high-frequency content is fixed in node index

The spec calls the spectral measurement "what separates a finding from an
anecdote".

**The implemented test was uninformative, and is reported as such.** It took the
argmax of the `r(θ)` spectrum. Peak index frequency × N/2 = **2.0 at every
rung**: it located the ellipse's own two-lobed shape, which is trivially mode 2,
not the boundary noise the spec asks about.

**Roughness answers the question instead.** Node-to-node radial variation:

| N | 96 | 128 | 192 | 256 | 384 |
|--:|--:|--:|--:|--:|--:|
| roughness | 0.00991 | 0.00916 | 0.00952 | 0.00919 | 0.00973 |

**Flat, from 96 to 384 nodes.** A smooth boundary sampled more densely would
show this falling as 1/N² — to one sixteenth of its N=96 value by N=384. It does
not move. The residual is per-node noise of roughly **1% of radius, fixed in
node index**: the model adds a constant jitter to every node however many there
are. That is the index-space prior the spec hypothesised, established from
roughness rather than from the peak.

Below 96 roughness rises (0.0213 at N=24), which is the ellipse's own curvature
dominating the second difference on a coarse polygon, not extra noise.

**The spec's roughness prediction is not met.** It predicted roughness at N=192
at least 2× the in-range value. Observed: **0.92×**. There is no blow-up at high
node count.

## 6. Protocol deviations

Dated, per [[../PROTOCOL]] §9.

- **2026-09-14** — **Ladder widened** from the registered seven rungs
  {40, 52, 64, 80, 96, 128, 192} to twelve {24, 32, 40, 48, 52, 64, 80, 96, 128,
  192, 256, 384}. More rungs, same rule.
- **2026-09-14** — Seeds: **3, as registered.** Not a deviation; recorded
  because five was considered and rejected before the run — it would not have
  resolved the ρ rule (SEM 2.24σ against 2.90σ) and was not needed for `Cd`.
- **2026-09-14** — **Uniform θ at every rung.** The spec assumed templates are
  uniformly spaced; the corpus is not. Uniform was chosen so that N is the only
  variable. **This choice is what exposed §3**, and it means the absolute `Cd`
  and ρ levels here are biased toward thick shapes and are not comparable to
  G01's. Comparisons *across N* are unaffected, since every rung shares it.
- **2026-09-14** — Targets are four of G01's twenty, on a 2×2 (thickness,
  Reynolds) factorial, drawn before any Cp existed.
- **2026-09-17** — The node-count analysis in §1 is **target-centred**, removing
  each target's own mean before testing across N. The spec does not specify
  this; without it, between-target spread is counted as a node-count effect.
- **2026-09-17** — The spectral test as implemented was uninformative (§5).
  Roughness is used instead. The grader is left unchanged so the uninformative
  output stays on the record.
- The grader's `f_peak` interpretation line, "constant ⇒ an index-space prior",
  was wrong as applied: it measured the ellipse, not the noise.

## 7. What this unblocks and what it changes

**C2 survives in a narrower form.** Node-count invariance is established, with a
capability the baselines lack. Spacing-invariance is not.

**G01 needs re-attribution, not re-running.** Part of its error is a sampling
template artefact. A **matched-template test** would measure how much: re-sample
G01's targets on a template whose spacing matches the corpus median aspect
(~0.64) rather than `dataset[0]`'s 0.511, and check whether the aspect regression
slope moves from 0.62 toward 1. No retraining; roughly the cost of G01.

**A specific architectural fix is now identified.** The positional channel
should not carry shape. Candidates: node-index or arc-length-fraction position in
place of `[cos θ, sin θ]` from the template, or removing `pos` from `p_cond`.
Each needs retraining, which puts it in scope for Paper 2 or a Paper 1
improvement, not a Paper 1 result.

**G02 is now interpretable** on node count, as the spec intended — the AoA10 set
has 80 nodes, inside the invariant range. It is **not** yet interpretable on
spacing.

## 8. Files

| | |
|---|---|
| Sampler (with verified template builder) | `scripts/paper1/p1_d02_sample.py` |
| Runner | `scripts/paper1/run_p1_d02_solver.sh` |
| Grader | `scripts/paper1/p1_d02_grade.py` |
| Bodies | `cfd/ellipse2d/shapes/d02_nodes/` |
| Data | `outputs/paper1/P1-D02_node-count-transfer/grading.json` |
