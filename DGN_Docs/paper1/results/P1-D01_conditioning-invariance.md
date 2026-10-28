---
id: P1-D01
spec: "../specs/P1-D01_conditioning-invariance.md"
title: "Conditioning invariance under resampling"
date_run: 2026-08-04
date_reported: 2026-08-04
status: complete
verdict: falsified
git_sha: d47682e
---

# P1-D01 — results

> **Spec:** [[../specs/P1-D01_conditioning-invariance]] · **Verdict:** ❌
> **falsified — the conditioning is not node-count independent, and the cause is
> a normalisation-convention error in `dct_ii`, not interpolation.**

## 1. Verdict against the pre-registered decision rule

The hypothesis was `< 1%` relative L2 deviation for `K = 8` across
`N ∈ [48, 192]`, with the residual attributed to interpolation error.

Measured worst median deviation: **41.2%**. The falsifier threshold was 5%.

The spec's fail branch applies: **rescope C2 to the range where it holds and
re-examine D02's design.** But the cause is not what the falsifier anticipated,
which changes the consequence — see §5.

## 2. Headline numbers

Median relative L2 deviation of the K=8 conditioning vector against its own
N=96 encoding, over 200 training curves:

| N | 48 | 64 | 80 | **96** | 128 | 192 |
|---|--:|--:|--:|--:|--:|--:|
| deviation | 29.1% | 18.2% | 8.6% | — | 15.4% | **41.2%** |

**The deviation is almost independent of `K`** (29.18% at K=3 vs 29.10% at
K=32) and almost independent of whether resampling preserves node clustering
or uses a uniform chord grid. Neither is what interpolation error looks like.
Interpolation error would grow with `K` as the retained modes approach the
Nyquist limit of the coarse grid. It does not.

## 3. Finding 1 — the deviation is entirely the `sqrt(N)` normalisation

`dct_ii` applies the **orthonormal** DCT-II convention: `sqrt(2/N)` for `k>0`,
`sqrt(1/N)` for `k=0`. By Parseval that preserves the *vector* norm — and the
norm of a sampled continuous function grows as `sqrt(N)`. So the coefficients
of a **fixed curve** scale as `sqrt(N)`.

| N | measured | `\|sqrt(N/96) − 1\|` | unexplained residual | under `1/N` convention |
|---|--:|--:|--:|--:|
| 48 | 29.08% | 29.29% | −0.21% | 2.12% |
| 64 | 18.20% | 18.35% | −0.15% | 1.12% |
| 80 | 8.64% | 8.71% | −0.07% | 0.44% |
| 128 | 15.37% | 15.47% | −0.10% | 0.55% |
| 192 | 41.17% | 41.42% | −0.25% | 1.07% |

The scaling law accounts for the deviation to within **0.25 percentage points
at every node count**. Under the `1/N` (forward-scaling) convention — which
approximates the continuous cosine coefficients and *is* sample-count
invariant — the deviation collapses to 0.4–2.1%, and it shrinks as the grid
refines, which is what genuine interpolation error does.

**The module docstring asserted the opposite of the code's behaviour:**

> *"so the encoding is scale-invariant under a change in sample count `N`. This
> is what makes the resulting descriptor source-agnostic across different node
> counts."*

That sentence is the stated justification for half of C2. It was never tested.
Corrected in this commit, with the true behaviour pinned by a regression test.

## 4. Finding 2 — node count is confounded with conditioning magnitude in training

The training corpus spans **52–96 nodes** ([[P1-F01_data-partition]]), so this
is not hypothetical — the confound is present in the data the model was
actually fitted on.

| | |
|---|---|
| `corr(‖cond‖, sqrt(N))` across 5,701 training samples | **0.708** |
| CV of `‖cond‖` | 16.4% |
| CV of `‖cond‖ / sqrt(N)` | 12.1% |

Roughly a quarter of the variance in conditioning magnitude is **node count,
not pressure**. The model has had a discretisation signal available in the
conditioning vector throughout training — exactly the leak C2 claims does not
exist.

Whether it *used* that signal is a separate question this experiment cannot
answer. It is now [[../specs/P1-D02_node-count-transfer]]'s central question,
and D02 must be read with this in mind: a transfer failure would no longer
distinguish "the model does not generalise across discretisations" from "the
conditioning changed scale underneath it."

## 5. Downstream impact — what this does and does not invalidate

**F01's OOD ladder survives in structure.** Recomputed under both conventions:

| | indist | highre | lowre | thin | aoa10 | thick |
|---|--:|--:|--:|--:|--:|--:|
| as-shipped | 0.99× | 2.28× | 2.64× | 4.32× | 6.77× | 18.1× |
| `1/N` convention | 0.95× | 1.94× | 3.26× | 12.96× | 9.20× | 37.8× |

The ordering is preserved except that **`thin` and `aoa10` swap**, and the
in-distribution rung stays at ~1× under both. The ladder is real, not an
artefact — the evaluation sets' node-count medians (64–76) sit inside the
training range (median 68), so the scaling factor largely cancels. But the
**rung spacings are not trustworthy as published** and F01's table must be
regenerated whichever convention is adopted.

*(These figures use a full nearest-neighbour search; F01's script subsamples
800 training points for the within-train reference, so its published numbers
differ slightly. The comparison above is internally consistent.)*

### 5.1 Superseded — authoritative regeneration, 2026-08-06

The table above was computed with a **proxy** for the corrected encoder
(dividing the legacy descriptor by `sqrt(N)`), before the fix was implemented.
It is superseded by `scripts/paper1/p1_b01_retrieval_index.py`, which uses the
real production encoder — forward normalisation **and** chordwise quadrature
abscissae — and F01's own 800-point within-train subsampling:

| | indist | highre | lowre | thin | aoa10 | thick |
|---|--:|--:|--:|--:|--:|--:|
| legacy (`ortho`) | 0.93× | 2.27× | 2.51× | 4.15× | 6.38× | 17.04× |
| **corrected** | **0.85×** | **2.01×** | **2.94×** | **12.10×** | **12.34×** | **37.13×** |

**The ladder ordering is unchanged** — `indist < highre < lowre < thin < aoa10
< thick` under both conventions. The proxy above wrongly predicted a
`thin`/`aoa10` swap; in the real encoder they do not swap, they converge to
near-equality (12.10 vs 12.34).

What does change is the **dynamic range**: 18× under the legacy encoder,
**44×** under the corrected one. The corrected ladder separates the rungs
better, which strengthens rather than weakens
[[../specs/P1-G03_dct-ood-ladder]].

`indist` sits at **0.85×** — below the within-train nearest-neighbour distance,
so held-out in-distribution targets are *closer* to training data than training
points are to each other. F01's "interpolation, not extrapolation" reading is
reinforced.

**What is blocked until the convention is decided:**

- [[../specs/P1-B01_retrieval-baseline]] — indexes training shapes by the DCT-Cp
  vector. Retrieval distances inherit the confound.
- [[../specs/P1-G03_dct-ood-ladder]] — built on F01's ladder.
- [[../specs/P1-D02_node-count-transfer]] — see §4.
- Any conditioning-space distance reported in the paper.

**What is not affected:** [[P1-F02_grading-metric]] (no DCT involvement),
[[P1-F05_reference-model]]'s training-half verdict (val loss is a
noise-prediction MSE, not a conditioning metric), and
[[../specs/P1-B03_basis-reachability]] (reconstruction round-trips through
`dct_ii_inverse`, so the normalisation cancels).

## 6. The decision this forces — not taken here

Three options, none of them free:

| Option | Cost | Consequence |
|---|---|---|
| **Keep `sqrt(N)`** | none | C2's conditioning claim is dropped; the paper states the descriptor is discretisation-dependent and reports the measured law. Honest, but it concedes half of C2. |
| **Switch to `1/N`** | retrain P1-REF (5 seeds, ~7.5 h) | C2's conditioning claim becomes demonstrable at 0.4–2.1%. Invalidates F05's checkpoints and every downstream comparison already anchored to them. |
| **Report both** | one extra 5-seed run | Turns the bug into a finding: *does* fixing the conditioning improve OOD transfer? Strongest scientifically, ~15 h of training. |

**This is not mine to decide** — it changes what the paper claims. Flagged for
the user. Nothing downstream of the conditioning space should run until it is
settled, because the numbers would be discarded.

`dct_ii` is **unchanged** in this commit. Only the docstring was corrected and
a test added. Changing the normalisation silently would have invalidated
[[P1-F05_reference-model]] without a record.

### 6.1 Decision taken — 2026-08-05

**Option 2, extended: fix the encoder and retrain.** Recorded here rather than
in the spec, which is closed.

A second defect was found while implementing the fix, and corrected with it:
the descriptor also **ignored where the nodes sat**, treating
LE/TE-clustered boundary nodes as evenly spaced. Passing the true chordwise
positions as quadrature abscissae is what makes the descriptor invariant to
node *spacing* as well as node *count*. On a clustered grid, ignoring positions
costs **36%** — an error of the same order as the one this experiment was
opened to investigate, and entirely independent of it.

Measured on the same 200 training curves, against the same N=96 reference:

| N | 48 | 64 | 80 | 128 | 192 |
|---|--:|--:|--:|--:|--:|
| legacy (as reported above) | 29.08% | 18.20% | 8.64% | 15.37% | 41.17% |
| normalisation fixed | 2.25% | 1.19% | 0.46% | 0.59% | 1.14% |
| **+ node positions** | **0.49%** | **0.23%** | **0.14%** | **0.12%** | **0.14%** |

**Worst-case deviation falls from 41.17% to 0.49%**, inside the 1% the
hypothesis originally predicted. The pre-registered verdict above is unchanged
— D01 tested what was shipped on 2026-08-04, and that was falsified. This
section records the remedy, not a re-grade.

Derivation, validity bound and the full numerical validation:
[[../CONDITIONING_MATHS]].

**Consequences, all now live:**

- **[[P1-F05_reference-model]] must be re-run.** Its five seeds were
  conditioned under the legacy convention. `configs/paper1/P1-F05_reference.yaml`
  now states `cond_norm: forward`; this will be attempt 3.
- B01, D02 and G03 are unblocked on the convention but now depend on the new
  reference checkpoints.
- F01's published ladder still reproduces — `p1_f01_partition.py` is pinned to
  `norm="ortho"` — but it should be regenerated under the corrected encoder
  before it appears in a figure, and the two versions compared (§5).
- [[P1-B03_basis-reachability]] is unaffected: both conventions differ by a
  positive diagonal scaling and span the same subspace, so its reconstruction
  errors are unchanged. Verified to float32 precision.

**A note for §5 of the paper.** The old and new conditioning differ by a factor
that correlates 0.708 with `sqrt(N)`, so the retrained model is not merely a
reseeded version of the old one — it is conditioned on a different signal. Any
comparison between attempt 2 and attempt 3 results is a comparison of two
methods, not two runs, and must be reported as such.

## 7. Validity range for C2, if the current convention is kept

Even under the `1/N` convention the claim is not unqualified — deviation
reaches 2.12% at N=48, above the 1% target. The honest statement for §3.3 is
a **bounded** one: invariance holds to ~1% for `N ≥ 64` and degrades below
that. The spec asked for "a stated validity range rather than an unqualified
claim"; this is it.

`K` is not the binding constraint anywhere in the tested range — the Nyquist
concern the spec anticipated does not materialise for `K ≤ 32`, `N ≥ 48`.

## 8. Reproduction

| | |
|---|---|
| Script | `scripts/paper1/p1_d01_conditioning_invariance.py` |
| Output | `outputs/paper1/P1-D01_conditioning-invariance/invariance.json` |
| Tests | `tests/test_pOnEllipseConditional.py` (2 added, 13 total) |
| Command | `uv run python scripts/paper1/p1_d01_conditioning_invariance.py` |
| Sample | 200 training curves (strided), 24 AoA10 curves, all 5,701 for §4 |

## 9. What this changes

**In the paper**
- §3.3 cannot assert conditioning invariance from construction. It must either
  state the measured bound (§7) or report the fix as a result.
- §4.4 gains a real measured quantity where it previously had an assumption.

**In the plan**
- **B01, G03, D02 are blocked** on the §6 decision.
- F01's ladder table must be regenerated before it appears in a figure.
- D02's design must control for the conditioning confound, or its result will
  not be interpretable.
