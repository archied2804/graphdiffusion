---
id: P1-F01
spec: "../specs/P1-F01_data-partition.md"
title: "Data partition and leakage audit"
date_run: 2026-08-03
date_reported: 2026-08-03
status: complete
verdict: falsified
git_sha: 0593e62
---

# P1-F01 — results

> **Spec:** [[../specs/P1-F01_data-partition]] · **Verdict:** ❌ **falsified — in an
> unanticipated way, and favourably**

## 1. Verdict against the pre-registered decision rule

The hypothesis was that `pOnEllipseTest.h5` is a disjoint held-out set. **That file
does not exist.** The URL in `DatasetUrl.TEST_H5` returns **404** and had never been
fetched, so the failure was invisible until now.

Neither pre-registered branch applies. The upstream repository instead publishes
**one in-distribution hold-out and four single-axis out-of-distribution sets**, none
of which had ever been downloaded. The correct action is neither "adopt the test
file" nor "carve a stratified split from train" — it is to adopt the publisher's
evaluation suite, which is materially better than either.

Two further findings materially change the plan. All three are below.

## 2. Headline numbers

| Partition | n | Nodes | Distinct node counts | Leaked | NN dist → train (design) | NN dist → train (DCT) |
|---|--:|---|--:|--:|--:|--:|
| **train** | 5701 | 52–96 | 12 | — | 1.00× (ref) | 1.00× (ref) |
| **indist** | 50 | 56–96 | 11 | 8 | **1.10×** | **0.99×** |
| highre | 50 | 52–96 | 11 | 0 | 3.49× | 1.78× |
| lowre | 50 | 56–88 | 9 | 0 | 3.54× | 2.39× |
| thin | 50 | 52–84 | 8 | 0 | 3.82× | **10.8×** |
| aoa10 | 24 | 68–80 | 4 | 0 | 1.41× | **12.9×** |
| thick | 50 | **64–100** | 10 | 0 | 7.18× | **33.3×** |

Leak tolerance `|ΔRe| < 1.0` **and** `|Δb| < 1e-3`. Distances are medians,
normalised by the within-train median nearest-neighbour distance (design space
0.0894; DCT space 0.01201).

Split frozen: **n_train = 5131, n_val = 570, n_test = 50**, plus 224 OOD samples
across five sets. `data/splits/paper1_partition.json`.

## 3. Finding 1 — node counts are **not** fixed at 96

**The training corpus spans 52–96 boundary nodes across 12 distinct values.** The
HDF5 array is `(5701, 96, 105)` — padded to its maximum — and reading that shape as
the node count is where the "fixed 96" belief came from.

```
52:  92   56: 622   60: 858   64: 745   68: 577   72: 561
76: 506   80: 608   84: 511   88: 327   92: 257   96:  37
```

Only **37 of 5701** samples (0.6%) actually have 96 nodes; the mode is 60.

**This corrects a load-bearing claim.** [[../../publication/PUBLICATION_PLAN]] §2.2
asserted that *"every sample in `pOnEllipseTrain.h5` has exactly 96 nodes, so the
model has never seen a different node count"*, and concluded resolution independence
was untested. The opposite is true: **the model has been trained across 12
discretisations from the start**, which is direct support for C2 and was available
all along.

Four consequences:

1. **C2 is stronger than claimed.** "Discretisation-native" already has in-range
   evidence. D02 is reframed from *"can a 96-node model transfer at all?"* to
   *"does it extrapolate outside 52–96?"* — a better-posed and harder question.
2. **A02's premise weakens, and in the graph's favour.** A 1-D CNN is the natural
   comparator on a *fixed-length* ordered ring. With variable node counts it needs
   padding or resampling, whereas the graph handles variable `N` natively. That is a
   real architectural argument the paper can now make — but it also means A02 must
   state its padding scheme as an unmatched factor.
3. **G02's confound largely dissolves.** The AoA10 set has 68–80 nodes, entirely
   *inside* the training range, so it is much closer to a pure physical-OOD test
   than the spec assumed. The `D02 → G02` ordering is no longer required for
   attribution (still useful, no longer blocking).
4. **D03 is partly already in effect** — training is de facto mixed-resolution.

**The `thick` set reaches 100 nodes, above the training maximum of 96.** That is a
free, real extrapolation test for D02 that no one had to construct.

## 4. Finding 2 — the evaluation suite that was never downloaded

`pOnEllipseTest.h5` does not exist. The repository publishes:

| File | Role |
|---|---|
| `pOnEllipseInDist.h5` | In-distribution hold-out (50) |
| `pOnEllipseLowRe.h5` / `pOnEllipseHighRe.h5` | OOD along Reynolds number (50 each) |
| `pOnEllipseThin.h5` / `pOnEllipseThick.h5` | OOD along thickness (50 each) |
| `pOnEllipseAoA10.h5` | OOD along incidence (24) — the only one previously local |
| `uvpAroundEllipse*.h5` | **Full velocity/pressure fields** (~266 MB each) — a data modality this project has never touched |

**This substantially improves the OOD story.** The DCT-space distances form a real,
graded, physically-interpretable ladder:

```
indist 0.99×  →  highre 1.78×  →  lowre 2.39×  →  thin 10.8×  →  aoa10 12.9×  →  thick 33.3×
```

Six rungs of real CFD spanning a 33× range in the space the model is actually
conditioned on. [[../specs/P1-G03_dct-ood-ladder]] was designed to *synthesise*
exactly this, and carried the weakness that synthetic targets may be physically
unrealisable — needing a realisability screen and an achievability bound. **Real
solved cases have no such problem.**

It also answers the viva squarely. *"Reconsider the construction of out-of-distribution
samples"* — these are the **publisher's** OOD sets, not ours, so the "you built your
own OOD set" objection cannot be raised. *"Potential variation in difficulty across
different principal components"* — four independent physical axes, each with a
measured distance, is precisely that.

Note the ordering is informative in itself: **thickness perturbations move the
conditioning vector 5–14× further than Reynolds perturbations do.** Difficulty is
strongly anisotropic, and geometry dominates.

## 5. Finding 3 — "held out" means held-out *samples*, not held-out *regions*

The `indist` set sits at **1.10×** the within-train nearest-neighbour distance in
design space and **0.99×** in DCT space. It is statistically indistinguishable from
another training sample.

8 of 50 `indist` samples have a training sample within `|ΔRe| < 1` and
`|Δb| < 1e-3`. On a densely-sampled two-parameter family that is expected rather
than sinister — but it must be stated plainly: **`indist` measures interpolation,
not generalisation to new regions.** The paper should say so, and let the five OOD
sets carry the extrapolation claim.

This makes [[../specs/P1-B01_retrieval-baseline]] more important, not less. At 0.99×
within-train distance, nearest-neighbour retrieval should be very strong on
`indist`, and the interesting question moves entirely to the OOD ladder.

## 6. Finding 4 — the Year-1 leak, confirmed

| Target | Year-1 claim | Membership under the frozen partition |
|---|---|---|
| sim #4849 | ρ = 0.964 "in-distribution" | **TRAIN** |
| sim #4066 | ρ = 0.966 "in-distribution" | **TRAIN** |

Both confirmed as training samples by direct index lookup against the frozen
partition.

## 7. Protocol deviations

The spec's method assumed `pOnEllipseTest.h5` existed. Steps 1–4 were executed
against the real file inventory instead, obtained from the HuggingFace API. Leakage
and NN-distance audits were run against **all five** evaluation sets rather than one.
The spec's decision rule did not anticipate a missing file; the adopted resolution
(use the publisher's evaluation suite) is recorded here rather than by editing the
spec.

## 8. Reproduction

| | |
|---|---|
| Script | `scripts/paper1/p1_f01_partition.py` |
| Partition | `data/splits/paper1_partition.json` (sha256 `e14e54b3…`) |
| Audit | `outputs/paper1/P1-F01_data-partition/audit.json` |
| Code changes | `DatasetUrl.TEST_H5` removed (404) → five evaluation URLs; `_VALID_VARIANTS` extended; `GraphDataLoader(train_indices=…, val_indices=…)` |
| Command | `python scripts/paper1/p1_f01_partition.py` |

## 9. What this changes

**In the paper**
- §3.5 states 52–96 nodes across 12 values, not "96-node ring". The method section's
  framing of the discretisation must be rewritten.
- §4.6 gains a six-rung real-CFD OOD ladder in place of a synthetic one.
- §4.1 must state that `indist` is interpolation, and that extrapolation rests on the
  five OOD sets.

**In the plan**
- [[../specs/P1-D02_node-count-transfer]] — reframe as extrapolation outside 52–96;
  `thick` (to 100 nodes) is a free real test.
- [[../specs/P1-G03_dct-ood-ladder]] — rebuild around the real sets; the synthetic
  ladder becomes optional, and the realisability screen is no longer needed.
- [[../specs/P1-G01_holdout-inverse-design]] — 50 `indist` targets, all of them,
  rather than 20 sampled from a larger set.
- [[../specs/P1-G02_aoa10-ood]] — confound dissolved; becomes one rung of the ladder.
- [[../specs/P1-A02_cnn1d-ring]] — must declare its padding/resampling scheme.
- [[../publication/PUBLICATION_PLAN]] §2.2 and §5.1 — the "fixed 96 nodes" claim is
  withdrawn.
- **New, unregistered:** the `uvpAroundEllipse*` field data is a Paper 2/3 asset. Log
  it, do not scope-creep into it.
