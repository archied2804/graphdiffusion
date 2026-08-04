---
id: P1-B03
spec: "../specs/P1-B03_basis-reachability.md"
title: "Basis reachability analysis"
date_run: 2026-08-04
date_reported: 2026-08-04
status: partial
verdict: falsified
git_sha: 3fec41d
---

# P1-B03 — results

> **Spec:** [[../specs/P1-B03_basis-reachability]] · **Verdict:** ❌ **falsified,
> and against C1's interest.** The shape manifold is effectively
> *one*-dimensional, not the hypothesised three to five. The AoA10 contrast the
> spec expected to carry C1 turns out to be a rotation, not a reach limitation.
>
> **Partial run** — the solver half (spec step 3) is deferred pending
> [[../specs/P1-F04_noise-floor]].

## 1. Verdict against the pre-registered decision rule

The hypothesis was **three to five modes**. Measured: **1 mode reaches 99.60%
explained variance, 2 modes 99.998%.** The spec's first branch applies, more
strongly than it anticipated:

> *"If ~3 modes suffice within the noise floor → state plainly in §4.3 and §5:
> on this dataset a low-dimensional basis is nearly sufficient. The paper then
> argues C1 on reach (AoA10, G03) rather than on in-distribution accuracy."*

The second half of that instruction is now also blocked — see §4.

## 2. Headline numbers

Median RMS reconstruction error, in units of normalised radius `r/mean(r)`,
against **held-out** shapes (the 570-sample val split of the frozen partition):

| M | 1 | 2 | 3 | 5 | 10 | 20 | 50 |
|---|--:|--:|--:|--:|--:|--:|--:|
| **PCA** | 3.1e-3 | 2.2e-4 | 1.9e-4 | 1.5e-4 | 7.0e-5 | 2.2e-5 | 3.7e-8 |
| **DCT** | 1.5e-1 | 1.5e-1 | 1.5e-1 | 2.5e-2 | 4.6e-3 | 4.7e-4 | 2.1e-4 |

| | |
|---|---|
| Modes to 99% variance | **1** |
| Modes to 99.9% variance | **2** |
| Held-out error at M=2 | 2.2e-4, i.e. **0.02% of mean radius** |

Two PCA modes describe this dataset's geometry to two parts in ten thousand.
For an ellipse family parameterised by `(Re, b)` this is the expected answer —
but it had never been measured, and C1's premise was asserted against it.

**The adapted basis beats the fixed one by ~700× at M=3** (1.9e-4 vs 1.5e-1).
That contrast is worth keeping: it is the honest form of A03's comparison, and
it says a *generic* spectral truncation is a much weaker representation than a
*fitted* one at equal DOF.

## 3. `thick` — genuine geometric extrapolation, and still reachable

`thick` occupies aspect ratios **0.850–0.923**, entirely outside the training
range of 0.587–0.846. It is the one evaluation set that is real geometric
extrapolation.

PCA reconstructs it at **2.9e-4 (M=3)** and 1.4e-4 (M=10) — the same order as
held-out. A basis fitted on thinner bodies extrapolates to thicker ones without
difficulty, because the manifold is smooth and low-dimensional.

This is a direct negative for C1 on this dataset.

## 4. The AoA10 contrast is a rotation, not a reach limitation

This is the finding that changes the section.

Raw AoA10 reconstruction looks catastrophic and **does not improve with more
modes** — 5.33e-2 at M=1, still 5.30e-2 at M=50. On its face that is exactly
the C1 evidence the spec wanted: *"a basis bounds the reachable design space."*

It is not. The geometry check:

| set | aspect ratio (minor/major) | major-axis angle |
|---|---|---|
| train | 0.587–0.846 (med 0.718) | 0° |
| **aoa10** | **0.587–0.846 (med 0.721)** | **170° (≡ −10°)** |
| thick | 0.850–0.923 (med 0.892) | 0° |

**AoA10 is the same ellipse family, rotated by 10°.** Identical aspect-ratio
range to three decimal places. The rotation control confirms it:

| AoA10 de-rotated by | 0° | 5° | **10°** | 15° |
|---|--:|--:|--:|--:|
| RMS at M=10 | 5.3e-2 | 2.7e-2 | **8.7e-5** | 2.7e-2 |
| × held-out | 759× | 387× | **1×** | 387× |

De-rotating by exactly 10° restores reconstruction to **held-out level**. The
apparent unreachability is entirely a property of the `r(θ)` representation,
which is anchored to the body-fixed frame and therefore **not
rotation-equivariant**. A physically trivial rigid rotation moves a shape 759×
outside a basis that otherwise reconstructs it perfectly.

Reporting the 759× as evidence for C1 would have been wrong, and it would have
been very easy to do — the number is large, it is in the expected direction,
and it appears without any prompting to look for it.

## 5. What this does to C1

C1's premise — that a low-dimensional parameterisation bottlenecks the
reachable design space — **is not supported analytically on pOnEllipse**:

- 2 modes cover held-out shapes to 2.2e-4
- 3 modes cover genuine thickness extrapolation (`thick`) to 2.9e-4
- the one apparently unreachable family is unreachable only by rotation

The honest position for §5 is that **this dataset cannot demonstrate the
parameterisation bottleneck**, because its shape manifold is nearly
one-dimensional. That is a limitation of the testbed, not a refutation of the
idea — but the paper must say so plainly rather than argue C1 from a dataset
that cannot support it.

Consequences:

- [[../specs/P1-B02_pca-generative]]'s hypothesis is now very likely to fail in
  the same direction. **Run it anyway** — a pre-registered baseline that beats
  the method is worth more than a quiet omission — but its expected result has
  changed, and B02's spec should not be edited to match (that would be a
  retrofit).
- [[../specs/P1-G02_aoa10-ood]] **must be reframed.** AoA10 is a flow-condition
  OOD test, not a shape OOD test. Its Cp field is genuinely out of distribution;
  its geometry is not.
- The paper's C1 argument, if it survives, has to come from
  [[../specs/P1-G03_dct-ood-ladder]] — which is currently blocked by
  [[P1-D01_conditioning-invariance]] §6.

## 6. A new finding the spec did not anticipate

**The `r(θ)` representation is orientation-locked**, and the training corpus is
entirely at 0° incidence. Every shape the model has ever seen is
phase-aligned. This is a property of the *method*, not of the dataset, and it
is not currently stated anywhere in the paper.

It matters beyond B03: a model trained on phase-aligned `r(θ)` has no mechanism
to produce a rotated body, so any evaluation that requires one is testing
something the representation forbids by construction. Whether that is a real
limitation or an acceptable design choice is a §5 discussion, but it must be
disclosed.

Not in scope for Paper 1 to fix — 1 DOF on a fixed ring is the pre-registered
scope, and rotation equivariance is a Paper 2 concern.

## 7. Protocol deviations

**Held-out set.** The spec says "held-out `r(θ)`". The G-group evaluation sets
are not yet graded, so this uses the 570-sample validation split of the frozen
partition. [[P1-F01_data-partition]] established that `indist` sits at ~1×
within-train nearest-neighbour distance, so val and `indist` are
interchangeable for this purpose. Stated rather than assumed.

**Solver half deferred.** Spec step 3 — pushing reconstructions at M=3 and
M=10 through the solver to convert geometric error into a Cp tolerance — needs
the [[../specs/P1-F04_noise-floor]] figure to be interpretable. 8 runs, to be
completed after F04. The reachability curves above do not depend on it.

**Common θ grid.** Node counts vary 52–96, so PCA requires resampling to a
fixed dimension (128 θ stations). This is resampling in *angle*, so
[[P1-D01_conditioning-invariance]]'s `sqrt(N)` finding does not apply; the DCT
rows round-trip through `dct_ii_inverse`, so the normalisation cancels there
too.

## 8. Reproduction

| | |
|---|---|
| Script | `scripts/paper1/p1_b03_basis_reachability.py` |
| Output | `outputs/paper1/P1-B03_basis-reachability/reachability.json` |
| Command | `uv run python scripts/paper1/p1_b03_basis_reachability.py` |
| Data | 5,131 train / 570 held-out / 24 aoa10 / 50 thick, `paper1_partition.json` |

## 9. What this changes

**In the paper**
- §4.3 reports a one-dimensional shape manifold and a reachable `thick` set.
- §5 must concede that pOnEllipse cannot demonstrate C1's premise.
- §3 must disclose that `r(θ)` is orientation-locked (§6).
- The AoA10 rotation control belongs in the paper as a **method** figure — it
  is a worked example of a plausible result that does not survive checking.

**In the plan**
- G02 reframed: flow-condition OOD, not shape OOD.
- B02's expected outcome revised; spec unchanged.
- B03 reopens after F04 for the 8 solver runs.
