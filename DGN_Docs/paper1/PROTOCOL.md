---
title: "Paper 1 — experimental protocol"
date: 2026-08-02
updated: 2026-08-02
tags: [paper1, protocol, methodology, reproducibility]
status: frozen-on-F04
---

# Paper 1 — experimental protocol

The rules every experiment in [specs/](specs/) obeys. This document is the paper's
§3.5 "Experimental protocol" in draft form — written once, cited everywhere, so no
individual experiment has to restate it.

> **Definitions here; the argument for each metric in [METRICS.md](METRICS.md)** —
> what claim it can carry, what it cannot do, what it costs, and how a reviewer
> will attack it. Use that document to choose what goes in the paper.

**Freeze point:** this document becomes read-only once **P1-F04** reports. After
that, changes require a dated amendment block at the bottom and a re-run of any
affected experiment. That discipline is the whole point: a protocol that can be
adjusted after seeing results is not a protocol.

---

## 1. Data partition

Defined by **P1-F01**, frozen to `data/splits/paper1_partition.json`, loaded by
every downstream experiment. Nothing may re-derive a split at runtime.

| Partition | Source | n | Used for |
|---|---|--:|---|
| **train** | `pOnEllipseTrain.h5` minus val | 5131 | Gradient updates only |
| **val** | held-out slice of `pOnEllipseTrain.h5` | 570 | Early stopping, checkpoint selection, hyperparameter choice |
| **test** | `pOnEllipseInDist.h5` | 50 | **Reported numbers only.** Touched once per experiment, after all choices are final |
| **ood-Re** | `pOnEllipseLowRe.h5`, `pOnEllipseHighRe.h5` | 50 each | G03 |
| **ood-thickness** | `pOnEllipseThin.h5`, `pOnEllipseThick.h5` | 50 each | G03 |
| **ood-incidence** | `pOnEllipseAoA10.h5` | 24 | G02, G03 |

> **`pOnEllipseTest.h5` does not exist** — the URL formerly in `DatasetUrl.TEST_H5`
> returns 404 and had never been fetched. The publisher instead provides the
> in-distribution hold-out and four single-axis OOD sets above. See
> [results/P1-F01_data-partition.md](results/P1-F01_data-partition.md).

### Two facts that shape every experiment

**Node counts vary: 52–96 across 12 distinct values.** The HDF5 array is padded to
its 96-node maximum, and reading that shape as the node count is wrong — only 37 of
5701 samples have 96 nodes; the mode is 60. The model is therefore trained across
discretisations by default, which is direct evidence for C2. Never describe this
work as operating on "a 96-node ring".

**`indist` is interpolation, not extrapolation.** It sits at 1.10× the within-train
nearest-neighbour distance in design space and 0.99× in DCT space — statistically
indistinguishable from another training sample. Report in-distribution numbers as
interpolation performance, and let the five OOD sets carry every generalisation
claim.

The JSON stores explicit index lists, the RNG seed, the file SHA-256 of each `.h5`,
and the leakage-audit result. It is committed to git.

### Rules

- **The test split is not a development set.** If you look at a test number and
  then change a hyperparameter, that experiment is contaminated and must be
  re-registered under a new ID.
- Target selection from the test split is **uniform random with a recorded seed**.
  Farthest-first selection is forbidden for headline numbers — it samples the
  extremes and inflates apparent difficulty. Where Year-1 numbers must be compared,
  report the farthest-first subset *separately and labelled*.
- Every experiment records `n_train / n_val / n_test` and the partition file hash
  in its `run_meta.json`.

> **Why this exists.** The Year-1 headline in-distribution result (ρ ≈ 0.95 on
> sims #4849 and #4066) was measured on **training samples** —
> `scripts/export_generated_for_openlb.py:163` calls
> `pick_targets_farthest_first()` over the full dataset and never sees the loader's
> split. This protocol exists so that cannot recur.

---

## 2. Metrics

### 2.1 Cp agreement — the primary grading metric

Defined once in `graph_diffusion.postprocessing.cp_metrics` (built by **P1-F02**).
No experiment may use a local copy.

Given a target curve `Cp_ref(x/c)` and a solver-measured curve `Cp_meas(x/c)`,
both interpolated onto a common 128-point chordwise grid:

| Symbol | Name | Definition | Reads as |
|:--:|---|---|---|
| **ρ** | Pearson correlation | `corr(Cp_ref, Cp_meas)` | **Shape fidelity.** Affine-invariant: unaffected by the solver's known offset/scale |
| **a** | affine slope | least squares `Cp_meas ≈ a·Cp_ref + b` | **Magnitude.** The dataset's own normalisation gives `a ≈ 1.74` for real shapes |
| **b** | affine offset | as above | Calibration offset; reported, rarely interpreted |
| **ε_aff** | post-affine RMS | `rms(Cp_meas − (a·Cp_ref + b))` | Residual shape error after removing calibration |

**Report ρ and a together, always.** ρ alone hides magnitude creep — which is
precisely how the Year-1 creep finding nearly went unnoticed.

**The affine model must be checked, not assumed.** `ε_aff` reports the residual
magnitude but says nothing about its *structure*. If the residual has systematic
chordwise structure, `Cp_meas ≈ a·Cp_ref + b` is the wrong model and both ρ and `a`
are contaminated by the misfit. F02 adds a residual-structure diagnostic (residual
vs `x/c`, plus a runs test for randomness); F03 reports it across the envelope.

### 2.2 Chordwise-resolved error

ρ is a single number over the whole curve, so it conflates a leading-edge error
with a wake error — and they mean entirely different things for design. Report the
post-affine residual over three chordwise bands:

| Band | `x/c` | Why it matters |
|---|---|---|
| Leading edge / stagnation | 0 – 0.15 | Sets the stagnation point; errors here indicate the wrong nose geometry |
| **Suction peak** | 0.15 – 0.5 | **Where design-relevant physics lives.** Peak magnitude and its chordwise location are the two numbers a designer reads |
| Aft / recovery | 0.5 – 1.0 | Pressure recovery and separation behaviour; noisiest region |

Also report **suction-peak magnitude and location** explicitly, target vs measured.
Free from data already collected, and far more legible than a whole-curve ρ.

Two further free quantities from the same file:

| Metric | Source | Reads as |
|---|---|---|
| **Stagnation-point location** | `argmax Cp` | Whether the nose geometry is right |
| **Base pressure** `Cp` at `x/c → 1` | Last node | Wake behaviour; correlates with drag |
| **`Cp_rms`** | **Third column of `cp.csv`, currently unused** | Wake unsteadiness — a physical-plausibility signal. If generated shapes produce systematically more unsteady wakes than dataset shapes at matched Re, that is a finding about what the model produces |

### 2.2a Reference scales — what a number *means*

Three calibration quantities. They measure nothing about the model; they are the
scales its numbers are read against, and the paper is much harder to attack with
them present.

| Scale | Set by | Without it |
|---|---|---|
| **Noise floor** `σ_ρ`, `σ_a`, `σ_Cd` | F04 | "Is 0.92 vs 0.86 real?" is unanswerable |
| **Chance level of ρ** | F04 companion analysis | ρ may be near-saturated for *any* plausible ellipse, making the headline vacuous |
| **Achievability bound** | B01's retrieval-best, per target | OOD degradation is over-attributed to the model |

The achievability bound gives an error decomposition that costs nothing:

```
total error  =  target infeasibility  +  model shortfall
                ^ retrieval-best         ^ what the model is responsible for
```

This matters most on G03's synthetic ladder, where a perturbed DCT target may
correspond to no realisable shape. Without the decomposition, ladder degradation is
charged to the model when part of it belongs to the ladder's construction — exactly
what a careful reviewer will probe.

### 2.3 Integrated aerodynamic coefficients

The solver already writes `tmp/ellipse2d_force.csv` with
`Cd_mean, Cd_std, Cl_mean, Cl_std, n_samples, t_start, t_end` for **every** run.

| Metric | Status in repo | Role here |
|---|---|---|
| **`Cd`** | Read by 6 analysis scripts | **Promoted to a primary metric** |
| **`Cl`** | Written by all 70 runs, **read by nothing** | Primary metric for G02 |
| `Cd_std`, `Cl_std` | Unused | Per-run unsteadiness → feeds F04 |

**Why this is not optional.** ρ is affine-invariant, so it is *structurally
incapable* of detecting a magnitude error; `a` catches magnitude only as a fitted
parameter of a calibration model. Neither is a physical quantity. Drag is the
quantity an aerodynamic design audience actually asks about, and a paper that
grades inverse design without ever reporting it invites the obvious question.

Three specific uses:

1. **An independent axis on the magnitude creep (G04).** If generated shapes are
   thinner and their Cp magnitude inflated, `Cd` must move too, in a predictable
   direction. Agreement between a fitted calibration parameter and an integrated
   physical force is far stronger evidence than either alone — and disagreement
   would mean the creep is a grading artefact.
2. **`Cl` is the natural metric for G02.** The AoA10 set is asymmetric and
   *lifting*. Grading a lifting case on pressure correlation while ignoring lift is
   a conspicuous omission, and `Cl ≈ 0` on a 10° case is a sharper statement of the
   symmetric-collapse failure mode than any Cp statistic.
3. **`Cd_std` is a free unsteadiness measure**, already time-averaged over a stated
   window. It feeds F04's noise floor directly and flags runs whose statistics have
   not converged.

Report `Cd` (and `Cl` where lift is non-trivial) alongside ρ and `a` in every
grading table. The cost is zero — the numbers are already on disk for all 70
existing runs.

### 2.4 Distributional metrics — the model is generative

Every metric above is a **per-sample conditional fidelity** measure. That evaluates
a regressor. This is a *generative* model going to an ML venue, where reviewers
will expect the output **distribution** to be evaluated, not only its members.

| Metric | Definition | Answers |
|---|---|---|
| **Coverage / recall** | Fraction of held-out real shapes with a generated sample inside their k-NN radius in shape-PCA space | Does the model reach the whole data manifold, or only its dense centre? |
| **Precision / density** | Fraction of generated samples falling inside the real data manifold | Are samples plausible, or off-manifold artefacts? |
| **MMD** | Maximum mean discrepancy between generated and real `r(θ)` distributions | A single distributional distance |
| **Unconditional sample quality** | The above, sampling with conditioning dropped | Did the model learn the shape distribution at all, separately from whether conditioning works? |

Use the standard precision/recall formulation for generative models
(Sajjadi et al. 2018; Kynkäänniemi et al. 2019) rather than an ad-hoc diversity
measure — the vocabulary is expected at MLST and the definitions are established.

**Unconditional quality is currently unmeasured on the aerodynamic data entirely.**
The circle series used a radius KS statistic; there is no aero analogue. It matters
because "the model learned the shape distribution" and "conditioning steers it
correctly" are separable claims, and only the second is currently tested. Primary
home: B04.

### 2.5 Sampling cost

Wall-clock per sampled design, and per **valid** design (dividing by validity
rate). The exploration claim implicitly competes with optimisation, and B01's
retrieval baseline is free — so accuracy alone is not the whole comparison.
Record for every sampling run; it costs nothing and is needed the moment a reviewer
asks what the method is *for*.

### 2.6 Geometry metrics

| Metric | Implementation | Purpose |
|---|---|---|
| Boundary roughness | `postprocessing.metrics.compute_boundary_roughness` | Sampling artefacts |
| Closure error | `scripts/metrics.compute_closure_error` | Ring consistency |
| **Validity** | new in F02: closed ∧ non-self-intersecting ∧ min node spacing > 0.1× mean | Fraction of samples that are usable designs at all. Becomes central in Paper 2 |
| Shape-PCA coefficients | `data.cp_manifold` machinery, refitted on `r(θ)` | Diversity and coverage (B03, B04) |

> **Roughness is a diagnostic, not a quality metric.** The Year-1 work treated
> boundary wobble as a defect and motivated `T = 1000` by it — then independent
> grading showed the wobble was cosmetic, with no measurable pressure consequence.
> Report roughness to explain *mechanisms* (D02's node-count behaviour, A01's
> missing neighbour coupling), never as a standalone measure of sample quality.
> Where it is reported, state its measured Cp consequence alongside. S03 tests this
> directly.

### 2.7 Model metrics

Train/val loss, val loss at best checkpoint, parameter count, epochs to best,
wall-clock. All in `loss_log.json`.

---

## 3. Grading pipeline

One path from a generated shape to a graded number. Any deviation is a protocol
violation.

```
sample r(θ)  →  polygon .dat  →  shapes/<EXP-ID>/manifest.csv
             →  runs/run_from_manifest.sh
             →  runs/<EXP-ID>/<case>/tmp/ellipse2d_cp.csv
             →  cp_metrics.cp_agreement()
             →  results/<EXP-ID>.md  +  outputs/paper1/<EXP-ID>/grading.json
```

### Frozen solver settings

Established and qualified by **P1-F03**. Do not vary these except in F03/F04,
where they are the independent variable.

| Setting | Value | Why |
|---|---|---|
| `RESOLUTION` | `max(100, round(0.2·Re))` | The validated stability rule; cells-per-chord must grow with Re or the Mach limit is exceeded |
| `MAX_PHYS_T` | 50 | |
| `AVG_WINDOW_CHORDS` | 25 | |
| `CHANNEL_HEIGHT` | per-row `H` from the manifest | The domain must match the sample the target came from |
| Re | per-row from the manifest | as above |

`--SEMI_MINOR_B` is deliberately omitted: the `.dat` polygon defines the body.

Every manifest carries `name, idx, Re, H, n_nodes` at minimum. `name` must begin
with the experiment ID.

---

## 4. Seeds and repetition

| Claim type | Seeds | Applies to |
|---|:--:|---|
| Headline result | **5** | F05, G01 |
| Any claim in the abstract or a figure with error bars | **3** | A, B, D, G groups |
| Sweep point within an ablation | 3 at the endpoints, 1 in the interior | C, S groups |
| Analytic / no-training experiment | n/a | F01, D01, B01, B03 |

Seeds are `0,1,2,3,4` and control **both** the training initialisation **and** the
sampling noise, recorded separately in `run_meta.json` as `train_seed` and
`sample_seed`. When only sampling is repeated (e.g. C03, which needs no
retraining), say so explicitly — sampling-only repeats measure a smaller variance
and must not be presented as full seed repeats.

Report **mean ± 95% CI** (t-interval, small n). Never report a bare best-of-n.

---

## 5. Baseline matching

A baseline that is not matched is not evidence. Every comparator in the **A** and
**B** groups must state, in its results file:

1. **Parameter count** within **±5%** of `P1-REF` (the F05 model). Rebalance width,
   not the comparison.
2. **Degrees of freedom** of the generated object — 96 for node-native; a truncated
   basis at M modes is *deliberately* unmatched and must be labelled as such (that
   is B02's whole point).
3. **Identical** data partition, conditioning vector, noise schedule, optimiser,
   LR schedule, epoch budget, early-stopping rule and grading pipeline.
4. **Identical seeds.**

Anything that cannot be matched is declared in the results file under
"Unmatched factors" with an argument for why it does not explain the result.

---

## 6. Significance — the noise floor rule

**P1-F04** measures the solver's own repeatability and produces a pooled σ for ρ
and for a. Call these `σ_ρ` and `σ_a`.

| Observed difference | Reported as |
|---|---|
| `< 2σ` | **"no detectable difference"** — not a win, not a tie-break |
| `2σ – 3σ` | "a difference of Δ, marginally above measurement uncertainty" |
| `> 3σ` | A result |

This rule is what lets the paper write "outside measurement uncertainty" honestly,
and it is the answer to the reviewer question *"n=2, no error bars — is 0.964 vs
0.860 outside sampling noise?"* The Year-1 OOD ladder was described as "noisier
than monotonic"; under this rule that sentence becomes a number.

**Apply the rule symmetrically.** It must be as capable of erasing a favourable
result as an unfavourable one. If A01 (no-edge control) lands within 2σ of the
graph, the honest report is "message passing shows no detectable benefit" — and
that is a publishable finding, not a failure.

### 6.1 Multiple comparisons

This suite makes on the order of **80 pairwise comparisons** across 27 experiments
and their sweeps. At a nominal 3σ threshold, roughly one spurious "result" is
expected by chance alone, and it will land wherever it is least convenient.

Two mitigations, both cheap:

1. **Pre-registration already does most of the work.** Each spec names *one*
   primary comparison in its decision rule. Only those ~27 comparisons carry
   confirmatory weight; everything else in a results file is exploratory and must
   be labelled as such in the text. This is the main reason the suite is registered
   in advance rather than analysed post hoc.
2. **Within a sweep, test the trend, not the best cell.** For ordered sweeps (C02's
   `K`, C03's `w`, S02's data fraction, G03's ladder) report a monotonicity or
   regression test across the whole sweep rather than the largest pairwise gap.
   Picking the best cell out of five and testing it at 3σ is the classic way to
   manufacture a result from noise.

State in §3.5 which comparisons are confirmatory and which are exploratory. Very
few papers do this, and it is close to free given the register already exists.

---

## 7. Reproducibility record

Every run writes `outputs/paper1/<EXP-ID>/<seed>/run_meta.json`:

```json
{
  "exp_id": "P1-F05",
  "git_sha": "...",
  "git_dirty": false,
  "config_path": "configs/paper1/P1-F05_reference.yaml",
  "config_sha256": "...",
  "partition_file": "data/splits/paper1_partition.json",
  "partition_sha256": "...",
  "train_seed": 0,
  "sample_seed": 0,
  "n_train": 0, "n_val": 0, "n_test": 0,
  "param_count": 147585,
  "device": "cuda", "torch": "...", "started": "...", "wall_clock_s": 0
}
```

A run with `git_dirty: true` may not be used for a reported number.

**Amended 2026-08-04 (P1-F05).** `git_dirty` is evaluated **at process start**,
over **code paths only** — `train.py`, `src/`, `configs/`, `data/splits/`. The
repo-wide list is still recorded as `git_dirty_files` for audit.

Two defects made the original rule unusable in practice. Git state was captured
when the run *finished*, so a run was attributed to whatever had been committed by
then rather than to the commit it actually executed. And the dirty check was
repo-wide, so an Obsidian vault autosave during a 7-hour training run marked it
unusable.

The rule's purpose is that the computation be reproducible from an identified
commit. A documentation file cannot change what a training run computes; a file
under `src/` can. The amended check is **stricter where it matters** (correct
attribution) and no longer fires on files that cannot affect the result.

---

## 8. Reporting a result

Each completed experiment gets `results/<EXP-ID>.md` from
[results/_template.md](results/_template.md), containing at minimum:

- The verdict against the **pre-registered decision rule**, stated first
- Whether the hypothesis was supported, falsified, or inconclusive against the
  noise floor
- Mean ± CI for every number, with n
- Unmatched factors and confounds
- What the paper will now say — and what it can no longer say

**Deviations from the pre-registered spec** are recorded in a dated
"Protocol deviations" block. Do not silently edit a spec after running it.

---

## 9. Amendments

_None yet. Protocol freezes when P1-F04 reports._
