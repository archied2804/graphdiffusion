---
title: "Paper 1 — experimental protocol"
date: 2026-08-02
updated: 2026-08-26
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

> [!warning] ρ is **not** the headline metric — amended 2026-08-17
> **P1-F04** measured the chance level of ρ on this corpus: two *unrelated*
> dataset shapes correlate at **ρ = 0.9954**, and a predictor that ignores its
> input and always returns the corpus mean scores **0.9969**
> ([[results/P1-F04_noise-floor]] §4). Matched cross-grading gives a
> discrimination signal of **+0.0023, or 2.1σ** — marginal by this document's
> own §6 rule — with 10.5% of *wrong* pairings beating the median correct one.
>
> ρ answers "is this a plausible ellipse Cp curve", not "is this the *correct*
> one". Three rules follow, binding on every experiment:
>
> 1. **The discriminative metric is rank / top-k**, not ρ: grade each generated
>    shape against *every* target and report where its own target ranks. Chance
>    is `1/N`. Measured on F03's envelope: top-1 33.3% against 6.7% chance.
> 2. **Never plot or quote ρ on a 0 → 1 axis.** State the 0.9954 chance floor
>    beside it. Its entire usable range is 0.9967 → 0.9990.
> 3. **`Cd` is the preferred scalar** where one is needed: 40% dynamic range
>    across the envelope against a 5% noise floor, where ρ has 0.2%.
>
> This restriction is corpus-relative — it says ρ cannot discriminate *within
> the design space this dataset spans*, which is the regime every Paper 1 claim
> lives in. It is not a general claim about correlation as a metric. Using ρ to
> assess *fit quality against a known curve* (F03, F06) remains valid.

#> [!tip] Running an experiment whose spec is written in ρ
> Several unrun specs (B01, A01–A03, D02, G03) state their decision rule in ρ,
> which §2.1 has since demoted. This does **not** block them and does **not**
> license editing them. Run the experiment, apply the registered ρ rule, report
> its verdict plainly — then report rank/top-k and `Cd` alongside it in the same
> results file. That adds information without retrofitting a rule. If the two
> disagree, the disagreement is itself a finding and belongs in the results file
> under "Protocol deviations", dated.

## 2.1a Dead surface probes — mandatory filter

Solver Cp curves produced before 2026-08-17 contain samples whose probe landed
inside the body: `Cp = −cpInf` exactly, `Cp_rms = 0.0` exactly, 3–4% of surface
nodes and up to 12% on some shapes. `read_cp_csv` drops them by default via
`dead_probe_mask`, and graders must report the count per run. The solver-side
fix is in `results.h`; filtered old runs and re-run new ones agree to within a
fifth of σ, so **regrading is sufficient — existing runs need not be repeated**.
Full diagnosis: [[results/P1-F04_noise-floor]] §2.

**The affine model must be checked, not assumed.** `ε_aff` reports the residual
magnitude but says nothing about its *structure*. If the residual has systematic
chordwise structure, `Cp_meas ≈ a·Cp_ref + b` is the wrong model and both ρ and `a`
are contaminated by the misfit. F02 adds a residual-structure diagnostic (residual
vs `x/c`, plus a runs test for randomness); F03 reports it across the envelope.

## 2.1b Generated bodies must be exported at physical scale — mandatory

`feature_mode="radial_norm"` trains on `r(θ)/mean(r)`, so **the model cannot
predict absolute size** and the sampler returns a body whose *mean radius* is 1.
Written straight to a `.dat` that gives a semi-major axis of 1.13–1.36 where the
corpus is 0.50.

**Every shape in the corpus has chord exactly 1.0** (verified across the
training file, std = 0). So the export must rescale to chord 1.0 before the body
reaches the solver. This is a corpus-wide constant, not the target's own size —
supplying it restores information the representation discards by design and
leaks nothing about the individual target.

**Why it matters.** `CHANNEL_HEIGHT` is passed in absolute units, so an
unrescaled body sits in the channel **2.1–2.8× oversized** and its blockage
ratio is wrong by the same factor. A confined oversized body accelerates the
surrounding flow, deepens the suction peak and inflates `|Cp|` — which is
indistinguishable, in the graded numbers, from the model overshooting.

⚠️ **Every Year-1 generated-shape run has this defect.** Measured chords:

| shape set | n | chord |
|---|--:|--:|
| `generated` (EXP-041/042) | 6 | 2.09 |
| `dps_sweep` | 4 | 2.48 |
| `ood_sweep` | 5 | 2.12 |
| `ood_systematic` | 16 | 2.77 |
| `node_sweep` | 5 | 2.48 |
| `baseline` *(dataset shapes)* | 3 | **1.00** ✅ |

Paper 1 regenerates every number under this protocol, so **no Paper 1 result is
affected**. But two consequences carry:

1. **[[specs/P1-G04_creep-attribution|G04]] gains a fourth candidate its spec
   does not list.** It asks whether the ~1.4× magnitude excess is model-side,
   solver-side or grading-side. It may be **export-side**. G04 has not run, so
   its spec can still be amended — that is a decision for the supervisor, not a
   silent edit.
2. Any Year-1 statement that magnitude creep is model-side is unsupported until
   re-measured.

Found 2026-08-28 during [[specs/P1-G01_holdout-inverse-design|G01]] setup, by a
geometry sanity check before the runs were launched. Fix:
`scripts/paper1/p1_g01_sample.py::to_physical`.

## 2.1c Vortex-shedding onset is uncontrolled — OPEN, found 2026-09-18

**Found, not fixed.** The averaging window is fixed at `t = MAX_PHYS_T − 25` to
`MAX_PHYS_T` (t = 25–50 at the standard settings). It is **not** triggered by the
flow reaching a statistically steady state. Shedding onset — the first time
`|Cl| > 0.05` in the logged history — ranges from t ≈ 8 to t ≈ 47.

**It differs systematically by shape source.** A smooth dataset body at zero
incidence is symmetric and waits for round-off to break the symmetry. A
generated body carries the ~1% per-node jitter of
[[results/P1-D02_node-count-transfer]] §5 — built-in asymmetry — and sheds early.
Runs whose `Cl` amplitude at least **doubles** inside the averaging window:

| experiment | shapes | median onset | amplitude still ×2 in window |
|---|---|--:|--:|
| F04 noise floor | dataset | 26.5 | **59%** |
| B01 retrieval | dataset | 22.0 | **26%** |
| G01 target geometries | dataset | 21.5 | **14%** |
| G01 generated | generated | 18.0 | 1% |
| D02 | generated | 12.0 | 0% |

Double precision sheds later still (less round-off): the F06 fp64 repeat of F03's
resolution sweep sheds at t = 31–47, which — not precision acting on converged
physics — is why its unsteady quantities disagree with fp32.

**What it affects, and what it does not.**

- **F04's noise floor** partly measures onset-time randomness rather than
  measurement noise on a converged flow. This is the likely mechanism behind
  F04's finding that `MAX_PHYS_T` dominates every σ.
- **Generated-against-dataset `Cd` comparisons** — G01 §4b and the B01/G01
  comparison — set a mostly-shedding flow against a sometimes-not-yet-shedding
  one. A systematic bias of unmeasured sign and size.
- **Solver-grid convergence (F03) is readable only where shedding is
  established.** At Re 751 it is, and the production rule is converged (drag
  within ~1–2% of 2×, Strouhal within ~1%). At Re 501 every run at ≥1× is still
  ramping up, so that case cannot show convergence either way.
- **Not affected:** anything purely geometric (B03 part 1, D01, D02's validity
  and shape results), and runs on generated shapes, which are almost all
  established.

**The fix is standard practice, and needs re-runs:** a brief asymmetric
perturbation at start-up so shedding begins early and consistently, and/or a
per-run stationarity gate before averaging. Not done — a decision for the
supervisor, since it re-opens F04's frozen σ.

Figures: `outputs/paper1/figures/fig8_shedding_onset`, `fig9_grid_convergence`.
Script: `scripts/paper1/p1_make_flow_figures.py`.

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

**P1-F04** measures the solver's own repeatability and produces a pooled σ per
metric. **Measured 2026-08-17 and frozen** — see
[[results/P1-F04_noise-floor]].

| Metric | **σ** | **2σ** — below this, no detectable difference | **3σ** — above this, a result |
|---|--:|--:|--:|
| **ρ** | 0.00108 | **0.00215** | **0.00323** |
| **a** | 0.03521 | **0.07042** | **0.10563** |
| **Cd** | 0.04742 | **0.09483** | **0.14225** |
| **Cl** | 0.01542 | **0.03084** | **0.04625** |

Pooled *within* shape across 27 runs varying resolution, node phase, averaging
window and physical time on a fixed geometry. The between-shape spread is
physics and is excluded by construction.

| Observed difference | Reported as |
|---|---|
| `< 2σ` | **"no detectable difference"** — not a win, not a tie-break |
| `2σ – 3σ` | "a difference of Δ, marginally above measurement uncertainty" |
| `> 3σ` | A result |

Two consequences worth carrying into every later experiment:

- **Node phase contributes σ_ρ = 2×10⁻⁵**, two orders below every other factor.
  Any node-placement or node-count effect measured downstream is the model's,
  not the instrument's. This is what makes the C2 claim measurable at all.
- **Statistical convergence dominates.** `MAX_PHYS_T` moves `Cd` more than
  resolution does, and the within-run unsteadiness exceeds the between-run
  scatter by only 1.4–1.8×. **Any experiment whose primary metric is `Cd` or
  `Cl` must use `MAX_PHYS_T ≥ 60`** — this applies to B04 and G01.

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
