---
title: "Paper 1 — metric assessment and menu"
date: 2026-08-02
updated: 2026-08-26
tags: [paper1, metrics, assessment, methodology]
status: active
---

# Paper 1 — metric assessment and menu

A **menu, not a mandate.** Every metric below is assessed for what claim it can
carry, what it cannot do, what it costs, and how a reviewer will attack it. Use it
to choose what goes in the paper — the recommended sets are in §5.

Definitions and the frozen pipeline live in [PROTOCOL.md §2](PROTOCOL.md). This
document is the *argument* for each one.

**Legend — tier:** 🟩 core (the paper is weak without it) · 🟨 optional (strengthens
a specific claim) · ⬜ diagnostic (explains mechanisms; not a quality measure).

---

## 1. The menu

| Metric | Measures | Carries | Tier | Marginal cost |
|---|---|:--:|:--:|---|
| **ρ** Pearson correlation | Cp shape fidelity — ⚠️ **corroboration only**, see below | C1 C4 | 🟩 | free |
| **`a`** affine slope | Cp magnitude | C4 | 🟩 | free |
| `b` affine offset | Calibration offset | — | ⬜ | free |
| **ε_aff** post-affine RMS | Residual shape error | C1 | 🟩 | free |
| **Residual structure test** | Is the affine model *valid*? | C3 | 🟩 | free |
| **Chordwise band residual** (LE / peak / aft) | *Where* the error is | C1 C4 | 🟩 | free |
| **Suction-peak magnitude & location** | The designer's two numbers | C1 | 🟨 | free |
| Stagnation-point location | Nose geometry correctness | C4 | 🟨 | free |
| Base pressure `Cp_te` | Wake/bluff-body behaviour | C4 | 🟨 | free |
| **`Cp_rms`** | Wake unsteadiness | C4 | 🟨 | **already in `cp.csv`, unused** |
| **`Cd`** | Drag — the engineering quantity | C1 C4 | 🟩 | **already in `force.csv`** |
| **`Cl`** | Lift — the G02 failure mode | C4 | 🟩 | **in `force.csv`, read by nothing** |
| `Cd_std`, `Cl_std` | Within-run unsteadiness | protocol | 🟨 | free |
| **Validity rate** | Are outputs usable designs? | C1 C2 | 🟩 | free |
| Boundary roughness | Sampling artefacts | — | ⬜ | free |
| Closure error | Ring consistency | — | ⬜ | free |
| Thickness / area / curvature extrema | Geometric mechanism | C4 | 🟨 | free |
| Power spectrum of `r(θ)` | Index- vs arc-length prior | C2 | 🟨 | free |
| **Precision / recall (generative)** | Distributional quality | C1 | 🟩 | analysis only |
| MMD | Single distributional distance | C1 | 🟨 | analysis only |
| **Unconditional sample quality** | Did it learn the distribution at all? | C1 | 🟨 | 1 sampling run |
| **Chance-level ρ** | What ρ scale *means here* — ✅ **measured: 0.9954** | C3 | 🟩 | analysis only |
| **Achievability bound** | Model error vs target infeasibility | C1 C4 | 🟩 | free (reuses B01) |
| Noise floor `σ_ρ σ_a σ_Cd` | Smallest detectable difference | protocol | 🟩 | F04 |
| Sampling cost / valid design | The practical case | C1 | 🟨 | free |
| Val loss, params, wall-clock | Standard reporting | — | 🟩 | free |

Everything marked "free" is computable from data the pipeline **already writes**.
The full menu costs no additional solver runs beyond F04, F06 and F07.

---

## 2. Assessment by family

### 2.1 The Cp agreement family

**ρ — the workhorse, and the most attackable number in the paper.**

Strength: affine-invariant, so it is immune to the calibration offset between
solver and dataset — which is exactly why it was chosen, and it was the right
choice.

Two weaknesses, both serious:

1. **It cannot see magnitude.** This is structural, not a limitation of degree.
   Never report ρ without `a` beside it.
2. **Its scale is uncalibrated.** All ellipse Cp curves look broadly alike. If two
   *randomly chosen* dataset samples correlate at ρ ≈ 0.85, then a headline of
   ρ = 0.92 is far less impressive than it reads — and a reviewer who suspects this
   will ask. **Chance-level ρ (§2.5) must be computed before ρ is used as a
   headline.** This is the single highest-risk item in the metric set.

> [!danger] Answered 2026-08-17, and it is worse than the hypothetical above.
> [[results/P1-F04_noise-floor]] §4 measured it. Two **unrelated** dataset shapes
> correlate at **ρ = 0.9954**. A predictor that ignores its input and returns the
> corpus mean scores **0.9969**. Correct-versus-wrong pairing separates by
> **+0.00226, which is 2.1σ — marginal by this suite's own rule**.
>
> Independently confirmed on B03's 24 runs: ρ spans **2.0σ in total** across the
> whole experiment while `Cd` spans **7.8σ**.
>
> **ρ is demoted.** It answers "is this a plausible ellipse Cp curve", not "is
> this the correct one". Report it — a reviewer will compute it — but on a
> chance-referenced axis, never 0→1, and never as the headline. The
> discriminating metrics are **rank / top-k** (top-1 33.3% against 6.7% chance)
> and **`Cd`**. See [[PROTOCOL]] §2.1 and [[REPORT_OUTLINE]] §4.

**`a` — carries the creep finding, and rests on an untested assumption.**

`a` is a *fitted parameter of a calibration model*, `Cp_meas ≈ a·Cp_ref + b`.
Nothing currently tests whether that affine model is the right one. If the residual
has systematic chordwise structure, the affine fit is a misfit and both ρ and `a`
are contaminated — so the **residual structure test is not optional**, because `a`
underpins C4's headline. Added to F02.

**Chordwise band residual and the suction peak.** A whole-curve ρ conflates a
nose error with a wake error, and those mean entirely different things for design.
Free, and it converts "ρ = 0.91" into "the model reproduces the suction peak to
within 3% but misses the aft recovery" — a far more useful sentence.

**`Cp_rms` is already in `cp.csv` and used by nothing.** It gives per-node
fluctuation amplitude, i.e. wake unsteadiness. Useful as a *physical plausibility*
signal: if generated shapes produce systematically more unsteady wakes than dataset
shapes at matched Re, that is a real finding about what the model is producing and
it is available at zero cost.

### 2.2 Integrated forces — the missing engineering axis

Assessed in full at [PROTOCOL §2.3](PROTOCOL.md). The essential points:

- ρ is affine-invariant and therefore **structurally incapable** of detecting a
  magnitude error; `a` sees magnitude only through a fitted calibration. Neither is
  a physical quantity.
- `Cd` is an integrated force the solver computes directly. Agreement between a
  fitted calibration parameter and an independent integrated force is much stronger
  evidence than either alone — and **disagreement is diagnostic**, telling you the
  creep lives in the grading rather than the model (G04).
- `Cl` is the natural metric for a lifting case (G02) and is currently read by
  nothing in the repository.

**What forces cannot do:** they are integrals, so they hide distribution. Two very
different pressure distributions can integrate to the same `Cd`. Forces and Cp
agreement are complements, not substitutes — report both.

### 2.3 Geometry metrics

**Validity** is the one that matters and it becomes primary in Paper 2. A shape
that is not closed and simple is not a design, whatever its ρ.

**Roughness — de-weight it.** Year 1 treated boundary wobble as a defect and
justified `T = 1000` by it; independent grading then showed the wobble was
cosmetic, with no measurable pressure consequence. It remains a good *mechanism*
diagnostic (D02's node-count behaviour, A01's missing neighbour coupling) but it
must not stand as a quality measure. Where reported, state its measured Cp
consequence alongside. S03 tests this directly.

**Thickness, area, curvature extrema** are the geometric route for G04 — they turn
"the model creeps in magnitude" into "the model produces measurably thinner bodies,
and thinner bodies produce this pressure signature."

One worth adding if a design audience is targeted: **enclosed area** as a
constraint-satisfaction proxy. Real design problems carry volume constraints, and a
shape that hits the Cp target with a wildly different area is a design-relevant
failure that no current metric would flag.

### 2.4 Distributional metrics — the ML-venue gap

Every metric above is **per-sample conditional fidelity**, which evaluates a
regressor. This is a generative model going to MLST, where reviewers expect the
output *distribution* evaluated with the established vocabulary — precision/recall
for generative models (Sajjadi et al. 2018; Kynkäänniemi et al. 2019), or MMD. An
ad-hoc pairwise-distance diversity measure will read as improvised.

**Unconditional sample quality is unmeasured on the aerodynamic data entirely.**
That conflates two separable claims — *the model learned the shape distribution*
and *conditioning steers it correctly*. Only the second is tested anywhere in the
suite. `p_uncond` already supports dropping conditioning, so the cost is one
sampling run.

**What these cannot do:** distributional metrics say nothing about whether an
individual design meets its specification, which is the actual use case. They
answer the ML reviewer, not the engineer. Both audiences are in scope for MLST.

### 2.5 Calibration metrics — what a number *means*

These are not measurements of the model. They are the scales against which the
model's numbers are read, and the paper is much harder to attack with them present.

| Metric | Answers | Without it |
|---|---|---|
| **Noise floor** `σ_ρ σ_a σ_Cd` (F04) | What is the smallest *detectable* difference? | "Is 0.92 vs 0.86 real?" is unanswerable |
| **Chance-level ρ** | What does ρ = 0.92 mean *on this data*? | ρ may be near-saturated for any plausible ellipse, making the headline vacuous |
| **Achievability bound** | Is the residual error the model's fault, or is the target unreachable? | OOD degradation is over-attributed to the model |

**Chance-level ρ** — compute the distribution of ρ between *pairs of dataset
samples*, and between each target and a random valid ellipse. This bounds the
useful range: if random pairs correlate at 0.85, the interesting axis is 0.85→1.0,
not 0→1, and every reported ρ should be positioned on that axis. **Do this before
committing to ρ as the headline metric.**

**Achievability bound** — B01's retrieval baseline already produces it for free.
The best ρ attainable by *any* real shape in the training set is an empirical
ceiling for that target. Reported per target it decomposes the error:

```
total error  =  (target infeasibility)  +  (model shortfall)
                 ^ from retrieval-best     ^ what the model is responsible for
```

This matters most on the G03 OOD ladder, where a perturbed DCT target may
correspond to no realisable shape at all. Without the decomposition, ladder
degradation is attributed to the model when part of it is the ladder's own
construction — precisely the weakness a reviewer will probe in a synthetic OOD set.

### 2.6 Cost metrics

Sampling wall-clock per design, and per **valid** design. The exploration claim
implicitly competes with optimisation, and B01's retrieval baseline is essentially
free — so accuracy alone is not the whole comparison. Cheap to record, and it is
the answer to "what is this method *for*?"

---

## 3. Claim → metric matrix

Which metrics can carry which contribution. **Bold** = load-bearing.

| Claim | Primary evidence | Supporting |
|---|---|---|
| **C1** Node-native generative design without a hand-designed basis | **ρ, ε_aff, `Cd` on held-out targets** (G01); **precision/recall** (B04) | chordwise bands, achievability bound, validity, sampling cost |
| **C2** Discretisation-independent conditioning | **Validity and ρ vs node count** (D02); conditioning-invariance deviation (D01) | power spectrum of `r(θ)`, roughness as mechanism |
| **C3** Independent, qualified grading instrument | **Schäfer–Turek `Cd`, `Cl`, Δp, St vs published bounds** (F06); noise floor (F04) | residual-structure test, chance-level ρ, envelope stability of `a` |
| **C4** Failure taxonomy | **`a` + `Cd` jointly** (G04); **`Cl` ≈ 0 on lifting targets** (G02); ρ vs DCT distance (G03) | `Cp_rms`, thickness/area, base pressure, stagnation location |

Note that **C3 is the only claim whose evidence is now *absolute* rather than
relative.** Everything in the Year-1 report grades the solver by correlation
against the dataset — which cannot detect a shared systematic error. F06 is what
changes that, and it is the reason the instrument framing becomes defensible.

---

## 4. How each is attacked

| Metric | The attack | Defence |
|---|---|---|
| ρ | "All these curves look alike — what's chance level?" | Compute it (§2.5) and report ρ on that axis |
| ρ | "n = 2, no error bars" | F04 noise floor + 5 seeds + CI |
| `a` | "Your affine model is a fudge factor hiding a solver error" | **F06 + F07** — absolute benchmark, then localise the constant |
| `a` | "The residual isn't white, so the fit is meaningless" | Residual-structure test in F02 |
| `Cd` | "Integrated forces hide distribution" | Report with Cp agreement, never instead of it |
| Roughness | "You justified `T` by an aesthetic metric" | De-weighted to diagnostic; S03 measures its Cp consequence |
| Diversity | "That's not a standard generative metric" | Precision/recall + MMD |
| OOD ρ decay | "Your synthetic targets may be unrealisable" | Achievability bound + physical-realisability screen (G03) |
| Everything | "27 experiments, some 3σ results are chance" | [PROTOCOL §6.1](PROTOCOL.md) — confirmatory vs exploratory |

---

## 5. Recommended sets

### Minimum viable — the paper is defensible with these

ρ · `a` · ε_aff · residual-structure test · **`Cd`** · validity · noise floor ·
**chance-level ρ** · F06 benchmark result · precision/recall.

Ten metrics, all free except F04 and F06. Anything less leaves a named reviewer
question unanswered.

### Recommended — the version to aim for

Minimum, plus: chordwise bands · suction-peak magnitude and location · **`Cl`**
(G02) · achievability bound · `Cp_rms` · thickness/area (G04) · sampling cost ·
unconditional sample quality.

Still zero additional solver runs. The marginal cost is analysis code, most of it
in the F02 module.

### Stretch — only if the argument needs it

MMD · base pressure · stagnation location · power spectrum · enclosed-area
constraint satisfaction · `Cd_std` unsteadiness comparison.

---

## 6. Deliberately excluded

| Metric | Why not |
|---|---|
| Skin-friction drag split | Not cleanly extractable from the current LBM output; adds solver work for no claim |
| Separation-point location | Needs wall shear; same reason. Interesting for Paper 2 |
| FID-style perceptual scores | No meaningful feature extractor for 96-node rings |
| Radius KS statistic | The circle-series metric; superseded by precision/recall on the aero data |
| Calibration of the conditional distribution (PIT / rank histograms) | Correct in principle, but the right question for Paper 2's uncertainty story — over-engineering here |
| Grid-convergence index | Instrument framing lowers this bar; F06's absolute benchmark serves the purpose more cheaply |

Excluded metrics are listed so the choice is visible. If a reviewer asks for one,
the answer is a stated scope decision rather than an oversight.
