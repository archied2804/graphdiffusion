---
id: P1-F07
spec: "../specs/P1-F07_normalisation-audit.md"
title: "Cp normalisation audit — the source of a ≈ 1.74"
date_run: 2026-08-26
date_reported: 2026-08-26
status: complete
verdict: partial
git_sha: e973fff
---

# P1-F07 — results

> **Spec:** [[../specs/P1-F07_normalisation-audit]] · **Verdict:** ⚠️ **partial.**
> The dominant mechanism is identified and sourced — a factor of two in the
> dataset's non-dimensionalisation — and it accounts for **90.6%** of the observed
> factor. A residual of **9.35% ± 0.91pp** remains, and it is real: 10.3 standard
> errors from zero. It is not a solver error, not a de-dimensionalisation error,
> and not structured in the operating point.

## 1. Verdict against the pre-registered decision rule

The spec's four branches, in order:

| Branch | Fires? | Why |
|---|:--:|---|
| F06 shows a solver Δp error near `a` → **STOP** | ❌ | F06's worst Δp error is 4%; the factor is 81% |
| Mechanism matches observed `a` **within the noise floor** | ❌ | Candidate 3 predicts 2.00; observed 1.813 [1.777, 1.849] |
| `a` **structured** in Re/`H`/`b` → escalate | ❌ | See §5 — the one resolved term is the instrument |
| **`a` constant, mechanism incomplete** | ✅ | **This is the operative branch** |

The paper reports a documented factor of two **plus** a stated residual
systematic with a bound, rather than a fitted calibration constant. That is
weaker than the best case the spec hoped for and stronger than the worst.

## 2. Headline numbers

| | |
|---|--:|
| Observed factor, raw (before conversion) | **1.81298** |
| 95% CI | **[1.7773, 1.8486]** |
| After the factor-of-two conversion, `a` | **0.90649** |
| σ_a (single measurement, F04 frozen) | 0.03521 |
| Standard error of the mean, n = 15 | 0.00909 |
| **Residual from unity** | **9.35% ± 0.91pp** |
| Residual significance | **10.3 SEM** |

The residual is not a marginal effect. Individual measurements of `a` scatter by
σ_a = 0.035, but the *mean* of fifteen is pinned to ±0.009, and a 0.0935 offset
is 10.3 standard errors from zero. **There is something left to explain.**

## 3. Step 1 — the solver is eliminated

[[P1-F06_schafer-turek]] qualified the absolute pressure scale: L0 in double
precision reproduces Schäfer–Turek Δp to **0.03%**, the production L1 case to
**1–4%**. The spec's STOP branch needs a solver Δp error *near* the factor being
explained. The largest available is 4% against a factor of 81%. Eliminated.

## 4. Step 2 — the solver's de-dimensionalisation is self-consistent

Cp is dimensionless, so at fixed Re it must be invariant to the choice of `U`,
and invariant to `ρ` outright. The case builds its converter with
`UnitConverterFromResolutionAndRelaxationTime`, so the *lattice* problem is set
by (`RESOLUTION`, τ, chord) alone. All six runs report `latticeU = 0.075` and
`Ma = 0.129904` identically — only `physDeltaT` and `physViscosity` rescale.
Anything that moves is therefore post-processing, not physics.

| run | `U` | `MAX_PHYS_T` | chord-flows | `a` vs base | dev | |
|---|--:|--:|--:|--:|--:|:--:|
| `u2m` | 2.0 | 25 | 50 | 1.00206 | 0.06σ | ✅ |
| `uhalfm` | 0.5 | 100 | 50 | 0.99025 | 0.28σ | ✅ |
| `rho2` | 1.0 | 50 | 50 | 1.00027 | 0.01σ | ✅ |
| `u2` | 2.0 | 50 | **100** | 1.03412 | 0.97σ | ⚠️ confounded |
| `uhalf` | 0.5 | 50 | **25** | 0.90409 | **2.72σ** | ⚠️ confounded |

**A 4× range in `U` and a 2× range in `ρ` move `a` by at most 0.28σ.** The
solver's normalisation is self-consistent and **candidate 9 is eliminated**, as
the spec anticipated it would be.

### The first pass was confounded, and the confound is instructive

`MAX_PHYS_T` is raw physical time, but a flow develops in **chord-flows**,
`U·t/chord`. Holding `MAX_PHYS_T = 50` while rescaling `U` put the three runs at
25, 50 and 100 chord-flows of development — and [[P1-F04_noise-floor]] found
`MAX_PHYS_T` the *dominant* noise factor for `a` (σ_a = 0.0502). The confound
alone moved `a` by 2.72σ, an order above the effect being looked for, and its
ordering is monotonic in chord-flows.

The bottom two rows are kept in the table deliberately. Read alone they say
"the solver's normalisation is broken"; they say nothing of the kind.
`AVG_WINDOW_CHORDS` needed no such correction — `results.h:135` already scales
it by `chord/U`.

## 5. Step 4 — `a` is constant within the measurement floor

Fifteen envelope cases, regressed with **F04's frozen σ_a as the error bar**
rather than the residual scatter.

**Constant model: χ² = 12.39 on 14 dof, χ²/dof = 0.885.** The entire spread of
`a` across the envelope is *inside* the measurement floor. There is no excess
variance for any mechanism to explain.

| term | slope | se | slope/se | Δχ² | span effect |
|---|--:|--:|--:|--:|--:|
| Re | +0.000119 | 0.000052 | **2.31** | 5.32 | 0.0593 (1.68σ) |
| `H` | −0.023605 | 0.022447 | −1.05 | 1.11 | −0.0236 (−0.67σ) |
| `b` | −0.047017 | 0.242128 | −0.19 | 0.04 | −0.0051 (−0.15σ) |

`H` and `b` are flat. Re resolves at 2.31σ — and that is the instrument, not the
convention.

### Step 4b — the Re term cannot be separated from resolution

`RESOLUTION = max(100, 0.2·Re)`, so across this sweep **corr(Re, RESOLUTION) =
+1.0000**. These fifteen cases *cannot in principle* distinguish a
convention-side Re dependence from a solver resolution artefact. Two pieces of
evidence say it is the latter:

1. F04 measured the resolution factor **directly, at fixed Re**: σ_a = 0.02514.
   The Re-span effect is 0.0593 — 2.4× that, on a resolution range that is itself
   2×. The trend is the size the instrument already accounts for.
2. F03's resolution sweep changes `a` by **+3.7%, −2.8%, +2.0%** on doubling.
   The signs disagree, so this is scatter, not a systematic under-resolution
   bias.

**Candidates 7, 8 and 10 all predict dependence on the operating point. None is
supported.** `a` is one constant.

## 6. Step 3 — candidate matching, and the trap in it

Observed raw factor **1.81298**, 95% CI **[1.7773, 1.8486]**.

| # | Mechanism | Predicted | Miss | In CI? | Status |
|:--:|---|--:|--:|:--:|---|
| 1 | Dataset `q_ref` = 1.8 × ½ρU² | 1.80 | +0.7% | ✅ | **struck before the run** |
| 2 | Dataset `U_ref` = 4/3 × ours | 1.78 | +1.9% | ✅ | **struck before the run** |
| **3** | **Missing factor of ½** | **2.00** | **−9.4%** | ❌ | **derived from source** |
| 4 | `U_max` vs `U_mean` (3/2) | 2.25 | −19.4% | ❌ | |
| 5 | Dataset velocity scaled 1.8× | 3.24 | −44.0% | ❌ | struck before the run |
| 6 | Momentum-average parabolic (6/5) | 1.44 | +25.9% | ❌ | |
| 7 | Channel blockage | ~1.38 | +31.9% | ❌ | needs `b` dependence — §5 says no |
| 8 | Compressibility O(Ma²) | 1.00 | +81.3% | ❌ | needs Re dependence — §5 says no |
| 9 | `ρ` mismatch | — | — | — | **eliminated by §4** |
| 10 | Averaging-window mismatch | 1.00 | +81.3% | ❌ | needs Re dependence — §5 says no |

**This is exactly the situation the pre-registered table exists for.** The two
candidates that land *inside* the confidence interval are the two with no
derivation, and the one candidate derived from both sides of the ratio sits
outside it.

Candidates 1, 2 and 5 were struck in [[../DATASET_PROVENANCE]] §5 **before this
experiment ran**, for reasons that have not changed: the "1.8" they rest on is
stated for **NsCircle**, the circular-cylinder family of Lino et al. (2022)
§III B, not the elliptical family of §III C; and it is a *network-input*
normalisation, listed beside "Reynolds scaled to [−1, 1]" and "edge attributes
divided by 0.2", not a physical non-dimensionalisation of stored CFD data.

Re-adopting them now **because they fit** would be the precise rationalisation
the spec's registered table was written to prevent:

> *"Each candidate predicts a specific number. This makes the audit a matching
> exercise rather than a rationalisation, which is why the table is registered
> before the work starts."*

The numerical closeness of 1.813 to 1.80 is recorded here as a **coincidence
worth stating and not acting on**. If a source is ever found showing the ellipse
data carries a 1.8 dynamic-pressure convention, candidate 1 becomes the answer
immediately — but that source does not currently exist.

## 7. Step 6 — no factor is introduced downstream

Traced end to end. `export_ellipse_for_openlb._reference_cp` takes the plain mean
of the 101 stored timestep columns and normalises `x/c` to a leading-edge origin
by chord. The solver writes `xocLoc = (x − leadingEdgeX)/chord` (`results.h:507`)
— the same convention. `cp_agreement` fits the affine model and applies nothing
else.

**The only transformation anywhere in the path is the documented
`DATASET_P_STAR_TO_CP = 2.0`.** No hidden factor, no ordering error, no origin
mismatch.

## 8. What remains unexplained

A **constant 9.35% ± 0.91pp**, real at 10.3 SEM, after a sourced factor of two.
It is provably not:

- a solver absolute-pressure error (§3, F06)
- a solver de-dimensionalisation error (§4)
- a dependence on Re, `H` or `b` (§5)
- a downstream artefact of the comparison code (§7)

What is left, from [[../DATASET_PROVENANCE]] §6, is a difference between the
dataset's spectral/hp-element solver and our lattice Boltzmann one, or an
undocumented rescaling in the ICLR 2025 re-release. Neither is cheaply testable,
and **the spec does not ask Paper 1 to close it** — closing the magnitude gap is
on the "deliberately not here" list in [[../EXPERIMENT_REGISTER]] §5.

**How the paper should carry it.** As a stated systematic with a bound. Because
`a` is constant across the envelope (§5), the offset is common to every
condition the paper compares, so it cancels in every comparison the paper
actually makes. It is a presentation problem, not a validity one — but §3.4 must
state it, with the CI, rather than reporting a fitted constant.

## 9. Protocol deviations

Dated, per [[../PROTOCOL]] §9.

- **2026-08-26** — Six solver runs, the spec's full budget, but not the split it
  describes. The spec allots two for step 2 and four spare; four were used for
  step 2 (`U` at 0.5×, 1×, 2× and `ρ` at 2×) and two more to repeat the `U`
  rescaling at matched chord-flows after the first pass was found confounded
  (§4). No candidate-testing runs were needed — steps 5 and 6 are archaeology.
- **2026-08-26** — Step 4 uses F04's frozen σ_a as the regression error bar.
  F03's own regression bootstrapped its residual scatter and reported the Re term
  as excluding zero; both are printed by the audit script. The two answer
  different questions and the spec asks for the first.
- **2026-08-26** — Step 4b is not in the spec. The Re/RESOLUTION confound has to
  be stated before the Re term is interpreted in either direction.
- **2026-08-26** — Steps 5 and 6 were substantially complete before the
  experiment ran, in [[../DATASET_PROVENANCE]] §§4–5. Candidates 1, 2 and 5 were
  struck there, on the record and before any number was produced. §6 above
  reports the trace; nothing in it changed the verdict.

## 10. What this unblocks

**G04** lists F07 as a dependency and is now clear. The attribution it has to
make is sharper than before: the reference scale is a factor of two plus a
constant 9.35% offset with a 0.91pp bound, so "generated shapes grade at `a` far
from 1" is now a deviation from a *known* value rather than from a fitted one.

**Paper 2's R2.2** targets `a → 1.74` through physical conditioning. That target
is now known to be **1.813 [1.777, 1.849]** raw, of which a factor of two is
definitional. R2.2 should be restated against the residual, not the raw number.

## 11. Files

| | |
|---|---|
| Runner (step 2) | `scripts/paper1/run_p1_f07_selfconsistency.sh` |
| Audit (steps 1, 3, 4, 4b) | `scripts/paper1/p1_f07_audit.py` |
| Archaeology (steps 5, 6) | [[../DATASET_PROVENANCE]] §§4–5 |
| Conversion constant | `cp_metrics.DATASET_P_STAR_TO_CP` |
| Data | `outputs/paper1/P1-F07_normalisation-audit/audit.json` |
