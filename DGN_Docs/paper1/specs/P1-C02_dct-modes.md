---
id: P1-C02
group: C — Conditioning
title: "DCT modes K"
tier: core
status: planned
depends_on: [P1-F05, P1-G01]
blocks: []
paper_section: "4.5"
contribution: C2
cost: {train_runs: 5, solver_runs: 20, effort: "~2 days compute"}
---

# P1-C02 — DCT modes K

## Question

Is `K = 8` justified, or inherited?

## Hypothesis *(pre-registered)*

Fidelity **saturates by `K = 8`** — the Cp curves of this smooth two-parameter
family are spectrally sparse, so the first eight cosine modes capture nearly all of
the target signal. Large `K` should be neutral in-distribution and **harmful
out-of-distribution**, because high-order modes carry solver noise that the model
then tries to reproduce.

Predicted: ρ flat within `2σ` for `K ≥ 8`; degraded at `K = 3`; and `K = 32`
measurably worse on the G03 OOD ladder than `K = 8`.

## Falsifier

Fidelity still improving at `K = 32`. Then the conditioning signal is being
truncated, `K = 8` costs accuracy, and the reported result understates what the
method can do.

## Method

1. Sweep `k_modes ∈ {3, 5, 8, 16, 32}` in the P1-REF recipe. Note this changes
   `global_dim` and `cond_dim` together — keep them consistent and record the
   parameter count, which moves slightly with `K`.
2. Seeds 0, 1, 2 at `K = 8` (incumbent) and `K = 32` (the falsifier point); single
   seed elsewhere. Five training runs.
3. Grade 4 held-out targets per setting.
4. **Also grade on the OOD subset** where the noise hypothesis is testable — that
   is where large `K` is predicted to hurt. In-distribution grading alone cannot
   distinguish "saturated" from "over-parameterised conditioning".
5. Report the **spectral energy** of the dataset's Cp curves per mode. If 99% of
   the energy is in the first 5 modes, that is a principled justification for
   `K = 8` independent of the sweep — and the two arguments should agree.

## Measurements

| Quantity | Purpose |
|---|---|
| Held-out ρ, `a` vs `K`, CI at the two decision points | The sweep result |
| OOD ρ vs `K` | Tests the noise-sensitivity hypothesis |
| Cumulative spectral energy of dataset Cp per mode | Independent, model-free justification |
| Conditioning reconstruction error vs `K` | Ties back to D01's Nyquist bound |

## Decision rule

- Saturation at or before `K = 8` → the choice is justified. Report both the sweep
  and the spectral-energy argument; two independent lines of support is what turns
  a defensible choice into an unremarkable one.
- Still improving at `K = 32` → report that `K = 8` was sub-optimal and quantify
  the cost. Do not re-run the suite at the new value after seeing test numbers.
- Large `K` worse OOD → a genuinely useful design finding: conditioning bandwidth
  should be matched to the *signal*, not maximised. It also connects C02 to G03.

## Consequence

Discharges part of viva point B1.1 — turning an inherited hyperparameter into a
measured one. Cheap: five training runs, twenty solver runs.

The OOD half is the part worth doing carefully. "More conditioning information is
better" is the intuitive expectation, and evidence against it in the extrapolation
regime is more interesting than the in-distribution saturation curve.

## Results

_Not yet run._
