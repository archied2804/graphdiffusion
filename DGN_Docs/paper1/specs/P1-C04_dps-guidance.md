---
id: P1-C04
group: C — Conditioning
title: "DPS guidance under the clean protocol"
tier: core
status: planned
depends_on: [P1-F05, P1-F04]
blocks: []
paper_section: "4.6"
contribution: C4
cost: {train_runs: 0, solver_runs: 48, effort: "~2 days"}
---

# P1-C04 — DPS guidance under the clean protocol

## Question

Does the Year-1 finding — that diffusion posterior sampling makes inverse design
**worse** — survive seed repeats, a held-out target set, and the measured noise
floor?

## Hypothesis *(pre-registered)*

Yes. The negative result holds and is outside the noise floor: ρ decreases
monotonically with DPS weight, with the endpoint difference exceeding `3σ_ρ`. The
mechanism is that DPS's likelihood gradient is computed through the auxiliary
pressure head, which is an approximation — so the guidance steers toward the head's
errors rather than toward the true physics.

## Falsifier

The degradation falls within `2σ` once seeds and held-out targets are used. Then
the Year-1 finding was single-seed noise on training targets, and the claim must be
withdrawn.

That possibility is why this is being re-run rather than carried over. The Year-1
version was fixed-seed, single-variable, on non-held-out targets — well-designed as
far as it went, but its conclusion was never tested against a measurement floor,
and it is one of the paper's advertised findings.

## Method

1. Sweep `sampling.dps_guidance_weight ∈ {0, 0.3, 0.5, 1.0}` — the Year-1 ladder —
   on frozen P1-REF checkpoints. No retraining.
2. **4 held-out targets × 4 weights × 3 seeds = 48 solver runs.** The Year-1 version
   used a single seed and non-held-out targets; both are corrected here.
3. Use both model seeds and sampling seeds so this is a full repeat rather than a
   sampling-only one — unlike C03, the claim here is a headline finding and needs
   the stronger variance estimate.
4. Report against the F04 noise floor explicitly, with the band on the figure.
5. **Test the mechanism.** Record the auxiliary head's own Cp prediction error on
   each generated sample. If the hypothesis is right, DPS damage should correlate
   with head error across targets — guidance is only as good as the surrogate
   gradient it follows. That correlation converts a negative result into an
   explained one.

## Measurements

| Quantity | Purpose |
|---|---|
| ρ, `a` vs DPS weight, mean ± CI, with noise-floor band | Whether the finding survives |
| Post-affine RMS vs DPS weight | Secondary fidelity measure |
| Correlation between head prediction error and DPS damage, per target | **The mechanism** |
| Sample validity vs DPS weight | Whether DPS also breaks geometry |

## Decision rule

- Monotone degradation, endpoints separated by `> 3σ` → the finding stands, now
  with seeds, held-out targets and a stated mechanism. Report as C4 evidence.
- Within `2σ` → **withdraw the claim.** Record the withdrawal in
  [[../publication/PUBLICATION_DIARY]] and state in §4.6 that the Year-1 result did
  not replicate under repetition. That is an uncomfortable paragraph to write and a
  strong one to have written.

## Consequence

Negative results about a popular technique are useful to the field and rarely
published — but only if they are robust. Under-powered negative results are worse
than none, because they are cited and then fail to replicate. Forty-eight solver
runs is the price of getting this one right.

## Results

_Not yet run._
