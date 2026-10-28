---
id: P1-A03
group: A — Architecture necessity
title: "Spectral-coefficient MLP"
tier: gate
status: planned
depends_on: [P1-F05]
blocks: [AUGUST-GATE]
paper_section: "4.2"
contribution: C1
cost: {train_runs: 6, solver_runs: 12, effort: "~3 days"}
---

# P1-A03 — Spectral-coefficient MLP

## Question

Does diffusing in a spectral basis at matched degrees of freedom match diffusing
per node?

*This is the dropped EXP-013 "Method E — Fourier coefficient MLP".*

## Hypothesis *(pre-registered)*

At **full rank** (96 DCT coefficients) the spectral model matches the node-native
model — the two are related by an orthogonal transform, so any gap is optimisation
dynamics, not expressivity. At **truncated** rank the spectral model degrades where
curvature is highest, and the truncation point is a hand-chosen basis decision of
exactly the kind C1 claims to avoid.

## Falsifier

Truncated spectral (K = 8 or 16) matches full-rank node-native within `2σ` on
held-out ρ. Then a low-dimensional spectral basis suffices for this dataset, and
the argument for per-node generation rests on richer shape families that this
paper does not have. That must be stated in §5, not buried.

## Method

Two things vary here, and separating them is the point:

1. **Full-rank control (matched DOF).** Diffuse all 96 DCT-II coefficients of
   `r(θ)`; MLP score network; same conditioning, schedule, seeds. Tests
   *representation basis* with expressivity held constant.
2. **Truncation ladder (unmatched DOF, deliberately).** K ∈ {8, 16, 32}. Tests
   *how much the low-dimensional basis costs*, which is the C1 argument in its
   sharpest form.
3. Parameters within ±5% of 147,585 for the full-rank run. The truncated runs are
   necessarily smaller at the input/output layer — state the counts, do not pretend
   they are matched.
4. Reuse `dct_ii` / `dct_ii_inverse` from
   `src/graph_diffusion/data/pOnEllipseConditional.py:58` — the same transform
   already used for conditioning, so the paper uses one DCT definition throughout.
5. Seeds 0, 1, 2 on full-rank; single seed on the truncation ladder interior.

## Measurements

| Quantity | Purpose |
|---|---|
| Held-out ρ and `a` vs K, with the full-rank and node-native points marked | The headline figure of §4.2 |
| Boundary roughness vs K | Where truncation shows up geometrically |
| Reconstruction error of held-out shapes at each K *(analytic, no training)* | Separates *basis expressivity* from *generative difficulty* |
| Parameter counts | Honest matching statement |

Computing the analytic reconstruction error alongside the generative result is
what lets §4.2 distinguish "the basis cannot represent this shape" from "the model
cannot find it" — two very different findings that a single ρ number conflates.

## Decision rule

- Full-rank spectral within `2σ` of node-native → expected; report as evidence
  that the contribution is the *per-node field*, not the coordinate system.
- Truncated K=8 within `2σ` of node-native → **the honest limitation.** Report in
  §4.3 and §5 alongside B03, and point at Paper 3.2 as where the advantage is
  expected to appear.
- Truncated spectral clearly worse, degrading monotonically with decreasing K →
  the cleanest possible support for C1. Show the ladder.

## Consequence

Together with **B02** and **B03** this is the evidential core of C1. A03 attacks
the question through the *basis of the diffusion variable*; B02 through a
*different low-dimensional basis* (shape-PCA); B03 *analytically*, with no model at
all. Three independent routes to the same question — if they agree, the finding is
robust; if they disagree, the disagreement is itself informative.

## Results

_Not yet run._
