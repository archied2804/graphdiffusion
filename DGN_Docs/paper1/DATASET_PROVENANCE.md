---
title: "pOnEllipse dataset — provenance and normalisation, from the source"
date: 2026-08-06
tags: [paper1, methods, dataset, provenance, normalisation]
status: reference
---

# pOnEllipse — what the source actually says

> **What this file is.** Facts about the training dataset, taken from the papers
> that produced it, with citations. Written because the affine slope `a ≈ 1.74`
> has been attributed to a "documented `u / 1.8` velocity scaling" that **this
> search could not find any source for** (§5).
>
> **What this file is not.** Report prose. Every claim here carries a citation
> or is explicitly marked as inference.

---

## 1. Provenance chain

| | |
|---|---|
| HuggingFace | `mariolinov/Ellipse` — **dataset card is empty**, no README content, no viewer |
| Consuming paper | Lino, Pfaff & Thuerey, *Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks*, **ICLR 2025**. [arXiv:2504.02843](https://arxiv.org/abs/2504.02843) |
| Code | [`tum-pbs/dgn4cfd`](https://github.com/tum-pbs/dgn4cfd) |
| **Originating paper** | Lino, Fotiadis, Bharath & Cantwell, *Multi-scale rotation-equivariant graph neural networks for unsteady Eulerian fluid dynamics*, **Physics of Fluids 34(8), 2022**. [doi:10.1063/5.0097679](https://doi.org/10.1063/5.0097679) |
| Original solver | **Nektar++** (Cantwell et al., 2015) — spectral/hp element method |

The ICLR 2025 datasets are described as *"low-resolution versions of the
datasets from Lino et al. (2022)"*, with resolution *"reduced by approximately a
factor of two"* (Appendix C).

**Our pipeline therefore grades a lattice-Boltzmann solver against
spectral-element data.** That is not wrong, but it is a stronger claim than
"same solver, different settings", and it should be stated in §3.

---

## 2. Geometry and flow configuration *(ICLR 2025, Appendix C)*

Verbatim where quoted:

- *"flow around an elliptical cylinder where the **free-stream velocity** is
  parallel to the ellipse's major axis"*
- *"The ellipse's major axis length is one (non-dimensional unit), and the
  **domain height ranges from five to six**."*
- *"The flow is laminar and unsteady, exhibiting periodic vortex shedding"*
- Each simulation is **101 consecutive time-steps** of fully developed 2-D flow.
- Node types are **inlet, wall, or inner node**.

**Confirms** our case's `CHANNEL_HEIGHT` 5.0–6.0 and chord = 1. The phrase
"free-stream velocity" supports the open-domain reading our case implements,
rather than a duct with a parabolic inlet — but the paper does **not** state the
top/bottom boundary condition explicitly, so this is not settled. It matters:
see §6.

---

## 3. Parameter ranges *(ICLR 2025, Table 2)*

| Dataset | Re | Minor axis `b` | AoA (°) | Avg # nodes (Ellipse) |
|---|---|---|:--:|:--:|
| Train | [500, 1000] | [0.5, 0.8] | 0 | 71 |
| InDist | [500, 1000] | [0.5, 0.8] | 0 | 70 |
| LowRe | **[400, 500]** | [0.5, 0.8] | 0 | 72 |
| HighRe | **[1000, 1100]** | [0.5, 0.8] | 0 | 71 |
| Thin | [500, 1000] | **[0.45, 0.5]** | 0 | 64 |
| Thick | [500, 1000] | **[0.8, 0.9]** | 0 | 76 |
| AoA | [500, 1000] | [0.5, 0.8] | **10** | 74 |

Three independent confirmations of our own audits:

1. **Node counts.** The paper's averages (64–76) match
   [[results/P1-F01_data-partition]]'s measured 52–96 with median 68. The
   Year-1 belief in a fixed 96 nodes is refuted by the source as well as by the
   data.
2. **`b` convention.** The paper's `b` ∈ [0.5, 0.8] is the **full** minor axis;
   our `eval_grid` manifest records `b` ∈ [0.26, 0.37], i.e. the **semi**-minor
   axis. Consistent, but the two must not be mixed — a factor-of-2 trap.
3. **AoA10 is a rotation.** The paper describes it as *"ellipses **rotated by
   10-degrees**"* with an otherwise identical parameter distribution. This is
   exactly [[results/P1-B03_basis-reachability]] §4's finding, arrived at
   independently from the geometry, and now confirmed by the source.

The OOD ladder in [[results/P1-F01_data-partition]] is therefore a **designed**
generalisation suite, not an accident of sampling — `thin`/`thick` extend
thickness, `lowre`/`highre` extend Reynolds, `aoa` rotates. Worth saying in §3.

---

## 4. The non-dimensionalisation — the key finding

The ICLR 2025 paper states the governing equations in non-dimensional form
(Appendix C). The momentum equation is printed as:

$$\frac{\partial u}{\partial t}+u\frac{\partial u}{\partial x}+v\frac{\partial u}{\partial y}+w\frac{\partial u}{\partial z}=-\frac{\partial p}{\partial x}+\frac{1}{Re}\left(\frac{\partial^2u}{\partial x^2}+\frac{\partial^2u}{\partial y^2}+\frac{\partial^2u}{\partial z^2}\right)$$

**The pressure-gradient term carries a coefficient of exactly 1** — no ½, no ρ.
That fixes the scaling. For the equation to be non-dimensional in this form, the
pressure must have been scaled as

$$p^{*}=\frac{p}{\rho U_\infty^{2}}$$

whereas the aerodynamic pressure coefficient our solver reports is

$$C_p=\frac{p-p_\infty}{\tfrac{1}{2}\rho U_\infty^{2}}=2\,(p^{*}-p^{*}_\infty)$$

$$\boxed{\;\text{predicted } a=\frac{C_p^{\text{solver}}}{p^{*}_{\text{dataset}}}=2.00\;}$$

This is **F07 candidate #3** ("missing factor of ½"), and it is now sourced from
the originating paper's own equations rather than guessed.

### How it compares to what we measure

| Measurement | `a` | vs 2.00 |
|---|--:|--:|
| [[results/P1-F03_solver-qualification]], well-fitted cases (ρ ≥ 0.99, n=4) | **1.8765 ± 0.0098** | −6.2% |
| F03, all 15 envelope cases | 1.739 | −13% |
| EXP-040 `sim_0` (Year-1) | 1.78 | −11% |

**A factor of 2 is the leading explanation**, and the cleanest measurement sits
6.2% below it.

⚠️ **Not yet confirmed.** The equations establish the convention used *in the
paper's exposition*. Whether the HDF5 columns store `p*` on that convention —
rather than, say, a re-scaled or re-referenced quantity — has **not** been
verified. Confirming it requires the Lino et al. (2022) description of the
Nektar++ setup, or a direct dimensional check against a known case.

---

## 5. ⚠️ The "u / 1.8" scaling could not be sourced

`DGN_Docs/reports/validation_report.md:118` states the observed `a ≈ 1.78` is

> *"strikingly close to the dataset's documented 'u / 1.8' velocity scaling,
> which gives a credible analytic explanation rather than a solver artefact."*

**This search found no such documentation.** Specifically:

| Source | Result |
|---|---|
| HuggingFace `mariolinov/Ellipse` card | empty — no README content at all |
| ICLR 2025 paper, full text | no occurrence of `1.8` as a scaling; only in results tables |
| `dgn4cfd` repository page | no normalisation documentation |
| `graphs4cfd` repository page | no normalisation documentation |

The Physics of Fluids 2022 paper has **not** been checked (paywalled; no arXiv
preprint located), so the claim is **unverified rather than disproven**.

**This matters disproportionately.** Three of F07's ten pre-registered
candidates — #1 (`q_ref = 1.8·½ρU²` → 1.80), #2 (`U_ref = 4/3` → 1.78) and #5
(velocity scaled 1.8 → 3.24) — are motivated by that 1.8. If it has no source,
those candidates lose their basis, and candidate #3 (factor of 2, §4 above)
becomes the only one with a documented derivation.

**Action for [[specs/P1-F07_normalisation-audit]]:** obtain Lino et al. (2022)
and either substantiate the 1.8 or record it as an unsourced Year-1 assumption.
This is a library request, not a compute task.

---

## 6. What is still unknown

| Unknown | Why it matters | How to close it |
|---|---|---|
| Top/bottom boundary condition in the source simulations | Decides whether our open-domain case matches the data's physics. [[results/P1-F06_schafer-turek]] L2 shows a duct-vs-open difference of 45% at high blockage | Lino et al. (2022) |
| Whether the HDF5 stores `p*` or something re-referenced | Decides whether the factor-of-2 prediction applies as stated | Lino et al. (2022), or a dimensional self-check |
| Definition of `p_∞` (reference pressure location) | Sets the affine offset `b`, and could contribute to the residual 6.2% | Lino et al. (2022) |
| Whether the 1.8 exists at all | §5 | Lino et al. (2022) |

**Every open question points at the same document.** Obtaining Lino et al.
(2022), *Physics of Fluids* 34(8), doi:10.1063/5.0097679 is the highest-value
next action for F07 — higher than any further solver run.

---

## 7. What this already changes

**In the paper**
- §3 must state that the training data comes from **Nektar++ spectral/hp
  element** simulations at roughly twice the released resolution, and that our
  grading solver is a *different method* (lattice Boltzmann), not a different
  configuration of the same one.
- §3 can cite Table 2 for the designed OOD structure rather than describing it
  as observed.
- §4.6 can state that AoA10 is a **rotation** on the source's own authority.

**In the plan**
- [[specs/P1-F07_normalisation-audit]]: candidate #3 is promoted — it now has a
  derivation from the source. Candidates #1, #2 and #5 are on notice pending
  §5.
- [[results/P1-F03_solver-qualification]]'s well-fitted `a` = 1.8765 should be
  read against **2.00**, not against 1.74. The residual is −6.2%, and
  [[results/P1-F06_schafer-turek]]'s open-vs-duct and fp32 findings are the two
  standing candidates for it.
