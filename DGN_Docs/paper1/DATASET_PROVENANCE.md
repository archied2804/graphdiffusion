---
title: "pOnEllipse dataset — provenance and normalisation, from the source"
date: 2026-08-06
tags: [paper1, methods, dataset, provenance, normalisation]
status: reference
---

# pOnEllipse — what the source actually says

> **What this file is.** Facts about the training dataset, taken from the papers
> that produced it, with citations.
>
> **Two results settle long-standing questions.** The affine slope `a` has a
> **derived prediction of exactly 2.00** from the source's own
> non-dimensionalisation (§4), and the "documented `u / 1.8` velocity scaling"
> the Year-1 work attributed it to is a **misreading** — that scaling belongs to
> a different dataset and is a machine-learning input normalisation, not a
> physical one (§5).
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

### Boundary conditions — settled, from Lino et al. (2022) §III C

> *"These datasets contain simulations of the two-dimensional incompressible
> flow around an elliptical cylinder with major axis a = 1 and minor axis b
> **inside a channel of height H**. … On the left, top and bottom boundaries
> **u = 1 and v = ∂p/∂x = 0**, the right boundary is an outlet with
> ∂u/∂x = ∂v/∂x = 0 and **p = 0**, and the cylinder wall has a no-slip
> condition."*

| Boundary | Source data | `ellipse2d` | |
|---|---|---|:--:|
| inlet | `u = 1, v = 0` uniform | material 3, uniform `(U,0)` | ✅ |
| top **and** bottom | `u = 1, v = 0` uniform | material 2, free-stream `(U,0)` | ✅ |
| outlet | `∂u/∂x = ∂v/∂x = 0`, `p = 0` | material 4, pressure outlet `p = 0` | ✅ |
| body | no-slip | material 5, Bouzidi no-slip | ✅ |

**The production case reproduces the source's boundary conditions exactly.**
The uniform inlet is not a simplification of this dataset — it is what the
dataset actually used. `lattice.h:15` records the uniform inlet as a deliberate
departure from `cylinder2d`; it is in fact a *faithful* reproduction of the
data's own configuration.

Two consequences:

- **The open-domain-versus-duct explanation for `a` is eliminated.** There is no
  BC mismatch to account for.
- [[results/P1-F06_schafer-turek]]'s L2 result (−31% vs the benchmark) is a
  statement about **Schäfer–Turek being a different problem** — a duct with a
  parabolic inlet — not about the production case being misconfigured. L1 is the
  benchmark configuration; **L2 is the configuration that matches our data.**

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

### Both sides of the ratio are now sourced

| | reference pressure | dynamic pressure |
|---|---|---|
| **Dataset** | `p = 0` at the outlet (Lino 2022 §III C) | `ρU²` (coefficient-1 momentum equation) |
| **`ellipse2d`** | `p_inf = 0` (`results.h:484`) | `qInf = 0.5·rho·U·U` (`results.h:380`) |

The reference pressures agree (both zero at the far field / outlet), so the
offset `b` should be ≈ 0 — which is what [[results/P1-F03_solver-qualification]]
measures on the well-fitted cases (`b` = −0.003, +0.012, +0.024, +0.022).

The dynamic pressures differ by exactly the factor of two. **The prediction
`a = 2.00` is now derived from both sides, not inferred from one.**

⚠️ **One step remains unverified**: that the HDF5 columns store `p*` rather than
a re-scaled quantity introduced during the ICLR 2025 re-release (which
*"reduced the resolution by approximately a factor of two"*). Nothing in either
paper suggests a rescaling, but the release step is not documented.

---

## 5. ⚠️ The "u / 1.8" scaling could not be sourced

`DGN_Docs/reports/validation_report.md:118` states the observed `a ≈ 1.78` is

> *"strikingly close to the dataset's documented 'u / 1.8' velocity scaling,
> which gives a credible analytic explanation rather than a solver artefact."*

**The 1.8 exists — but it is the wrong dataset and the wrong kind of quantity.**
Lino et al. (2022) §III **B**, *Incompressible flow around a **circular**
cylinder*:

> *"The u is scaled by dividing by 1.8, and the Reynolds number is linearly
> scaled to have a minimum equal to −1 and a maximum equal to 1 in dataset
> **NsCircle**, and the input edge-attributes e¹ᵢⱼ are normalized by dividing
> them by 0.2."*

Two independent reasons this cannot explain `a`:

1. **Wrong dataset.** It is stated for **NsCircle** — the circular-cylinder
   datasets of §III B. The elliptical-cylinder datasets are §III **C**, which
   contains no scaling statement of any kind. Searching the full text after the
   ellipse section returns no further normalisation statement.
2. **Wrong kind of quantity.** It is a **machine-learning input scaling**,
   listed alongside "Reynolds number linearly scaled to [−1, 1]" and "edge
   attributes divided by 0.2". These normalise *network inputs*; they are not
   the physical non-dimensionalisation of the stored CFD data.

**The Year-1 attribution is therefore a misreading on both counts**, and the
numerical closeness of 1.8 to the observed 1.78 is a coincidence.

**Action for [[specs/P1-F07_normalisation-audit]]:** strike candidates #1, #2
and #5 — all three are motivated by this 1.8 and none has a basis. Candidate
**#3 (factor of two)** is the only one with a derivation from the source, and
§4 now supplies both halves of it.

---

## 6. What is still unknown

Lino et al. (2022) closed all four of the previously open questions. What
remains is the **6.2% gap** between the predicted 2.00 and the measured
1.8765 ± 0.0098.

| Candidate | Direction | Test |
|---|---|---|
| Our solver under-resolves the suction peak, attenuating `\|Cp\|` | correct sign (`a` < 2) and consistent with F03's ρ-dependence: low-ρ cases give *lower* `a` | fp64 + resolution sweep, running |
| fp32 round-off | [[results/P1-F06_schafer-turek]] §3 | same sweep |
| Spectral/hp element vs lattice Boltzmann | unknown sign | not cheaply testable; state as a limitation |
| A rescaling in the ICLR 2025 re-release | unknown | undocumented; would need the release code |

The first two are being measured now. Note the sign works: our LBM at
RES = 100–200 smoothing a suction peak would depress `\|Cp_solver\|` and pull
`a` below 2.00, and F03 §5 already showed `a` rising toward ~1.877 precisely on
the cases where agreement is best.

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
