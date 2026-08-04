---
tags: [report, presentation, aerodynamic, cfd, openlb, diffusion, inverse-design]
title: "Diffusion-based inverse design of aerodynamic shapes, validated against an independent lattice-Boltzmann solver"
date: 2026-06-17
---

# Diffusion-based inverse design of aerodynamic shapes

**Validated against an independent lattice-Boltzmann solver.**

## Abstract

We build a graph-diffusion generative model that, conditioned on a target surface-pressure
curve `Cp(x/c)`, produces the 2D shape that should carry it — an *inverse-design* generator.
The model denoises the radial field `r(θ)` of a closed boundary using a graph-network score
function, with the target pressure encoded as a fixed-size DCT (discrete cosine transform) spectrum and injected through a jointly-trained DeepSets [4] head. The decisive contribution of this work is *independent physical validation*: generated shapes run through a from-scratch 2D OpenLB lattice-Boltzmann (LBM) CFD solve, and its measured `Cp` is compared to the target the shape was conditioned on a check that does not use the model's own pressure head. The loop reproduces dataset pressure on known ellipses at correlation 0.998, and grades generated shapes at correlation ≈ 0.95 in-distribution. Detail lives in [[EXP-040_openlb_cp_baseline]] and
[[EXP-041_openlb_generated_inverse_design]].

---

## 1. Theory & architecture

### Denoising diffusion on graph node features

The generator is a denoising diffusion probabilistic model (DDPM) [1] operating on graph node
features. A clean boundary is represented by its radial field `r(θ)` sampled at the surface nodes. The forward process gradually corrupts it with Gaussian noise over `T` steps; in closed form, the marginal at step `t` is

> q(x_t | x_0) = N( x_t ; √(ᾱ_t) · x_0 , (1 − ᾱ_t) · I ),

where `ᾱ_t` is the cumulative product of the schedule. We use a cosine `ᾱ_t` schedule [2], which
spends fewer steps in the near-pure-noise regime than the original linear schedule. A learned
network reverses this process, denoising `x_T ~ N(0, I)` back to a shape. The schedule buffers
(`betas`, `alphas_cumprod`, …) are owned by `NoiseSchedule`, registered as module buffers so they move with the device and are saved in checkpoints.

The network is trained to predict the injected noise ε. The objective is the standard simple loss

> L = E_{t, x_0, ε} [ ‖ ε − ε_θ(x_t, t, c) ‖² ],

an MSE between the true and predicted noise (`GraphDiffusionModel.compute_loss`), with `c` the conditioning context.

### Graph-network score model

The score network `ε_θ` (`ScoreNetwork`) stacks `n_layers` Battaglia-style graph-network blocks
[3]. Each `GraphNetworkBlock` applies the canonical three-stage update `EdgeModel (φ^e) → NodeModel (φ^v) → GlobalModel (φ^u)`, aggregating messages along the bidirectional ring edges of the boundary. The timestep is mapped to a sinusoidal embedding 
(`SinusoidalTimeEmbedding`) and the conditioning vector is appended; crucially both are injected
*once* into the per-graph global attribute `u`, not concatenated onto every node, and then propagate to the nodes through the `GlobalModel → NodeModel` message-passing pathway. The model is implemented in the `graph_diffusion` package under a strict one-way dependency DAG
(`building_blocks/ → model/`, with `data/` independent of both).

### Pressure conditioning

The target steady-mean pressure curve `Cp(x/c)` is encoded as the first `K = 8` type-II DCT
coefficients. This makes the conditioning vector fixed-size and independent of node count, and
captures the smooth low-frequency morphology of the pressure curve in a few numbers. To force the model to actually learn the forward shape→pressure map (rather than memorise training pairs), a DeepSets [4] pressure head `h(x̂_0, pos) → ℝ^K` (`PressurePredictionHead`) is trained jointly: it must predict the DCT spectrum of the pressure that the *generated* shape would produce, supplying an auxiliary gradient that ties geometry to pressure.

### Inference amplification

At sampling time the conditioning signal is amplified with classifier-free guidance [5]. The model
is trained with the condition randomly dropped, and at inference the conditional and unconditional noise predictions are combined,

> ε = (1 + w) · ε_cond − w · ε_uncond,

with guidance weight `w` (here `w = 3.0`). For out-of-distribution targets we optionally fall back
to Diffusion Posterior Sampling [6], which steers the reverse process toward the measured target
via a likelihood gradient.

### Independent validation solver

The validation loop is a 2D incompressible lattice-Boltzmann solve [7] built on OpenLB [8]. The
generated boundary polygon is stamped into the lattice as a solid via a winding-number signed-
distance indicator, with a curved Bouzidi no-slip body wall, a velocity inlet, free-stream-velocity
Dirichlet top/bottom, and a pressure outlet. A graded `SpongeLayerDynamics` absorbing zone [9] sits upstream of the outlet to suppress reflected acoustic waves. The solver measures time-averaged surface `Cp(x/c)` from first principles and never sees the diffusion model.

---

## 2. The pipeline

![[assets/report_pipeline_schematic.png]]

The inverse-design loop runs left to right: a **target `Cp(x/c)`** conditions the **conditional
GN-DDPM** (EXP-022), which samples a **generated boundary** `r(θ)`; that boundary is meshed and run
through the **2D OpenLB LBM solver** (validated in EXP-040), which **measures** the shape's true
`Cp`; finally we **correlate** measured against target. The crucial property is that the OpenLB step grades the generator *independently* — it uses no part of the model, so a high measured-vs-target correlation is a genuine physical confirmation of inverse design, not the model's pressure head grading itself.

---

## 3. Dataset

The training data is the pOnEllipse dataset [10]: 2D incompressible flow around an elliptical
cylinder, major-axis length 1 (semi-major 0.5), variable minor axis `b` and channel height `H`,
at Reynolds numbers 500–1000 — the laminar vortex-shedding regime. It comprises **5701 training simulations**, each storing per-surface-node coordinates and 101 unsteady pressure timesteps, which we collapse to a steady-mean `Cp`.

![[assets/report_dataset_example.png]]

Each sample is a closed elliptical boundary (left) and its surface pressure curve `Cp(x/c)`
(right). The HDF5 layout per node is `[x, y, Re, H, p0..p100]`.

> [!NOTE]
> The fourth column `H` is the **channel height** (≈ 5–6), not angle of attack — it was previously
> mis-labelled AoA, which had downstream consequences (see [[EXP-030_dual_cp]]).

---

## 4. Results

### Baseline validation — EXP-040

Before trusting the loop on generated shapes, we gate it on *known* dataset ellipses: does it
reproduce the dataset's own `Cp`?

![[assets/EXP-040_cp_comparison.png]]

| Sim | Re | `Cp` correlation | affine slope `a` | offset `b` |
|---|---|---|---|---|
| sim_0 | 500 | **0.998** | 1.78 | ≈ 0 |
| sim_2850 | 754 | 0.93 | 1.55 | ≈ 0 |
| sim_5700 | 1000 | 0.92 | 1.70 | ≈ 0 |

The solver reproduces the dataset `Cp` curve, differing only by a single multiplicative normalisation `a ≈ 1.78` — strikingly close to the dataset's documented "u / 1.8" velocity
scaling, which gives a credible analytic explanation rather than a solver artefact. The correlation tapers slightly with Re due to genuine under-resolution of stronger vortex shedding, not a correctness failure.

The headline engineering fix: the instantaneous force trace showed sustained ±15% density
"breathing", which was diagnosed as a **domain quarter-wave acoustic resonance** trapped by the
reflecting boundaries. Removing it with a sponge layer plus rest-initialisation collapsed the
drag-coefficient noise from ±1.5 to ±0.05 and *improved* the `Cp` match (sim_0: 0.98 → 0.998),
confirming the old "Cd noise" was never physics. Full narrative: [[EXP-040_openlb_cp_baseline]].

### Inverse-design validation — EXP-041

We then feed *generated* shapes through the validated loop and grade their measured `Cp` against
the target each was conditioned on.

![[assets/report_generated_shapes.png]]

![[assets/EXP-041_cp_inverse_design.png]]

| Target | Type | Re | `Cp` corr (raw) | `Cp` corr (smoothed) |
|---|---|---|---|---|
| #4849 | in-distribution | 960 | **0.964** | 0.952 |
| #4066 | in-distribution | 884 | **0.966** | 0.963 |
| synthetic | out-of-distribution | 754 | **0.860** | 0.876 |

Generated shapes hit their target `Cp` at correlation **≈ 0.95 in-distribution** and, across a
five-target out-of-distribution spread, **0.70–0.93 (declining with OOD distance)** — verified by
the independent solver, not the model's own head. This is the first independent confirmation of the
inverse-design claim. The OOD drop is exactly the extrapolation failure the loop was built to
surface. Boundary roughness turns out to be cosmetic:
raw versus Fourier-smoothed variants differ by only Δcorr ~0.01 in `Cp` and ~5% in `Cd`, so the
characteristic wobble does not corrupt the physics. Full detail:
[[EXP-041_openlb_generated_inverse_design]].

To check that the out-of-distribution behaviour is general rather than a quirk of one target, we
ran a spread of five synthetic OOD targets (perturbing the dataset-mean `Cp` in amplitude, sign,
frequency, and tilt) and plotted fidelity against each target's distance from the training set:

![[assets/EXP-041_ood_spread.png]]

Correlation declines with OOD distance (0.70–0.93, versus ≈ 0.96 in-distribution), while the
magnitude slope `a` stays elevated (2.3–3.4) for *every* OOD target — the amplitude creep is a
general model property, not specific to one curve. The per-target measured-vs-target overlays:

![[assets/EXP-041_ood_cp_overlays.png]]

> [!WARNING]
> Treat **correlation** (morphology) as the inverse-design verdict. The model captures `Cp` shape,
> not absolute amplitude — the affine slope on generated targets (2.2–3.6) exceeds the dataset
> factor (~1.78), a known limitation from [[EXP-020_fourier_pressure_conditioning]].

### Does posterior guidance help out-of-distribution? — DPS sweep

The natural lever for the OOD gap is Diffusion Posterior Sampling [6], which adds a likelihood
gradient `∇‖h(x̂_0, pos) − c_target‖²` from the model's own pressure head at each reverse step.
We swept its weight on the single synthetic OOD target — same seed, so shapes differ *only* by the guidance — and graded each by the independent solver.

![[assets/EXP-041_dps_sweep.png]]

| DPS weight | `Cp` corr | affine slope `a` | `Cd` |
|---|---|---|---|
| 0.0 (plain CFG) | **0.860** | 3.46 | 1.59 |
| 0.3 | 0.818 | 5.22 | 2.46 |
| 0.5 | 0.807 | 5.32 | 2.33 |
| 1.0 | 0.762 | 6.48 | 3.10 |

The result is an unambiguous **negative**: every metric degrades monotonically with guidance
strength — correlation falls, the magnitude slope blows up from 3.5 to 6.5, and the body inflates
(drag nearly doubles). DPS steers toward the *self-trained head's* idea of the target, but OOD that
head is unreliable, so stronger guidance optimises the surrogate's own error. **Plain classifier-free guidance (`dps = 0`) is the correct OOD configuration**; posterior guidance is the wrong knob. This rules out a cheap surrogate-only fix and motivates true CFD-in-the-loop optimisation for Phase 2.

### Flow-field visualisation (ParaView)

Instantaneous OpenLB flow fields around a generated in-distribution shape, rendered in ParaView, give the qualitative picture behind the validation numbers.

![[assets/report_paraview_velocity_wake.png]]

**Velocity magnitude.** The laminar boundary layer separates off the elliptical body and rolls up
into a regular **von Kármán vortex street** — the alternating wake that defines the Re 500–1000
shedding regime the dataset was built in. The accelerated flow over the shoulders (yellow) and the low-momentum wake (dark) are the signature of the steady-mean pressure distribution the model is conditioned on.

![[assets/report_paraview_pressure.png]]

**Pressure field.** The same instant, coloured by pressure: a high-pressure stagnation region on the
upstream nose (red) and a train of discrete low-pressure vortex cores shed into the wake (blue). The surface trace of exactly this field *is* the `Cp(x/c)` curve that the inverse-design loop grades —
making concrete what the correlation numbers above are measuring.

---

## 5. Conclusions & next steps

**Phase 1 (physical validation) is complete.** An independent lattice-Boltzmann solver reproduces
the dataset pressure on known ellipses (corr 0.998) and confirms that diffusion-generated shapes
carry their target `Cp` in-distribution (corr ≈ 0.95). The inverse-design loop is therefore
*trustworthy in-distribution*, which unblocks **Phase 2**: optimisation directly in the model's
gradient / latent space with OpenLB as ground truth.

Near-term:

- The **DPS sweep** has settled the OOD question: posterior guidance *degrades* true fidelity
  (it overfits the model's own pressure head), so Phase 2 will use plain CFG for sampling and put
  the optimisation pressure on CFD-in-the-loop rather than the surrogate head.
- The **`Cp` magnitude creep** has been traced to the model: a roughness-, node-density- and
  resolution-controlled study ([[EXP-042_ood_fidelity_magnitude_creep]]) shows the affine slope
  `a` is unmoved by any CFD/discretisation lever and is already ~2.8–3.0 *in-distribution* (vs the
  solver's validated 1.78). It is a generative amplitude bias (pressure-head calibration), not a
  solver artefact — so it will be addressed model-side, with the solver trusted as ground truth.
- Pinning the `a ≈ 1.78` normalisation analytically from the dataset card.

---

## References

[1] J. Ho, A. Jain, P. Abbeel. "Denoising Diffusion Probabilistic Models." *NeurIPS*, 2020.

[2] A. Nichol, P. Dhariwal. "Improved Denoising Diffusion Probabilistic Models." *ICML*, 2021.

[3] P. W. Battaglia et al. "Relational inductive biases, deep learning, and graph networks."
*arXiv:1806.01261*, 2018.

[4] M. Zaheer, S. Kottur, S. Ravanbakhsh, B. Póczos, R. Salakhutdinov, A. J. Smola. "Deep Sets."
*NeurIPS*, 2017.

[5] J. Ho, T. Salimans. "Classifier-Free Diffusion Guidance." *NeurIPS Workshop / arXiv:2207.12598*,
2022.

[6] H. Chung, J. Kim, M. T. Mccann, M. L. Klasky, J. C. Ye. "Diffusion Posterior Sampling for
General Noisy Inverse Problems." *ICLR*, 2023.

[7] T. Krüger, H. Kusumaatmaja, A. Kuzmin, O. Shardt, G. Silva, E. M. Viggen. *The Lattice
Boltzmann Method: Principles and Practice.* Springer, 2017.

[8] J. Latt et al. "Palabos / OpenLB: lattice Boltzmann based open-source software."
*Computers & Mathematics with Applications*, 2021.

[9] X. Xu, P. Sagaut. "Analysis of the absorbing layers for the weakly-compressible lattice
Boltzmann methods." *Journal of Computational Physics*, 245, 2013.

[10] M. Lino, S. Fotiadis, A. A. Bharath, C. D. Cantwell. "Multi-scale rotation-equivariant graph
neural networks for unsteady Eulerian fluid dynamics." *Physics of Fluids*, 34, 087110, 2022.

---

*Related experiment logs:* [[EXP-040_openlb_cp_baseline]] · [[EXP-041_openlb_generated_inverse_design]]
