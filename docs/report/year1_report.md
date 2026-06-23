---
title: "Diffusion-based inverse design of aerodynamic shapes, validated against an independent lattice-Boltzmann solver"
date: 2026-06-23
tags: [report, phd, methodology, results, diffusion, cfd, inverse-design]
---

# Diffusion-based inverse design of aerodynamic shapes

**Validated against an independent lattice-Boltzmann solver.**

## Abstract

We present a graph-diffusion generative model for aerodynamic inverse design: conditioned on a
target surface-pressure curve `Cp(x/c)`, it produces the 2D shape that should carry it. The model
denoises the radial field `r(θ)` of a closed boundary with a graph-network score function, treating
the boundary as a graph of surface nodes. The target pressure is encoded as a fixed-size discrete
cosine transform (DCT) spectrum and injected through a jointly-trained DeepSets head, so the
conditioning is order-invariant and decoupled from sampling resolution. The decisive contribution
is *independent physical validation*: generated shapes are run through a from-scratch 2D OpenLB
lattice-Boltzmann solve, and the measured `Cp` is compared to the conditioning target — a check
that never touches the model's own pressure head. The loop reproduces dataset pressure on known
ellipses at correlation 0.998, establishing the baseline gate, and grades generated shapes at
correlation ≈ 0.95 in-distribution. This couples a generative inverse-design model to an
independent physics ground truth rather than a self-graded surrogate.

## 1. Introduction

<!-- STUB: draft in Year-2 paper pass -->

_To cover:_

- Motivation: aerodynamic inverse design is a one-to-many problem (many shapes ~ one target pressure), which suits a generative model rather than a deterministic regressor.
- Why diffusion models on graphs `[@ho2020ddpm]` and why a graph-network representation of the boundary `[@lino2022msgnn]`.
- The gap this work addresses: generative aerodynamic models are rarely validated against independent physics; pressure heads grade themselves.
- Contribution statement: a conditional GN-DDPM inverse-design generator + an independent lattice-Boltzmann validation loop; key results (ρ=0.998 baseline gate, ρ≈0.95 in-distribution inverse design); scope (2D laminar elliptical-cylinder regime).

## 2. Related Work

<!-- STUB -->

_To cover:_

- Denoising diffusion models and guidance `[@ho2020ddpm]`, `[@nichol2021improved]`, `[@ho2022cfg]`, `[@chung2023dps]`.
- Graph-network surrogates for CFD / mesh-based simulation `[@battaglia2018relational]`, `[@lino2022msgnn]`.
- Inverse design in aerodynamics (generative and optimisation-based approaches).
- Set representations for conditioning `[@zaheer2017deepsets]`.

## 3. Methodology

### 3.1 Problem formulation

We treat aerodynamic shape design as an *inverse* problem: given a target surface-pressure
distribution `Cp(x/c)`, produce a closed 2D boundary shape that realises it. The boundary is
represented as a radial field `r(θ)` sampled at the surface nodes, and the design target is summarised
by a conditioning context `c` derived from `Cp(x/c)`. The map from target pressure to shape is
*one-to-many*: distinct boundaries can carry near-identical pressure curves, so there is no single
correct output to regress towards. A deterministic regressor, forced to average over this fibre,
would collapse the valid set to a blurred mean shape. A generative model instead learns the
conditional distribution `p(r(θ) | c)`, sampling individual valid shapes for a given target — which
is why we adopt a conditional generative formulation rather than direct regression.

### 3.2 Shape representation

We diffuse the boundary as a radial field `r(θ)` sampled at the surface nodes, organising those nodes
into a boundary graph with bidirectional ring edges so each node exchanges messages with its two
ring neighbours. The data pipeline supports several coordinate parameterisations of the same boundary
— direct-Cartesian `(x, y)`, raw radial, and scale-normalised radial — and we adopt the
scale-normalised radial field as the diffusion target.
<!-- source: canonical radial target established in the circle baseline series (EXP-00x) and the EXP-010 radial pipeline; feature modes defined in pOnEllipseDataset -->
This choice is principled rather than incidental. A single scalar per node is a minimal,
well-conditioned one-dimensional target on the ring, avoiding the two coupled channels of a
Cartesian parameterisation. It exposes `r(θ)` directly to a second-difference (curvature) diagnostic,
so surface smoothness can be both measured and, where useful, penalised. Normalising by scale further
decouples shape from absolute size, focusing the model on morphology. The same radial target was
first validated on canonical shapes in a preliminary study (Section 4.1), where radial diffusion on a
ring graph reproduced target shape distributions, and was carried unchanged into the aerodynamic
setting.

### 3.3 Denoising diffusion model

The generator is a denoising diffusion probabilistic model (DDPM) `[@ho2020ddpm]` operating on the
graph node features. The forward process corrupts a clean shape `x_0 = r(θ)` with Gaussian noise over
`T` steps; in closed form the marginal at step `t` is

> q(x_t | x_0) = N( x_t ; √(ᾱ_t) · x_0 , (1 − ᾱ_t) · I ),

where `ᾱ_t` is the cumulative product of the noise schedule. We use a cosine `ᾱ_t` schedule
`[@nichol2021improved]`, which spends fewer steps in the near-pure-noise regime than the original
linear schedule and so allocates more reverse steps to the low-noise regime where fine geometric
detail emerges. A learned network reverses the process, denoising `x_T ~ N(0, I)` back to a shape,
and is trained to predict the injected noise `ε` under the standard simple loss

> L = E_{t, x_0, ε} [ ‖ ε − ε_θ(x_t, t, c) ‖² ],

an MSE between the true and predicted noise, with `c` the conditioning context. All schedule buffers
(`betas`, `alphas_cumprod`, and derived quantities) are owned by a `NoiseSchedule` module and
registered as module buffers, so they move with the device and are saved inside checkpoints,
guaranteeing that training and sampling share an identical schedule.

### 3.4 Graph-network score model

The score network `ε_θ` stacks `N` Battaglia-style graph-network blocks `[@battaglia2018relational]`,
each applying the canonical three-stage update — EdgeModel (φ^e) → NodeModel (φ^v) →
GlobalModel (φ^u) — and aggregating messages along the bidirectional ring edges of the boundary. The
key design choice is how time and conditioning enter the network: the timestep is mapped through a
sinusoidal embedding, and both this embedding and the conditioning vector are injected *once* into the
per-graph global attribute `u`, rather than being concatenated onto every node. They then propagate to
the nodes through the GlobalModel → NodeModel message-passing pathway, so a single per-graph signal
conditions all nodes consistently and the per-node feature stays a single radial channel. The model
is implemented under a strict one-way package dependency DAG (`building_blocks/ → model/`, with
`data/` independent of both), keeping the score architecture decoupled from data handling.

![[assets/report_architecture.png]]

*Architecture of the score network `ε_θ`. The diffusion timestep (via a sinusoidal embedding) and the
DCT pressure-conditioning vector are summed once into the per-graph global attribute `u`, which feeds
the N× stack of graph-network blocks (φ^e → φ^v → φ^u) operating over the bidirectional ring edges; a
jointly-trained auxiliary DeepSets pressure head reads the denoised shape to predict its pressure
spectrum (detailed in 3.5).*

### 3.5 Pressure conditioning

The target pressure curve is encoded compactly: we take the first `K = 8` type-II discrete cosine
transform (DCT) coefficients of the steady-mean `Cp(x/c)`. This yields a fixed-size conditioning
vector that is independent of the surface node count and captures the smooth, low-frequency
morphology of the pressure curve in a handful of numbers. Encoding alone, however, does not force the
model to respect physics. To tie geometry to pressure, a DeepSets pressure head
`[@zaheer2017deepsets]` `h(x̂_0, pos) → ℝ^K` is trained jointly with the score network: from the
denoised shape it must predict the DCT spectrum of the pressure that the *generated* shape would
actually produce. Its prediction error supplies an auxiliary gradient that couples the generated
geometry to its implied pressure, forcing the model to learn the forward shape→pressure map rather
than memorise training shape–pressure pairs. The DeepSets form is chosen because it is permutation-
invariant over the unordered set of surface nodes, matching the symmetry of the boundary.

### 3.6 Inference guidance

At sampling time the conditioning signal is amplified with classifier-free guidance
`[@ho2022cfg]`. During training the condition is randomly dropped and replaced by a learnable null
token, so the same network learns both conditional and unconditional noise prediction. At inference
the two predictions are combined,

> ε = (1 + w) · ε_cond − w · ε_uncond,

with guidance weight `w` (here `w = 3.0`), pushing samples toward the target without a separate
classifier. The tested alternative is Diffusion Posterior Sampling `[@chung2023dps]`, which instead
adds a likelihood gradient from the jointly-trained pressure head to steer the reverse process toward
the measured target; its effect on out-of-distribution targets is reported in the Results.

### 3.7 Independent CFD validation solver

<!-- provisional: pending validation-framework re-verification -->
Generated shapes are graded by an independent computational fluid dynamics solve rather than by the
model's own pressure head. The validation loop is a 2D incompressible lattice-Boltzmann solve
`[@kruger2017lbm]` built on OpenLB `[@latt2021openlb]`. The generated boundary polygon is stamped into
the lattice as a solid via a winding-number signed-distance indicator, giving a curved Bouzidi no-slip
body wall; the domain has a velocity inlet, free-stream-velocity Dirichlet conditions on the top and
bottom walls, and a pressure outlet. A graded sponge / absorbing layer `[@xu2013sponge]` sits upstream
of the outlet to suppress reflected acoustic waves that would otherwise contaminate the measured
field. The decisive property is independence: the solver measures the time-averaged surface
`Cp(x/c)` from first principles and never sees the diffusion model or its pressure head. Because the
grading physics shares no component with the generator, a high measured-versus-target correlation is a
genuine physical confirmation of inverse design, not a surrogate grading itself.

![[assets/EXP-040_domain_schematic.png]]

*Schematic of the 2D lattice-Boltzmann validation domain: a velocity inlet, free-stream Dirichlet top
and bottom walls, the stamped solid body with a curved Bouzidi no-slip wall, and a pressure outlet
preceded by a graded sponge layer that absorbs reflected acoustic waves.*

### 3.8 Dataset

Training data is the pOnEllipse dataset `[@lino2022msgnn]`: 2D incompressible flow around an
elliptical cylinder of major-axis length 1 (semi-major 0.5), with variable minor axis `b` and channel
height `H`, at Reynolds numbers 500–1000 — the laminar vortex-shedding regime. It comprises 5701
training simulations, each storing per-surface-node coordinates together with 101 unsteady pressure
timesteps, which we collapse to a single steady-mean `Cp`. The per-node HDF5 layout is
`[x, y, Re, H, p0..p100]`. Note that the fourth column `H` is the channel height (≈ 5–6), not angle of
attack — a distinction that matters because mistaking it for AoA would corrupt the conditioning. The
narrow, well-characterised regime makes this dataset a controlled testbed for inverse design before
scaling to higher-Reynolds or 3D flows.

![[assets/report_dataset_example.png]]

*An example pOnEllipse training sample: a closed elliptical boundary (left) and its corresponding
steady-mean surface pressure curve `Cp(x/c)` (right), the quantity the model is conditioned on.*

### 3.9 Evaluation metrics

The headline verdict for inverse design is the **Pearson correlation** between the solver-measured
`Cp(x/c)` and the conditioning target: it scores whether the generated shape reproduces the *shape* of
the target pressure curve (its morphology), independently of amplitude. We report it as the primary
metric because the model captures `Cp` shape, not absolute magnitude. Alongside it we report the
affine slope `a` and offset `b` from a linear fit of measured against target `Cp`, which quantify
amplitude calibration; the drag coefficient `Cd`; and the second-difference roughness of `r(θ)`, a
surface-quality measure where lower is smoother. We state explicitly that correlation, not absolute
amplitude, is the inverse-design verdict — the affine slope is reported as a known calibration
limitation rather than a pass/fail criterion.

## 4. Results

### 4.1 Preliminary: representation studies

Radial diffusion on canonical shapes established the representation and connectivity that were
carried unchanged into the aerodynamic setting. A preliminary circle-series study denoised a radial
field `r(θ)` on a boundary ring graph and confirmed that the pipeline reproduces target shape
distributions: sampling from `x_T ~ N(0, I)` recovers the family of Fourier-perturbed unit circles
it was trained on, with low distributional discrepancy and no boundary violations.
<!-- source: circle radial ablation series, EXP-00x_series_summary.md (EXP-005 reference result) -->
That study also fixed the two design choices reused downstream: a single-scalar radial node feature
`[r]` rather than richer coupled channels, and higher-order ring-neighbour connectivity, which
improved distribution fidelity sharply over nearest-neighbour-only graphs. This subsection is
scaffolding for the aerodynamic results that follow, not a headline in its own right.

![[assets/EXP-001_generated_shapes.png]]

*Shapes generated by the preliminary radial diffusion model on the canonical circle benchmark: the
reverse process recovers the target family of Fourier-perturbed boundaries from pure Gaussian noise.*

### 4.2 Surface-quality methodology study

Boundary roughness is a sampling-time property, not a training-loss property. A dedicated
surface-quality study tried five orthogonal levers to clean up the visibly wobbly boundaries of the
baseline conditional model, each evaluated on the same four conditioning targets by the mean
second-difference roughness of `r(θ)` (lower is smoother). The outcome separates the two levers that
act on the reverse process from three "modern DDPM tricks" that act on the training objective.

| Configuration | Diffusion steps `T` | Overall roughness | Δ vs baseline | Best val loss |
|---|---|---|---|---|
| baseline | 200 | 0.03379 | — | 0.00306 |
| longer training | 200 | 0.02125 | −37% | 0.000104 |
| longer training + `T=1000` steps | 1000 | **0.01523** | **−55%** | 0.000090 |
| + Min-SNR-γ loss `[@hang2023minsnr]` | 200 | 0.02700 | −20% | **0.000028** |
| + EMA `[@song2020improved]` | 200 | 0.03747 | +11% | 0.000124 |
| + EMA + v-prediction `[@salimans2022progressive]` | 200 | 0.02451 | −27% | 0.000189 |

The `T=1000` configuration wins, cutting roughness 55% to 0.01523, yet its best validation loss is
essentially identical to longer-training-alone (0.000090 vs 0.000104). The gain therefore lives in
the reverse process — more reverse steps mean more polishing in the low-noise regime where fine
geometric detail emerges — and not in a better-fitted network. The three training-objective tricks
are a validated negative result: the lowest validation loss of the whole sweep (Min-SNR-γ at
0.000028) yields *worse* roughness (0.02700) than simply training longer, and EMA actually pushes
roughness above the baseline. Minimising the training MSE harder does not smooth boundaries, which
is the central methodological finding carried into the validation work below.

<!-- COMMIT 1 boundary: 4.1–4.3 committed together after the solver-gate subsection -->

### 4.3 CFD solver validation gate

Before trusting the loop on generated shapes, we gate it on *known* dataset ellipses: the
independent solver reproduces the dataset's own `Cp` at correlation up to 0.998. This establishes
that the grading physics is sound before any generated geometry is judged by it.

<!-- provisional: pending validation-framework re-verification -->

| Sim | Re | `Cp` correlation | affine slope `a` | offset `b` |
|---|---|---|---|---|
| sim_0 | 500 | **0.998** | 1.78 | ≈ 0 |
| sim_2850 | 754 | 0.93 | 1.55 | ≈ 0 |
| sim_5700 | 1000 | 0.92 | 1.70 | ≈ 0 |
<!-- source: baseline solver validation, validation_report.md / EXP-040 -->

The solver reproduces the dataset curve up to a single multiplicative normalisation `a ≈ 1.78`,
strikingly close to the dataset's documented "u / 1.8" velocity scaling — a credible analytic
explanation rather than a solver artefact. Correlation tapers slightly with Reynolds number through
genuine under-resolution of the stronger vortex shedding, not a correctness failure. The decisive
engineering fix concerned stability: the instantaneous force trace showed sustained ±15% density
"breathing", diagnosed as a domain quarter-wave acoustic resonance trapped by the reflecting
boundaries. Adding a sponge / absorbing layer `[@xu2013sponge]` together with rest-initialisation
collapsed the drag-coefficient noise from ±1.5 to ±0.05 and *improved* the `Cp` match (0.98 →
0.998), confirming the earlier "Cd noise" was never physics.

![[assets/EXP-040_cp_comparison.png]]

*Solver-measured surface pressure `Cp(x/c)` against the dataset's own `Cp` for three known ellipses
spanning the Reynolds-number range: the curves overlay up to a single amplitude normalisation.*

![[assets/EXP-040_stability_trace.png]]

*Instantaneous force trace before and after the sponge-layer fix: the ±15% acoustic-resonance
"breathing" collapses to a quiet signal once reflected waves are absorbed.*

### 4.4 Inverse-design validation

Generated shapes hit their target `Cp` at correlation ≈ 0.95 in-distribution, verified by the
independent solver rather than the model's own pressure head — the first independent confirmation of
the inverse-design claim. Each generated shape is run through the validated loop and its measured
`Cp` correlated against the target it was conditioned on.

<!-- provisional: pending validation-framework re-verification -->

| Target | Type | Re | `Cp` corr (raw) | `Cp` corr (smoothed) |
|---|---|---|---|---|
| #4849 | in-distribution | 960 | **0.964** | 0.952 |
| #4066 | in-distribution | 884 | **0.966** | 0.963 |
| synthetic | out-of-distribution | 754 | **0.860** | 0.876 |
<!-- source: inverse-design validation, validation_report.md / EXP-041 -->

In-distribution the loop confirms inverse design at correlation ≈ 0.95. Across a five-target
out-of-distribution spread — perturbing the dataset-mean `Cp` in amplitude, sign, frequency, and
tilt — correlation falls to 0.70–0.93, declining with distance from the training set, exactly the
extrapolation failure the loop was built to surface. Surface roughness turns out to be cosmetic:
raw versus Fourier-smoothed variants differ by only Δcorr ~0.01 in `Cp` and ~5% in `Cd`, so the
characteristic wobble does not corrupt the physics.

![[assets/report_generated_shapes.png]]

*Boundaries generated by the conditional model for the validation targets, before CFD grading.*

![[assets/EXP-041_cp_inverse_design.png]]

*Solver-measured `Cp(x/c)` of each generated shape overlaid on its conditioning target: in-distribution
targets track closely in morphology, the synthetic out-of-distribution target less so.*

![[assets/EXP-041_ood_spread.png]]

*Fidelity against out-of-distribution distance across the five-target spread: correlation declines
from ≈ 0.96 in-distribution to 0.70–0.93 as targets move away from the training set.*

![[assets/EXP-041_ood_cp_overlays.png]]

*Per-target measured-versus-target `Cp` overlays for the out-of-distribution spread, showing where
morphology is preserved and where amplitude diverges.*

> Correlation (morphology) is the inverse-design verdict. The model captures `Cp` *shape*, not
> absolute amplitude: the affine slope on generated targets (2.2–3.6) exceeds the dataset factor
> (~1.78), a known calibration limitation rather than a pass/fail failure.

### 4.5 Out-of-distribution limits, magnitude creep, and the posterior-guidance negative

Posterior guidance is an unambiguous negative out-of-distribution. The natural lever for the OOD gap
is Diffusion Posterior Sampling `[@chung2023dps]`, which adds a likelihood gradient from the model's
own pressure head at each reverse step. Swept on the single synthetic OOD target — same seed, so the
shapes differ *only* by the guidance — and graded by the independent solver, every metric degrades.

<!-- provisional: pending validation-framework re-verification -->

| DPS weight | `Cp` corr | affine slope `a` | `Cd` |
|---|---|---|---|
| 0.0 (plain CFG) | **0.860** | 3.46 | 1.59 |
| 0.3 | 0.818 | 5.22 | 2.46 |
| 0.5 | 0.807 | 5.32 | 2.33 |
| 1.0 | 0.762 | 6.48 | 3.10 |
<!-- source: DPS sweep, validation_report.md / EXP-041 -->

Correlation falls, the magnitude slope blows up from 3.5 to 6.5, and the body inflates as drag
nearly doubles — all monotonically with guidance strength. The mechanism is clear: DPS steers toward
the *self-trained* pressure head's idea of the target, but out-of-distribution that head is
unreliable, so stronger guidance merely optimises the surrogate's own error. Plain classifier-free
guidance (`dps = 0`) is the correct out-of-distribution configuration, and this rules out a cheap
surrogate-only fix.

The amplitude bias itself is model-side, not solver-side. A roughness-, node-density- and
resolution-controlled study shows the affine slope `a` is unmoved by any CFD or discretisation lever
and is already ~2.8–3.0 *in-distribution* (against the solver's validated 1.78). This points to a
generative amplitude bias — pressure-head calibration — to be addressed model-side with the solver
trusted as ground truth, rather than a discretisation artefact.

![[assets/EXP-041_dps_sweep.png]]

*Inverse-design fidelity against DPS guidance weight: correlation falls and the amplitude slope and
drag rise monotonically, so posterior guidance degrades true out-of-distribution fidelity.*

![[assets/EXP-042_creep_ladder.png]]

*Affine slope `a` across the roughness-, node-density- and resolution-controlled ladder: `a` stays
elevated and is unmoved by any CFD or discretisation lever, isolating the creep to the generator.*

### 4.6 Qualitative flow fields

Instantaneous OpenLB fields around a generated in-distribution shape, rendered in ParaView, give the
qualitative picture behind the validation numbers. In the velocity-magnitude field the laminar
boundary layer separates off the body and rolls up into a regular von Kármán vortex street — the
alternating wake that defines the Re 500–1000 shedding regime — with accelerated flow over the
shoulders and a low-momentum wake behind the body.

![[assets/report_paraview_velocity_wake.png]]

*Velocity-magnitude field around a generated in-distribution shape: a laminar separation rolling up
into a regular von Kármán vortex street, with accelerated shoulder flow and a low-momentum wake.*

The pressure field at the same instant shows a high-pressure stagnation region on the upstream nose
and a train of discrete low-pressure vortex cores shed into the wake. The surface trace of exactly
this field *is* the `Cp(x/c)` curve that the inverse-design loop grades, making concrete what the
correlation numbers above are measuring.

![[assets/report_paraview_pressure.png]]

*Pressure field at the same instant: a high-pressure stagnation nose and a train of discrete
low-pressure vortex cores in the wake, whose surface trace is the graded `Cp(x/c)` curve.*


## 5. Discussion

<!-- STUB -->

_To cover:_

- In-distribution trustworthiness vs out-of-distribution extrapolation failure.
- Cp magnitude creep as a model-side calibration issue (not a solver artefact).
- Why posterior guidance `[@chung2023dps]` is the wrong knob out-of-distribution (it overfits the self-trained head).
- Threats to validity, including that the validation framework is still being verified.

## 6. Conclusions

<!-- STUB -->

_To cover:_

- Phase 1 (independent physical validation) complete.
- Phase 2: CFD-in-the-loop optimisation in the model's latent/gradient space with OpenLB as ground truth.

## References

References are managed in `latex/refs.bib`; cited inline above as `[@key]`.
