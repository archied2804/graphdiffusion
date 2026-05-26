---
experiment_id: "EXP-020"
title: "Fourier-DCT pressure conditioning with joint pressure-head loss and CFG"
date: 2026-05-19
status: complete
parent: "EXP-010"
tags: [ellipse, aerodynamic, pressure-conditioning, fourier-dct, classifier-free-guidance, dps, inverse-design, pOnEllipse]
config: "configs/EXP-020_fourier_pressure_conditioning.yaml"
output_dir: "outputs/EXP-020_fourier_pressure_conditioning/"
---

# EXP-020: Fourier-DCT pressure conditioning

> **One-line summary:** Condition the radial DDPM on a fixed-size DCT encoding of the steady-state ⟨Cp⟩(x/c) curve, with a jointly-trained pressure prediction head to drive shape→pressure physics, classifier-free guidance for inference amplification, and optional DPS gradient correction for out-of-distribution targets.

## Motivation

EXP-010/011/012 produced an unconditional radial DDPM that reproduces ellipse-like geometry. EXP-020 turns the pipeline into an **inverse design** tool: given a target pressure distribution, generate a closed boundary shape whose induced pressure matches it. The target may come from a non-ellipse source (e.g. an aerofoil), so the model must generalise beyond simple lookup of the training ellipses.

### Why DCT modes of Cp(x/c)

The HDF5 stores 101 unsteady Cp timesteps per node. We follow Mariolinov's *Diffusion Graph Networks*: collapse to the steady-state mean per node, order by chordwise x/c, and encode as the first K=8 type-II DCT coefficients (orthonormalised). This descriptor is fixed-size and node-count-independent — an aerofoil Cp curve can be encoded the same way and fed in at inference.

### Why a pressure prediction head

A naïve conditional DDPM trained on `(ellipse_shape, ellipse_Cp)` pairs tends to memorise the mapping (target_Cp → nearest training ellipse). Adding a learnable head `h(x̂₀, pos) → ℝ^K` and a joint loss `λ·‖h(x̂₀,pos) − c_target‖²` forces the model to internalise the **forward** map (shape → pressure), so any shape that satisfies the pressure target — including shapes outside the training distribution — becomes a valid generation. The head is only active at lower-noise timesteps (t ≤ T/2) where x̂₀ is reliable.

### Why CFG and DPS

- **CFG** with `p_uncond = 0.15` and inference scale `w = 3.0` lets us amplify the conditioning signal at sampling time without overfitting at training time.
- **DPS** (`dps_guidance_weight > 0`) optionally injects `∇‖h(x̂₀,pos) − c_target‖²` into the reverse step. This is the safety valve for aggressive OOD targets (aerofoil Cp); off by default.

## Hypothesis

With the joint head loss + CFG, conditioning on distinct training Cp curves should produce visibly distinct generated shapes (not noise variance around a mean ellipse), and a synthetic asymmetric Cp target should produce a vertically-asymmetric shape — the first step toward lift-producing geometry.

## Setup

- Dataset: `pOnEllipseConditionalDataset(cond_mode="fourier", k_modes=8)` on `pOnEllipseTrain.h5`. Attaches `Data.cond ∈ ℝ^{1×8}`.
- Score network: same backbone as EXP-010 plus `cond_dim=8`, `p_uncond=0.15`, and learnable `null_cond ∈ ℝ^8`.
- Pressure head: DeepSets `MLP → scatter_mean → MLP` mapping `(x̂₀, pos) → ℝ^8`, jointly trained with `λ_pressure = 0.1` (active for `t ≤ T/2`).
- Inference: `guidance_scale = 3.0`; DPS off by default. Smoke test with `dps_guidance_weight = 0.5` on an aerofoil target.

## Conditioning pipeline

End-to-end walkthrough of how a target pressure curve flows from raw CFD output into the score network, how the joint training objective is assembled, and how inference uses the conditioning to generate matching shapes.

### Stage 1 — Raw Cp → fixed-size descriptor

Implemented in [`pOnEllipseConditional.py:158-198`](../../src/graph_diffusion/data/pOnEllipseConditional.py#L158-L198), specifically `pOnEllipseConditionalDataset._build_graphs`.

For each CFD simulation in the HDF5:

1. **Steady-state collapse.** The HDF5 stores 101 unsteady Cp timesteps per node in columns `[_H5_COL_P_START : _H5_COL_P_START + 101]`. We collapse to `cp_mean = cp.mean(axis=-1)`, shape `(N_nodes,)`. This discards transient vortex shedding and keeps only the time-averaged pressure field — the quantity an inverse-design user can specify.

2. **Chordwise reordering.** The raw node order in the HDF5 follows mesh-generation order, not geometry. We compute `x_over_c = (x_c - x_c.min()) / chord` and sort `cp_mean` by `argsort(x_over_c)`. The resulting `cp_ordered` is a 1-D signal indexed by chordwise position — `Cp(x/c)` — which is the canonical aerodynamic plot.

3. **DCT-II encoding.** [`dct_ii`](../../src/graph_diffusion/data/pOnEllipseConditional.py#L54-L81) applies the orthonormalised type-II discrete cosine transform and truncates to the first `K=8` modes:

   ```
   X[k] = sum_{n=0..N-1} cp_ordered[n] * cos(pi * (2n+1) * k / (2N))
   ```

   with `sqrt(2/N)` orthonormalisation (`sqrt(1/N)` for `k=0`). The orthonormalisation makes the encoding **scale-invariant in N**: re-sampling the same physical Cp curve to a different node count produces (approximately) the same K coefficients, so an aerofoil dataset with a different mesh density can be encoded identically without re-tuning.

4. **Attachment.** The K-vector is stored on the graph as `Data.cond ∈ ℝ^{1×K}` (shape `[1, K]` so PyG batches it correctly to `[B, K]`). This is what the score network and the head both see.

**Why DCT, not the raw Cp curve.** A naive `Data.cp = Cp(x/c)` would force the conditioning vector to vary with node count, breaking transfer to aerofoils. K=8 modes capture the dominant low-frequency variation in steady-state Cp curves (the high-frequency content is mostly numerical noise from the CFD solver).

### Stage 2 — Conditioning injection in the score network

Implemented in [`score_network.py:122-126,196-213`](../../src/graph_diffusion/model/score_network.py#L122-L213). Three pieces are added on top of the unconditional backbone:

- **`cond_proj: Linear(K, global_dim)`** — projects the K=8 DCT vector into the same dimensionality as the per-graph global attribute `u`. Built only when `cond_dim` is set in the config.
- **`null_cond: nn.Parameter(zeros(K))`** — a learnable "unconditional" token. CFG needs an explicit unconditional input; this parameter is what the model is trained to interpret as "no conditioning specified."
- **u-level injection** — at every forward pass:

  ```python
  u = u + self.time_proj(t_emb)        # time injection (always)
  u = u + self.cond_proj(cond_input)   # conditioning injection (EXP-020)
  ```

  The conditioning is added to `u` *once*, and then the standard `GraphNetworkBlock` cascade propagates it through `GlobalModel → NodeModel` for `n_layers` rounds, so every node ultimately receives the conditioning signal via message passing. This is cheaper and more inductive-bias-friendly than concatenating the cond vector onto every node.

**CFG training-time dropout** (`score_network.py:197-213`): with probability `p_uncond = 0.15`, each graph in the batch has its `cond` swapped for `null_cond` via a Bernoulli mask. This is what trains the model to produce both `ε_cond` and `ε_uncond` predictions; without it CFG at inference would have no `ε_uncond` to combine. The dropout is per-graph (not per-batch), so each minibatch sees a mix of conditional and unconditional examples.

A `force_uncond=True` flag overrides the dropout and forces the null token regardless of training mode — used during CFG sampling to obtain the unconditional pass.

### Stage 3 — Joint training objective

The total loss is assembled in [`graph_diffusion_model.py:226-283`](../../src/graph_diffusion/model/graph_diffusion_model.py#L226-L283):

```
L_total = MSE(ε_pred, ε_true)                      # standard DDPM noise loss
        + λ_smooth · L_smoothness(x̂₀, batch)       # off (EXP-013 work)
        + λ_pressure · L_head(h(x̂₀, pos), cond)    # EXP-020 pressure-head loss
```

Three things to note about the head term:

1. **`x̂₀` is derived, not directly available.** The score network predicts `ε`. We back out the implied clean estimate:

   ```
   x̂₀ = (x_t − √(1-ᾱ_t) · ε_pred) / √ᾱ_t
   ```

   This is the standard Tweedie reconstruction. At high noise (`t ≈ T`) `√ᾱ_t → 0` so `x̂₀` is dominated by `ε_pred / √ᾱ_t` and is noisy. At low noise (`t ≈ 0`) `√ᾱ_t → 1` and `x̂₀ ≈ x_t` is reliable.

2. **Head loss is half-time-gated.** Because `x̂₀` is unreliable at high noise, the head loss is masked off for `t > T/2`:

   ```python
   half_t = self.noise_schedule.T // 2
   active = (t_idx <= half_t).float()
   head_loss = (active * per_graph_sq).sum() / active.sum()
   ```

   Without this gate the head would receive noisy `x̂₀` early in training, the gradients would dominate, and the head would learn to predict random `cond`. Gating focuses the head's learning on timesteps where it can actually see geometry.

3. **The head is a [`PressurePredictionHead`](../../src/graph_diffusion/model/pressure_head.py)** with DeepSets architecture: `MLP(per-node) → scatter_mean → MLP(per-graph)`. The per-node MLP `φ([x̂₀; pos]) → ℝ^{node_embed_dim}` runs on every node, then `scatter_mean` produces one graph-level vector, then `ρ: ℝ^{node_embed_dim} → ℝ^K` predicts the DCT modes. DeepSets is the right structure because:
   - **Permutation invariance.** Cp modes are a property of the *shape*, not the node ordering. DeepSets is provably the universal architecture for permutation-invariant functions over sets.
   - **Variable node counts.** pOnEllipse graphs have 52–96 nodes; DeepSets handles this with no change.

**Why this prevents memorisation.** Consider what happens *without* the head loss: a conditional DDPM trained on `(ellipse_i, cp_i)` pairs sees each training Cp at most once per epoch. The shortest-path solution is to memorise a hash map `cp_target → ellipse_index`. At inference on a new `cp_target`, the model falls back to the nearest training ellipse — there's no learned forward map. The head loss closes this loophole by forcing every generated `x̂₀` to back-translate into `cond_target` via the head. The model can satisfy this in two ways: copy a training ellipse with matching Cp (the lazy path), or learn shape→Cp physics that generalises (the desired path). With `λ_pressure = 0.1` the lazy path is still partially possible but heavily penalised when the target is OOD.

### Stage 4 — Inference: CFG sampling

Implemented in [`graph_diffusion_model.py:285-460`](../../src/graph_diffusion/model/graph_diffusion_model.py#L285-L460), specifically `GraphDiffusionModel.sample`.

Start from `x_T ~ N(0, I)`. For each step `t = T, T-1, …, 1`:

1. **Build the noisy graph.** `noisy_data.x = x_t` (or `cat([x_t, p_cond], dim=-1)` when `n_noise_channels` indicates partial diffusion).

2. **Conditional noise prediction.** `ε_cond = score_network(noisy_data, t, cond=cond)`.

3. **CFG combination** (when `w ≠ 1` and `null_cond` is set):

   ```
   ε_null = score_network(noisy_data, t, cond=cond, force_uncond=True)
   ε     = (1 + w) · ε_cond − w · ε_null
   ```

   This is the Ho & Salimans (2022) form. Intuitively, `(ε_cond − ε_null)` is the "direction of pull toward the target," and `w` amplifies that direction. At `w = 1` you get just `ε_cond`. At `w = 3.0` (the configured value) the model takes a 3× step in the conditioning direction relative to the unconditional baseline, sharpening the influence of the target Cp.

4. **DDPM reverse step:**

   ```
   x_{t-1} = (1/√α_t) · (x_t − β_t/√(1-ᾱ_t) · ε) + √β_t · z   (z ~ N(0,I), except at the last step)
   ```

5. **Clamp** (radial features): `x_t = x_t.clamp(0.5, 2.0)` per step. This bounds the implicit radius and prevents a small number of outlier nodes from drifting to negative `r` (which would produce a degenerate shape).

### Stage 5 — Optional DPS for OOD targets

When `dps_guidance_weight > 0` ([`graph_diffusion_model.py:419-437`](../../src/graph_diffusion/model/graph_diffusion_model.py#L419-L437)), each reverse step inserts an extra gradient correction:

```
x̂₀_hat = (x_t − √(1-ᾱ_t) · ε) / √ᾱ_t
loss   = ‖h(x̂₀_hat, pos) − cond_target‖²
grad   = ∂loss/∂x̂₀_hat
x̂₀_corr = x̂₀_hat − dps_w · grad / ‖grad‖
ε      = (x_t − √ᾱ_t · x̂₀_corr) / √(1-ᾱ_t)   # rewrite ε from corrected x̂₀
```

Conceptually: we run the head *forward* on the current best estimate of the clean shape, compare against the target Cp, and backprop the mismatch into a correction on `x̂₀`. The correction is L2-normalised so its magnitude is set by `dps_w` independently of how strong the gradient happens to be at that step.

DPS is off by default (`dps_guidance_weight = 0.0`) because the head is already in the training loss, and CFG plus the head loss are usually sufficient for in-distribution and mildly-OOD targets. DPS is the safety valve for aggressive OOD: an aerofoil Cp curve (negative pressure peak, sharp recovery) is far enough outside ellipse Cp curves that the model needs explicit forward-model guidance to stay near the manifold of valid shapes that match the target.

### Tying it together

The conditioning pipeline is intentionally redundant at three levels:

| Where | What enforces "this shape matches the target Cp" |
|---|---|
| Architecture | `cond_proj` adds the target into `u`; message passing propagates to nodes |
| Training | Joint head loss `λ·‖h(x̂₀) − cond‖²` pushes the model toward learning forward physics |
| Inference | CFG amplifies the conditioning direction; DPS (optional) injects explicit forward-model gradients |

Each layer alone is insufficient: pure architectural injection memorises, head loss alone produces weak conditioning at inference, CFG alone doesn't generalise to OOD. The combination is what enables true inverse design.

## Status

Complete — trained 2026-05-26, early-stopped at epoch 79 with best val_loss `0.003058` at epoch 59 (patience=20). Final train_loss `0.004841`. The plan target was val_loss < 0.02; converged 6× below it.

See `docs/superpowers/specs/2026-05-25-exp020-conditioning-visuals-design.md` for the design and `docs/superpowers/plans/2026-05-25-exp020-conditioning-visuals.md` for the executed implementation plan.

## Results

Checkpoint: `outputs/EXP-020_fourier_pressure_conditioning/checkpoint_best.pt` (epoch 59).
Loss log: `outputs/EXP-020_fourier_pressure_conditioning/loss_log.json`.

Figures (post-processed via `scripts/postprocess_exp020.py`):

| File | What it shows |
|---|---|
| `outputs/EXP-020_fourier_pressure_conditioning/figure_a_conditioning_grid.png` | 3 farthest-first training targets + 1 OOD synthetic asymmetric target, with target Cp / head-predicted Cp ± σ / 4 generated shapes per row. Distinct conditioning vectors produce visibly distinct shapes; the synthetic OOD row shows the model still produces a smooth closed boundary, with the head prediction tracking the asymmetric perturbation. |
| `outputs/EXP-020_fourier_pressure_conditioning/figure_b_trajectory.png` | Forward (top) + reverse (bottom) diffusion filmstrip at t ∈ {0, 1, 4, 14, 53, 199} on the synthetic Cp target. Forward goes from clean ellipse to pure noise; reverse mirrors back from noise to clean ellipse. |
| `outputs/EXP-020_fourier_pressure_conditioning/figure_b_reverse.mp4` / `.gif` | Full-resolution reverse-diffusion animation (200 frames, 25 fps) targeting the synthetic asymmetric Cp. |
| `outputs/EXP-020_fourier_pressure_conditioning/figure_c_cfg_sweep.png` | Same OOD synthetic target sampled at CFG `w ∈ {1, 3, 7}`. Head-predicted Cp follows the target shape; higher `w` introduces more boundary detail without diverging. |

### Reproduce

```bash
python train.py --config configs/EXP-020_fourier_pressure_conditioning.yaml --epochs 200 --device cuda --output outputs/EXP-020_fourier_pressure_conditioning/generated_shapes.png
python scripts/postprocess_exp020.py --experiment-dir outputs/EXP-020_fourier_pressure_conditioning --config configs/EXP-020_fourier_pressure_conditioning.yaml --device cuda --n-samples 4 --n-targets 3 --target-seed 0
```

### Caveats

- The head-predicted Cp magnitude in Figure C is biased ~0.1 below the target across all `w`, but the shape of the curve matches. This is consistent with `lambda_pressure=0.1` weighting the head loss lightly relative to the DDPM MSE — the head learns the right Cp morphology but not absolute amplitude. Worth revisiting in EXP-021 if absolute amplitude matters.
- DPS guidance was not exercised here (`dps_guidance_weight=0.0`). The synthetic target is mild enough that CFG suffices; aerofoil-style OOD targets should retry with `dps_guidance_weight=0.5`.
- The boundary nodes in the HDF5 are not stored in chordwise order; `postprocess_exp020.py:radial_to_xy` sorts by theta before plotting so the closed boundary draws correctly.
