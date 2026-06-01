"""
postprocess_exp020.py — Inverse-design figures for the EXP-020 pipeline
=========================================================================

Loads a trained EXP-020 checkpoint and produces:
  * figure_a_conditioning_grid.png  — n_targets training + 1 OOD synthetic,
                                       n_samples shapes per target.
  * figure_b_trajectory.png         — forward + reverse filmstrip on the
                                       OOD synthetic target.
  * figure_b_reverse.mp4 / .gif     — animated reverse diffusion.
  * figure_c_cfg_sweep.png          — same OOD target, w in {1, 3, 7}.

Usage:
    python scripts/postprocess_exp020.py \\
        --experiment-dir outputs/EXP-020_fourier_pressure_conditioning \\
        --config configs/EXP-020_fourier_pressure_conditioning.yaml \\
        --device cuda \\
        --n-samples 4 \\
        --n-targets 3 \\
        --target-seed 0
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data

from graph_diffusion.data.pOnEllipseConditional import (
    dct_ii,
    dct_ii_inverse,
    pOnEllipseConditionalDataset,
)
from graph_diffusion.model.graph_diffusion_model import GraphDiffusionModel
from graph_diffusion.postprocessing.inference import (
    radial_to_xy,
    sample_shapes_from_cond,
    template_thetas,
)
from graph_diffusion.postprocessing.loaders import (
    build_dataset,
    build_model,
    load_config,
)
from graph_diffusion.postprocessing.metrics import compute_boundary_roughness
from graph_diffusion.visualisation.plotting import (
    plot_conditioning_grid,
    plot_trajectory_filmstrip,
    write_trajectory_animation,
)
from graph_diffusion.visualisation.trajectory import (
    collect_forward,
    collect_reverse,
)

N_CP_GRID = 128


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-dir", required=True, type=Path)
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--n-samples", default=4, type=int)
    p.add_argument("--n-targets", default=3, type=int)
    p.add_argument("--target-seed", default=0, type=int)
    p.add_argument(
        "--checkpoint",
        default=None,
        type=Path,
        help=(
            "Override checkpoint path " "(default: <experiment-dir>/checkpoint_best.pt)"
        ),
    )
    return p.parse_args()


def pick_targets_farthest_first(
    dataset: pOnEllipseConditionalDataset,
    n_targets: int,
    seed: int,
) -> list[int]:
    """Greedy farthest-first picking over dataset.cond vectors."""
    conds = torch.stack([g.cond.squeeze(0) for g in dataset], dim=0).numpy()
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, conds.shape[0]))
    picked = [first]
    while len(picked) < n_targets:
        picked_vecs = conds[picked]
        dists = np.linalg.norm(conds[:, None, :] - picked_vecs[None, :, :], axis=-1)
        min_dists = dists.min(axis=1)
        next_idx = int(np.argmax(min_dists))
        picked.append(next_idx)
    return picked


def make_synthetic_target(
    dataset: pOnEllipseConditionalDataset, k_modes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build an asymmetric synthetic Cp curve and DCT-encode it.

    For ``cond_mode="fourier_dual"`` returns a dense ``(2 * N_CP_GRID,)``
    curve (upper concatenated with lower) and a ``(2K,)`` cond vector.
    Otherwise returns ``(N_CP_GRID,)`` dense and ``(K,)`` cond.
    """
    x_over_c = np.linspace(0.0, 1.0, N_CP_GRID)
    mean_cond = (
        torch.stack([g.cond.squeeze(0) for g in dataset], dim=0).mean(dim=0).numpy()
    )
    if dataset.cond_mode == "fourier_dual":
        upper_mean = dct_ii_inverse(mean_cond[:k_modes], N_CP_GRID)
        lower_mean = dct_ii_inverse(mean_cond[k_modes:], N_CP_GRID)
        # Lift-producing perturbation: stronger suction on the upper surface,
        # weaker pressure recovery on the lower surface.
        upper_synth = upper_mean + 0.5 * np.sin(np.pi * x_over_c)
        lower_synth = lower_mean - 0.2 * np.sin(np.pi * x_over_c)
        upper_modes = dct_ii(upper_synth.astype(np.float32), k_modes)
        lower_modes = dct_ii(lower_synth.astype(np.float32), k_modes)
        target_cond = np.concatenate([upper_modes, lower_modes]).astype(np.float32)
        cp_dense = np.concatenate(
            [upper_synth.astype(np.float32), lower_synth.astype(np.float32)]
        )
        return cp_dense, target_cond

    cp_mean_dense = dct_ii_inverse(mean_cond, N_CP_GRID)
    cp_synth = cp_mean_dense + 0.3 * np.sin(np.pi * x_over_c)
    target_cond = dct_ii(cp_synth.astype(np.float32), k_modes)
    return cp_synth.astype(np.float32), target_cond


def shape_template_from_dataset(
    dataset: pOnEllipseConditionalDataset, device: str
) -> Data:
    """Use dataset[0]'s topology as the sampling template."""
    return copy.copy(dataset[0]).to(device)


def sample_shapes_for_target(
    model: GraphDiffusionModel,
    template: Data,
    cond_vec: torch.Tensor,
    n_samples: int,
    guidance_scale: float,
    device: str,
    clamp_range: tuple[float, float],
    k_modes: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generate ``n_samples`` shapes for a target cond.

    Thin adapter around :func:`sample_shapes_from_cond` that returns
    legacy list-of-arrays types (Cartesian boundaries and dense
    head-predicted Cp curves on the ``N_CP_GRID``). When ``k_modes`` is
    set and the cond vector has length ``2 * k_modes`` (fourier_dual
    mode), the head-predicted modes are split into upper/lower halves
    and each is inverse-DCT'd separately; the returned dense curves are
    the concatenation ``[upper (N_CP_GRID), lower (N_CP_GRID)]``.
    """
    radii, head_modes = sample_shapes_from_cond(
        model=model,
        template=template,
        cond_vec=cond_vec,
        n_samples=n_samples,
        guidance_scale=guidance_scale,
        device=device,
        clamp_range=clamp_range,
        seed=0,
    )
    theta = template_thetas(template)
    shapes = [radial_to_xy(radii[i], theta) for i in range(n_samples)]
    is_dual = k_modes is not None and head_modes.shape[1] == 2 * k_modes
    head_preds_dense: list[np.ndarray] = []
    for i in range(n_samples):
        if is_dual:
            assert k_modes is not None
            upper = dct_ii_inverse(head_modes[i, :k_modes], N_CP_GRID)
            lower = dct_ii_inverse(head_modes[i, k_modes:], N_CP_GRID)
            head_preds_dense.append(np.concatenate([upper, lower]))
        else:
            head_preds_dense.append(dct_ii_inverse(head_modes[i], N_CP_GRID))
    return shapes, head_preds_dense


def _resolve_clamp_range(
    config: dict[str, Any],
) -> tuple[float, float]:
    cr = config.get("clamp_range")
    if cr is None:
        return (0.5, 2.0)
    return (float(cr[0]), float(cr[1]))


def _resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint
    best = args.experiment_dir / "checkpoint_best.pt"
    if best.exists():
        return best
    return args.experiment_dir / "checkpoint.pt"


def _figure_a(
    args: argparse.Namespace,
    config: dict[str, Any],
    model: GraphDiffusionModel,
    dataset: pOnEllipseConditionalDataset,
    template: Data,
    train_target_indices: list[int],
    synth_cp_dense: np.ndarray,
    synth_cond_np: np.ndarray,
    clamp_range: tuple[float, float],
) -> tuple[torch.Tensor, list[list[np.ndarray]], list[str]]:
    """Build Figure A and return the synthetic-cond tensor and per-target shapes.

    The shapes are returned so callers can compute summary metrics (e.g.
    boundary roughness) without re-running the reverse diffusion.
    """
    sampling_cfg = config.get("sampling", {})
    guidance_scale = float(sampling_cfg.get("guidance_scale", 1.0))
    device = args.device
    k_modes = int(config["ellipse_dataset"].get("k_modes", 8))
    is_dual = config["ellipse_dataset"].get("cond_mode", "fourier") == "fourier_dual"

    target_cps_dense: list[np.ndarray] = []
    head_pred_cps: list[np.ndarray] = []
    head_pred_stds: list[np.ndarray] = []
    sample_shapes: list[list[np.ndarray]] = []
    row_labels: list[str] = []

    def _decode_cond(cond_np: np.ndarray) -> np.ndarray:
        """Decode a (K,) or (2K,) cond vector to a dense Cp array.

        Returns ``(N_CP_GRID,)`` for single mode or ``(2*N_CP_GRID,)``
        (upper concatenated with lower) for dual mode.
        """
        if is_dual:
            upper = dct_ii_inverse(cond_np[:k_modes], N_CP_GRID)
            lower = dct_ii_inverse(cond_np[k_modes:], N_CP_GRID)
            return np.concatenate([upper, lower])
        return dct_ii_inverse(cond_np, N_CP_GRID)

    for rank, idx in enumerate(train_target_indices):
        cond_vec = dataset[idx].cond.squeeze(0)
        target_cps_dense.append(_decode_cond(cond_vec.numpy()))
        shapes, head_preds = sample_shapes_for_target(
            model,
            template,
            cond_vec,
            args.n_samples,
            guidance_scale,
            device,
            clamp_range,
            k_modes=k_modes,
        )
        sample_shapes.append(shapes)
        stacked = np.stack(head_preds, axis=0)
        head_pred_cps.append(stacked.mean(axis=0))
        head_pred_stds.append(stacked.std(axis=0))
        row_labels.append(f"train #{rank}")

    synth_cond_t = torch.tensor(synth_cond_np)
    target_cps_dense.append(synth_cp_dense)
    shapes, head_preds = sample_shapes_for_target(
        model,
        template,
        synth_cond_t,
        args.n_samples,
        guidance_scale,
        device,
        clamp_range,
        k_modes=k_modes,
    )
    sample_shapes.append(shapes)
    stacked = np.stack(head_preds, axis=0)
    head_pred_cps.append(stacked.mean(axis=0))
    head_pred_stds.append(stacked.std(axis=0))
    row_labels.append("synth asym.")

    # Split arrays for dual-mode plotting.
    target_cps_lower = head_pred_cps_lower = head_pred_stds_lower = None
    if is_dual:
        target_cps_lower = [c[N_CP_GRID:] for c in target_cps_dense]
        target_cps_dense = [c[:N_CP_GRID] for c in target_cps_dense]
        head_pred_cps_lower = [c[N_CP_GRID:] for c in head_pred_cps]
        head_pred_cps = [c[:N_CP_GRID] for c in head_pred_cps]
        head_pred_stds_lower = [c[N_CP_GRID:] for c in head_pred_stds]
        head_pred_stds = [c[:N_CP_GRID] for c in head_pred_stds]

    fig_a = plot_conditioning_grid(
        target_cps=target_cps_dense,
        sample_shapes=sample_shapes,
        head_pred_cps=head_pred_cps,
        head_pred_stds=head_pred_stds,
        row_labels=row_labels,
        target_cps_lower=target_cps_lower,
        head_pred_cps_lower=head_pred_cps_lower,
        head_pred_stds_lower=head_pred_stds_lower,
    )
    fig_a_path = args.experiment_dir / "figure_a_conditioning_grid.png"
    fig_a.savefig(fig_a_path, dpi=140, bbox_inches="tight")
    plt.close(fig_a)
    print(f"Saved Figure A to {fig_a_path}")
    return synth_cond_t, sample_shapes, row_labels


def _figure_c(
    args: argparse.Namespace,
    config: dict[str, Any],
    model: GraphDiffusionModel,
    template: Data,
    synth_cond_t: torch.Tensor,
    synth_cp_dense: np.ndarray,
    clamp_range: tuple[float, float],
) -> None:
    k_modes = int(config["ellipse_dataset"].get("k_modes", 8))
    is_dual = config["ellipse_dataset"].get("cond_mode", "fourier") == "fourier_dual"
    cfg_w_values = [1.0, 3.0, 7.0]
    cfg_shapes: list[np.ndarray] = []
    cfg_head_preds: list[np.ndarray] = []
    for w in cfg_w_values:
        shapes_w, head_preds_w = sample_shapes_for_target(
            model,
            template,
            synth_cond_t,
            n_samples=1,
            guidance_scale=w,
            device=args.device,
            clamp_range=clamp_range,
            k_modes=k_modes,
        )
        cfg_shapes.append(shapes_w[0])
        cfg_head_preds.append(head_preds_w[0])

    fig_c, axes = plt.subplots(2, 3, figsize=(12, 6))
    x_grid = np.linspace(0.0, 1.0, N_CP_GRID)
    for col, w in enumerate(cfg_w_values):
        ax = axes[0, col]
        xy = cfg_shapes[col]
        closed = np.vstack([xy, xy[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color="C2", lw=1.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"w = {w}", fontsize=10)

        ax = axes[1, col]
        if is_dual:
            head_upper = cfg_head_preds[col][:N_CP_GRID]
            head_lower = cfg_head_preds[col][N_CP_GRID:]
            target_upper = synth_cp_dense[:N_CP_GRID]
            target_lower = synth_cp_dense[N_CP_GRID:]
            ax.plot(x_grid, head_upper, color="C1", lw=2, label="head pred (upper)")
            ax.plot(x_grid, head_lower, color="C4", lw=2, label="head pred (lower)")
            ax.plot(
                x_grid, target_upper, color="C0", lw=1, ls="--", label="target (upper)"
            )
            ax.plot(
                x_grid, target_lower, color="C3", lw=1, ls="--", label="target (lower)"
            )
        else:
            ax.plot(x_grid, cfg_head_preds[col], color="C1", lw=2, label="head pred")
            ax.plot(x_grid, synth_cp_dense, color="C0", lw=1, ls="--", label="target")
        ax.grid(alpha=0.3)
        if col == 0:
            ax.legend(fontsize=7)
    fig_c.tight_layout()
    fig_c_path = args.experiment_dir / "figure_c_cfg_sweep.png"
    fig_c.savefig(fig_c_path, dpi=140, bbox_inches="tight")
    plt.close(fig_c)
    print(f"Saved Figure C to {fig_c_path}")


def _figure_b(
    args: argparse.Namespace,
    config: dict[str, Any],
    model: GraphDiffusionModel,
    dataset: pOnEllipseConditionalDataset,
    template: Data,
    synth_cond_t: torch.Tensor,
    synth_cp_dense: np.ndarray,
) -> None:
    # In dual mode synth_cp_dense is the (2*N_CP_GRID,) [upper, lower]
    # concatenation; plot_trajectory_filmstrip expects a single curve,
    # so we display the upper half only here.
    if (
        config["ellipse_dataset"].get("cond_mode", "fourier") == "fourier_dual"
        and synth_cp_dense.shape[0] == 2 * N_CP_GRID
    ):
        synth_cp_dense = synth_cp_dense[:N_CP_GRID]
    device = args.device
    sampling_cfg = config.get("sampling", {})
    guidance_scale = float(sampling_cfg.get("guidance_scale", 1.0))
    total_T = config["noise_schedule"]["T"]  # noqa: N806

    log_steps = np.unique(
        np.round(np.logspace(np.log10(1), np.log10(total_T - 1), num=5)).astype(int)
    ).tolist()
    snapshot_steps = sorted({0, *log_steps})[:6]
    print(f"Trajectory snapshot timesteps: {snapshot_steps}")

    # Use dataset[0] for the forward template too so its theta grid matches
    # the reverse template's, otherwise the two rows of the filmstrip have
    # different node counts and can't be plotted on the same axes.
    clamp_range = _resolve_clamp_range(config)

    fwd_template = copy.copy(dataset[0]).to(device)
    forward_snaps_t = collect_forward(
        model, fwd_template, snapshot_steps=snapshot_steps, seed=0
    )

    template_with_synth = copy.copy(template)
    template_with_synth.cond = synth_cond_t.unsqueeze(0).to(device)
    reverse_snaps_t = collect_reverse(
        model,
        template_with_synth,
        cond=template_with_synth.cond,
        snapshot_steps=snapshot_steps,
        guidance_scale=guidance_scale,
        seed=0,
        clamp_range=clamp_range,
    )

    theta = template_thetas(dataset[0])
    forward_xy = [radial_to_xy(s[:, 0].numpy(), theta) for s in forward_snaps_t]
    reverse_xy = [radial_to_xy(s[:, 0].numpy(), theta) for s in reverse_snaps_t]

    fig_b = plot_trajectory_filmstrip(
        forward_snapshots=forward_xy,
        reverse_snapshots=reverse_xy,
        timesteps=snapshot_steps,
        target_cp=synth_cp_dense,
    )
    fig_b_path = args.experiment_dir / "figure_b_trajectory.png"
    fig_b.savefig(fig_b_path, dpi=140, bbox_inches="tight")
    plt.close(fig_b)
    print(f"Saved Figure B to {fig_b_path}")

    all_steps = list(range(total_T - 1, -1, -1))
    full_reverse_t = collect_reverse(
        model,
        template_with_synth,
        cond=template_with_synth.cond,
        snapshot_steps=all_steps,
        guidance_scale=guidance_scale,
        seed=0,
        clamp_range=clamp_range,
    )
    full_reverse_xy = [radial_to_xy(s[:, 0].numpy(), theta) for s in full_reverse_t]
    write_trajectory_animation(
        reverse_snapshots=full_reverse_xy,
        target_cp=synth_cp_dense,
        out_path_mp4=args.experiment_dir / "figure_b_reverse.mp4",
        out_path_gif=args.experiment_dir / "figure_b_reverse.gif",
        fps=25,
    )
    print(
        f"Saved Figure B animation to "
        f"{args.experiment_dir / 'figure_b_reverse.mp4'} and "
        f"{args.experiment_dir / 'figure_b_reverse.gif'}"
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = args.device

    dataset = build_dataset(config)
    model = build_model(config, device)

    ckpt_path = _resolve_checkpoint(args)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint from {ckpt_path}")

    train_target_indices = pick_targets_farthest_first(
        dataset, args.n_targets, args.target_seed
    )
    print(f"Picked training target indices: {train_target_indices}")

    k_modes = config["ellipse_dataset"]["k_modes"]
    synth_cp_dense, synth_cond_np = make_synthetic_target(dataset, k_modes)
    print(f"Built synthetic asymmetric target ({k_modes} DCT modes)")

    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.experiment_dir / "exp020_targets.npz",
        train_indices=np.array(train_target_indices),
        synth_cp_dense=synth_cp_dense,
        synth_cond=synth_cond_np,
    )
    print(f"Saved target manifest to {args.experiment_dir / 'exp020_targets.npz'}")

    template = shape_template_from_dataset(dataset, device)
    clamp_range = _resolve_clamp_range(config)

    synth_cond_t, sample_shapes_per_target, row_labels = _figure_a(
        args,
        config,
        model,
        dataset,
        template,
        train_target_indices,
        synth_cp_dense,
        synth_cond_np,
        clamp_range,
    )

    # Per-target + overall boundary-roughness for EXP-021..025 comparison.
    per_target_roughness: dict[str, float] = {}
    for label, shapes in zip(row_labels, sample_shapes_per_target, strict=False):
        per_target_roughness[label] = compute_boundary_roughness(
            np.stack(shapes, axis=0)
        )
    overall_roughness = float(np.mean(list(per_target_roughness.values())))
    roughness_path = args.experiment_dir / "roughness_report.json"
    with open(roughness_path, "w") as fh:  # noqa: PTH123
        json.dump(
            {
                "per_target": per_target_roughness,
                "overall_mean": overall_roughness,
            },
            fh,
            indent=2,
        )
    print(
        f"Saved roughness report to {roughness_path} "
        f"(overall={overall_roughness:.5f})"
    )

    _figure_c(args, config, model, template, synth_cond_t, synth_cp_dense, clamp_range)
    _figure_b(
        args,
        config,
        model,
        dataset,
        template,
        synth_cond_t,
        synth_cp_dense,
    )


if __name__ == "__main__":
    main()
