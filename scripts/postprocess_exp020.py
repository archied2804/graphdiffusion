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
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch_geometric.data import Data

from graph_diffusion.building_blocks.noise_schedule import NoiseSchedule
from graph_diffusion.data.pOnEllipseConditional import (
    dct_ii,
    pOnEllipseConditionalDataset,
)
from graph_diffusion.data.transforms import (
    ComputeAngularEdgeFeatures,
    ComputeArcLengthEdgeFeatures,
)
from graph_diffusion.model.graph_diffusion_model import GraphDiffusionModel
from graph_diffusion.model.pressure_head import PressurePredictionHead
from graph_diffusion.model.score_network import ScoreNetwork
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


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:  # noqa: PTH123
        return yaml.safe_load(f)  # type: ignore[no-any-return]


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


def build_dataset(config: dict[str, Any]) -> pOnEllipseConditionalDataset:
    ds_cfg = config["ellipse_dataset"]
    feature_mode = ds_cfg.get("feature_mode", "radial_norm")
    pre_transform = (
        ComputeArcLengthEdgeFeatures()
        if feature_mode == "cartesian"
        else ComputeAngularEdgeFeatures()
    )
    return pOnEllipseConditionalDataset(
        root=ds_cfg.get("root", "data/ellipse"),
        cond_mode=ds_cfg.get("cond_mode", "fourier"),
        k_modes=ds_cfg.get("k_modes", 8),
        feature_mode=feature_mode,
        split=ds_cfg.get("split", "train"),
        n_samples=ds_cfg.get("n_samples", None),
        k_neighbors=ds_cfg.get("k_neighbors", 6),
        global_dim=ds_cfg.get("global_dim", 8),
        pre_transform=pre_transform,
    )


def build_model(config: dict[str, Any], device: str) -> GraphDiffusionModel:
    ns_cfg = config["noise_schedule"]
    schedule = NoiseSchedule(
        T=ns_cfg["T"],
        schedule_type=ns_cfg.get("schedule_type", "cosine"),
        beta_start=ns_cfg.get("beta_start", 1.0e-4),
        beta_end=ns_cfg.get("beta_end", 0.02),
    )
    sn_cfg = config["score_network"]
    mlp_cfg = config["mlp"]
    sn = ScoreNetwork(
        node_dim=sn_cfg["node_dim"],
        edge_dim=sn_cfg["edge_dim"],
        global_dim=sn_cfg["global_dim"],
        time_embed_dim=sn_cfg["time_embed_dim"],
        n_layers=sn_cfg["n_layers"],
        hidden_dims=sn_cfg.get("hidden_dims", [64, 64]),
        activation=mlp_cfg.get("activation", "silu"),
        layer_norm=mlp_cfg.get("layer_norm", True),
        residual=mlp_cfg.get("residual", True),
        input_dim=sn_cfg.get("input_dim", None),
        cond_dim=sn_cfg.get("cond_dim", None),
        p_uncond=float(sn_cfg.get("p_uncond", 0.0)),
        output_dim=sn_cfg.get("output_dim", None),
    )
    ph_cfg = config["pressure_head"]
    head = PressurePredictionHead(
        in_dim=ph_cfg["in_dim"],
        out_dim=ph_cfg["out_dim"],
        node_hidden=ph_cfg.get("node_hidden", [64, 64]),
        global_hidden=ph_cfg.get("global_hidden", [64, 64]),
        node_embed_dim=ph_cfg.get("node_embed_dim", 64),
        activation=mlp_cfg.get("activation", "silu"),
        layer_norm=mlp_cfg.get("layer_norm", True),
    )
    model_cfg = config.get("model", {})
    return GraphDiffusionModel(
        score_network=sn,
        noise_schedule=schedule,
        n_noise_channels=model_cfg.get("n_noise_channels", None),
        pressure_head=head,
        lambda_pressure=float(model_cfg.get("lambda_pressure", 0.0)),
    ).to(device)


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


def _inverse_dct(modes: np.ndarray, n_grid: int) -> np.ndarray:
    """Inverse type-II DCT of K modes onto an n_grid sample grid."""
    k_modes = modes.shape[0]
    n_idx = np.arange(n_grid, dtype=np.float32)
    k_idx = np.arange(k_modes, dtype=np.float32)[:, None]
    basis = np.cos(np.pi * (2.0 * n_idx + 1.0) * k_idx / (2.0 * n_grid))
    norm = np.full(k_modes, np.sqrt(2.0 / n_grid), dtype=np.float32)
    norm[0] = np.sqrt(1.0 / n_grid)
    return (modes * norm) @ basis  # type: ignore[no-any-return]


def make_synthetic_target(
    dataset: pOnEllipseConditionalDataset, k_modes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build an asymmetric synthetic Cp curve and DCT-encode it."""
    x_over_c = np.linspace(0.0, 1.0, N_CP_GRID)
    mean_cond = (
        torch.stack([g.cond.squeeze(0) for g in dataset], dim=0).mean(dim=0).numpy()
    )
    cp_mean_dense = _inverse_dct(mean_cond, N_CP_GRID)
    cp_synth = cp_mean_dense + 0.3 * np.sin(np.pi * x_over_c)
    target_cond = dct_ii(cp_synth.astype(np.float32), k_modes)
    return cp_synth.astype(np.float32), target_cond


def shape_template_from_dataset(
    dataset: pOnEllipseConditionalDataset, device: str
) -> Data:
    """Use dataset[0]'s topology as the sampling template."""
    return copy.copy(dataset[0]).to(device)


def radial_to_xy(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """(N,) r and (N,) theta -> (N, 2) Cartesian, sorted by theta.

    Sorting by theta gives a proper boundary ordering for plt.plot to
    draw a closed curve; the raw dataset order is not chordwise.
    """
    order = np.argsort(theta)
    r_sorted = r[order]
    theta_sorted = theta[order]
    return np.stack(
        [r_sorted * np.cos(theta_sorted), r_sorted * np.sin(theta_sorted)], axis=1
    )


def template_thetas(dataset: pOnEllipseConditionalDataset) -> np.ndarray:
    """Recover the (N,) theta vector from dataset[0].pos = (cos t, sin t)."""
    pos = dataset[0].pos.numpy()
    return np.arctan2(pos[:, 1], pos[:, 0])


def sample_shapes_for_target(
    model: GraphDiffusionModel,
    template: Data,
    cond_vec: torch.Tensor,
    n_samples: int,
    guidance_scale: float,
    device: str,
    clamp_range: tuple[float, float],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generate n_samples shapes for a target cond.

    Returns:
        shapes: list of (N, 2) Cartesian arrays.
        head_pred_cps_dense: list of (N_cp_grid,) head-predicted Cp curves.
    """
    template_with_cond = copy.copy(template)
    template_with_cond.cond = cond_vec.unsqueeze(0).to(device)

    shapes: list[np.ndarray] = []
    head_preds_dense: list[np.ndarray] = []
    pos = template.pos
    theta = np.arctan2(pos[:, 1].cpu().numpy(), pos[:, 0].cpu().numpy())

    for i in range(n_samples):
        torch.manual_seed(i)
        out = model.sample(
            template_with_cond,
            clamp_range=clamp_range,
            guidance_scale=guidance_scale,
        )
        r = out.x[:, 0].detach().cpu().numpy()
        shapes.append(radial_to_xy(r, theta))

        assert model.pressure_head is not None
        batch_vec = torch.zeros(pos.size(0), dtype=torch.long, device=device)
        with torch.no_grad():
            pred = model.pressure_head(out.x, pos, batch_vec)
        head_preds_dense.append(_inverse_dct(pred[0].cpu().numpy(), N_CP_GRID))
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
) -> torch.Tensor:
    """Build Figure A and return the synthetic-cond tensor for downstream re-use."""
    sampling_cfg = config.get("sampling", {})
    guidance_scale = float(sampling_cfg.get("guidance_scale", 1.0))
    device = args.device

    target_cps_dense: list[np.ndarray] = []
    head_pred_cps: list[np.ndarray] = []
    head_pred_stds: list[np.ndarray] = []
    sample_shapes: list[list[np.ndarray]] = []
    row_labels: list[str] = []

    for rank, idx in enumerate(train_target_indices):
        cond_vec = dataset[idx].cond.squeeze(0)
        target_cps_dense.append(_inverse_dct(cond_vec.numpy(), N_CP_GRID))
        shapes, head_preds = sample_shapes_for_target(
            model,
            template,
            cond_vec,
            args.n_samples,
            guidance_scale,
            device,
            clamp_range,
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
    )
    sample_shapes.append(shapes)
    stacked = np.stack(head_preds, axis=0)
    head_pred_cps.append(stacked.mean(axis=0))
    head_pred_stds.append(stacked.std(axis=0))
    row_labels.append("synth asym.")

    fig_a = plot_conditioning_grid(
        target_cps=target_cps_dense,
        sample_shapes=sample_shapes,
        head_pred_cps=head_pred_cps,
        head_pred_stds=head_pred_stds,
        row_labels=row_labels,
    )
    fig_a_path = args.experiment_dir / "figure_a_conditioning_grid.png"
    fig_a.savefig(fig_a_path, dpi=140, bbox_inches="tight")
    plt.close(fig_a)
    print(f"Saved Figure A to {fig_a_path}")
    return synth_cond_t


def _figure_c(
    args: argparse.Namespace,
    model: GraphDiffusionModel,
    template: Data,
    synth_cond_t: torch.Tensor,
    synth_cp_dense: np.ndarray,
    clamp_range: tuple[float, float],
) -> None:
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
        ax.plot(x_grid, cfg_head_preds[col], color="C1", lw=2, label="head pred")
        ax.plot(x_grid, synth_cp_dense, color="C0", lw=1, ls="--", label="target")
        ax.grid(alpha=0.3)
        if col == 0:
            ax.legend(fontsize=8)
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

    theta = template_thetas(dataset)
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

    synth_cond_t = _figure_a(
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
    _figure_c(args, model, template, synth_cond_t, synth_cp_dense, clamp_range)
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
