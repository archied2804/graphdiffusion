"""
postprocess_circle.py — Post-processing & evaluation for circle diffusion
==========================================================================

Loads a trained checkpoint and evaluates generated shapes against the
training data distribution. Produces quantitative metrics and diagnostic
plots.

Usage:
python scripts/postprocess_circle.py \
    --experiment-dir outputs/EXP-001_circle_radial_baseline \
    --config configs/circle_radial.yaml \
    --n-samples 50 \
    --visualize-diffusion \
    --save-samples

Outputs (saved to --experiment-dir):
    evaluation_report.json   Quantitative metrics (per-sample & aggregate)
    loss_curves.png          Training / validation loss over epochs
    radii_histogram.png      Reference vs generated radii distributions
    sample_gallery.png       Gallery of 16 generated shapes
    quality_distributions.png  Smoothness & circularity histograms
    diffusion_process.png    (--visualize-diffusion) Reverse diffusion stages
    generated_samples.pt     (--save-samples) Raw radii, angles, & metrics
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from graph_diffusion.building_blocks.feature_transforms import LogitNormTransform

from metrics import (
    compute_boundary_violations,
    compute_circularity,
    compute_closure_error,
    compute_radii_stats,
    compute_smoothness,
    extract_sorted_radii,
    ks_statistic,
)

from graph_diffusion.building_blocks.noise_schedule import NoiseSchedule
from graph_diffusion.data.circledataset import UnitCircleDataset
from graph_diffusion.data.transforms import ComputeAngularEdgeFeatures
from graph_diffusion.model.graph_diffusion_model import GraphDiffusionModel
from graph_diffusion.model.score_network import ScoreNetwork
from graph_diffusion.postprocessing import load_checkpoint


def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path) as f:  # noqa: PTH123
        return yaml.safe_load(f)


def build_model(config: dict) -> GraphDiffusionModel:
    """Reconstruct model architecture from config."""
    ns_cfg = config.get("noise_schedule", {})
    noise_schedule = NoiseSchedule(
        T=ns_cfg.get("T", 200),
        schedule_type=ns_cfg.get("schedule_type", "cosine"),
        beta_start=ns_cfg.get("beta_start", 1e-4),
        beta_end=ns_cfg.get("beta_end", 0.02),
    )

    sn_cfg = config.get("score_network", {})
    mlp_cfg = config.get("mlp", {})
    score_network = ScoreNetwork(
        node_dim=sn_cfg.get("node_dim", 32),
        edge_dim=sn_cfg.get("edge_dim", 2),
        global_dim=sn_cfg.get("global_dim", 8),
        time_embed_dim=sn_cfg.get("time_embed_dim", 64),
        n_layers=sn_cfg.get("n_layers", 4),
        hidden_dims=sn_cfg.get("hidden_dims", [64, 64]),
        activation=mlp_cfg.get("activation", "silu"),
        layer_norm=mlp_cfg.get("layer_norm", True),
        residual=mlp_cfg.get("residual", True),
        input_dim=sn_cfg.get("input_dim", 1),
    )

    feature_transform = None
    ft_cfg = config.get("feature_transform")
    if ft_cfg and ft_cfg.get("type") == "logit_norm":
        feature_transform = LogitNormTransform(
            r_min=float(ft_cfg.get("r_min", 0.5)),
            r_max=float(ft_cfg.get("r_max", 1.5)),
        )

    return GraphDiffusionModel(
        score_network=score_network,
        noise_schedule=noise_schedule,
        feature_transform=feature_transform,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-process and evaluate circle diffusion results"
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Directory containing checkpoint.pt and loss_log.json",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/circle_radial.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of shapes to generate for evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference",
    )
    parser.add_argument(
        "--visualize-diffusion",
        action="store_true",
        help="Generate a multi-panel plot showing the reverse diffusion process",
    )
    parser.add_argument(
        "--save-samples",
        action="store_true",
        help="Save raw generated radii, angles, and metrics to a .pt file",
    )
    parser.add_argument(
        "--n-snapshots",
        type=int,
        default=8,
        help="Number of intermediate snapshots for diffusion visualisation",
    )
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    config = load_config(args.config)
    device = torch.device(args.device)

    # --- Load model ---
    model = build_model(config)
    checkpoint = load_checkpoint(str(exp_dir / "checkpoint.pt"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # --- Load dataset for reference stats ---
    ds_cfg = config.get("circle_dataset", {})
    data_root = ds_cfg.get("root", "data/circle")
    pre_transform = ComputeAngularEdgeFeatures()
    dataset = UnitCircleDataset(
        root=data_root,
        n_graphs=ds_cfg.get("n_graphs", 2000),
        n_nodes=ds_cfg.get("n_nodes", 64),
        n_fourier_modes=ds_cfg.get("n_fourier_modes", 5),
        amplitude_scale=ds_cfg.get("amplitude_scale", 0.15),
        r_min=ds_cfg.get("r_min", 0.5),
        r_max=ds_cfg.get("r_max", 1.5),
        k_neighbors=ds_cfg.get("k_neighbors", 2),
        global_dim=ds_cfg.get("global_dim", 8),
        include_curvature=ds_cfg.get("include_curvature", False),
        include_arc_length=ds_cfg.get("include_arc_length", False),
        seed=ds_cfg.get("seed", 42),
        pre_transform=pre_transform,
    )

    # Determine r bounds: prefer clamp_range, fall back to feature_transform, then defaults
    clamp_cfg = config.get("clamp_range")
    ft_cfg = config.get("feature_transform")
    if clamp_cfg:
        r_min = float(clamp_cfg[0])
        r_max = float(clamp_cfg[1])
    elif ft_cfg:
        r_min = float(ft_cfg.get("r_min", 0.5))
        r_max = float(ft_cfg.get("r_max", 1.5))
    else:
        r_min, r_max = 0.5, 1.5
    clamp_range: tuple[float, float] | None = (r_min, r_max) if clamp_cfg else None

    # --- Reference distribution from training data ---
    ref_radii = []
    for i in range(min(200, len(dataset))):
        ref_radii.append(dataset[i].x[:, 0].numpy())
    ref_radii_all = np.concatenate(ref_radii)
    ref_stats = compute_radii_stats(ref_radii_all)
    print("\n--- Reference (training data) radii stats ---")
    for k, v in ref_stats.items():
        print(f"  {k}: {v:.4f}")

    # --- Generate samples ---
    template = dataset[0].to(device)
    all_radii = []
    per_sample_metrics = []

    print(f"\nGenerating {args.n_samples} samples...")
    with torch.no_grad():
        for i in range(args.n_samples):
            torch.manual_seed(i)
            result = model.sample(template, clamp_range=clamp_range)
            r_sorted, theta_sorted = extract_sorted_radii(result, template)
            all_radii.append(r_sorted)

            metrics = {
                "sample": i,
                "radii_stats": compute_radii_stats(r_sorted),
                "smoothness": compute_smoothness(r_sorted),
                "closure_error": compute_closure_error(r_sorted),
                "boundary_violations": compute_boundary_violations(
                    r_sorted, r_min, r_max
                ),
                "circularity_cv": compute_circularity(r_sorted),
            }
            per_sample_metrics.append(metrics)

    # Pre-compute sorted angles from template for reuse
    theta_template = np.arctan2(
        template.pos[:, 1].cpu().numpy(),
        template.pos[:, 0].cpu().numpy(),
    )
    theta_order = np.argsort(theta_template)
    theta_sorted = theta_template[theta_order]

    gen_radii_all = np.concatenate(all_radii)
    gen_stats = compute_radii_stats(gen_radii_all)

    # --- Aggregate metrics ---
    avg_smoothness = np.mean([m["smoothness"] for m in per_sample_metrics])
    avg_closure = np.mean([m["closure_error"] for m in per_sample_metrics])
    avg_cv = np.mean([m["circularity_cv"] for m in per_sample_metrics])
    avg_violations = np.mean(
        [m["boundary_violations"]["total"] for m in per_sample_metrics]
    )

    print(f"\n--- Generated samples radii stats ({args.n_samples} shapes) ---")
    for k, v in gen_stats.items():
        print(f"  {k}: {v:.4f}")

    print(f"\n--- Quality metrics (averaged over {args.n_samples} samples) ---")
    print(f"  Smoothness (2nd-order diff):  {avg_smoothness:.4f}")
    print(f"  Closure error (|r_0 - r_N|): {avg_closure:.4f}")
    print(f"  Circularity (CV of radii):   {avg_cv:.4f}")
    print(f"  Boundary violation rate:     {avg_violations:.4f}")

    # --- Distribution comparison ---
    ks_stat = ks_statistic(ref_radii_all, gen_radii_all)
    print(f"  KS statistic (ref vs gen):   {ks_stat:.4f}")

    # --- Save metrics ---
    report = {
        "reference_stats": ref_stats,
        "generated_stats": gen_stats,
        "aggregate_metrics": {
            "smoothness": float(avg_smoothness),
            "closure_error": float(avg_closure),
            "circularity_cv": float(avg_cv),
            "boundary_violation_rate": float(avg_violations),
            "ks_statistic": float(ks_stat),
        },
        "per_sample_metrics": per_sample_metrics,
        "n_samples": args.n_samples,
        "epochs": checkpoint["epochs"],
    }

    report_path = exp_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved evaluation report to {report_path}")

    # --- Save raw samples ---
    if args.save_samples:
        samples_path = exp_dir / "generated_samples.pt"
        torch.save(
            {
                "radii": [torch.from_numpy(r) for r in all_radii],
                "theta": torch.from_numpy(theta_sorted),
                "per_sample_metrics": per_sample_metrics,
                "config": config,
            },
            samples_path,
        )
        print(f"Saved raw samples to {samples_path}")

    # --- Plots ---
    _plot_loss_curves(exp_dir)
    _plot_radii_histogram(ref_radii_all, gen_radii_all, exp_dir)
    _plot_sample_gallery(all_radii, template, exp_dir)
    _plot_smoothness_distribution(per_sample_metrics, exp_dir)

    # --- Diffusion process visualisation ---
    if args.visualize_diffusion:
        print("\nGenerating diffusion process visualisation...")
        torch.manual_seed(0)
        _, trajectory = model.sample_with_trajectory(
            template,
            n_snapshots=args.n_snapshots,
            clamp_range=clamp_range,
        )
        _plot_diffusion_process(trajectory, template, exp_dir)

    # --- Output summary ---
    print(f"\n{'=' * 50}")
    print("Output files:")
    for p in sorted(exp_dir.iterdir()):
        if p.is_file():
            print(f"  {p}")
    print(f"{'=' * 50}")


def _plot_loss_curves(exp_dir: Path) -> None:
    """Plot training and validation loss curves from loss_log.json."""
    log_path = exp_dir / "loss_log.json"
    if not log_path.exists():
        print("  No loss_log.json found, skipping loss curves.")
        return

    with open(log_path) as f:
        loss_log = json.load(f)

    epochs = [e["epoch"] for e in loss_log]
    train_loss = [e["train_loss"] for e in loss_log]
    val_loss = [e["val_loss"] for e in loss_log]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="Train Loss", linewidth=1.5)
    ax.plot(epochs, val_loss, label="Val Loss", linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(exp_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved loss_curves.png")


def _plot_radii_histogram(
    ref_radii: np.ndarray, gen_radii: np.ndarray, exp_dir: Path
) -> None:
    """Overlay histograms of reference vs generated radii distributions."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ref_radii, bins=60, alpha=0.5, density=True, label="Training data")
    ax.hist(gen_radii, bins=60, alpha=0.5, density=True, label="Generated")
    ax.axvline(1.0, color="k", linestyle="--", alpha=0.5, label="Unit circle")
    ax.set_xlabel("Radius")
    ax.set_ylabel("Density")
    ax.set_title("Radii Distribution: Training vs Generated")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(exp_dir / "radii_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved radii_histogram.png")


def _plot_sample_gallery(
    all_radii: list[np.ndarray], template: Data, exp_dir: Path
) -> None:
    """Plot a gallery of 16 generated shapes overlaid on the unit circle."""
    n_show = min(16, len(all_radii))
    cols = 4
    rows = (n_show + cols - 1) // cols

    theta = np.arctan2(
        template.pos[:, 1].cpu().numpy(),
        template.pos[:, 0].cpu().numpy(),
    )
    order = np.argsort(theta)
    theta_sorted = theta[order]

    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))
    axes = np.array(axes).flatten()

    t_ref = np.linspace(0, 2 * np.pi, 100)

    for i in range(n_show):
        ax = axes[i]
        r = all_radii[i]
        x_cart = r * np.cos(theta_sorted)
        y_cart = r * np.sin(theta_sorted)
        x_cart = np.append(x_cart, x_cart[0])
        y_cart = np.append(y_cart, y_cart[0])

        ax.plot(np.cos(t_ref), np.sin(t_ref), "k--", alpha=0.3, linewidth=0.5)
        ax.plot(x_cart, y_cart, "b-", linewidth=1.2)
        ax.set_aspect("equal")
        ax.set_title(f"Sample {i + 1}", fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)

    for i in range(n_show, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Generated Shape Gallery", fontsize=13)
    fig.tight_layout()
    fig.savefig(exp_dir / "sample_gallery.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved sample_gallery.png")


def _plot_smoothness_distribution(
    per_sample_metrics: list[dict], exp_dir: Path
) -> None:
    """Histograms of per-sample smoothness and circularity."""
    smoothness = [m["smoothness"] for m in per_sample_metrics]
    cv = [m["circularity_cv"] for m in per_sample_metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.hist(smoothness, bins=25, edgecolor="black", alpha=0.7)
    ax1.set_xlabel("Smoothness (2nd-order diff)")
    ax1.set_ylabel("Count")
    ax1.set_title("Smoothness Distribution")
    ax1.grid(True, alpha=0.3)

    ax2.hist(cv, bins=25, edgecolor="black", alpha=0.7, color="orange")
    ax2.set_xlabel("Circularity (CV of radii)")
    ax2.set_ylabel("Count")
    ax2.set_title("Circularity Distribution")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(exp_dir / "quality_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved quality_distributions.png")


def _plot_diffusion_process(
    trajectory: list[tuple[int, torch.Tensor]],
    template: Data,
    exp_dir: Path,
) -> None:
    """Visualise the reverse diffusion process from noise to final shape.

    Each panel shows the shape at an intermediate timestep, progressing
    from pure noise (t = T) on the left to the clean sample (t = 0) on
    the right.
    """
    theta = np.arctan2(
        template.pos[:, 1].cpu().numpy(),
        template.pos[:, 0].cpu().numpy(),
    )
    order = np.argsort(theta)
    theta_sorted = theta[order]
    t_ref = np.linspace(0, 2 * np.pi, 100)

    n_panels = len(trajectory)
    cols = min(n_panels, 5)
    rows = (n_panels + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))
    axes = np.array(axes).flatten()

    for i, (timestep, x_t) in enumerate(trajectory):
        ax = axes[i]
        r = x_t[:, 0].numpy()[order]

        x_cart = r * np.cos(theta_sorted)
        y_cart = r * np.sin(theta_sorted)
        # Close the curve
        x_cart = np.append(x_cart, x_cart[0])
        y_cart = np.append(y_cart, y_cart[0])

        # Unit circle reference
        ax.plot(np.cos(t_ref), np.sin(t_ref), "k--", alpha=0.3, linewidth=0.5)
        ax.plot(x_cart, y_cart, "b-", linewidth=1.2, alpha=0.9)
        ax.set_aspect("equal")
        ax.set_title(f"t = {timestep}", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.2)

        # Adaptive axis limits: wider for noisy early steps
        lim = max(2.0, float(np.max(np.abs(r))) * 1.3)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.tick_params(labelsize=7)

    for i in range(n_panels, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        "Reverse Diffusion Process: Noise → Shape",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(exp_dir / "diffusion_process.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved diffusion_process.png")


if __name__ == "__main__":
    main()
