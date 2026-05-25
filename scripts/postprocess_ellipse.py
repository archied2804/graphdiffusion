"""
postprocess_ellipse.py — Post-processing & evaluation for ellipse diffusion
============================================================================

Mirrors postprocess_circle.py for the EXP-01x radial shape generation series.
Supports all three experiment types:
  - EXP-010: shape (radial, no conditioning)
  - EXP-011: conditional (radial + global pressure summary)
  - EXP-012: conditional (radial + per-node pressure)

Usage:
    python scripts/postprocess_ellipse.py \\
        --experiment-dir outputs/EXP-010_ellipse_radial_mesh \\
        --config configs/EXP-010_ellipse_data_pipeline.yaml \\
        --n-samples 50 \\
        --smooth-modes 20 \\
        --upsample 512 \\
        --visualize-diffusion

Outputs (saved to --experiment-dir):
    evaluation_report.json    Quantitative metrics (per-sample & aggregate)
    loss_curves.png           Training / validation loss (log scale)
    radii_histogram.png       Reference vs generated radii distributions
    sample_gallery.png        Gallery of 16 generated shapes (post-processed)
    quality_distributions.png Smoothness & circularity histograms
    surface_finish.png        (--smooth-modes / --upsample) Raw vs post-processed overlay
    diffusion_process.png     (--visualize-diffusion) Reverse diffusion stages
    generated_samples.pt      (--save-samples) Raw radii, angles & metrics
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
from scipy.interpolate import CubicSpline
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
from graph_diffusion.data.ellipsedataset import (
    EllipseConditionalDataset,
    EllipseShapeDataset,
)
from graph_diffusion.data.transforms import ComputeAngularEdgeFeatures
from graph_diffusion.model.graph_diffusion_model import GraphDiffusionModel
from graph_diffusion.model.score_network import ScoreNetwork
from graph_diffusion.postprocessing import load_checkpoint


def fourier_smooth(r: np.ndarray, n_modes: int) -> np.ndarray:
    """Fourier low-pass filter: retain only the lowest ``n_modes`` frequency components.

    Args:
        r: Radii array of shape ``(N,)``, ordered by ascending angle.
        n_modes: Number of complex Fourier modes to keep (including DC).
            ``n_modes=1`` → circle; ``n_modes=N//2+1`` → no change.

    Returns:
        Smoothed radii of shape ``(N,)``.
    """
    R = np.fft.rfft(r)
    R[n_modes:] = 0.0
    return np.fft.irfft(R, n=len(r))


def spline_upsample(
    theta: np.ndarray,
    r: np.ndarray,
    n_out: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a closed radial curve to ``n_out`` uniformly-spaced angles via cubic spline.

    Extends the knot sequence by one period to enforce periodic closure before
    fitting a not-a-knot cubic spline, then evaluates at ``n_out`` angles
    uniformly distributed over ``[theta[0], theta[0] + 2π)``.

    Args:
        theta: Angles in ascending order, shape ``(N,)``.
        r: Radii at each angle, shape ``(N,)``.
        n_out: Number of output nodes.

    Returns:
        ``(theta_out, r_out)``: Arrays of shape ``(n_out,)``.
    """
    # Periodic extension: append first knot one full period forward
    angular_step = (theta[-1] - theta[0]) / (len(theta) - 1)
    theta_ext = np.append(theta, theta[-1] + angular_step)
    r_ext = np.append(r, r[0])
    cs = CubicSpline(theta_ext, r_ext)
    theta_out = np.linspace(theta[0], theta[0] + 2 * np.pi, n_out, endpoint=False)
    return theta_out, np.clip(cs(theta_out), 0.0, None)


def load_config(path: str) -> dict:  # type: ignore[type-arg]
    """Load YAML configuration file."""
    with open(path) as f:  # noqa: PTH123
        return yaml.safe_load(f)


def build_model(config: dict) -> GraphDiffusionModel:  # type: ignore[type-arg]
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
        cond_dim=sn_cfg.get("cond_dim", None),
        output_dim=sn_cfg.get("output_dim", None),
    )
    model_cfg = config.get("model", {})
    return GraphDiffusionModel(
        score_network=score_network,
        noise_schedule=noise_schedule,
        n_noise_channels=model_cfg.get("n_noise_channels", None),
    )


def build_dataset(
    config: dict,  # type: ignore[type-arg]
) -> EllipseShapeDataset | EllipseConditionalDataset:
    """Instantiate the reference dataset from config."""
    ds_cfg = config.get("ellipse_dataset", {})
    root = ds_cfg.get("root", "data/ellipse")
    split = ds_cfg.get("split", "train")
    k_neighbors = ds_cfg.get("k_neighbors", 6)
    global_dim = ds_cfg.get("global_dim", 8)
    time_index = ds_cfg.get("time_index", -1)
    pre_transform = ComputeAngularEdgeFeatures()

    dataset_type = config.get("dataset_type", "shape")
    feature_mode = ds_cfg.get("feature_mode", "radial")

    if dataset_type == "conditional":
        cond_type = ds_cfg.get("cond_type", "global_summary")
        return EllipseConditionalDataset(
            root=root,
            feature_mode=feature_mode,
            cond_type=cond_type,
            split=split,
            k_neighbors=k_neighbors,
            global_dim=global_dim,
            time_index=time_index,
            pre_transform=pre_transform,
        )

    return EllipseShapeDataset(
        root=root,
        feature_mode=feature_mode,
        split=split,
        k_neighbors=k_neighbors,
        global_dim=global_dim,
        pre_transform=pre_transform,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-process and evaluate ellipse radial diffusion results"
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
        required=True,
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
    parser.add_argument(
        "--smooth-modes",
        type=int,
        default=0,
        help="Fourier low-pass: number of modes to keep (0 = disabled)",
    )
    parser.add_argument(
        "--upsample",
        type=int,
        default=0,
        help="Spline upsample to this many boundary nodes (0 = disabled)",
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
    print(f"Loaded checkpoint (epoch {checkpoint['epoch']})")

    # --- Load dataset for reference stats ---
    dataset = build_dataset(config)
    print(f"Dataset: {len(dataset)} graphs")

    # --- Clamp range ---
    clamp_cfg = config.get("clamp_range")
    if clamp_cfg:
        r_min, r_max = float(clamp_cfg[0]), float(clamp_cfg[1])
    else:
        r_min, r_max = 0.1, 3.0
    clamp_range: tuple[float, float] | None = (r_min, r_max) if clamp_cfg else None

    # --- Reference distribution (up to 200 graphs) ---
    # For radial_norm, dataset.x stores r/r_mean.  Collect both:
    #   ref_radii_norm  — normalised (shape-only, for shape-space KS)
    #   ref_radii_all   — physical  (for radii histogram and stats)
    ref_radii: list[np.ndarray] = []
    ref_radii_norm: list[np.ndarray] = []
    for i in range(min(200, len(dataset))):
        g = dataset[i]
        r_norm_vals = g.x[:, 0].numpy()
        ref_radii_norm.append(r_norm_vals)
        r_phys_vals = (
            r_norm_vals * g.r_scale.item() if hasattr(g, "r_scale") else r_norm_vals
        )
        ref_radii.append(r_phys_vals)
    ref_radii_all = np.concatenate(ref_radii)
    ref_radii_norm_all = np.concatenate(ref_radii_norm)
    ref_stats = compute_radii_stats(ref_radii_all)
    print("\n--- Reference (training data) radii stats ---")
    for k, v in ref_stats.items():
        print(f"  {k}: {v:.4f}")

    # --- Generate samples ---
    template = dataset[0].to(device)
    all_radii: list[np.ndarray] = []
    per_sample_metrics: list[dict] = []  # type: ignore[type-arg]

    # Scale factor for radial_norm reconstruction
    r_scale_factor = (
        template.r_scale.item() if hasattr(template, "r_scale") else 1.0
    )

    all_radii_norm: list[np.ndarray] = []

    do_smooth = args.smooth_modes > 0
    do_upsample = args.upsample > 0
    post_processing_active = do_smooth or do_upsample
    if do_smooth:
        print(f"  Fourier smooth: keeping {args.smooth_modes} modes")
    if do_upsample:
        print(f"  Spline upsample: {len(template.pos)} → {args.upsample} nodes")

    # Raw (pre-post-processing) radii kept for comparison plot
    all_radii_raw: list[np.ndarray] = []

    print(f"\nGenerating {args.n_samples} samples...")
    with torch.no_grad():
        for i in range(args.n_samples):
            torch.manual_seed(i)
            result = model.sample(template, clamp_range=clamp_range)
            # r_sorted_norm: in the model's feature space (normalised for radial_norm)
            r_sorted_norm, _ = extract_sorted_radii(result, template)
            # boundary violations checked in feature space (against sampling bounds)
            bv = compute_boundary_violations(r_sorted_norm, r_min, r_max)
            all_radii_norm.append(r_sorted_norm)
            # scale to physical coordinates
            r_sorted = r_sorted_norm * r_scale_factor
            all_radii_raw.append(r_sorted)

            # --- Post-processing ---
            r_proc = r_sorted
            theta_proc = theta_template[np.argsort(theta_template)]
            if do_smooth:
                r_proc = fourier_smooth(r_proc, args.smooth_modes)
            if do_upsample:
                theta_proc, r_proc = spline_upsample(theta_proc, r_proc, args.upsample)
            all_radii.append(r_proc)

            metrics: dict = {  # type: ignore[type-arg]
                "sample": i,
                "radii_stats": compute_radii_stats(r_proc),
                "smoothness": compute_smoothness(r_proc),
                "smoothness_raw": compute_smoothness(r_sorted),
                "closure_error": compute_closure_error(r_proc),
                "boundary_violations": bv,
                "circularity_cv": compute_circularity(r_proc),
            }
            per_sample_metrics.append(metrics)

    theta_template = np.arctan2(
        template.pos[:, 1].cpu().numpy(),
        template.pos[:, 0].cpu().numpy(),
    )
    theta_sorted = theta_template[np.argsort(theta_template)]

    gen_radii_all = np.concatenate(all_radii)
    gen_stats = compute_radii_stats(gen_radii_all)

    # --- Aggregate metrics ---
    avg_smoothness = float(np.mean([m["smoothness"] for m in per_sample_metrics]))
    avg_closure = float(np.mean([m["closure_error"] for m in per_sample_metrics]))
    avg_cv = float(np.mean([m["circularity_cv"] for m in per_sample_metrics]))
    avg_violations = float(
        np.mean([m["boundary_violations"]["total"] for m in per_sample_metrics])
    )
    ks_stat = ks_statistic(ref_radii_all, gen_radii_all)
    # Shape-space KS: normalised radii — removes cross-ellipse scale variation
    # and measures pure shape distribution match (meaningful for radial_norm)
    gen_radii_norm_all = np.concatenate(all_radii_norm)
    ks_shape = ks_statistic(ref_radii_norm_all, gen_radii_norm_all)

    print(f"\n--- Generated ({args.n_samples} shapes) radii stats ---")
    for k, v in gen_stats.items():
        print(f"  {k}: {v:.4f}")
    print(f"\n--- Quality metrics (averaged over {args.n_samples} samples) ---")
    print(f"  Smoothness (2nd-order diff):  {avg_smoothness:.4f}")
    print(f"  Closure error (|r_0 - r_N|): {avg_closure:.4f}")
    print(f"  Circularity (CV of radii):   {avg_cv:.4f}")
    print(f"  Boundary violation rate:     {avg_violations:.4f}")
    print(f"  KS statistic (physical r):   {ks_stat:.4f}")
    print(f"  KS statistic (shape-space):  {ks_shape:.4f}  [r/r_mean]")

    avg_smoothness_raw = float(
        np.mean([m["smoothness_raw"] for m in per_sample_metrics])
    )

    # --- Save metrics ---
    report = {
        "reference_stats": ref_stats,
        "generated_stats": gen_stats,
        "aggregate_metrics": {
            "smoothness": avg_smoothness,
            "smoothness_raw": avg_smoothness_raw,
            "closure_error": avg_closure,
            "circularity_cv": avg_cv,
            "boundary_violation_rate": avg_violations,
            "ks_statistic": ks_stat,
            "ks_shape_space": ks_shape,
        },
        "post_processing": {
            "fourier_smooth_modes": args.smooth_modes if do_smooth else None,
            "spline_upsample_nodes": args.upsample if do_upsample else None,
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
    _plot_quality_distributions(per_sample_metrics, exp_dir)
    if post_processing_active:
        _plot_surface_finish(
            all_radii_raw,
            all_radii,
            theta_template[np.argsort(theta_template)],
            exp_dir,
            smooth_modes=args.smooth_modes if do_smooth else None,
            upsample=args.upsample if do_upsample else None,
        )

    if args.visualize_diffusion:
        print("\nGenerating diffusion process visualisation...")
        torch.manual_seed(0)
        _, trajectory = model.sample_with_trajectory(
            template,
            n_snapshots=args.n_snapshots,
            clamp_range=clamp_range,
        )
        _plot_diffusion_process(trajectory, template, exp_dir)

    print(f"\n{'=' * 50}")
    print("Output files:")
    for p in sorted(exp_dir.iterdir()):
        if p.suffix in (".png", ".json", ".pt"):
            print(f"  {p}")
    print(f"{'=' * 50}")


def _plot_loss_curves(exp_dir: Path) -> None:
    """Training and validation loss with best-epoch marker."""
    log_path = exp_dir / "loss_log.json"
    if not log_path.exists():
        print("  No loss_log.json found, skipping loss curves.")
        return

    with open(log_path) as f:
        loss_log = json.load(f)

    epochs = [e["epoch"] for e in loss_log]
    train_loss = [e["train_loss"] for e in loss_log]
    val_loss = [e["val_loss"] for e in loss_log]
    best_ep = epochs[int(np.argmin(val_loss))]
    best_vl = min(val_loss)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="Train loss", linewidth=1.5)
    ax.plot(epochs, val_loss, label="Val loss", linewidth=1.5, alpha=0.85)
    ax.axvline(
        best_ep,
        color="r",
        linestyle="--",
        alpha=0.6,
        label=f"Best val {best_vl:.4f} @ ep{best_ep}",
    )
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
    ref_radii: np.ndarray,
    gen_radii: np.ndarray,
    exp_dir: Path,
) -> None:
    """Overlay histograms of reference vs generated radii distributions."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ref_radii, bins=60, alpha=0.5, density=True, label="Training data")
    ax.hist(gen_radii, bins=60, alpha=0.5, density=True, label="Generated")
    ax.set_xlabel("Radius r")
    ax.set_ylabel("Density")
    ax.set_title("Radii Distribution: Training vs Generated")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(exp_dir / "radii_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved radii_histogram.png")


def _plot_sample_gallery(
    all_radii: list[np.ndarray],
    template: Data,
    exp_dir: Path,
) -> None:
    """Gallery of up to 16 generated shapes with template overlay."""
    n_show = min(16, len(all_radii))
    cols = 4
    rows = (n_show + cols - 1) // cols

    theta = np.arctan2(
        template.pos[:, 1].cpu().numpy(),
        template.pos[:, 0].cpu().numpy(),
    )
    order = np.argsort(theta)
    theta_sorted = theta[order]

    # Template boundary for reference overlay (scale back for radial_norm)
    r_scale_factor = template.r_scale.item() if hasattr(template, "r_scale") else 1.0
    r_template = template.x[:, 0].cpu().numpy()[order] * r_scale_factor
    xt = np.append(
        r_template * np.cos(theta_sorted),
        r_template[0] * np.cos(theta_sorted[0]),
    )
    yt = np.append(
        r_template * np.sin(theta_sorted),
        r_template[0] * np.sin(theta_sorted[0]),
    )

    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))
    axes = np.array(axes).flatten()

    for i in range(n_show):
        ax = axes[i]
        r = all_radii[i]
        xc = np.append(r * np.cos(theta_sorted), r[0] * np.cos(theta_sorted[0]))
        yc = np.append(r * np.sin(theta_sorted), r[0] * np.sin(theta_sorted[0]))

        ax.plot(xt, yt, color="0.7", linewidth=0.8, label="template")
        ax.plot(xc, yc, "b-", linewidth=1.2, label="generated")
        ax.set_aspect("equal")
        ax.set_title(f"Sample {i + 1}", fontsize=9)
        ax.grid(True, alpha=0.2)

    for i in range(n_show, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Generated Shape Gallery", fontsize=13)
    fig.tight_layout()
    fig.savefig(exp_dir / "sample_gallery.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved sample_gallery.png")


def _plot_quality_distributions(
    per_sample_metrics: list[dict],  # type: ignore[type-arg]
    exp_dir: Path,
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
    """Visualise the reverse diffusion process from noise to final shape."""
    theta = np.arctan2(
        template.pos[:, 1].cpu().numpy(),
        template.pos[:, 0].cpu().numpy(),
    )
    order = np.argsort(theta)
    theta_sorted = theta[order]

    n_panels = len(trajectory)
    cols = min(n_panels, 5)
    rows = (n_panels + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))
    axes = np.array(axes).flatten()

    for i, (timestep, x_t) in enumerate(trajectory):
        ax = axes[i]
        r = x_t[:, 0].numpy()[order]
        xc = np.append(r * np.cos(theta_sorted), r[0] * np.cos(theta_sorted[0]))
        yc = np.append(r * np.sin(theta_sorted), r[0] * np.sin(theta_sorted[0]))

        ax.plot(xc, yc, "b-", linewidth=1.2, alpha=0.9)
        ax.set_aspect("equal")
        ax.set_title(f"t = {timestep}", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.2)

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


def _plot_surface_finish(
    all_radii_raw: list[np.ndarray],
    all_radii_proc: list[np.ndarray],
    theta_raw: np.ndarray,
    exp_dir: Path,
    smooth_modes: int | None,
    upsample: int | None,
) -> None:
    """Side-by-side overlay of raw vs post-processed boundary for 4 samples.

    Left column: full shape (Cartesian). Right column: zoomed curvature detail
    (radius vs angle) showing the smoothing effect.
    """
    n_show = min(4, len(all_radii_raw))
    fig, axes = plt.subplots(n_show, 2, figsize=(12, 3.5 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    label_parts = []
    if smooth_modes:
        label_parts.append(f"Fourier K={smooth_modes}")
    if upsample:
        label_parts.append(f"spline ×{upsample}")
    proc_label = " + ".join(label_parts)

    for i in range(n_show):
        r_raw = all_radii_raw[i]
        r_proc = all_radii_proc[i]

        # Determine angle arrays
        theta_r = theta_raw  # angles for raw (same length as r_raw)
        if upsample and len(r_proc) != len(r_raw):
            theta_p = np.linspace(
                theta_raw[0], theta_raw[0] + 2 * np.pi, len(r_proc), endpoint=False
            )
        else:
            theta_p = theta_raw

        # --- Left: full Cartesian shape ---
        ax_shape = axes[i, 0]
        xr = np.append(r_raw * np.cos(theta_r), r_raw[0] * np.cos(theta_r[0]))
        yr = np.append(r_raw * np.sin(theta_r), r_raw[0] * np.sin(theta_r[0]))
        xp = np.append(r_proc * np.cos(theta_p), r_proc[0] * np.cos(theta_p[0]))
        yp = np.append(r_proc * np.sin(theta_p), r_proc[0] * np.sin(theta_p[0]))
        ax_shape.plot(xr, yr, color="0.65", linewidth=1.0, label="raw")
        ax_shape.plot(xp, yp, "b-", linewidth=1.4, label=proc_label)
        ax_shape.set_aspect("equal")
        ax_shape.set_title(f"Sample {i + 1} — shape", fontsize=9)
        ax_shape.legend(fontsize=7, loc="upper right")
        ax_shape.grid(True, alpha=0.2)

        # --- Right: r(θ) curvature detail ---
        ax_curve = axes[i, 1]
        ax_curve.plot(theta_r, r_raw, color="0.65", linewidth=1.0, label="raw")
        ax_curve.plot(theta_p, r_proc, "b-", linewidth=1.4, label=proc_label)
        ax_curve.set_xlabel("θ (rad)", fontsize=8)
        ax_curve.set_ylabel("r", fontsize=8)
        ax_curve.set_title(f"Sample {i + 1} — r(θ)", fontsize=9)
        ax_curve.legend(fontsize=7)
        ax_curve.grid(True, alpha=0.2)

    fig.suptitle("Surface finish: raw generated vs post-processed", fontsize=12)
    fig.tight_layout()
    fig.savefig(exp_dir / "surface_finish.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved surface_finish.png")


if __name__ == "__main__":
    main()
