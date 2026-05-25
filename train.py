"""
train.py — Aerodynamic boundary mesh shape generation
=======================================================

Train a DDPM over pOnEllipse boundary node positions so that sampling
produces novel, physically plausible surface mesh configurations.

The script loads a YAML config, instantiates pOnEllipseDataset, builds the
ScoreNetwork + GraphDiffusionModel pipeline, runs the training loop, and
saves a checkpoint plus generated-shape plots.

Supported experiments:
  EXP-010  radial shape generation
           (feature_mode=radial, dataset_type=shape)
  EXP-011  radial + cartesian shape ablation
           (feature_mode=radial or cartesian)
  EXP-012  radial_norm shape generation
           (feature_mode=radial_norm)

Usage:
    python train.py --config configs/EXP-010_ellipse_data_pipeline.yaml --epochs 200
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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from graph_diffusion.building_blocks.feature_transforms import (
    FeatureTransform,
    LogitNormTransform,
)
from graph_diffusion.building_blocks.noise_schedule import NoiseSchedule
from graph_diffusion.data.dataloader import GraphDataLoader
from graph_diffusion.data.pOnEllipse import pOnEllipseDataset
from graph_diffusion.data.pOnEllipseConditional import pOnEllipseConditionalDataset
from graph_diffusion.data.transforms import (
    ComputeAngularEdgeFeatures,
    ComputeArcLengthEdgeFeatures,
)
from graph_diffusion.model.graph_diffusion_model import GraphDiffusionModel
from graph_diffusion.model.pressure_head import PressurePredictionHead
from graph_diffusion.model.score_network import ScoreNetwork


def load_config(path: str) -> dict:  # type: ignore[type-arg]
    """Load YAML configuration file."""
    with open(path) as f:  # noqa: PTH123
        return yaml.safe_load(f)


def _build_dataset(config: dict) -> pOnEllipseDataset:  # type: ignore[type-arg]
    """Instantiate pOnEllipseDataset (or its conditional variant) from config."""
    ds_cfg = config.get("ellipse_dataset", {})
    feature_mode = ds_cfg.get("feature_mode", "radial")
    pre_transform = (
        ComputeArcLengthEdgeFeatures()
        if feature_mode == "cartesian"
        else ComputeAngularEdgeFeatures()
    )
    common = {
        "root": ds_cfg.get("root", "data/ellipse"),
        "feature_mode": feature_mode,
        "split": ds_cfg.get("split", "train"),
        "n_samples": ds_cfg.get("n_samples", None),
        "k_neighbors": ds_cfg.get("k_neighbors", 2),
        "global_dim": ds_cfg.get("global_dim", 8),
        "pre_transform": pre_transform,
    }
    if "cond_mode" in ds_cfg:
        return pOnEllipseConditionalDataset(
            cond_mode=ds_cfg.get("cond_mode", "fourier"),
            k_modes=ds_cfg.get("k_modes", 8),
            **common,
        )
    return pOnEllipseDataset(**common)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train aerodynamic boundary mesh shape generator"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/EXP-010_ellipse_data_pipeline.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="Number of shapes to generate after training",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="generated_shapes.png",
        help="Path to save generated shapes plot",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device)

    # --- Feature transform (optional bounded diffusion) ---
    feature_transform: FeatureTransform | None = None
    ft_cfg = config.get("feature_transform")
    if ft_cfg and ft_cfg.get("type") == "logit_norm":
        feature_transform = LogitNormTransform(
            r_min=float(ft_cfg.get("r_min", 0.5)),
            r_max=float(ft_cfg.get("r_max", 1.5)),
        )
        print(
            f"Using LogitNormTransform(r_min={ft_cfg.get('r_min')}, "
            f"r_max={ft_cfg.get('r_max')})"
        )

    # --- Dataset ---
    dataset = _build_dataset(config)

    # --- Data Loader ---
    data_cfg = config.get("data", {})
    loader = GraphDataLoader(
        dataset,
        batch_size=data_cfg.get("batch_size", 32),
        val_split=data_cfg.get("val_split", 0.1),
        num_workers=data_cfg.get("num_workers", 0),
        shuffle=data_cfg.get("shuffle", True),
        seed=42,
    )

    # --- Noise schedule ---
    ns_cfg = config.get("noise_schedule", {})
    noise_schedule = NoiseSchedule(
        T=ns_cfg.get("T", 200),
        schedule_type=ns_cfg.get("schedule_type", "cosine"),
        beta_start=ns_cfg.get("beta_start", 1e-4),
        beta_end=ns_cfg.get("beta_end", 0.02),
    )

    # --- Score network ---
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
        input_dim=sn_cfg.get("input_dim", None),
        cond_dim=sn_cfg.get("cond_dim", None),
        p_uncond=float(sn_cfg.get("p_uncond", 0.0)),
        output_dim=sn_cfg.get("output_dim", None),
    )

    # --- Pressure prediction head (EXP-020) ---
    pressure_head: PressurePredictionHead | None = None
    ph_cfg = config.get("pressure_head")
    if ph_cfg is not None:
        pressure_head = PressurePredictionHead(
            in_dim=ph_cfg["in_dim"],
            out_dim=ph_cfg["out_dim"],
            node_hidden=ph_cfg.get("node_hidden", [64, 64]),
            global_hidden=ph_cfg.get("global_hidden", [64, 64]),
            node_embed_dim=ph_cfg.get("node_embed_dim", 64),
            activation=mlp_cfg.get("activation", "silu"),
            layer_norm=mlp_cfg.get("layer_norm", True),
        )

    # --- Diffusion model ---
    model_cfg = config.get("model", {})
    model = GraphDiffusionModel(
        score_network=score_network,
        noise_schedule=noise_schedule,
        feature_transform=feature_transform,
        n_noise_channels=model_cfg.get("n_noise_channels", None),
        smoothness_weight=float(model_cfg.get("smoothness_weight", 0.0)),
        pressure_head=pressure_head,
        lambda_pressure=float(model_cfg.get("lambda_pressure", 0.0)),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Training on: {device}")
    print(f"Dataset: {len(dataset)} graphs, {args.epochs} epochs")

    clamp_cfg = config.get("clamp_range")
    clamp_range: tuple[float, float] | None = None
    if clamp_cfg is not None:
        clamp_range = (float(clamp_cfg[0]), float(clamp_cfg[1]))

    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    training_cfg = config.get("training", {})

    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    if training_cfg.get("scheduler") == "cosine_annealing":
        eta_min = float(training_cfg.get("eta_min", 1e-5))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=eta_min
        )

    early_stopping_patience: int | None = training_cfg.get("early_stopping_patience")
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    loss_log: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(
            loader.train_loader(),
            desc=f"Epoch {epoch:3d}/{args.epochs}",
            leave=False,
        )
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = model.compute_loss(batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(n_batches, 1)
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in loader.val_loader():
                batch = batch.to(device)
                loss = model.compute_loss(batch)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        loss_log.append(
            {"epoch": epoch, "train_loss": avg_loss, "val_loss": avg_val_loss}
        )
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={avg_loss:.4f}  val_loss={avg_val_loss:.4f}"
        )

        if scheduler is not None:
            scheduler.step()

        if early_stopping_patience is not None:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                patience_counter = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config,
                        "epoch": epoch,
                        "lr": args.lr,
                    },
                    output_dir / "checkpoint_best.pt",
                )
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(
                        f"Early stopping at epoch {epoch} "
                        f"(best val_loss={best_val_loss:.4f} at epoch {best_epoch})"
                    )
                    break

    writer.close()
    print("Training complete.")

    checkpoint_path = output_dir / "checkpoint.pt"
    epochs_run = loss_log[-1]["epoch"] if loss_log else 0
    if (
        early_stopping_patience is not None
        and (output_dir / "checkpoint_best.pt").exists()
    ):
        import shutil
        shutil.copy(output_dir / "checkpoint_best.pt", checkpoint_path)
        print(f"Saved best checkpoint (epoch {best_epoch}) to {checkpoint_path}")
    else:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "epoch": epochs_run,
                "lr": args.lr,
            },
            checkpoint_path,
        )

    with open(output_dir / "loss_log.json", "w") as f:
        json.dump(loss_log, f, indent=2)
    print(f"Saved loss log to {output_dir / 'loss_log.json'}")

    # --- Generate and plot sample shapes ---
    model.eval()
    ds_cfg = config.get("ellipse_dataset", {})
    feature_mode = ds_cfg.get("feature_mode", "radial")
    template = dataset[0].to(device)

    fig, axes = plt.subplots(1, args.n_samples, figsize=(4 * args.n_samples, 4))
    if args.n_samples == 1:
        axes = [axes]

    sampling_cfg = config.get("sampling", {})
    guidance_scale = float(sampling_cfg.get("guidance_scale", 1.0))
    dps_guidance_weight = float(sampling_cfg.get("dps_guidance_weight", 0.0))

    for i, ax in enumerate(axes):
        torch.manual_seed(i)
        result = model.sample(
            template,
            clamp_range=clamp_range,
            guidance_scale=guidance_scale,
            dps_guidance_weight=dps_guidance_weight,
        )

        if feature_mode in ("radial", "radial_norm", "normalised"):
            r_raw = result.x[:, 0].cpu().numpy()
            theta = np.arctan2(
                template.pos[:, 1].cpu().numpy(),
                template.pos[:, 0].cpu().numpy(),
            )
            order = np.argsort(theta)
            theta_s = theta[order]
            r_scale = (
                template.r_scale.item() if hasattr(template, "r_scale") else 1.0
            )
            r_s = r_raw[order] * r_scale
            xc = np.append(r_s * np.cos(theta_s), r_s[0] * np.cos(theta_s[0]))
            yc = np.append(r_s * np.sin(theta_s), r_s[0] * np.sin(theta_s[0]))
            r_ref = template.x[:, 0].cpu().numpy()[order] * r_scale
            xr = np.append(r_ref * np.cos(theta_s), r_ref[0] * np.cos(theta_s[0]))
            yr = np.append(r_ref * np.sin(theta_s), r_ref[0] * np.sin(theta_s[0]))
            ax.plot(xr, yr, color="0.7", linewidth=1.0, label="template")
            ax.plot(xc, yc, "b-", linewidth=1.5, label="generated")
            ax.set_aspect("equal")
            ax.legend(fontsize=7, loc="upper right")
        else:  # cartesian
            xc_pred = result.x[:, 0].cpu().numpy()
            yc_pred = result.x[:, 1].cpu().numpy()
            xc_pred = np.append(xc_pred, xc_pred[0])
            yc_pred = np.append(yc_pred, yc_pred[0])
            xt = template.x[:, 0].cpu().numpy()
            yt = template.x[:, 1].cpu().numpy()
            xt = np.append(xt, xt[0])
            yt = np.append(yt, yt[0])
            ax.plot(xt, yt, color="0.7", linewidth=1.0, label="template")
            ax.plot(xc_pred, yc_pred, "b-", linewidth=1.5, label="generated")
            ax.set_aspect("equal")
            ax.legend(fontsize=7, loc="upper right")

        ax.set_title(f"Sample {i + 1}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved generated shapes to {args.output}")


if __name__ == "__main__":
    main()
