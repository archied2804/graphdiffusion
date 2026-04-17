"""
train_circle.py — Training script for radial diffusion on unit circle
======================================================================

Loads configuration from YAML, instantiates the circle experiment pipeline
(UnitCircleDataset, angular edge features, noise schedule, score network
with input projection, diffusion model), trains, and generates sample
shapes via matplotlib.

Usage:
    python train_circle.py --config configs/circle_radial.yaml --epochs 100
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

# Ensure src/ is on the path for editable installs
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from graph_diffusion.building_blocks.noise_schedule import NoiseSchedule
from graph_diffusion.data.circledataset import UnitCircleDataset
from graph_diffusion.data.dataloader import GraphDataLoader
from graph_diffusion.data.transforms import ComputeAngularEdgeFeatures
from graph_diffusion.model.graph_diffusion_model import GraphDiffusionModel
from graph_diffusion.model.score_network import ScoreNetwork


def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path) as f:  # noqa: PTH123
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train radial diffusion on unit circle"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/circle_radial.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
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

    # --- Dataset ---
    ds_cfg = config.get("circle_dataset", {})
    pre_transform = ComputeAngularEdgeFeatures()

    dataset = UnitCircleDataset(
        root="data/circle",
        n_graphs=ds_cfg.get("n_graphs", 2000),
        n_nodes=ds_cfg.get("n_nodes", 64),
        n_fourier_modes=ds_cfg.get("n_fourier_modes", 5),
        amplitude_scale=ds_cfg.get("amplitude_scale", 0.15),
        r_min=ds_cfg.get("r_min", 0.5),
        r_max=ds_cfg.get("r_max", 1.5),
        k_neighbors=ds_cfg.get("k_neighbors", 2),
        global_dim=ds_cfg.get("global_dim", 8),
        seed=ds_cfg.get("seed", 42),
        pre_transform=pre_transform,
    )

    # --- Data Loader ---
    data_cfg = config.get("data", {})
    loader = GraphDataLoader(
        dataset,
        batch_size=data_cfg.get("batch_size", 64),
        val_split=data_cfg.get("val_split", 0.1),
        num_workers=data_cfg.get("num_workers", 0),
        shuffle=data_cfg.get("shuffle", True),
        seed=42,
    )

    # --- Model ---
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

    model = GraphDiffusionModel(
        score_network=score_network,
        noise_schedule=noise_schedule,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Training on: {device}")
    print(f"Dataset: {len(dataset)} graphs, {args.epochs} epochs")

    # --- Clamp range ---
    clamp_cfg = config.get("clamp_range")
    clamp_range: tuple[float, float] | None = None
    if clamp_cfg is not None:
        clamp_range = (float(clamp_cfg[0]), float(clamp_cfg[1]))

    # --- Output directory ---
    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- TensorBoard ---
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    # --- Training ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
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

        # --- Validation ---
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
        writer.add_scalar("Loss/test", avg_val_loss, epoch)
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={avg_loss:.4f}  val_loss={avg_val_loss:.4f}"
        )

    writer.close()
    print("Training complete.")

    # --- Save checkpoint and loss log ---
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "epoch": args.epochs,
            "epochs": args.epochs,
            "lr": args.lr,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")

    loss_log_path = output_dir / "loss_log.json"
    with open(loss_log_path, "w") as f:
        json.dump(loss_log, f, indent=2)
    print(f"Saved loss log to {loss_log_path}")

    # --- Generate and plot sample shapes ---
    model.eval()
    template = dataset[0].to(device)

    fig, axes = plt.subplots(1, args.n_samples, figsize=(4 * args.n_samples, 4))
    if args.n_samples == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        torch.manual_seed(i)
        result = model.sample(template, clamp_range=clamp_range)

        r = result.x[:, 0].cpu().numpy()
        theta = np.arctan2(
            template.pos[:, 1].cpu().numpy(),
            template.pos[:, 0].cpu().numpy(),
        )

        # Sort by angle for smooth plotting
        order = np.argsort(theta)
        theta_sorted = theta[order]
        r_sorted = r[order]

        # Cartesian for polar plot
        x_cart = r_sorted * np.cos(theta_sorted)
        y_cart = r_sorted * np.sin(theta_sorted)

        # Close the loop
        x_cart = np.append(x_cart, x_cart[0])
        y_cart = np.append(y_cart, y_cart[0])

        ax.plot(x_cart, y_cart, "b-", linewidth=1.5)
        ax.set_aspect("equal")
        ax.set_title(f"Sample {i + 1}")
        ax.grid(True, alpha=0.3)

        # Draw reference unit circle
        t_ref = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(t_ref), np.sin(t_ref), "k--", alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved generated shapes to {args.output}")


if __name__ == "__main__":
    main()
