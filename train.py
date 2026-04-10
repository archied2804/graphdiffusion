"""
train.py — Training script for graph_diffusion
================================================

Loads configuration from YAML, instantiates the full pipeline
(dataset, data loader, noise schedule, score network, diffusion model),
and runs a training loop.

Usage:
    python train.py --config configs/default.yaml --epochs 100 --device cuda
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch_geometric.data import Data
from tqdm import tqdm

# Ensure src/ is on the path for editable installs
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from graph_diffusion.building_blocks.noise_schedule import NoiseSchedule
from graph_diffusion.data.dataloader import GraphDataLoader
from graph_diffusion.data.dataset import SyntheticGraphDataset
from graph_diffusion.model.graph_diffusion_model import GraphDiffusionModel
from graph_diffusion.model.score_network import ScoreNetwork


def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path) as f:  # noqa: PTH123
        return yaml.safe_load(f)


def add_edge_attr(batch: Data) -> Data:
    """Compute edge lengths as edge features from node positions."""
    src, dst = batch.edge_index
    diff = batch.pos[src] - batch.pos[dst]
    batch.edge_attr = diff.norm(dim=-1, keepdim=True)
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Train graph diffusion model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
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
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device)

    # --- Dataset ---
    ds_cfg = config.get("synthetic_dataset", {})
    n_nodes_range = tuple(ds_cfg.get("n_nodes_range", [20, 50]))
    dataset = SyntheticGraphDataset(
        root="data/synthetic",
        n_graphs=ds_cfg.get("n_graphs", 1000),
        n_nodes_range=n_nodes_range,
        node_feature_dim=ds_cfg.get("node_feature_dim", 8),
        k=ds_cfg.get("k", 6),
        global_dim=ds_cfg.get("global_dim", 16),
        seed=ds_cfg.get("seed", 42),
    )

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

    # --- Model ---
    ns_cfg = config.get("noise_schedule", {})
    noise_schedule = NoiseSchedule(
        T=ns_cfg.get("T", 1000),
        schedule_type=ns_cfg.get("schedule_type", "cosine"),
        beta_start=ns_cfg.get("beta_start", 1e-4),
        beta_end=ns_cfg.get("beta_end", 0.02),
    )

    sn_cfg = config.get("score_network", {})
    mlp_cfg = config.get("mlp", {})

    # Edge dim is 1 (edge length) for synthetic data
    edge_dim = 1
    # Align score network dims with synthetic dataset
    ds_global_dim = ds_cfg.get("global_dim", 16)
    ds_node_dim = ds_cfg.get("node_feature_dim", 8)
    score_network = ScoreNetwork(
        node_dim=ds_node_dim,
        edge_dim=edge_dim,
        global_dim=ds_global_dim,
        time_embed_dim=sn_cfg.get("time_embed_dim", 128),
        n_layers=sn_cfg.get("n_layers", 6),
        hidden_dims=sn_cfg.get("hidden_dims", [256, 256]),
        activation=mlp_cfg.get("activation", "silu"),
        layer_norm=mlp_cfg.get("layer_norm", True),
        residual=mlp_cfg.get("residual", True),
    )

    model = GraphDiffusionModel(
        score_network=score_network,
        noise_schedule=noise_schedule,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Training on: {device}")
    print(f"Dataset: {len(dataset)} graphs, {args.epochs} epochs")

    # --- Training ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
            batch = add_edge_attr(batch)
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
                batch = add_edge_attr(batch)
                batch = batch.to(device)
                loss = model.compute_loss(batch)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={avg_loss:.4f}  val_loss={avg_val_loss:.4f}"
        )

    print("Training complete.")


if __name__ == "__main__":
    main()
