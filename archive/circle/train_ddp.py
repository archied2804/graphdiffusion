"""
train_ddp.py — Multi-GPU DDP training script for graph_diffusion
=================================================================

Distributed Data Parallel training template.  Launched via ``torchrun``::

    # Single-node, 4 GPUs:
    torchrun --standalone --nproc_per_node=4 train_ddp.py \\
        --config configs/default.yaml --epochs 100

    # With AMP (bfloat16):
    torchrun --standalone --nproc_per_node=4 train_ddp.py \\
        --config configs/default.yaml --epochs 100 --amp

Environment variables set by ``torchrun``:
    LOCAL_RANK, RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817
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


def setup_ddp(backend: str = "nccl") -> tuple[int, int, int]:
    """Initialise the distributed process group.

    Returns:
        tuple[int, int, int]: ``(local_rank, global_rank, world_size)``.
    """
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return local_rank, global_rank, world_size


def cleanup_ddp() -> None:
    """Destroy the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train graph diffusion model (multi-GPU DDP)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        help="Scale learning rate linearly by world_size",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable automatic mixed precision (bfloat16)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints (rank 0 only)",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="Save checkpoint every N epochs (0 = final only)",
    )
    args = parser.parse_args()

    # --- DDP setup ---
    local_rank, global_rank, world_size = setup_ddp(backend="nccl")
    device = torch.device("cuda", local_rank)
    is_main = global_rank == 0

    config = load_config(args.config)

    if is_main:
        print(f"DDP: {world_size} processes, backend=nccl")
        print(f"Local rank {local_rank} on {torch.cuda.get_device_name(device)}")

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

    # --- Data Loader (distributed) ---
    data_cfg = config.get("data", {})
    loader = GraphDataLoader(
        dataset,
        batch_size=data_cfg.get("batch_size", 32),
        val_split=data_cfg.get("val_split", 0.1),
        num_workers=data_cfg.get("num_workers", 0),
        shuffle=data_cfg.get("shuffle", True),
        seed=42,
        distributed=True,
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

    edge_dim = 1
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

    # Wrap in DDP
    dist_cfg = config.get("distributed", {})
    model_ddp: nn.Module = DDP(
        model,
        device_ids=[local_rank],
        find_unused_parameters=dist_cfg.get("find_unused_parameters", False),
    )

    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"Model parameters: {n_params:,}")
        print(f"Training on: {world_size}x {torch.cuda.get_device_name(device)}")
        print(f"Dataset: {len(dataset)} graphs, {args.epochs} epochs")

    # --- Optimiser ---
    lr = args.lr
    if args.scale_lr:
        lr = lr * world_size
        if is_main:
            print(f"LR scaled by world_size: {lr}")

    optimizer = torch.optim.Adam(model_ddp.parameters(), lr=lr)

    # --- AMP setup ---
    amp_dtype = torch.bfloat16 if args.amp else None
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    if is_main and args.amp:
        print("AMP enabled (bfloat16)")

    # --- Checkpoint directory ---
    if is_main:
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # --- Training ---
    for epoch in range(1, args.epochs + 1):
        model_ddp.train()
        loader.set_epoch(epoch)

        epoch_loss = 0.0
        n_batches = 0

        train_iter = loader.train_loader()
        pbar = (
            tqdm(train_iter, desc=f"Epoch {epoch:3d}/{args.epochs}", leave=False)
            if is_main
            else train_iter
        )
        for batch in pbar:
            batch = add_edge_attr(batch)
            batch = batch.to(device)

            optimizer.zero_grad()

            with torch.autocast("cuda", dtype=amp_dtype, enabled=args.amp):
                loss = model_ddp.module.compute_loss(batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1
            if is_main:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(n_batches, 1)

        # --- Validation ---
        model_ddp.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in loader.val_loader():
                batch = add_edge_attr(batch)
                batch = batch.to(device)
                with torch.autocast("cuda", dtype=amp_dtype, enabled=args.amp):
                    loss = model_ddp.module.compute_loss(batch)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)

        if is_main:
            print(
                f"Epoch {epoch:3d}/{args.epochs}  "
                f"train_loss={avg_loss:.4f}  val_loss={avg_val_loss:.4f}"
            )

        # --- Checkpoint (rank 0 only) ---
        if is_main and args.save_every > 0 and epoch % args.save_every == 0:
            ckpt_path = Path(args.checkpoint_dir) / f"model_epoch{epoch:04d}.pt"
            torch.save(model_ddp.module.state_dict(), ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # --- Final checkpoint ---
    if is_main:
        ckpt_path = Path(args.checkpoint_dir) / "model_final.pt"
        torch.save(model_ddp.module.state_dict(), ckpt_path)
        print(f"Training complete. Final checkpoint: {ckpt_path}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
