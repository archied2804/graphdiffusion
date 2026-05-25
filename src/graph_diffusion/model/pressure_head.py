"""
graph_diffusion.model.pressure_head
=====================================

Auxiliary prediction head that maps a clean-shape estimate to a
fixed-size pressure descriptor.

Used in the EXP-020 pressure-conditioned diffusion pipeline as an
inductive-bias term in the training loss: the diffusion model must
generate shapes whose head-predicted pressure matches the target
conditioning. This pushes the model toward learning a *forward map*
(shape → pressure) rather than a *lookup* (target → nearest training
ellipse), which is what enables novel-shape generation at inference
time when the target lies outside the training distribution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from graph_diffusion.building_blocks.mlp import MLP

__all__ = [
    "PressurePredictionHead",
]


class PressurePredictionHead(nn.Module):
    """DeepSets-style shape → pressure-descriptor regressor.

    Architecture:

        per-node MLP φ([x_0; pos])  →  scatter_mean over nodes within each
        graph  →  global MLP ρ  →  R^K

    The DeepSets form (per-node MLP then permutation-invariant
    aggregation) means the head is invariant to node ordering and works
    for variable node counts — required since pOnEllipseDataset graphs
    have 52–96 nodes.

    Args:
        in_dim (int): Width of the per-node input ``[x_0; pos]``. For
            ``feature_mode="radial_norm"`` this is ``1 + 2 = 3``.
        out_dim (int): Number of pressure descriptors to predict. For
            DCT conditioning this matches ``k_modes`` in the dataset.
        node_hidden (list[int]): Per-node MLP φ hidden widths.
        global_hidden (list[int]): Global MLP ρ hidden widths.
        node_embed_dim (int): Width of the per-node embedding produced
            by φ and aggregated. Defaults to ``64``.
        activation (str): MLP activation. Defaults to ``"silu"``.
        layer_norm (bool): Whether MLPs apply LayerNorm. Defaults to ``True``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        node_hidden: list[int],
        global_hidden: list[int],
        node_embed_dim: int = 64,
        activation: str = "silu",
        layer_norm: bool = True,
    ) -> None:
        super().__init__()

        if in_dim < 1:
            raise ValueError(f"in_dim must be >= 1, got {in_dim}")
        if out_dim < 1:
            raise ValueError(f"out_dim must be >= 1, got {out_dim}")

        self.node_mlp = MLP(
            in_dim=in_dim,
            hidden_dims=node_hidden,
            out_dim=node_embed_dim,
            activation=activation,
            layer_norm=layer_norm,
        )
        self.global_mlp = MLP(
            in_dim=node_embed_dim,
            hidden_dims=global_hidden,
            out_dim=out_dim,
            activation=activation,
            layer_norm=layer_norm,
        )

    def forward(
        self,
        x_0: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Predict pressure descriptors from a batched graph.

        Args:
            x_0 (torch.Tensor): Clean (or estimated-clean) node features,
                shape ``(N_total, F_x)``.
            pos (torch.Tensor): Node positions, shape ``(N_total, F_pos)``.
            batch (torch.Tensor): Batch assignment, shape ``(N_total,)``.

        Returns:
            torch.Tensor: Pressure descriptors, shape ``(B, out_dim)``,
                where ``B = batch.max() + 1``.
        """
        node_in = torch.cat([x_0, pos], dim=-1)
        node_emb = self.node_mlp(node_in)  # (N_total, node_embed_dim)

        n_graphs = int(batch.max().item()) + 1
        graph_emb = scatter(
            node_emb, batch, dim=0, dim_size=n_graphs, reduce="mean"
        )  # (B, node_embed_dim)

        return self.global_mlp(graph_emb)  # type: ignore[no-any-return]
