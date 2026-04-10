"""
graph_diffusion.model.score_network
=====================================

Score network that predicts noise given a noisy graph and timestep.
Composes sinusoidal time embedding, stacked graph network blocks,
and an output projection MLP.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data

from graph_diffusion.building_blocks.graph_network import GraphNetworkBlock
from graph_diffusion.building_blocks.mlp import MLP, SinusoidalTimeEmbedding

__all__ = [
    "ScoreNetwork",
]


class ScoreNetwork(nn.Module):
    """Noise prediction network for graph diffusion.

    Takes a noisy graph ``(G_t, t)`` and predicts the noise ``epsilon``.
    The network is a pure function: it does not know about the diffusion
    process itself.

    Architecture:
        1. Sinusoidal time embedding projected to ``global_dim`` and added
           to the global attribute ``u``.
        2. ``n_layers`` stacked ``GraphNetworkBlock`` layers.
        3. Output MLP projecting node features back to the original
           node feature dimensionality.

    Args:
        node_dim (int): Dimensionality of node features in the GN blocks.
        edge_dim (int): Dimensionality of edge features.
        global_dim (int): Dimensionality of global features.
        time_embed_dim (int): Dimensionality of the sinusoidal time
            embedding before projection.
        n_layers (int): Number of stacked ``GraphNetworkBlock`` layers.
        hidden_dims (list[int]): Hidden layer widths for all internal MLPs.
        activation (str): Activation function name. Defaults to ``"silu"``.
        layer_norm (bool): Whether to apply LayerNorm. Defaults to ``True``.
        residual (bool): Whether to use residual connections.
            Defaults to ``True``.

    Raises:
        ValueError: If ``n_layers < 1``.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        global_dim: int,
        time_embed_dim: int,
        n_layers: int,
        hidden_dims: list[int],
        activation: str = "silu",
        layer_norm: bool = True,
        residual: bool = True,
    ) -> None:
        super().__init__()

        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_proj = nn.Linear(time_embed_dim, global_dim)

        self.gn_layers = nn.ModuleList(
            [
                GraphNetworkBlock(
                    node_dim=node_dim,
                    edge_dim=edge_dim,
                    global_dim=global_dim,
                    hidden_dims=hidden_dims,
                    activation=activation,
                    layer_norm=layer_norm,
                    residual=residual,
                )
                for _ in range(n_layers)
            ]
        )

        self.output_proj = MLP(
            in_dim=node_dim,
            hidden_dims=hidden_dims,
            out_dim=node_dim,
            activation=activation,
            layer_norm=layer_norm,
            residual=residual,
        )

    def forward(
        self,
        data: Data,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise from a noisy graph and timestep.

        Args:
            data (Data): A PyG ``Data`` object containing ``x`` (noisy node
                features), ``edge_index``, ``edge_attr``, ``u`` (global), and
                ``batch``.
            t (torch.Tensor): Integer timesteps, shape ``(B,)``.

        Returns:
            torch.Tensor: Predicted noise ``eps_pred``, shape
                ``(N_total, F)`` — same shape as ``data.x``.
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        u = data.u
        batch = data.batch

        # Inject time embedding into global attribute
        t_emb = self.time_embedding(t)  # (B, time_embed_dim)
        u = u + self.time_proj(t_emb)  # (B, global_dim)

        # Pass through stacked GN layers
        for gn_layer in self.gn_layers:
            x, edge_attr, u = gn_layer(x, edge_index, edge_attr, u, batch)

        # Project node features to output
        eps_pred = self.output_proj(x)
        return eps_pred  # type: ignore[no-any-return]
