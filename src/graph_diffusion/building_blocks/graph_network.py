"""
graph_diffusion.building_blocks.graph_network
===============================================

Full Battaglia et al. (2018) graph network block with optional edge, node,
and global update functions.  Uses ``torch_scatter.scatter`` directly for
all aggregation operations.
"""

import torch
import torch.nn as nn
from torch_scatter import scatter

from graph_diffusion.building_blocks.mlp import MLP

__all__ = [
    "EdgeModel",
    "NodeModel",
    "GlobalModel",
    "GraphNetworkBlock",
]


class EdgeModel(nn.Module):
    """Edge update function (phi^e) from the Battaglia GN formulation.

    Concatenates per-edge: ``[e_k || v_{s_k} || v_{r_k} || u]`` and passes
    through an MLP to produce updated edge features.

    Args:
        edge_dim (int): Dimensionality of edge features.
        node_dim (int): Dimensionality of node features.
        global_dim (int): Dimensionality of global features.
        hidden_dims (list[int]): Hidden layer widths for the internal MLP.
        activation (str): Activation function name. Defaults to ``"silu"``.
        layer_norm (bool): Whether to apply LayerNorm in the MLP.
            Defaults to ``True``.
        residual (bool): Whether to use a residual connection in the MLP.
            Defaults to ``True``.
    """

    def __init__(
        self,
        edge_dim: int,
        node_dim: int,
        global_dim: int,
        hidden_dims: list[int],
        activation: str = "silu",
        layer_norm: bool = True,
        residual: bool = True,
    ) -> None:
        super().__init__()
        # Input: [e_k || v_{s_k} || v_{r_k} || u]
        in_dim = edge_dim + 2 * node_dim + global_dim
        self.mlp = MLP(
            in_dim,
            hidden_dims,
            edge_dim,
            activation=activation,
            layer_norm=layer_norm,
            residual=residual,
        )

    def forward(
        self,
        src: torch.Tensor,
        dest: torch.Tensor,
        edge_attr: torch.Tensor,
        u: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Compute updated edge features.

        Args:
            src (torch.Tensor): Source node features gathered per edge,
                shape ``(E, node_dim)``.
            dest (torch.Tensor): Destination node features gathered per edge,
                shape ``(E, node_dim)``.
            edge_attr (torch.Tensor): Current edge features,
                shape ``(E, edge_dim)``.
            u (torch.Tensor): Global features expanded to edges,
                shape ``(E, global_dim)``.
            batch (torch.Tensor): Batch assignment vector (unused here, kept
                for interface consistency), shape ``(N_total,)``.

        Returns:
            torch.Tensor: Updated edge features, shape ``(E, edge_dim)``.
        """
        inp = torch.cat([edge_attr, src, dest, u], dim=-1)
        return self.mlp(inp)  # type: ignore[no-any-return]


class NodeModel(nn.Module):
    """Node update function (phi^v) from the Battaglia GN formulation.

    First aggregates incoming updated edge features per node via scatter
    mean, then concatenates ``[agg_edges || v_i || u]`` and passes through
    an MLP.

    Args:
        edge_dim (int): Dimensionality of edge features.
        node_dim (int): Dimensionality of node features.
        global_dim (int): Dimensionality of global features.
        hidden_dims (list[int]): Hidden layer widths for the internal MLP.
        activation (str): Activation function name. Defaults to ``"silu"``.
        layer_norm (bool): Whether to apply LayerNorm in the MLP.
            Defaults to ``True``.
        residual (bool): Whether to use a residual connection in the MLP.
            Defaults to ``True``.
    """

    def __init__(
        self,
        edge_dim: int,
        node_dim: int,
        global_dim: int,
        hidden_dims: list[int],
        activation: str = "silu",
        layer_norm: bool = True,
        residual: bool = True,
    ) -> None:
        super().__init__()
        # Input: [agg_edges || v_i || u]
        in_dim = edge_dim + node_dim + global_dim
        self.mlp = MLP(
            in_dim,
            hidden_dims,
            node_dim,
            activation=activation,
            layer_norm=layer_norm,
            residual=residual,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        u: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Compute updated node features.

        Args:
            x (torch.Tensor): Current node features, shape ``(N, node_dim)``.
            edge_index (torch.Tensor): COO edge list, shape ``(2, E)``.
            edge_attr (torch.Tensor): Updated edge features,
                shape ``(E, edge_dim)``.
            u (torch.Tensor): Global features expanded to nodes,
                shape ``(N, global_dim)``.
            batch (torch.Tensor): Batch assignment vector (unused directly
                here), shape ``(N,)``.

        Returns:
            torch.Tensor: Updated node features, shape ``(N, node_dim)``.
        """
        # Aggregate incoming edges per receiver node
        row, col = edge_index  # row = source, col = receiver
        agg_edges = scatter(edge_attr, col, dim=0, dim_size=x.size(0), reduce="mean")
        inp = torch.cat([agg_edges, x, u], dim=-1)
        return self.mlp(inp)  # type: ignore[no-any-return]


class GlobalModel(nn.Module):
    """Global update function (phi^u) from the Battaglia GN formulation.

    Aggregates edge and node features per graph via scatter mean, then
    concatenates ``[agg_edges || agg_nodes || u]`` and passes through an MLP.

    Args:
        edge_dim (int): Dimensionality of edge features.
        node_dim (int): Dimensionality of node features.
        global_dim (int): Dimensionality of global features.
        hidden_dims (list[int]): Hidden layer widths for the internal MLP.
        activation (str): Activation function name. Defaults to ``"silu"``.
        layer_norm (bool): Whether to apply LayerNorm in the MLP.
            Defaults to ``True``.
        residual (bool): Whether to use a residual connection in the MLP.
            Defaults to ``True``.
    """

    def __init__(
        self,
        edge_dim: int,
        node_dim: int,
        global_dim: int,
        hidden_dims: list[int],
        activation: str = "silu",
        layer_norm: bool = True,
        residual: bool = True,
    ) -> None:
        super().__init__()
        # Input: [agg_edges || agg_nodes || u]
        in_dim = edge_dim + node_dim + global_dim
        self.mlp = MLP(
            in_dim,
            hidden_dims,
            global_dim,
            activation=activation,
            layer_norm=layer_norm,
            residual=residual,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        u: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Compute updated global features.

        Args:
            x (torch.Tensor): Updated node features, shape ``(N, node_dim)``.
            edge_index (torch.Tensor): COO edge list, shape ``(2, E)``.
            edge_attr (torch.Tensor): Updated edge features,
                shape ``(E, edge_dim)``.
            u (torch.Tensor): Current global features, shape ``(B, global_dim)``.
            batch (torch.Tensor): Batch assignment vector,
                shape ``(N,)``.

        Returns:
            torch.Tensor: Updated global features, shape ``(B, global_dim)``.
        """
        n_graphs = u.size(0)

        # Map each edge to its graph via its source node's batch index
        edge_batch = batch[edge_index[0]]
        agg_edges = scatter(
            edge_attr, edge_batch, dim=0, dim_size=n_graphs, reduce="mean"
        )

        # Aggregate node features per graph
        agg_nodes = scatter(x, batch, dim=0, dim_size=n_graphs, reduce="mean")

        inp = torch.cat([agg_edges, agg_nodes, u], dim=-1)
        return self.mlp(inp)  # type: ignore[no-any-return]


class GraphNetworkBlock(nn.Module):
    """One step of the full Battaglia et al. (2018) graph network update.

    Composes optional edge, node, and global update functions. Each
    sub-model can be individually disabled to produce simpler GNN variants
    without code duplication.

    Args:
        node_dim (int): Dimensionality of node features.
        edge_dim (int): Dimensionality of edge features.
        global_dim (int): Dimensionality of global features.
        hidden_dims (list[int]): Hidden layer widths shared by all sub-MLPs.
        activation (str): Activation function name. Defaults to ``"silu"``.
        layer_norm (bool): Whether to apply LayerNorm. Defaults to ``True``.
        residual (bool): Whether to use residual connections.
            Defaults to ``True``.
        use_edge_model (bool): If ``True``, include the edge update.
            Defaults to ``True``.
        use_node_model (bool): If ``True``, include the node update.
            Defaults to ``True``.
        use_global_model (bool): If ``True``, include the global update.
            Defaults to ``True``.

    # TODO(extensibility): disable individual sub-models via constructor
    # flags to produce simpler GNN variants (e.g. edge-only, node-only).
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        global_dim: int,
        hidden_dims: list[int],
        activation: str = "silu",
        layer_norm: bool = True,
        residual: bool = True,
        use_edge_model: bool = True,
        use_node_model: bool = True,
        use_global_model: bool = True,
    ) -> None:
        super().__init__()

        self.use_edge_model = use_edge_model
        self.use_node_model = use_node_model
        self.use_global_model = use_global_model

        if use_edge_model:
            self.edge_model = EdgeModel(
                edge_dim=edge_dim,
                node_dim=node_dim,
                global_dim=global_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                layer_norm=layer_norm,
                residual=residual,
            )

        if use_node_model:
            self.node_model = NodeModel(
                edge_dim=edge_dim,
                node_dim=node_dim,
                global_dim=global_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                layer_norm=layer_norm,
                residual=residual,
            )

        if use_global_model:
            self.global_model = GlobalModel(
                edge_dim=edge_dim,
                node_dim=node_dim,
                global_dim=global_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                layer_norm=layer_norm,
                residual=residual,
            )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        u: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one full graph network update step.

        Args:
            x (torch.Tensor): Node features, shape ``(N_total, node_dim)``.
            edge_index (torch.Tensor): COO edge list, shape ``(2, E)``.
            edge_attr (torch.Tensor): Edge features, shape ``(E, edge_dim)``.
            u (torch.Tensor): Global features, shape ``(B, global_dim)``.
            batch (torch.Tensor): Batch vector, shape ``(N_total,)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Updated node features, shape ``(N_total, node_dim)``.
                - Updated edge features, shape ``(E, edge_dim)``.
                - Updated global features, shape ``(B, global_dim)``.
        """
        row, col = edge_index

        # 1. Edge update
        if self.use_edge_model:
            # Expand global to per-edge via source node's batch index
            edge_batch = batch[row]
            u_edge = u[edge_batch]
            edge_attr = self.edge_model(
                src=x[row],
                dest=x[col],
                edge_attr=edge_attr,
                u=u_edge,
                batch=batch,
            )

        # 2. Node update
        if self.use_node_model:
            # Expand global to per-node
            u_node = u[batch]
            x = self.node_model(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                u=u_node,
                batch=batch,
            )

        # 3. Global update
        if self.use_global_model:
            u = self.global_model(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                u=u,
                batch=batch,
            )

        return x, edge_attr, u
