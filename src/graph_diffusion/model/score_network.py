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
        input_dim (int | None): Dimensionality of raw input node features
            when it differs from ``node_dim``.  When set, a linear
            ``input_proj`` lifts features to ``node_dim`` before the GN
            layers.  When ``None`` (default), no input projection is used.
        cond_dim (int | None): Dimensionality of an optional per-graph
            conditioning vector (e.g. global pressure summary in EXP-015).
            When set, a ``cond_proj: Linear(cond_dim, global_dim)`` is added
            and its output is summed into ``u`` alongside the time embedding.
            A learnable ``null_cond`` parameter of shape ``(cond_dim,)`` is
            also created for classifier-free guidance.
            Defaults to ``None``.
        p_uncond (float): Probability of dropping the conditioning signal
            during training (per graph) and replacing it with ``null_cond``.
            Only active when ``cond_dim`` is set. Defaults to ``0.0``
            (no dropout — disables CFG).
        output_dim (int | None): Override the output dimensionality of
            ``output_decode``.  Useful when the score network predicts noise
            for a subset of input channels (e.g. EXP-016 where only the
            radial channel is noised but the pressure channel is also in the
            input).  When ``None``, falls back to ``input_dim`` for backward
            compatibility.  Defaults to ``None``.

    Note:
        ``cond_dim`` and ``output_dim`` are future-work parameters for
        conditional inverse design (EXP-015: global pressure conditioning,
        EXP-016: node-level pressure conditioning, EXP-020: DCT-Fourier
        pressure conditioning with CFG). Leave as ``None`` for the
        unconditional shape-generation baseline.

    Raises:
        ValueError: If ``n_layers < 1``.
        ValueError: If ``p_uncond`` is not in ``[0, 1]``.
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
        input_dim: int | None = None,
        cond_dim: int | None = None,
        p_uncond: float = 0.0,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()

        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        if not 0.0 <= p_uncond <= 1.0:
            raise ValueError(f"p_uncond must be in [0, 1], got {p_uncond}")

        self.p_uncond = p_uncond

        # Optional input projection when raw feature dim != node_dim
        self.input_proj: nn.Linear | None = None
        if input_dim is not None and input_dim != node_dim:
            self.input_proj = nn.Linear(input_dim, node_dim)

        # Output decode: projects node_dim back to the desired output size.
        # Priority: output_dim (explicit) > input_dim (backward compat) > None.
        self.output_decode: nn.Linear | None = None
        effective_out = output_dim if output_dim is not None else input_dim
        if effective_out is not None and effective_out != node_dim:
            self.output_decode = nn.Linear(node_dim, effective_out)

        # Optional conditioning projection (EXP-015 global pressure summary,
        # EXP-020 DCT-Fourier pressure conditioning).
        self.cond_proj: nn.Linear | None = None
        self.null_cond: nn.Parameter | None = None
        if cond_dim is not None:
            self.cond_proj = nn.Linear(cond_dim, global_dim)
            # Learnable null token for classifier-free guidance.
            self.null_cond = nn.Parameter(torch.zeros(cond_dim))

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
        cond: torch.Tensor | None = None,
        force_uncond: bool = False,
    ) -> torch.Tensor:
        """Predict noise from a noisy graph, timestep, and optional conditioning.

        Args:
            data (Data): A PyG ``Data`` object containing ``x`` (noisy node
                features), ``edge_index``, ``edge_attr``, ``u`` (global), and
                ``batch``.
            t (torch.Tensor): Integer timesteps, shape ``(B,)``.
            cond (torch.Tensor | None): Optional per-graph conditioning vector,
                shape ``(B, cond_dim)``.  Ignored when ``cond_proj`` is
                ``None``.  Defaults to ``None``.
            force_uncond (bool): If ``True``, override ``cond`` with the
                learnable ``null_cond`` token (the unconditional pass of
                classifier-free guidance). Has no effect when ``cond_proj``
                is ``None``. Defaults to ``False``.

        Returns:
            torch.Tensor: Predicted noise ``eps_pred``, shape
                ``(N_total, output_dim)`` or ``(N_total, node_dim)`` when
                no output projection is set.
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        u = data.u
        batch = data.batch

        # Lift raw features to internal node_dim when input_proj is set
        if self.input_proj is not None:
            x = self.input_proj(x)

        # Inject time embedding into global attribute
        t_emb = self.time_embedding(t)  # (B, time_embed_dim)
        u = u + self.time_proj(t_emb)  # (B, global_dim)

        # Inject optional conditioning into global attribute (EXP-015 / EXP-020)
        if self.cond_proj is not None and self.null_cond is not None:
            n_graphs = u.size(0)
            if force_uncond or cond is None:
                cond_input = self.null_cond.unsqueeze(0).expand(n_graphs, -1)
            elif self.training and self.p_uncond > 0.0:
                # Per-graph CFG dropout: swap a Bernoulli(p_uncond) subset
                # of conditions for the learned null token.
                mask = (
                    torch.rand(n_graphs, device=cond.device) < self.p_uncond
                ).unsqueeze(
                    -1
                )  # (B, 1) boolean
                null_broadcast = self.null_cond.unsqueeze(0).expand_as(cond)
                cond_input = torch.where(mask, null_broadcast, cond)
            else:
                cond_input = cond
            u = u + self.cond_proj(cond_input)

        # Pass through stacked GN layers
        for gn_layer in self.gn_layers:
            x, edge_attr, u = gn_layer(x, edge_index, edge_attr, u, batch)

        # Project node features to output
        eps_pred = self.output_proj(x)

        # Decode back to target feature dim when output_decode is set
        if self.output_decode is not None:
            eps_pred = self.output_decode(eps_pred)

        return eps_pred  # type: ignore[no-any-return]
