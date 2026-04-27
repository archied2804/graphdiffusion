"""
graph_diffusion.model.graph_diffusion_model
=============================================

Top-level DDPM model that owns the full diffusion pipeline: forward
diffusion, loss computation, and reverse sampling.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch_geometric.data import Data

from graph_diffusion.building_blocks.noise_schedule import NoiseSchedule
from graph_diffusion.model.score_network import ScoreNetwork

__all__ = [
    "GraphDiffusionModel",
]

# TODO(extensibility): add "ddim" and "sde" samplers by implementing new
# sampler strategy classes and registering them in _SAMPLERS.
_SAMPLERS = {"ddpm"}


class GraphDiffusionModel(nn.Module):
    """Top-level model that owns the full DDPM pipeline.

    Composes a ``ScoreNetwork`` (noise predictor) with a ``NoiseSchedule``
    (pre-computed diffusion quantities) to provide forward diffusion,
    loss computation, and reverse sampling.

    Args:
        score_network (ScoreNetwork): The noise prediction network.
        noise_schedule (NoiseSchedule): Pre-computed diffusion schedule.
    """

    def __init__(
        self,
        score_network: ScoreNetwork,
        noise_schedule: NoiseSchedule,
    ) -> None:
        super().__init__()
        self.score_network = score_network
        self.noise_schedule = noise_schedule

    def forward_diffusion(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample x_t from q(x_t | x_0).

        Implements the forward process:
        ``x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon``

        Args:
            x_0 (torch.Tensor): Clean node features, shape ``(N_total, F)``.
            t (torch.Tensor): Per-graph timestep indices, shape ``(B,)``.
            batch (torch.Tensor): Batch assignment vector mapping each node
                to its graph index, shape ``(N_total,)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - ``x_t``: Noisy node features at step t, same shape as x_0.
                - ``epsilon``: The noise sample used, same shape as x_0.
        """
        # Expand per-graph coefficients to per-node via batch vector
        # get_t returns (B, 1), index by batch → (N_total, 1)
        sqrt_alpha_bar = self.noise_schedule.get_t(t, "sqrt_alphas_cumprod")[batch]
        sqrt_one_minus_alpha_bar = self.noise_schedule.get_t(
            t, "sqrt_one_minus_alphas_cumprod"
        )[batch]

        epsilon = torch.randn_like(x_0)

        # Variance-preserving scaling per Ho et al. Eq. (4)
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * epsilon

        return x_t, epsilon

    def compute_loss(
        self,
        batch: Data,
    ) -> torch.Tensor:
        """Compute the denoising score-matching loss.

        ``L = E_{t, x_0, epsilon} [ || eps_theta(x_t, t) - epsilon ||^2 ]``

        Procedure:
            1. Sample ``t ~ Uniform({1, ..., T})`` for each graph in the batch.
            2. Compute ``x_t, epsilon`` via ``forward_diffusion``.
            3. Build a noisy ``Data`` object (same structure, but ``x = x_t``).
            4. Call ``score_network(noisy_data, t) -> eps_pred``.
            5. Return ``F.mse_loss(eps_pred, epsilon)``.

        Args:
            batch (Data): A batched PyG ``Data`` object containing clean
                node features in ``batch.x``.

        Returns:
            torch.Tensor: Scalar loss tensor (differentiable).
        """
        x_0 = batch.x
        batch_vec = batch.batch

        # Number of graphs in the batch
        n_graphs = int(batch_vec.max().item()) + 1

        # Sample t ~ Uniform({1, ..., T}) for each graph
        # NOTE: timesteps are 1-indexed in the DDPM formulation but
        # 0-indexed in our buffers, so we sample [1, T] then subtract 1
        # for buffer indexing.
        t = torch.randint(1, self.noise_schedule.T + 1, (n_graphs,), device=x_0.device)
        t_idx = t - 1  # 0-indexed for buffer access

        # Forward diffusion
        x_t, epsilon = self.forward_diffusion(x_0, t_idx, batch_vec)

        # Build noisy Data object — clone structure, replace x
        noisy_data = batch.clone()
        noisy_data.x = x_t

        # Predict noise
        eps_pred = self.score_network(noisy_data, t_idx)

        return F.mse_loss(eps_pred, epsilon)

    @torch.no_grad()
    def sample(
        self,
        graph_template: Data,
        n_steps: int | None = None,
        sampler: str = "ddpm",
        clamp_range: tuple[float, float] | None = None,
    ) -> Data:
        """Reverse diffusion: generate new node features for a given graph.

        Args:
            graph_template (Data): A ``Data`` object whose topology
                (``edge_index``, ``edge_attr``, ``pos``, ``batch``) defines
                the graph structure. The ``x`` attribute is ignored —
                generation starts from ``x_T ~ N(0, I)``.
            n_steps (int | None): Number of denoising steps. Defaults to
                ``self.noise_schedule.T``.
            sampler (str): Sampling strategy. Currently supports ``"ddpm"``.
                Defaults to ``"ddpm"``.
            clamp_range (tuple[float, float] | None): If provided, clamp
                node features to ``(min, max)`` after each reverse step.
                Useful for bounded diffusion (e.g. radial coordinates).
                Defaults to ``None`` (no clamping).

        Returns:
            Data: A new ``Data`` object with the same topology as
                ``graph_template`` but with ``x`` set to the generated
                (denoised) node features.

        Raises:
            ValueError: If ``sampler`` is not a supported strategy.

        # TODO(extensibility): add "ddim" and "sde" samplers by
        # implementing new sampler classes and registering in _SAMPLERS.
        """
        if sampler not in _SAMPLERS:
            raise ValueError(
                f"sampler must be one of {sorted(_SAMPLERS)}, got '{sampler}'"
            )

        total_steps = self.noise_schedule.T
        steps = n_steps if n_steps is not None else total_steps

        result = copy.copy(graph_template)

        # Determine node feature dim from the template
        if graph_template.pos is not None:
            n_nodes = graph_template.pos.size(0)
        else:
            n_nodes = graph_template.x.size(0)

        if graph_template.x is not None:
            node_dim = graph_template.x.size(1)
        else:
            node_dim = graph_template.u.size(1)

        # Start from pure noise x_T ~ N(0, I)
        x_t = torch.randn(n_nodes, node_dim, device=graph_template.edge_index.device)

        batch_vec = graph_template.batch
        if batch_vec is None:
            batch_vec = torch.zeros(n_nodes, dtype=torch.long, device=x_t.device)

        n_graphs = int(batch_vec.max().item()) + 1

        # DDPM reverse process: iterate from t = T down to t = 1
        for step in range(steps, 0, -1):
            t_idx = torch.full(
                (n_graphs,), step - 1, dtype=torch.long, device=x_t.device
            )

            # Build Data for score network
            noisy_data = copy.copy(graph_template)
            noisy_data.x = x_t
            noisy_data.batch = batch_vec

            eps_pred = self.score_network(noisy_data, t_idx)

            # DDPM reverse step coefficients
            beta_t = self.noise_schedule.get_t(t_idx, "betas")[batch_vec]
            alpha_t = self.noise_schedule.get_t(t_idx, "alphas")[batch_vec]
            sqrt_one_minus_alpha_bar_t = self.noise_schedule.get_t(
                t_idx, "sqrt_one_minus_alphas_cumprod"
            )[batch_vec]

            # x_{t-1} = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps)
            x_t = (1.0 / torch.sqrt(alpha_t)) * (
                x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * eps_pred
            )

            # Add noise for all steps except the last (t=1)
            if step > 1:
                z = torch.randn_like(x_t)
                x_t = x_t + torch.sqrt(beta_t) * z

            # Clamp to bounded range if specified
            if clamp_range is not None:
                x_t = x_t.clamp(min=clamp_range[0], max=clamp_range[1])

        result.x = x_t
        return result

    @torch.no_grad()
    def sample_with_trajectory(
        self,
        graph_template: Data,
        n_snapshots: int = 8,
        n_steps: int | None = None,
        sampler: str = "ddpm",
        clamp_range: tuple[float, float] | None = None,
    ) -> tuple[Data, list[tuple[int, torch.Tensor]]]:
        """Reverse diffusion with trajectory snapshots for visualisation.

        Identical to :meth:`sample` but additionally captures intermediate
        ``x_t`` tensors at evenly-spaced timesteps throughout the reverse
        process.

        Args:
            graph_template (Data): Graph topology template (see :meth:`sample`).
            n_snapshots (int): Number of intermediate snapshots to capture
                (excluding the final result). Defaults to ``8``.
            n_steps (int | None): Number of denoising steps. Defaults to
                ``self.noise_schedule.T``.
            sampler (str): Sampling strategy. Currently ``"ddpm"`` only.
            clamp_range (tuple[float, float] | None): Optional per-step
                clamping bounds.

        Returns:
            tuple[Data, list[tuple[int, torch.Tensor]]]:
                - Final denoised ``Data`` object (same as :meth:`sample`).
                - Trajectory list of ``(timestep, x_t)`` pairs ordered from
                  ``t = T`` (noise) to ``t = 0`` (clean). The final ``t = 0``
                  snapshot is always included.
        """
        if sampler not in _SAMPLERS:
            raise ValueError(
                f"sampler must be one of {sorted(_SAMPLERS)}, got '{sampler}'"
            )

        total_steps = self.noise_schedule.T
        steps = n_steps if n_steps is not None else total_steps

        result = copy.copy(graph_template)

        if graph_template.pos is not None:
            n_nodes = graph_template.pos.size(0)
        else:
            n_nodes = graph_template.x.size(0)

        if graph_template.x is not None:
            node_dim = graph_template.x.size(1)
        else:
            node_dim = graph_template.u.size(1)

        x_t = torch.randn(n_nodes, node_dim, device=graph_template.edge_index.device)

        batch_vec = graph_template.batch
        if batch_vec is None:
            batch_vec = torch.zeros(n_nodes, dtype=torch.long, device=x_t.device)

        n_graphs = int(batch_vec.max().item()) + 1

        # Determine which timesteps to snapshot
        snapshot_steps = set()
        for i in range(n_snapshots):
            # Evenly-spaced from steps (noise) down to 1
            s = steps - int(i * steps / n_snapshots)
            snapshot_steps.add(s)

        trajectory: list[tuple[int, torch.Tensor]] = []
        # Capture the initial noise (t = T)
        trajectory.append((steps, x_t.clone().cpu()))

        for step in range(steps, 0, -1):
            t_idx = torch.full(
                (n_graphs,), step - 1, dtype=torch.long, device=x_t.device
            )

            noisy_data = copy.copy(graph_template)
            noisy_data.x = x_t
            noisy_data.batch = batch_vec

            eps_pred = self.score_network(noisy_data, t_idx)

            beta_t = self.noise_schedule.get_t(t_idx, "betas")[batch_vec]
            alpha_t = self.noise_schedule.get_t(t_idx, "alphas")[batch_vec]
            sqrt_one_minus_alpha_bar_t = self.noise_schedule.get_t(
                t_idx, "sqrt_one_minus_alphas_cumprod"
            )[batch_vec]

            x_t = (1.0 / torch.sqrt(alpha_t)) * (
                x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * eps_pred
            )

            if step > 1:
                z = torch.randn_like(x_t)
                x_t = x_t + torch.sqrt(beta_t) * z

            if clamp_range is not None:
                x_t = x_t.clamp(min=clamp_range[0], max=clamp_range[1])

            # Capture snapshot at selected timesteps
            if step - 1 in snapshot_steps or step == 1:
                trajectory.append((step - 1, x_t.clone().cpu()))

        result.x = x_t
        return result, trajectory
