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

from graph_diffusion.building_blocks.feature_transforms import FeatureTransform
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
        feature_transform (FeatureTransform | None): Optional invertible
            transform applied to node features before diffusion and inverted
            after sampling.  When set, the model operates in the transformed
            (unbounded) space and ``clamp_range`` in :meth:`sample` is
            ignored.  Defaults to ``None``.
        n_noise_channels (int | None): When set, only the first
            ``n_noise_channels`` columns of ``batch.x`` are noised during
            training and generated during sampling.  The remaining columns are
            treated as fixed conditioning (e.g. pressure in EXP-016) and are
            concatenated from ``batch.p_cond`` / ``graph_template.p_cond``.
            Defaults to ``None`` (all channels noised).
        smoothness_weight (float): Weight ``λ`` for the second-order finite
            difference regularisation term added to the denoising loss.
            At each training step the model's ``x̂₀`` reconstruction is
            used to compute ``λ · mean(Δ²x̂₀)²`` over all nodes (periodic
            ring boundary).  Set to ``0.0`` to disable.  Defaults to ``0.0``.

    Note:
        ``feature_transform``, ``n_noise_channels``, and
        ``smoothness_weight`` are future-work parameters for bounded
        diffusion and conditional/regularised training (EXP-013+,
        EXP-015+). Leave at their defaults (``None``, ``None``, ``0.0``)
        for the unconditional shape-generation baseline.
    """

    def __init__(
        self,
        score_network: ScoreNetwork,
        noise_schedule: NoiseSchedule,
        feature_transform: FeatureTransform | None = None,
        n_noise_channels: int | None = None,
        smoothness_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.score_network = score_network
        self.noise_schedule = noise_schedule
        self.feature_transform = feature_transform
        self.n_noise_channels = n_noise_channels
        self.smoothness_weight = smoothness_weight

    def _smoothness_loss(
        self,
        x0_hat: torch.Tensor,
        batch_vec: torch.Tensor,
        snr_weight: torch.Tensor,
    ) -> torch.Tensor:
        """SNR-weighted mean squared second finite difference of x̂₀.

        Nodes within each graph are assumed to be ordered sequentially around
        the boundary (as in ``UnitCircleDataset`` and ``EllipseShapeDataset``).
        The periodic boundary condition wraps the last node back to the first.

        The penalty is weighted by ``ᾱ_t`` per node so that it is suppressed at
        high noise timesteps where the x̂₀ reconstruction is unreliable (large
        ``1/√ᾱ_t`` amplification), and active at low noise timesteps where the
        predicted clean signal should already be smooth.

        Args:
            x0_hat (torch.Tensor): Reconstructed clean features, shape
                ``(N_total, F)``.
            batch_vec (torch.Tensor): Batch assignment, shape ``(N_total,)``.
            snr_weight (torch.Tensor): Per-node weight ``ᾱ_t``, shape
                ``(N_total, 1)``.

        Returns:
            torch.Tensor: Scalar SNR-weighted mean squared curvature proxy.
        """
        n_total = x0_hat.size(0)
        device = x0_hat.device

        counts = torch.bincount(batch_vec)
        ptr = torch.zeros(counts.size(0) + 1, dtype=torch.long, device=device)
        ptr[1:] = torch.cumsum(counts, dim=0)

        arange = torch.arange(n_total, device=device)
        local_idx = arange - ptr[batch_vec]
        n_g = counts[batch_vec]

        next_idx = ptr[batch_vec] + (local_idx + 1) % n_g
        prev_idx = ptr[batch_vec] + (local_idx - 1) % n_g

        kappa = x0_hat[next_idx] - 2.0 * x0_hat + x0_hat[prev_idx]
        return (snr_weight * kappa.pow(2)).mean()

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
        # Apply feature transform to operate in unbounded diffusion space
        if self.feature_transform is not None:
            x_0 = self.feature_transform.forward(x_0)
        batch_vec = batch.batch

        # Split into noised channels and fixed conditioning channels (EXP-016)
        if self.n_noise_channels is not None:
            x_0_noise = x_0[:, : self.n_noise_channels]
        else:
            x_0_noise = x_0

        # Number of graphs in the batch
        n_graphs = int(batch_vec.max().item()) + 1

        # Sample t ~ Uniform({1, ..., T}) for each graph
        # NOTE: timesteps are 1-indexed in the DDPM formulation but
        # 0-indexed in our buffers, so we sample [1, T] then subtract 1
        # for buffer indexing.
        t = torch.randint(1, self.noise_schedule.T + 1, (n_graphs,), device=x_0.device)
        t_idx = t - 1  # 0-indexed for buffer access

        # Forward diffusion on the noised channels only
        x_t, epsilon = self.forward_diffusion(x_0_noise, t_idx, batch_vec)

        # Build noisy Data object — clone structure, replace x
        noisy_data = batch.clone()
        if self.n_noise_channels is not None:
            # Concatenate noised shape channels with clean pressure conditioning
            p_cond = batch.p_cond  # (N_total, n_cond_channels)
            noisy_data.x = torch.cat([x_t, p_cond], dim=-1)
        else:
            noisy_data.x = x_t

        # Extract optional global conditioning vector (EXP-015)
        cond = getattr(batch, "cond", None)

        # Predict noise
        eps_pred = self.score_network(noisy_data, t_idx, cond=cond)

        mse = F.mse_loss(eps_pred, epsilon)

        if self.smoothness_weight > 0.0:
            # Reconstruct x̂₀ from the predicted noise via DDPM Eq. (15)
            sqrt_alpha_bar_t = self.noise_schedule.get_t(
                t_idx, "sqrt_alphas_cumprod"
            )[batch_vec]
            sqrt_one_minus_alpha_bar_t = self.noise_schedule.get_t(
                t_idx, "sqrt_one_minus_alphas_cumprod"
            )[batch_vec]
            x0_hat = (
                (x_t - sqrt_one_minus_alpha_bar_t * eps_pred)
                / sqrt_alpha_bar_t.clamp(min=1e-8)
            )
            # ᾱ_t = (√ᾱ_t)² — weight suppresses penalty at high noise levels
            snr_weight = sqrt_alpha_bar_t.pow(2)
            mse = mse + self.smoothness_weight * self._smoothness_loss(
                x0_hat, batch_vec, snr_weight
            )

        return mse

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

        # Determine node count and noise dimensionality
        if graph_template.pos is not None:
            n_nodes = graph_template.pos.size(0)
        else:
            n_nodes = graph_template.x.size(0)

        # When n_noise_channels is set we only generate the noised channels;
        # the remaining channels come from graph_template.p_cond at each step.
        if self.n_noise_channels is not None:
            noise_dim = self.n_noise_channels
        elif graph_template.x is not None:
            noise_dim = graph_template.x.size(1)
        else:
            noise_dim = graph_template.u.size(1)

        # Start from pure noise x_T ~ N(0, I)
        x_t = torch.randn(n_nodes, noise_dim, device=graph_template.edge_index.device)

        batch_vec = graph_template.batch
        if batch_vec is None:
            batch_vec = torch.zeros(n_nodes, dtype=torch.long, device=x_t.device)

        n_graphs = int(batch_vec.max().item()) + 1

        # Optional conditioning tensors propagated via graph_template shallow copy
        cond = getattr(graph_template, "cond", None)
        p_cond = getattr(graph_template, "p_cond", None)

        # DDPM reverse process: iterate from t = T down to t = 1
        for step in range(steps, 0, -1):
            t_idx = torch.full(
                (n_graphs,), step - 1, dtype=torch.long, device=x_t.device
            )

            # Build Data for score network
            noisy_data = copy.copy(graph_template)
            if self.n_noise_channels is not None and p_cond is not None:
                noisy_data.x = torch.cat([x_t, p_cond], dim=-1)
            else:
                noisy_data.x = x_t
            noisy_data.batch = batch_vec

            eps_pred = self.score_network(noisy_data, t_idx, cond=cond)

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
            # (skipped when feature_transform is set)
            if clamp_range is not None and self.feature_transform is None:
                x_t = x_t.clamp(min=clamp_range[0], max=clamp_range[1])

        # Invert feature transform to recover bounded domain
        if self.feature_transform is not None:
            x_t = self.feature_transform.inverse(x_t)

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

        if self.n_noise_channels is not None:
            noise_dim = self.n_noise_channels
        elif graph_template.x is not None:
            noise_dim = graph_template.x.size(1)
        else:
            noise_dim = graph_template.u.size(1)

        x_t = torch.randn(n_nodes, noise_dim, device=graph_template.edge_index.device)

        batch_vec = graph_template.batch
        if batch_vec is None:
            batch_vec = torch.zeros(n_nodes, dtype=torch.long, device=x_t.device)

        n_graphs = int(batch_vec.max().item()) + 1

        cond = getattr(graph_template, "cond", None)
        p_cond = getattr(graph_template, "p_cond", None)

        # Determine which timesteps to snapshot
        snapshot_steps = set()
        for i in range(n_snapshots):
            s = steps - int(i * steps / n_snapshots)
            snapshot_steps.add(s)

        trajectory: list[tuple[int, torch.Tensor]] = []
        trajectory.append((steps, x_t.clone().cpu()))

        for step in range(steps, 0, -1):
            t_idx = torch.full(
                (n_graphs,), step - 1, dtype=torch.long, device=x_t.device
            )

            noisy_data = copy.copy(graph_template)
            if self.n_noise_channels is not None and p_cond is not None:
                noisy_data.x = torch.cat([x_t, p_cond], dim=-1)
            else:
                noisy_data.x = x_t
            noisy_data.batch = batch_vec

            eps_pred = self.score_network(noisy_data, t_idx, cond=cond)

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

            if clamp_range is not None and self.feature_transform is None:
                x_t = x_t.clamp(min=clamp_range[0], max=clamp_range[1])

            # Capture snapshot at selected timesteps (in diffusion space)
            if step - 1 in snapshot_steps or step == 1:
                trajectory.append((step - 1, x_t.clone().cpu()))

        if self.feature_transform is not None:
            x_t = self.feature_transform.inverse(x_t)

        result.x = x_t
        return result, trajectory
