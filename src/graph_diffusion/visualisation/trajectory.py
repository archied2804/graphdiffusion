"""
graph_diffusion.visualisation.trajectory
==========================================

Capture intermediate ``x_t`` tensors from forward and reverse diffusion
runs for filmstrip and animation plotting.
"""

from __future__ import annotations

import copy

import torch
from torch_geometric.data import Data

from graph_diffusion.model.graph_diffusion_model import GraphDiffusionModel

__all__ = [
    "collect_forward",
    "collect_reverse",
]


def collect_forward(
    model: GraphDiffusionModel,
    template: Data,
    snapshot_steps: list[int],
    seed: int = 0,
) -> list[torch.Tensor]:
    """Run forward diffusion on ``template.x`` and snapshot at given ``t``.

    Args:
        model: Trained diffusion model providing the ``noise_schedule``.
        template: A graph ``Data`` whose ``x`` provides the clean signal
            ``x_0``. Must be on the same device as the model.
        snapshot_steps: 0-indexed timesteps to record (``t in [0, T-1]``).
            Each ``t`` corresponds to one entry in the returned list, in
            the same order as ``snapshot_steps``.
        seed: Seed for the additive Gaussian noise.

    Returns:
        A list of CPU ``torch.float32`` tensors, each of shape
        ``(N_nodes, n_noise_channels_or_x_cols)``, one per snapshot.
    """
    device = template.x.device
    generator = torch.Generator(device=device).manual_seed(seed)
    x_0 = template.x
    schedule = model.noise_schedule

    snapshots: list[torch.Tensor] = []
    for t in snapshot_steps:
        t_idx = torch.tensor([t], device=device)
        sqrt_alpha_bar = schedule.get_t(t_idx, "sqrt_alphas_cumprod")
        sqrt_one_minus = schedule.get_t(t_idx, "sqrt_one_minus_alphas_cumprod")
        epsilon = torch.randn(x_0.shape, generator=generator, device=device)
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus * epsilon
        snapshots.append(x_t.detach().cpu().to(torch.float32))
    return snapshots


def collect_reverse(
    model: GraphDiffusionModel,
    template: Data,
    cond: torch.Tensor | None,
    snapshot_steps: list[int],
    guidance_scale: float = 1.0,
    seed: int = 0,
) -> list[torch.Tensor]:
    """Run reverse diffusion and snapshot ``x_t`` at the requested steps.

    Duplicates the inner loop of :meth:`GraphDiffusionModel.sample` so we
    can record intermediate ``x_t`` without modifying the model API.
    Supports classifier-free guidance via ``guidance_scale``; does NOT
    support DPS (kept simple — the spec only animates the standard CFG
    reverse path).

    Args:
        model: Trained diffusion model.
        template: Graph ``Data`` template. ``template.x`` is ignored;
            generation starts from ``x_T ~ N(0, I)``.
        cond: Optional conditioning vector of shape ``(B, cond_dim)``.
        snapshot_steps: 0-indexed timesteps to record. ``T - 1`` records
            the pure-noise initial state; ``0`` records the final
            denoised output.
        guidance_scale: CFG scale ``w``. ``1.0`` disables guidance.
        seed: Seed for the initial noise and per-step stochasticity.

    Returns:
        A list of CPU ``torch.float32`` tensors, each shape
        ``(N_nodes, n_noise_channels)``, one per snapshot in the input
        order of ``snapshot_steps``.
    """
    schedule = model.noise_schedule
    total_T = schedule.T  # noqa: N806
    device = template.edge_index.device

    n_nodes = template.pos.size(0) if template.pos is not None else template.x.size(0)

    if model.n_noise_channels is not None:
        noise_dim = model.n_noise_channels
    elif template.x is not None:
        noise_dim = template.x.size(1)
    else:
        noise_dim = template.u.size(1)

    generator = torch.Generator(device=device).manual_seed(seed)
    x_t = torch.randn(n_nodes, noise_dim, generator=generator, device=device)

    batch_vec = template.batch
    if batch_vec is None:
        batch_vec = torch.zeros(n_nodes, dtype=torch.long, device=device)
    n_graphs = int(batch_vec.max().item()) + 1

    p_cond = getattr(template, "p_cond", None)
    use_cfg = (
        guidance_scale != 1.0
        and cond is not None
        and getattr(model.score_network, "null_cond", None) is not None
    )

    requested = set(snapshot_steps)
    captured: dict[int, torch.Tensor] = {}

    if total_T - 1 in requested:
        captured[total_T - 1] = x_t.detach().cpu().to(torch.float32)

    with torch.no_grad():
        for step in range(total_T, 0, -1):
            t_idx = torch.full((n_graphs,), step - 1, dtype=torch.long, device=device)
            noisy_data = copy.copy(template)
            if model.n_noise_channels is not None and p_cond is not None:
                noisy_data.x = torch.cat([x_t, p_cond], dim=-1)
            else:
                noisy_data.x = x_t
            noisy_data.batch = batch_vec

            eps_pred = model.score_network(noisy_data, t_idx, cond=cond)
            if use_cfg:
                eps_null = model.score_network(
                    noisy_data, t_idx, cond=cond, force_uncond=True
                )
                eps_pred = (1.0 + guidance_scale) * eps_pred - (
                    guidance_scale * eps_null
                )

            beta_t = schedule.get_t(t_idx, "betas")[batch_vec]
            alpha_t = schedule.get_t(t_idx, "alphas")[batch_vec]
            sqrt_one_minus = schedule.get_t(t_idx, "sqrt_one_minus_alphas_cumprod")[
                batch_vec
            ]

            x_t = (1.0 / torch.sqrt(alpha_t)) * (
                x_t - (beta_t / sqrt_one_minus) * eps_pred
            )
            if step > 1:
                z = torch.randn(x_t.shape, generator=generator, device=device)
                x_t = x_t + torch.sqrt(beta_t) * z

            new_t = step - 1
            if new_t in requested:
                captured[new_t] = x_t.detach().cpu().to(torch.float32)

    return [captured[t] for t in snapshot_steps]
