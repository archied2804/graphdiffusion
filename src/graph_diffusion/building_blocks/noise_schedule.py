"""
graph_diffusion.building_blocks.noise_schedule
================================================

Pre-computed DDPM noise schedule registered as ``nn.Module`` buffers.
Supports linear and cosine schedules.
"""

import math

import torch
import torch.nn as nn

__all__ = [
    "NoiseSchedule",
]

_SCHEDULE_TYPES = {"linear", "cosine"}


class NoiseSchedule(nn.Module):
    """Pre-computes and registers all DDPM diffusion quantities as buffers.

    This is not a learned module — it contains no ``nn.Parameter``. All
    tensors are registered via ``register_buffer`` so they automatically
    move with ``.to(device)`` and are saved in checkpoints.

    Args:
        T (int): Total number of diffusion timesteps. Defaults to ``1000``.
        schedule_type (str): One of ``"linear"`` or ``"cosine"``.
            Defaults to ``"cosine"``.
        beta_start (float): Starting beta value for the linear schedule.
            Defaults to ``1e-4``.
        beta_end (float): Ending beta value for the linear schedule.
            Defaults to ``0.02``.

    Raises:
        ValueError: If ``schedule_type`` is not ``"linear"`` or ``"cosine"``.
        ValueError: If ``T < 1``.
    """

    def __init__(
        self,
        T: int = 1000,  # noqa: N803
        schedule_type: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()

        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}")
        if schedule_type not in _SCHEDULE_TYPES:
            raise ValueError(
                f"schedule_type must be one of {sorted(_SCHEDULE_TYPES)}, "
                f"got '{schedule_type}'"
            )

        self.T = T

        if schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, T)
        else:
            # Cosine schedule (Nichol & Dhariwal 2021)
            s = 0.008
            t_vals = torch.linspace(0, T, T + 1)
            f = torch.cos(((t_vals / T) + s) / (1 + s) * math.pi / 2) ** 2
            alphas_cumprod_full = f / f[0]
            betas = torch.clamp(
                1 - alphas_cumprod_full[1:] / alphas_cumprod_full[:-1],
                max=0.999,
            )

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

    def get_t(self, t: torch.Tensor, buffer_name: str) -> torch.Tensor:
        """Index a registered buffer by timestep.

        Args:
            t (torch.Tensor): Integer indices, shape ``(B,)``.
            buffer_name (str): One of the registered buffer names
                (e.g. ``"sqrt_alphas_cumprod"``).

        Returns:
            torch.Tensor: Tensor of shape ``(B, 1)`` — ready to broadcast
                over ``(N_total, F)`` node features.

        Raises:
            ValueError: If ``buffer_name`` is not a registered buffer.
        """
        buf = getattr(self, buffer_name, None)
        if buf is None:
            raise ValueError(
                f"'{buffer_name}' is not a registered buffer. "
                f"Available: betas, alphas, alphas_cumprod, "
                f"sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod"
            )
        return buf[t].unsqueeze(-1)  # type: ignore[no-any-return]
