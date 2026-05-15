"""
graph_diffusion.model.fourier_score_network
=============================================

Score network operating in Fourier coefficient space (EXP-013 — Method E).

Represents each ellipse boundary as the first K Fourier coefficients of the
radial function r(θ): ``{A_k, B_k}`` for k=0..K-1 (2K real scalars per
shape).  A plain MLP-based DDPM diffuses these coefficients directly,
bypassing graph architecture altogether.

This enables a key ablation: *is the GN graph inductive bias necessary, or
does low-dimensional coefficient diffusion match graph diffusion quality?*

The network concatenates the coefficient vector with a sinusoidal time
embedding and passes the result through a single ``MLP``.

.. note::
    This is **not** a ``GraphDiffusionModel`` component — it has its own
    standalone DDPM loop in ``train_ellipse_fourier.py``.
"""

import torch
import torch.nn as nn

from graph_diffusion.building_blocks.mlp import MLP, SinusoidalTimeEmbedding

__all__ = [
    "FourierScoreNetwork",
]


class FourierScoreNetwork(nn.Module):
    """Noise prediction network operating on Fourier coefficients.

    Takes a noisy coefficient vector ``c_t`` and a diffusion timestep ``t``,
    and predicts the noise ``ε`` in coefficient space.

    Architecture:
        1. ``SinusoidalTimeEmbedding`` maps ``t`` → ``(B, time_embed_dim)``.
        2. Concatenate ``[c_t, t_emb]`` → ``(B, coeff_dim + time_embed_dim)``.
        3. ``MLP`` projects to ``coeff_dim``.

    Args:
        n_fourier_modes (int): Number of Fourier modes ``K``.  The coefficient
            vector has dimension ``coeff_dim = 2 * K`` (cosine + sine per mode).
        time_embed_dim (int): Dimensionality of the sinusoidal time embedding.
            Defaults to ``64``.
        hidden_dims (list[int]): Hidden layer widths for the MLP.
            Defaults to ``[256, 256, 256]``.
        activation (str): Activation function name (``"silu"`` / ``"relu"`` /
            ``"gelu"``). Defaults to ``"silu"``.
        layer_norm (bool): Whether to apply LayerNorm in the MLP.
            Defaults to ``True``.

    Raises:
        ValueError: If ``n_fourier_modes < 1``.
        ValueError: If ``time_embed_dim < 1``.
    """

    def __init__(
        self,
        n_fourier_modes: int,
        time_embed_dim: int = 64,
        hidden_dims: list[int] | None = None,
        activation: str = "silu",
        layer_norm: bool = True,
    ) -> None:
        super().__init__()

        if n_fourier_modes < 1:
            raise ValueError(f"n_fourier_modes must be >= 1, got {n_fourier_modes}")
        if time_embed_dim < 1:
            raise ValueError(f"time_embed_dim must be >= 1, got {time_embed_dim}")

        if hidden_dims is None:
            hidden_dims = [256, 256, 256]

        self.n_fourier_modes = n_fourier_modes
        coeff_dim = 2 * n_fourier_modes  # A_k and B_k for each mode

        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)

        self.mlp = MLP(
            in_dim=coeff_dim + time_embed_dim,
            hidden_dims=hidden_dims,
            out_dim=coeff_dim,
            activation=activation,
            layer_norm=layer_norm,
            residual=False,
        )

    @property
    def coeff_dim(self) -> int:
        """Dimensionality of the Fourier coefficient vector (2 * n_fourier_modes)."""
        return 2 * self.n_fourier_modes

    def forward(
        self,
        coeff: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise from a noisy coefficient vector and timestep.

        Args:
            coeff (torch.Tensor): Noisy Fourier coefficients, shape
                ``(B, 2 * n_fourier_modes)``.
            t (torch.Tensor): Integer timestep indices, shape ``(B,)``.

        Returns:
            torch.Tensor: Predicted noise in coefficient space, shape
                ``(B, 2 * n_fourier_modes)``.
        """
        t_emb = self.time_embedding(t)  # (B, time_embed_dim)
        x = torch.cat([coeff, t_emb], dim=-1)  # (B, coeff_dim + time_embed_dim)
        return self.mlp(x)  # type: ignore[no-any-return]  # MLP.forward returns Any

    @staticmethod
    def coeffs_to_radii(
        coeffs: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct radial values from Fourier coefficients via synthesis.

        Given coefficient vector ``[A_0, B_0, A_1, B_1, ..., A_{K-1}, B_{K-1}]``
        and angular positions ``θ``, compute:

        ``r(θ_i) = Σ_{k=0}^{K-1} A_k * cos(k θ_i) + B_k * sin(k θ_i)``

        Args:
            coeffs (torch.Tensor): Fourier coefficients, shape
                ``(B, 2 * K)`` — interleaved as ``[A_0, B_0, A_1, B_1, ...]``.
            theta (torch.Tensor): Angular positions in radians, shape ``(N,)``.

        Returns:
            torch.Tensor: Radial values, shape ``(B, N)``.
        """
        k = coeffs.size(-1) // 2  # number of modes
        modes = torch.arange(k, device=coeffs.device, dtype=theta.dtype)

        # (K, N): angles for each mode at each node
        angles = modes.unsqueeze(1) * theta.unsqueeze(0)  # (K, N)

        cos_basis = torch.cos(angles)  # (K, N)
        sin_basis = torch.sin(angles)  # (K, N)

        # Extract A and B coefficients: (B, K)
        a_coeffs = coeffs[:, 0::2]  # even indices: A_0, A_1, ...
        b_coeffs = coeffs[:, 1::2]  # odd indices: B_0, B_1, ...

        # Synthesise: (B, K) @ (K, N) → (B, N)
        r = a_coeffs @ cos_basis + b_coeffs @ sin_basis

        return r
