"""
graph_diffusion.building_blocks.mlp
====================================

Configurable multi-layer perceptron used as the learnable kernel inside
every GraphNetworkBlock update function, plus a sinusoidal timestep
embedding module for diffusion conditioning.
"""

import math

import torch
import torch.nn as nn

__all__ = [
    "MLP",
    "SinusoidalTimeEmbedding",
]

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}


class MLP(nn.Module):
    """Configurable multi-layer perceptron.

    Supports optional LayerNorm per hidden layer and a residual connection
    when ``in_dim == out_dim``.

    Args:
        in_dim (int): Dimensionality of the input tensor.
        hidden_dims (list[int]): Width of each hidden layer, in order.
        out_dim (int): Dimensionality of the output tensor.
        activation (str): One of ``"relu"``, ``"silu"``, ``"gelu"``.
            Defaults to ``"silu"``.
        layer_norm (bool): If ``True``, apply LayerNorm after each hidden
            activation. Defaults to ``False``.
        residual (bool): If ``True`` **and** ``in_dim == out_dim``, add a
            skip connection from input to output. Defaults to ``False``.

    Raises:
        ValueError: If ``activation`` is not one of the supported strings.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        activation: str = "silu",
        layer_norm: bool = False,
        residual: bool = False,
    ) -> None:
        super().__init__()

        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {list(_ACTIVATIONS.keys())}, "
                f"got '{activation}'"
            )

        self.residual = residual and (in_dim == out_dim)

        layers: list[nn.Module] = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(_ACTIVATIONS[activation]())
            prev_dim = h_dim

        # Final linear — no activation, no norm
        layers.append(nn.Linear(prev_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        Args:
            x (torch.Tensor): Input of shape ``(*, in_dim)``.

        Returns:
            torch.Tensor: Output of shape ``(*, out_dim)``.
        """
        out = self.net(x)
        if self.residual:
            out = out + x
        return out  # type: ignore[no-any-return]


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps.

    Maps a scalar timestep ``t`` to a fixed sinusoidal encoding, then
    projects to ``embed_dim`` via a small MLP. This is the standard
    transformer / diffusion time encoding.

    Args:
        embed_dim (int): Output embedding dimensionality.
        max_period (int): Controls the range of frequencies used in the
            sinusoidal encoding. Defaults to ``10000``.
    """

    def __init__(self, embed_dim: int, max_period: int = 10000) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period
        self.mlp = MLP(embed_dim, [embed_dim * 4], embed_dim, activation="silu")

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute sinusoidal time embedding.

        Args:
            t (torch.Tensor): Integer timestep indices, shape ``(B,)`` or
                scalar.

        Returns:
            torch.Tensor: Embedding tensor, shape ``(B, embed_dim)``.
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)

        half_dim = self.embed_dim // 2
        # Frequency scaling: exp(-log(max_period) * i / half_dim)
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, dtype=torch.float32, device=t.device)
            / half_dim
        )
        # Outer product: (B, 1) * (half_dim,) → (B, half_dim)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        # Concatenate sin and cos → (B, embed_dim)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return self.mlp(embedding)  # type: ignore[no-any-return]
