"""
graph_diffusion.building_blocks.feature_transforms
====================================================

Feature-space transforms for bounded diffusion.  A ``FeatureTransform``
maps node features to an unbounded domain before the forward diffusion
process and inverts them after the reverse process, allowing the model
to operate in an unconstrained space without post-hoc clamping.
"""

from __future__ import annotations

import abc

import torch

__all__ = [
    "FeatureTransform",
    "LogitNormTransform",
]


class FeatureTransform(abc.ABC):
    """Abstract base for invertible feature-space transforms.

    Subclasses implement a bijective mapping between the bounded feature
    domain used by the dataset and the unbounded domain used by the
    diffusion process.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map features from bounded space to unbounded diffusion space.

        Args:
            x (torch.Tensor): Node features in the original bounded domain,
                shape ``(N, F)``.

        Returns:
            torch.Tensor: Transformed features in unbounded space,
                same shape as ``x``.
        """

    @abc.abstractmethod
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Map features from unbounded diffusion space back to bounded domain.

        Args:
            z (torch.Tensor): Features in unbounded diffusion space,
                shape ``(N, F)``.

        Returns:
            torch.Tensor: Features in the original bounded domain,
                same shape as ``z``.
        """


class LogitNormTransform(FeatureTransform):
    """Logit-normalisation transform for radii bounded to ``[r_min, r_max]``.

    Maps ``r ∈ [r_min, r_max]`` to ``ℝ`` via logit normalisation:

    .. code-block:: text

        forward:  r  →  logit( (r - r_min) / (r_max - r_min) )
        inverse:  z  →  sigmoid(z) * (r_max - r_min) + r_min

    The forward transform guarantees the diffusion operates in an
    unconstrained space so that the generated samples naturally satisfy
    the boundary condition without post-hoc clamping.

    Args:
        r_min (float): Lower bound of the original feature domain.
        r_max (float): Upper bound of the original feature domain.
        eps (float): Clamping epsilon to avoid ``logit(0)`` / ``logit(1)``
            singularities. Defaults to ``1e-6``.

    Raises:
        ValueError: If ``r_min >= r_max``.
        ValueError: If ``eps <= 0`` or ``eps >= 0.5``.
    """

    def __init__(
        self,
        r_min: float,
        r_max: float,
        eps: float = 1e-6,
    ) -> None:
        if r_min >= r_max:
            raise ValueError(f"r_min must be < r_max, got r_min={r_min}, r_max={r_max}")
        if eps <= 0 or eps >= 0.5:
            raise ValueError(f"eps must be in (0, 0.5), got {eps}")

        self.r_min = r_min
        self.r_max = r_max
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map r ∈ [r_min, r_max] → ℝ via logit normalisation.

        Args:
            x (torch.Tensor): Radii in ``[r_min, r_max]``, shape ``(N, F)``.

        Returns:
            torch.Tensor: Logit-normalised features in ``ℝ``, same shape.
        """
        p = (x - self.r_min) / (self.r_max - self.r_min)
        return torch.logit(p.clamp(self.eps, 1.0 - self.eps))

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Map ℝ → r ∈ [r_min, r_max] via sigmoid.

        Args:
            z (torch.Tensor): Features in unbounded diffusion space,
                shape ``(N, F)``.

        Returns:
            torch.Tensor: Radii in ``[r_min, r_max]``, same shape.
        """
        p = torch.sigmoid(z)
        return p * (self.r_max - self.r_min) + self.r_min
