"""
graph_diffusion.model.ema
==========================

Exponential-moving-average (EMA) wrapper used by EXP-024+. A thin
adapter over :class:`torch.optim.swa_utils.AveragedModel` configured
with an EMA averaging function. The wrapper is intentionally minimal —
just enough to construct an EMA copy of a model and validate its decay
parameter — so the training loop owns the update-and-checkpoint
plumbing.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


def build_ema(model: nn.Module, decay: float) -> AveragedModel:
    """Wrap ``model`` in an EMA :class:`AveragedModel`.

    The EMA model's parameters are initialised to a clone of ``model``
    and updated towards ``model`` on each :meth:`update_parameters`
    call as ``ema = decay · ema + (1 − decay) · model``.

    Args:
        model: The model whose parameters should be tracked.
        decay: EMA decay coefficient in the open interval ``(0, 1)``.
            Standard DDPM training uses ``0.9999``. Higher values give
            a smoother EMA at the cost of slower tracking.

    Returns:
        An :class:`AveragedModel` whose ``.module`` attribute is the
        EMA-tracked copy of ``model``.

    Raises:
        ValueError: If ``decay`` is not in the open interval ``(0, 1)``.
    """
    if not (0.0 < decay < 1.0):
        raise ValueError(f"decay must be in (0, 1); got {decay}")
    return AveragedModel(
        model,
        multi_avg_fn=get_ema_multi_avg_fn(decay),  # type: ignore[no-untyped-call]
        device=next(model.parameters()).device,
    )


def save_ema_state_dict(
    ema: AveragedModel,
    path: str,
    extra: dict[str, object] | None = None,
) -> None:
    """Save the EMA-tracked module's ``state_dict`` to ``path``.

    The output format matches :class:`GraphDiffusionModel`'s regular
    checkpoint shape so ``load_exp020(..., checkpoint_name=...)`` can
    consume it without surgery.

    Args:
        ema: The :class:`AveragedModel` returned by :func:`build_ema`.
        path: Destination ``.pt`` path.
        extra: Optional extra keys to merge into the checkpoint dict
            (e.g. ``epoch``, ``lr``, ``config``).
    """
    payload: dict[str, object] = {"model_state_dict": ema.module.state_dict()}
    if extra is not None:
        payload.update(extra)
    torch.save(payload, path)
