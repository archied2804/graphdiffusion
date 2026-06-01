"""
Tests for graph_diffusion.model.ema
====================================

Exercises the thin EMA wrapper used by EXP-024. The wrapper is a
:class:`torch.optim.swa_utils.AveragedModel` configured for EMA; the
tests confirm divergence from raw weights over updates and shape
preservation of the saved state dict.
"""

from __future__ import annotations

import copy

import torch
from context import graph_diffusion  # noqa: F401

from graph_diffusion.building_blocks.mlp import MLP
from graph_diffusion.model.ema import build_ema


def _make_simple_model() -> MLP:
    torch.manual_seed(0)
    return MLP(in_dim=4, hidden_dims=[8], out_dim=2)


def test_ema_initial_matches_raw_model() -> None:
    model = _make_simple_model()
    ema = build_ema(model, decay=0.99)
    raw_state = model.state_dict()
    ema_state = ema.module.state_dict()
    for k in raw_state:
        if not k.startswith("n_averaged"):  # internal counter, not a param
            assert torch.allclose(raw_state[k], ema_state[k])


def test_ema_diverges_from_raw_after_updates() -> None:
    model = _make_simple_model()
    ema = build_ema(model, decay=0.9)  # aggressive decay so divergence is fast

    initial_first_param = copy.deepcopy(next(iter(model.parameters())))

    # Mutate the raw model weights deliberately to simulate optimiser steps.
    for _ in range(100):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.01 * torch.randn_like(p))
        ema.update_parameters(model)

    # Raw weights have drifted.
    assert not torch.allclose(
        next(iter(model.parameters())), initial_first_param, atol=1e-3
    )
    # EMA weights lag behind the raw weights.
    raw_first = next(iter(model.parameters()))
    ema_first = next(iter(ema.module.parameters()))
    assert not torch.allclose(raw_first, ema_first, atol=1e-3)


def test_ema_state_dict_round_trip() -> None:
    model = _make_simple_model()
    ema = build_ema(model, decay=0.99)
    # Do a few updates so the EMA holds non-trivial state.
    for _ in range(5):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.01)
        ema.update_parameters(model)

    saved = ema.module.state_dict()

    fresh_model = _make_simple_model()
    fresh_model.load_state_dict(saved)
    fresh_state = fresh_model.state_dict()
    for k in saved:
        assert torch.allclose(saved[k], fresh_state[k])


def test_ema_decay_validation() -> None:
    model = _make_simple_model()
    for bad in (-0.1, 0.0, 1.0, 1.5):
        try:
            build_ema(model, decay=bad)
        except ValueError as e:
            assert "decay" in str(e)
        else:
            raise AssertionError(f"expected ValueError for decay={bad}")
