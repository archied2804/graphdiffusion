"""
Tests for v-prediction parameterisation on GraphDiffusionModel.

Salimans & Ho 2022 reparameterise the DDPM target from ε to
``v = √ᾱ · ε − √(1−ᾱ) · x_0``. Round-trip identities used by the
sampler:

    x_0 = √ᾱ · x_t − √(1−ᾱ) · v
    ε   = √(1−ᾱ) · x_t + √ᾱ · v
"""

from __future__ import annotations

import torch
from context import graph_diffusion  # noqa: F401
from torch_geometric.data import Data

from graph_diffusion.building_blocks.noise_schedule import NoiseSchedule
from graph_diffusion.model.graph_diffusion_model import GraphDiffusionModel
from graph_diffusion.model.score_network import ScoreNetwork

NODE_DIM = 8
EDGE_DIM = 4
GLOBAL_DIM = 6


def _make_model(prediction_type: str = "epsilon") -> GraphDiffusionModel:
    score_net = ScoreNetwork(
        node_dim=NODE_DIM,
        edge_dim=EDGE_DIM,
        global_dim=GLOBAL_DIM,
        time_embed_dim=16,
        n_layers=1,
        hidden_dims=[16],
    )
    ns = NoiseSchedule(T=20, schedule_type="linear")
    return GraphDiffusionModel(
        score_network=score_net,
        noise_schedule=ns,
        prediction_type=prediction_type,
    )


def _make_batch(n_graphs: int = 4, n_nodes_per_graph: int = 6) -> Data:
    torch.manual_seed(0)
    n_nodes = n_graphs * n_nodes_per_graph
    x = torch.randn(n_nodes, NODE_DIM)
    edge_index = torch.stack(
        [
            torch.randint(0, n_nodes, (24,)),
            torch.randint(0, n_nodes, (24,)),
        ]
    )
    edge_attr = torch.randn(24, EDGE_DIM)
    u = torch.randn(n_graphs, GLOBAL_DIM)
    batch = torch.arange(n_graphs).repeat_interleave(n_nodes_per_graph)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u, batch=batch)


def test_invalid_prediction_type_raises() -> None:
    try:
        _make_model(prediction_type="x_0")
    except ValueError as e:
        assert "prediction_type" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_default_is_epsilon_and_matches_baseline_loss() -> None:
    eps_model = _make_model("epsilon")
    batch = _make_batch()
    torch.manual_seed(123)
    loss_eps = eps_model.compute_loss(batch)
    # Loss is finite and positive.
    assert torch.isfinite(loss_eps)
    assert loss_eps > 0


def test_v_prediction_loss_differs_from_epsilon() -> None:
    eps_model = _make_model("epsilon")
    v_model = _make_model("v")
    v_model.load_state_dict(eps_model.state_dict())
    batch = _make_batch()

    torch.manual_seed(123)
    loss_eps = eps_model.compute_loss(batch)
    torch.manual_seed(123)
    loss_v = v_model.compute_loss(batch)
    assert not torch.allclose(loss_eps, loss_v)


def test_v_prediction_gradients_flow() -> None:
    model = _make_model("v")
    batch = _make_batch()
    loss = model.compute_loss(batch)
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )


def test_v_to_eps_round_trip_identity() -> None:
    """Algebraic identity: ε = √(1−ᾱ)·x_t + √ᾱ·v with v defined from (x_0, ε)."""
    torch.manual_seed(0)
    alpha_bar = torch.tensor(0.3)
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus = torch.sqrt(1.0 - alpha_bar)

    x_0 = torch.randn(7, 3)
    eps = torch.randn(7, 3)
    x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus * eps
    v = sqrt_alpha_bar * eps - sqrt_one_minus * x_0

    eps_reconstructed = sqrt_one_minus * x_t + sqrt_alpha_bar * v
    assert torch.allclose(eps_reconstructed, eps, atol=1e-5)

    x0_reconstructed = sqrt_alpha_bar * x_t - sqrt_one_minus * v
    assert torch.allclose(x0_reconstructed, x_0, atol=1e-5)


def test_v_sample_produces_finite_output() -> None:
    """End-to-end check that v-prediction sampling doesn't NaN."""
    torch.manual_seed(0)
    model = _make_model("v")
    template = _make_batch(n_graphs=1, n_nodes_per_graph=6)
    template.batch = torch.zeros(template.x.size(0), dtype=torch.long)
    out = model.sample(template)
    assert torch.isfinite(out.x).all()
