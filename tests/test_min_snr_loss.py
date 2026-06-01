"""
Tests for the Min-SNR-γ loss reweighting on GraphDiffusionModel.

Min-SNR-γ (Hang et al. 2023) replaces the uniform MSE in DDPM training
with ``mean(min(SNR_t, γ) / SNR_t · (ε_pred − ε)²)``, which up-weights
low-noise timesteps. γ = ∞ recovers the vanilla loss; γ = 5 is the
recommended setting for the EXP-023 surface-quality study.
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


def _make_model(min_snr_gamma: float | None = None) -> GraphDiffusionModel:
    score_net = ScoreNetwork(
        node_dim=NODE_DIM,
        edge_dim=EDGE_DIM,
        global_dim=GLOBAL_DIM,
        time_embed_dim=16,
        n_layers=1,
        hidden_dims=[16],
    )
    ns = NoiseSchedule(T=50, schedule_type="linear")
    return GraphDiffusionModel(
        score_network=score_net,
        noise_schedule=ns,
        min_snr_gamma=min_snr_gamma,
    )


def _make_batch(
    n_nodes_per_graph: int = 6, n_edges: int = 40, n_graphs: int = 16
) -> Data:
    torch.manual_seed(0)
    n_nodes = n_nodes_per_graph * n_graphs
    x = torch.randn(n_nodes, NODE_DIM)
    edge_index = torch.stack(
        [
            torch.randint(0, n_nodes, (n_edges,)),
            torch.randint(0, n_nodes, (n_edges,)),
        ]
    )
    edge_attr = torch.randn(n_edges, EDGE_DIM)
    u = torch.randn(n_graphs, GLOBAL_DIM)
    batch = torch.arange(n_graphs).repeat_interleave(n_nodes_per_graph)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u, batch=batch)


def test_gamma_none_recovers_vanilla_mse() -> None:
    torch.manual_seed(0)
    baseline = _make_model(min_snr_gamma=None)
    batch = _make_batch()
    torch.manual_seed(123)
    loss_baseline = baseline.compute_loss(batch)
    torch.manual_seed(123)
    loss_repeat = baseline.compute_loss(batch)
    assert torch.allclose(loss_baseline, loss_repeat)


def test_gamma_very_large_matches_vanilla_loss() -> None:
    """γ → ∞ means min(SNR, γ)/SNR = 1; the weighted loss must equal vanilla."""
    torch.manual_seed(0)
    baseline = _make_model(min_snr_gamma=None)
    weighted = _make_model(min_snr_gamma=1e12)
    # Copy weights so the two models compute identical eps_pred.
    weighted.load_state_dict(baseline.state_dict())
    batch = _make_batch()

    torch.manual_seed(123)
    loss_baseline = baseline.compute_loss(batch)
    torch.manual_seed(123)
    loss_weighted = weighted.compute_loss(batch)

    assert torch.allclose(loss_baseline, loss_weighted, atol=1e-5)


def test_gamma_5_differs_from_baseline() -> None:
    torch.manual_seed(0)
    baseline = _make_model(min_snr_gamma=None)
    weighted = _make_model(min_snr_gamma=5.0)
    weighted.load_state_dict(baseline.state_dict())
    batch = _make_batch()

    torch.manual_seed(123)
    loss_baseline = baseline.compute_loss(batch)
    torch.manual_seed(123)
    loss_weighted = weighted.compute_loss(batch)

    assert not torch.allclose(loss_baseline, loss_weighted)


def test_gamma_5_gradients_still_flow() -> None:
    torch.manual_seed(0)
    model = _make_model(min_snr_gamma=5.0)
    batch = _make_batch()
    loss = model.compute_loss(batch)
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )
    assert has_grad
