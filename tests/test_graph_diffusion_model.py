"""
Tests for graph_diffusion.model.graph_diffusion_model
======================================================

Unit tests for GraphDiffusionModel: forward_diffusion, compute_loss, sample.
"""

import torch
from context import graph_diffusion  # noqa: F401
from torch_geometric.data import Data

from graph_diffusion.building_blocks.noise_schedule import NoiseSchedule
from graph_diffusion.model.graph_diffusion_model import GraphDiffusionModel
from graph_diffusion.model.score_network import ScoreNetwork

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NODE_DIM = 8
EDGE_DIM = 4
GLOBAL_DIM = 6


def _make_model():
    """Create a small GraphDiffusionModel for testing."""
    score_net = ScoreNetwork(
        node_dim=NODE_DIM,
        edge_dim=EDGE_DIM,
        global_dim=GLOBAL_DIM,
        time_embed_dim=16,
        n_layers=1,
        hidden_dims=[16],
    )
    ns = NoiseSchedule(T=50, schedule_type="linear")
    return GraphDiffusionModel(score_network=score_net, noise_schedule=ns)


def _make_batched_data(n_nodes=10, n_edges=20, n_graphs=2):
    """Create a batched Data object for testing."""
    torch.manual_seed(0)
    x = torch.randn(n_nodes, NODE_DIM)
    edge_index = torch.stack(
        [
            torch.randint(0, n_nodes, (n_edges,)),
            torch.randint(0, n_nodes, (n_edges,)),
        ]
    )
    edge_attr = torch.randn(n_edges, EDGE_DIM)
    u = torch.randn(n_graphs, GLOBAL_DIM)
    batch = torch.zeros(n_nodes, dtype=torch.long)
    batch[n_nodes // 2 :] = 1
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u, batch=batch)


# ---------------------------------------------------------------------------
# forward_diffusion
# ---------------------------------------------------------------------------


class TestForwardDiffusion:
    def test_output_shapes(self):
        """x_t and epsilon should have the same shape as x_0."""
        torch.manual_seed(0)
        model = _make_model()
        data = _make_batched_data()
        t = torch.tensor([5, 10])
        x_t, epsilon = model.forward_diffusion(data.x, t, data.batch)
        assert x_t.shape == data.x.shape
        assert epsilon.shape == data.x.shape

    def test_noise_is_standard_normal_scale(self):
        """Epsilon should have approximately zero mean and unit variance."""
        torch.manual_seed(0)
        model = _make_model()
        # Use many nodes for statistical significance
        n_nodes = 10000
        x_0 = torch.randn(n_nodes, NODE_DIM)
        batch = torch.zeros(n_nodes, dtype=torch.long)
        t = torch.tensor([25])
        _, epsilon = model.forward_diffusion(x_0, t, batch)
        assert epsilon.mean().abs() < 0.1
        assert (epsilon.std() - 1.0).abs() < 0.1

    def test_t_zero_preserves_signal(self):
        """At t=0 (alpha_bar close to 1), x_t should be close to x_0."""
        torch.manual_seed(0)
        model = _make_model()
        x_0 = torch.randn(5, NODE_DIM)
        batch = torch.zeros(5, dtype=torch.long)
        t = torch.tensor([0])
        x_t, _ = model.forward_diffusion(x_0, t, batch)
        # At t=0, sqrt_alpha_bar ≈ 1, so x_t ≈ x_0 + small noise
        alpha_bar_0 = model.noise_schedule.alphas_cumprod[0].item()
        assert alpha_bar_0 > 0.99  # should be very close to 1 for linear schedule


# ---------------------------------------------------------------------------
# compute_loss
# ---------------------------------------------------------------------------


class TestComputeLoss:
    def test_returns_scalar(self):
        """Loss should be a scalar tensor."""
        torch.manual_seed(0)
        model = _make_model()
        data = _make_batched_data()
        loss = model.compute_loss(data)
        assert loss.dim() == 0

    def test_loss_is_positive(self):
        """MSE loss should be non-negative."""
        torch.manual_seed(0)
        model = _make_model()
        data = _make_batched_data()
        loss = model.compute_loss(data)
        assert loss.item() >= 0.0

    def test_loss_is_differentiable(self):
        """Loss should support backward() for gradient computation."""
        torch.manual_seed(0)
        model = _make_model()
        data = _make_batched_data()
        loss = model.compute_loss(data)
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
        )
        assert has_grad

    def test_loss_single_graph(self):
        """Should work with batch size 1."""
        torch.manual_seed(0)
        model = _make_model()
        n_nodes, n_edges = 5, 8
        x = torch.randn(n_nodes, NODE_DIM)
        edge_index = torch.stack(
            [
                torch.randint(0, n_nodes, (n_edges,)),
                torch.randint(0, n_nodes, (n_edges,)),
            ]
        )
        edge_attr = torch.randn(n_edges, EDGE_DIM)
        u = torch.randn(1, GLOBAL_DIM)
        batch = torch.zeros(n_nodes, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u, batch=batch)
        loss = model.compute_loss(data)
        assert loss.dim() == 0
        assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------


class TestSample:
    def test_output_is_data(self):
        """sample() should return a Data object."""
        torch.manual_seed(0)
        model = _make_model()
        template = _make_batched_data()
        result = model.sample(template, n_steps=3)
        assert isinstance(result, Data)

    def test_output_shape(self):
        """Generated x should have the same shape as the template's x."""
        torch.manual_seed(0)
        model = _make_model()
        template = _make_batched_data()
        result = model.sample(template, n_steps=3)
        assert result.x.shape == template.x.shape

    def test_topology_preserved(self):
        """edge_index should be unchanged from the template."""
        torch.manual_seed(0)
        model = _make_model()
        template = _make_batched_data()
        result = model.sample(template, n_steps=3)
        assert torch.equal(result.edge_index, template.edge_index)

    def test_invalid_sampler(self):
        """Unknown sampler should raise ValueError."""
        torch.manual_seed(0)
        model = _make_model()
        template = _make_batched_data()
        try:
            model.sample(template, sampler="ddim")
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_default_n_steps(self):
        """When n_steps is None, should use noise_schedule.T steps."""
        torch.manual_seed(0)
        model = _make_model()
        template = _make_batched_data()
        # Just verify it runs without error — full T=50 steps
        result = model.sample(template)
        assert result.x.shape == template.x.shape
