"""
Tests for graph_diffusion.model.score_network
===============================================

Unit tests for ScoreNetwork.
"""

import torch
from context import graph_diffusion  # noqa: F401
from torch_geometric.data import Data

from graph_diffusion.model.score_network import ScoreNetwork

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NODE_DIM = 8
EDGE_DIM = 4
GLOBAL_DIM = 6
TIME_EMBED_DIM = 16
N_LAYERS = 2
HIDDEN_DIMS = [16]


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


def _make_score_network():
    """Create a ScoreNetwork instance for testing."""
    return ScoreNetwork(
        node_dim=NODE_DIM,
        edge_dim=EDGE_DIM,
        global_dim=GLOBAL_DIM,
        time_embed_dim=TIME_EMBED_DIM,
        n_layers=N_LAYERS,
        hidden_dims=HIDDEN_DIMS,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestScoreNetwork:
    def test_output_shape(self):
        """Output should have the same shape as data.x."""
        torch.manual_seed(0)
        net = _make_score_network()
        data = _make_batched_data()
        t = torch.tensor([5, 10])
        eps_pred = net(data, t)
        assert eps_pred.shape == data.x.shape

    def test_output_shape_single_graph(self):
        """Should work with a single graph (batch size 1)."""
        torch.manual_seed(0)
        net = _make_score_network()
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
        t = torch.tensor([50])
        eps_pred = net(data, t)
        assert eps_pred.shape == (n_nodes, NODE_DIM)

    def test_gradient_flow(self):
        """Gradients should flow from loss back to network parameters."""
        torch.manual_seed(0)
        net = _make_score_network()
        data = _make_batched_data()
        t = torch.tensor([0, 99])
        eps_pred = net(data, t)
        loss = eps_pred.sum()
        loss.backward()
        # Check that at least some parameters got gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters()
        )
        assert has_grad

    def test_deterministic(self):
        """Same input should produce same output."""
        torch.manual_seed(0)
        net = _make_score_network()
        data = _make_batched_data()
        t = torch.tensor([5, 10])
        out1 = net(data, t)
        out2 = net(data, t)
        assert torch.allclose(out1, out2)

    def test_different_timesteps_produce_different_output(self):
        """Different timesteps should produce different predictions."""
        torch.manual_seed(0)
        net = _make_score_network()
        data1 = _make_batched_data()
        data2 = _make_batched_data()
        t1 = torch.tensor([0, 0])
        t2 = torch.tensor([500, 500])
        out1 = net(data1, t1)
        out2 = net(data2, t2)
        assert not torch.allclose(out1, out2)

    def test_invalid_n_layers(self):
        """n_layers < 1 should raise ValueError."""
        try:
            ScoreNetwork(
                node_dim=8,
                edge_dim=4,
                global_dim=6,
                time_embed_dim=16,
                n_layers=0,
                hidden_dims=[16],
            )
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_parameter_count_increases_with_layers(self):
        """More layers should mean more parameters."""
        torch.manual_seed(0)
        net_small = ScoreNetwork(
            node_dim=NODE_DIM,
            edge_dim=EDGE_DIM,
            global_dim=GLOBAL_DIM,
            time_embed_dim=TIME_EMBED_DIM,
            n_layers=1,
            hidden_dims=HIDDEN_DIMS,
        )
        net_big = ScoreNetwork(
            node_dim=NODE_DIM,
            edge_dim=EDGE_DIM,
            global_dim=GLOBAL_DIM,
            time_embed_dim=TIME_EMBED_DIM,
            n_layers=4,
            hidden_dims=HIDDEN_DIMS,
        )
        n_small = sum(p.numel() for p in net_small.parameters())
        n_big = sum(p.numel() for p in net_big.parameters())
        assert n_big > n_small

    def test_input_dim_output_shape(self):
        """With input_dim=1, output should be (N, 1), not (N, node_dim)."""
        torch.manual_seed(0)
        input_dim = 1
        node_dim = 32
        edge_dim = 2
        global_dim = 8
        n_nodes = 10
        n_edges = 20
        n_graphs = 2

        net = ScoreNetwork(
            node_dim=node_dim,
            edge_dim=edge_dim,
            global_dim=global_dim,
            time_embed_dim=16,
            n_layers=2,
            hidden_dims=[16],
            input_dim=input_dim,
        )

        x = torch.randn(n_nodes, input_dim)
        edge_index = torch.stack(
            [
                torch.randint(0, n_nodes, (n_edges,)),
                torch.randint(0, n_nodes, (n_edges,)),
            ]
        )
        edge_attr = torch.randn(n_edges, edge_dim)
        u = torch.randn(n_graphs, global_dim)
        batch = torch.zeros(n_nodes, dtype=torch.long)
        batch[n_nodes // 2 :] = 1
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u, batch=batch)

        t = torch.tensor([5, 10])
        eps_pred = net(data, t)
        assert eps_pred.shape == (n_nodes, input_dim)

    def test_input_dim_none_unchanged_behaviour(self):
        """With input_dim=None, output shape should match node_dim."""
        torch.manual_seed(0)
        net = _make_score_network()
        data = _make_batched_data()
        t = torch.tensor([5, 10])
        eps_pred = net(data, t)
        assert eps_pred.shape == (data.x.shape[0], NODE_DIM)

    def test_input_dim_gradient_flow(self):
        """Gradients should flow through input/output projections."""
        torch.manual_seed(0)
        net = ScoreNetwork(
            node_dim=16,
            edge_dim=2,
            global_dim=4,
            time_embed_dim=8,
            n_layers=1,
            hidden_dims=[8],
            input_dim=1,
        )

        x = torch.randn(6, 1)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        edge_attr = torch.randn(4, 2)
        u = torch.randn(1, 4)
        batch = torch.zeros(6, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u, batch=batch)

        t = torch.tensor([3])
        eps_pred = net(data, t)
        loss = eps_pred.sum()
        loss.backward()

        # Check input_proj and output_decode have gradients
        assert net.input_proj is not None
        assert net.input_proj.weight.grad is not None
        assert net.output_decode is not None
        assert net.output_decode.weight.grad is not None
