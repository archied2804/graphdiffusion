"""
Tests for graph_diffusion.building_blocks.graph_network
========================================================

Unit tests for EdgeModel, NodeModel, GlobalModel, and GraphNetworkBlock.
"""

import torch
from context import graph_diffusion  # noqa: F401

from graph_diffusion.building_blocks.graph_network import (
    EdgeModel,
    GlobalModel,
    GraphNetworkBlock,
    NodeModel,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NODE_DIM = 8
EDGE_DIM = 4
GLOBAL_DIM = 6
HIDDEN_DIMS = [16]


def _make_graph_tensors(n_nodes=10, n_edges=20, n_graphs=2):
    """Create minimal tensors for a batched graph."""
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
    # Assign first half of nodes to graph 0, second half to graph 1
    batch = torch.zeros(n_nodes, dtype=torch.long)
    batch[n_nodes // 2 :] = 1
    return x, edge_index, edge_attr, u, batch


# ---------------------------------------------------------------------------
# EdgeModel
# ---------------------------------------------------------------------------


class TestEdgeModel:
    def test_output_shape(self):
        """Output should have shape (E, edge_dim)."""
        torch.manual_seed(0)
        model = EdgeModel(
            edge_dim=EDGE_DIM,
            node_dim=NODE_DIM,
            global_dim=GLOBAL_DIM,
            hidden_dims=HIDDEN_DIMS,
        )
        x, edge_index, edge_attr, u, batch = _make_graph_tensors()
        row, col = edge_index
        edge_batch = batch[row]
        u_edge = u[edge_batch]
        out = model(src=x[row], dest=x[col], edge_attr=edge_attr, u=u_edge, batch=batch)
        assert out.shape == edge_attr.shape

    def test_deterministic(self):
        """Same input should produce same output."""
        torch.manual_seed(0)
        model = EdgeModel(
            edge_dim=EDGE_DIM,
            node_dim=NODE_DIM,
            global_dim=GLOBAL_DIM,
            hidden_dims=HIDDEN_DIMS,
        )
        x, edge_index, edge_attr, u, batch = _make_graph_tensors()
        row, col = edge_index
        edge_batch = batch[row]
        u_edge = u[edge_batch]
        out1 = model(
            src=x[row], dest=x[col], edge_attr=edge_attr, u=u_edge, batch=batch
        )
        out2 = model(
            src=x[row], dest=x[col], edge_attr=edge_attr, u=u_edge, batch=batch
        )
        assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# NodeModel
# ---------------------------------------------------------------------------


class TestNodeModel:
    def test_output_shape(self):
        """Output should have shape (N, node_dim)."""
        torch.manual_seed(0)
        model = NodeModel(
            edge_dim=EDGE_DIM,
            node_dim=NODE_DIM,
            global_dim=GLOBAL_DIM,
            hidden_dims=HIDDEN_DIMS,
        )
        x, edge_index, edge_attr, u, batch = _make_graph_tensors()
        u_node = u[batch]
        out = model(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            u=u_node,
            batch=batch,
        )
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# GlobalModel
# ---------------------------------------------------------------------------


class TestGlobalModel:
    def test_output_shape(self):
        """Output should have shape (B, global_dim)."""
        torch.manual_seed(0)
        model = GlobalModel(
            edge_dim=EDGE_DIM,
            node_dim=NODE_DIM,
            global_dim=GLOBAL_DIM,
            hidden_dims=HIDDEN_DIMS,
        )
        x, edge_index, edge_attr, u, batch = _make_graph_tensors()
        out = model(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u, batch=batch)
        assert out.shape == u.shape


# ---------------------------------------------------------------------------
# GraphNetworkBlock
# ---------------------------------------------------------------------------


class TestGraphNetworkBlock:
    def test_output_shapes(self):
        """All three outputs should preserve input shapes."""
        torch.manual_seed(0)
        block = GraphNetworkBlock(
            node_dim=NODE_DIM,
            edge_dim=EDGE_DIM,
            global_dim=GLOBAL_DIM,
            hidden_dims=HIDDEN_DIMS,
        )
        x, edge_index, edge_attr, u, batch = _make_graph_tensors()
        x_out, edge_out, u_out = block(x, edge_index, edge_attr, u, batch)
        assert x_out.shape == x.shape
        assert edge_out.shape == edge_attr.shape
        assert u_out.shape == u.shape

    def test_no_edge_model(self):
        """With use_edge_model=False, edge_attr should pass through unchanged."""
        torch.manual_seed(0)
        block = GraphNetworkBlock(
            node_dim=NODE_DIM,
            edge_dim=EDGE_DIM,
            global_dim=GLOBAL_DIM,
            hidden_dims=HIDDEN_DIMS,
            use_edge_model=False,
        )
        x, edge_index, edge_attr, u, batch = _make_graph_tensors()
        _, edge_out, _ = block(x, edge_index, edge_attr, u, batch)
        assert torch.equal(edge_out, edge_attr)

    def test_no_node_model(self):
        """With use_node_model=False, x should pass through unchanged."""
        torch.manual_seed(0)
        block = GraphNetworkBlock(
            node_dim=NODE_DIM,
            edge_dim=EDGE_DIM,
            global_dim=GLOBAL_DIM,
            hidden_dims=HIDDEN_DIMS,
            use_node_model=False,
        )
        x, edge_index, edge_attr, u, batch = _make_graph_tensors()
        x_out, _, _ = block(x, edge_index, edge_attr, u, batch)
        assert torch.equal(x_out, x)

    def test_no_global_model(self):
        """With use_global_model=False, u should pass through unchanged."""
        torch.manual_seed(0)
        block = GraphNetworkBlock(
            node_dim=NODE_DIM,
            edge_dim=EDGE_DIM,
            global_dim=GLOBAL_DIM,
            hidden_dims=HIDDEN_DIMS,
            use_global_model=False,
        )
        x, edge_index, edge_attr, u, batch = _make_graph_tensors()
        _, _, u_out = block(x, edge_index, edge_attr, u, batch)
        assert torch.equal(u_out, u)

    def test_all_models_disabled(self):
        """With all models disabled, all outputs should pass through."""
        torch.manual_seed(0)
        block = GraphNetworkBlock(
            node_dim=NODE_DIM,
            edge_dim=EDGE_DIM,
            global_dim=GLOBAL_DIM,
            hidden_dims=HIDDEN_DIMS,
            use_edge_model=False,
            use_node_model=False,
            use_global_model=False,
        )
        x, edge_index, edge_attr, u, batch = _make_graph_tensors()
        x_out, edge_out, u_out = block(x, edge_index, edge_attr, u, batch)
        assert torch.equal(x_out, x)
        assert torch.equal(edge_out, edge_attr)
        assert torch.equal(u_out, u)

    def test_gradient_flow(self):
        """Gradients should flow through the full block."""
        torch.manual_seed(0)
        block = GraphNetworkBlock(
            node_dim=NODE_DIM,
            edge_dim=EDGE_DIM,
            global_dim=GLOBAL_DIM,
            hidden_dims=HIDDEN_DIMS,
        )
        x, edge_index, edge_attr, u, batch = _make_graph_tensors()
        x.requires_grad_(True)
        x_out, edge_out, u_out = block(x, edge_index, edge_attr, u, batch)
        loss = x_out.sum() + edge_out.sum() + u_out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_single_graph(self):
        """Block should work with a single graph (batch size 1)."""
        torch.manual_seed(0)
        block = GraphNetworkBlock(
            node_dim=NODE_DIM,
            edge_dim=EDGE_DIM,
            global_dim=GLOBAL_DIM,
            hidden_dims=HIDDEN_DIMS,
        )
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
        x_out, edge_out, u_out = block(x, edge_index, edge_attr, u, batch)
        assert x_out.shape == (n_nodes, NODE_DIM)
        assert edge_out.shape == (n_edges, EDGE_DIM)
        assert u_out.shape == (1, GLOBAL_DIM)
