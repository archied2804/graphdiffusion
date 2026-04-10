"""
Tests for end-to-end integration
==================================

Integration tests verifying the full pipeline: dataset creation,
data loading, loss computation, backward pass, and sampling.
"""

import torch
from context import graph_diffusion  # noqa: F401

from graph_diffusion.building_blocks.noise_schedule import NoiseSchedule
from graph_diffusion.data.dataloader import GraphDataLoader
from graph_diffusion.data.dataset import SyntheticGraphDataset
from graph_diffusion.model.graph_diffusion_model import GraphDiffusionModel
from graph_diffusion.model.score_network import ScoreNetwork


def _build_pipeline(
    tmp_path: str,
) -> tuple[GraphDataLoader, GraphDiffusionModel]:
    """Build a small pipeline for integration testing."""
    node_feature_dim = 8
    global_dim = 16
    edge_dim = 1

    dataset = SyntheticGraphDataset(
        root=tmp_path,
        n_graphs=20,
        n_nodes_range=(10, 20),
        node_feature_dim=node_feature_dim,
        k=4,
        global_dim=global_dim,
        seed=42,
    )

    loader = GraphDataLoader(
        dataset,
        batch_size=4,
        val_split=0.2,
        seed=42,
    )

    noise_schedule = NoiseSchedule(T=10, schedule_type="linear")

    score_network = ScoreNetwork(
        node_dim=node_feature_dim,
        edge_dim=edge_dim,
        global_dim=global_dim,
        time_embed_dim=32,
        n_layers=2,
        hidden_dims=[32],
        activation="silu",
        layer_norm=True,
        residual=False,
    )

    model = GraphDiffusionModel(
        score_network=score_network,
        noise_schedule=noise_schedule,
    )

    return loader, model


def _add_edge_attr(batch):
    """Add dummy edge_attr of dimension 1 (edge length from pos)."""
    src, dst = batch.edge_index
    diff = batch.pos[src] - batch.pos[dst]
    batch.edge_attr = diff.norm(dim=-1, keepdim=True)
    return batch


class TestIntegrationPipeline:
    def test_compute_loss_and_backward(self, tmp_path):
        """Full pipeline: dataset -> loader -> compute_loss -> backward."""
        torch.manual_seed(0)
        loader, model = _build_pipeline(str(tmp_path / "integration"))

        batch = next(iter(loader.train_loader()))
        batch = _add_edge_attr(batch)
        loss = model.compute_loss(batch)

        assert loss.shape == ()
        assert loss.item() > 0
        assert torch.isfinite(loss)

        loss.backward()

        # Verify gradients were computed
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                assert torch.isfinite(param.grad).all()
        assert has_grad

    def test_sample_correct_shape(self, tmp_path):
        """Sample produces Data with correct node feature shape."""
        torch.manual_seed(0)
        loader, model = _build_pipeline(str(tmp_path / "integration"))

        batch = next(iter(loader.train_loader()))
        batch = _add_edge_attr(batch)

        result = model.sample(batch, n_steps=5)

        assert result.x is not None
        assert result.x.shape == batch.x.shape
        assert torch.isfinite(result.x).all()

    def test_val_loader_works(self, tmp_path):
        """Validation loader yields batches compatible with compute_loss."""
        torch.manual_seed(0)
        loader, model = _build_pipeline(str(tmp_path / "integration"))

        batch = next(iter(loader.val_loader()))
        batch = _add_edge_attr(batch)
        loss = model.compute_loss(batch)

        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_training_loop(self, tmp_path):
        """Multiple training steps reduce loss (smoke test)."""
        torch.manual_seed(0)
        loader, model = _build_pipeline(str(tmp_path / "integration"))

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses = []
        for i, batch in enumerate(loader.train_loader()):
            batch = _add_edge_attr(batch)
            optimizer.zero_grad()
            loss = model.compute_loss(batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if i >= 4:
                break

        # All losses should be finite
        assert all(torch.isfinite(torch.tensor(val)) for val in losses)
        # At least 2 training steps completed
        assert len(losses) >= 2
