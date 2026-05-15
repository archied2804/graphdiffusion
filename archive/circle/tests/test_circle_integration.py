"""
Tests for circle experiment end-to-end integration
====================================================

Integration tests verifying the full circle experiment pipeline:
UnitCircleDataset with angular edge features, ScoreNetwork with
input_dim projection, loss computation, sampling with clamp_range.
"""

import torch
from context import graph_diffusion  # noqa: F401

from graph_diffusion.building_blocks.noise_schedule import NoiseSchedule
from graph_diffusion.data.circledataset import UnitCircleDataset
from graph_diffusion.data.dataloader import GraphDataLoader
from graph_diffusion.data.transforms import ComputeAngularEdgeFeatures
from graph_diffusion.model.graph_diffusion_model import GraphDiffusionModel
from graph_diffusion.model.score_network import ScoreNetwork


def _build_circle_pipeline(
    tmp_path: str,
) -> tuple[GraphDataLoader, GraphDiffusionModel]:
    """Build a small circle experiment pipeline for integration testing."""
    input_dim = 1
    node_dim = 16
    edge_dim = 2
    global_dim = 4

    pre_transform = ComputeAngularEdgeFeatures()

    dataset = UnitCircleDataset(
        root=tmp_path,
        n_graphs=20,
        n_nodes=16,
        n_fourier_modes=3,
        amplitude_scale=0.1,
        r_min=0.5,
        r_max=1.5,
        k_neighbors=2,
        global_dim=global_dim,
        seed=42,
        pre_transform=pre_transform,
    )

    loader = GraphDataLoader(
        dataset,
        batch_size=4,
        val_split=0.2,
        seed=42,
    )

    noise_schedule = NoiseSchedule(T=10, schedule_type="linear")

    score_network = ScoreNetwork(
        node_dim=node_dim,
        edge_dim=edge_dim,
        global_dim=global_dim,
        time_embed_dim=16,
        n_layers=2,
        hidden_dims=[16],
        activation="silu",
        layer_norm=True,
        residual=False,
        input_dim=input_dim,
    )

    model = GraphDiffusionModel(
        score_network=score_network,
        noise_schedule=noise_schedule,
    )

    return loader, model


class TestCircleIntegration:
    def test_compute_loss_and_backward(self, tmp_path):
        """Full circle pipeline: dataset -> loader -> compute_loss -> backward."""
        torch.manual_seed(0)
        loader, model = _build_circle_pipeline(str(tmp_path / "circle"))

        batch = next(iter(loader.train_loader()))
        loss = model.compute_loss(batch)

        assert loss.shape == ()
        assert loss.item() > 0
        assert torch.isfinite(loss)

        loss.backward()

        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                assert torch.isfinite(param.grad).all()
        assert has_grad

    def test_sample_with_clamp_range(self, tmp_path):
        """Sample with clamp_range should produce radii within bounds."""
        torch.manual_seed(0)
        loader, model = _build_circle_pipeline(str(tmp_path / "circle"))

        batch = next(iter(loader.train_loader()))
        r_min, r_max = 0.5, 1.5

        result = model.sample(batch, n_steps=5, clamp_range=(r_min, r_max))

        assert result.x is not None
        assert result.x.shape[1] == 1  # radial dim
        assert result.x.min().item() >= r_min - 1e-6
        assert result.x.max().item() <= r_max + 1e-6

    def test_sample_correct_shape(self, tmp_path):
        """Sample should produce Data with correct node feature shape."""
        torch.manual_seed(0)
        loader, model = _build_circle_pipeline(str(tmp_path / "circle"))

        batch = next(iter(loader.train_loader()))
        result = model.sample(batch, n_steps=5)

        assert result.x is not None
        assert result.x.shape == batch.x.shape
        assert torch.isfinite(result.x).all()

    def test_val_loader_works(self, tmp_path):
        """Validation loader should yield compatible batches."""
        torch.manual_seed(0)
        loader, model = _build_circle_pipeline(str(tmp_path / "circle"))

        batch = next(iter(loader.val_loader()))
        loss = model.compute_loss(batch)

        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_training_loop(self, tmp_path):
        """Multiple training steps should produce finite losses."""
        torch.manual_seed(0)
        loader, model = _build_circle_pipeline(str(tmp_path / "circle"))

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses = []
        for i, batch in enumerate(loader.train_loader()):
            optimizer.zero_grad()
            loss = model.compute_loss(batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if i >= 4:
                break

        assert all(torch.isfinite(torch.tensor(val)) for val in losses)
        assert len(losses) >= 2
