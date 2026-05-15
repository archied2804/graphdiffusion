"""
End-to-end integration tests for the ellipse pipeline.

Tests:
  - EllipseShapeDataset → GraphDataLoader → GraphDiffusionModel.compute_loss
  - Full 1-epoch train + sample with radial representation (EXP-010, plain radial)
  - Full 1-epoch train + sample with radial_norm + positional conditioning
  - EllipseConditionalDataset (radial + global_summary) → train + sample (EXP-011)
  - EllipseConditionalDataset (radial + node_concat) → train + sample (EXP-012)

"""

import tempfile
from pathlib import Path

import numpy as np
import torch
from context import graph_diffusion  # noqa: F401

from graph_diffusion.building_blocks.noise_schedule import NoiseSchedule
from graph_diffusion.data.dataloader import GraphDataLoader
from graph_diffusion.data.ellipsedataset import (
    _H5_N_TIMESTEPS,
    EllipseConditionalDataset,
    EllipseShapeDataset,
)
from graph_diffusion.data.transforms import ComputeAngularEdgeFeatures
from graph_diffusion.model.graph_diffusion_model import GraphDiffusionModel
from graph_diffusion.model.score_network import ScoreNetwork

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_N_SIMS = 8  # number of CFD simulations in the fake HDF5
_MAX_NODES = 8  # NaN-padded node dimension
_N_NODES_A = 6  # node count for even simulations
_N_NODES_B = 8  # node count for odd simulations


def _write_fake_h5(path: Path) -> None:
    import h5py

    rng = np.random.default_rng(0)
    n_feat = 4 + _H5_N_TIMESTEPS
    data_arr = np.full((_N_SIMS, _MAX_NODES, n_feat), np.nan, dtype=np.float32)
    for i in range(_N_SIMS):
        n_nodes = _N_NODES_A if i % 2 == 0 else _N_NODES_B
        theta = np.linspace(0.0, 2.0 * np.pi, n_nodes, endpoint=False)
        a = rng.uniform(1.0, 2.0)
        b = rng.uniform(0.5, 1.5)
        x = (a * np.cos(theta)).astype(np.float32)
        y = (b * np.sin(theta)).astype(np.float32)
        re_col = np.full(n_nodes, 500.0, dtype=np.float32)
        aoa_col = np.full(n_nodes, 5.5, dtype=np.float32)
        p = rng.standard_normal((n_nodes, _H5_N_TIMESTEPS)).astype(np.float32)
        rows = np.concatenate(
            [x[:, None], y[:, None], re_col[:, None], aoa_col[:, None], p], axis=1
        )
        data_arr[i, :n_nodes, :] = rows
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=data_arr)


def _make_root(tmpdir: str, suffix: str = "") -> Path:
    root = Path(tmpdir) / f"ds{suffix}"
    (root / "raw").mkdir(parents=True)
    _write_fake_h5(root / "raw" / "pOnEllipseTrain.h5")
    return root


def _make_score_net(
    input_dim: int = 1,
    cond_dim: int | None = None,
    output_dim: int | None = None,
) -> ScoreNetwork:
    return ScoreNetwork(
        node_dim=16,
        edge_dim=2,
        global_dim=8,
        time_embed_dim=16,
        n_layers=1,
        hidden_dims=[16],
        input_dim=input_dim,
        cond_dim=cond_dim,
        output_dim=output_dim,
    )


def _make_model(
    input_dim: int = 1,
    cond_dim: int | None = None,
    output_dim: int | None = None,
    n_noise_channels: int | None = None,
) -> GraphDiffusionModel:
    ns = NoiseSchedule(T=10, schedule_type="linear")
    return GraphDiffusionModel(
        score_network=_make_score_net(input_dim, cond_dim, output_dim),
        noise_schedule=ns,
        n_noise_channels=n_noise_channels,
    )


# ---------------------------------------------------------------------------
# Shape dataset (Method A — radial)
# ---------------------------------------------------------------------------


def test_radial_shape_train_loop() -> None:
    """1-epoch training loop with EllipseShapeDataset (radial)."""
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_root(tmpdir)
        ds = EllipseShapeDataset(
            root=str(root),
            feature_mode="radial",
            n_samples=_N_SIMS,
            k_neighbors=2,
            global_dim=8,
            pre_transform=ComputeAngularEdgeFeatures(),
        )
        loader = GraphDataLoader(ds, batch_size=8, val_split=0.2, seed=0)
        model = _make_model(input_dim=1)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses: list[float] = []
        for batch in loader.train_loader():
            opt.zero_grad()
            loss = model.compute_loss(batch)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert len(losses) > 0
        assert all(v >= 0 for v in losses)


def test_radial_shape_sample() -> None:
    """sample() returns x with (N, 1) shape for a radial template."""
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_root(tmpdir, "s")
        ds = EllipseShapeDataset(
            root=str(root),
            feature_mode="radial",
            n_samples=10,
            k_neighbors=2,
            global_dim=8,
            pre_transform=ComputeAngularEdgeFeatures(),
        )
        model = _make_model(input_dim=1)
        template = ds[0]
        result = model.sample(template, n_steps=2)
        assert result.x.shape == (_N_NODES_A, 1)


def test_radial_norm_pos_train_loop() -> None:
    """radial_norm with [cos θ, sin θ] conditioning: input_dim=3, n_noise_channels=1."""
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_root(tmpdir, "rnp")
        ds = EllipseShapeDataset(
            root=str(root),
            feature_mode="radial_norm",
            n_samples=_N_SIMS,
            k_neighbors=2,
            global_dim=8,
            pre_transform=ComputeAngularEdgeFeatures(),
        )
        # radial_norm graphs carry p_cond=[cos θ, sin θ] for positional context
        assert hasattr(ds[0], "p_cond") and ds[0].p_cond.shape == (_N_NODES_A, 2)
        loader = GraphDataLoader(ds, batch_size=8, val_split=0.2, seed=0)
        model = _make_model(input_dim=3, output_dim=1, n_noise_channels=1)

        for batch in loader.train_loader():
            loss = model.compute_loss(batch)
            assert loss.ndim == 0 and loss.item() >= 0
            break


def test_radial_norm_pos_sample() -> None:
    """sample() with radial_norm positional conditioning returns (N, 1)."""
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_root(tmpdir, "rnps")
        ds = EllipseShapeDataset(
            root=str(root),
            feature_mode="radial_norm",
            n_samples=10,
            k_neighbors=2,
            global_dim=8,
            pre_transform=ComputeAngularEdgeFeatures(),
        )
        model = _make_model(input_dim=3, output_dim=1, n_noise_channels=1)
        template = ds[0]
        result = model.sample(template, n_steps=2)
        assert result.x.shape == (_N_NODES_A, 1)


# ---------------------------------------------------------------------------
# Conditional dataset — radial + global_summary (EXP-011)
# ---------------------------------------------------------------------------


def test_global_summary_train_loop() -> None:
    """1-epoch training — EXP-011: radial_norm + pos context + global pressure cond."""
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_root(tmpdir, "c11")
        ds = EllipseConditionalDataset(
            root=str(root),
            feature_mode="radial_norm",
            cond_type="global_summary",
            n_samples=_N_SIMS,
            k_neighbors=2,
            global_dim=8,
            pre_transform=ComputeAngularEdgeFeatures(),
        )
        # p_cond=[cos θ, sin θ], cond=[p_mean,p_std,p_max,p_min]
        assert ds[0].p_cond.shape == (_N_NODES_A, 2)
        loader = GraphDataLoader(ds, batch_size=8, val_split=0.2, seed=0)
        model = _make_model(input_dim=3, cond_dim=4, output_dim=1, n_noise_channels=1)

        for batch in loader.train_loader():
            loss = model.compute_loss(batch)
            assert loss.ndim == 0 and loss.item() >= 0
            break


def test_global_summary_sample_with_cond() -> None:
    """sample() with radial_norm + global cond produces (N, 1) output (EXP-011)."""
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_root(tmpdir, "c11s")
        ds = EllipseConditionalDataset(
            root=str(root),
            feature_mode="radial_norm",
            cond_type="global_summary",
            n_samples=10,
            k_neighbors=2,
            global_dim=8,
            pre_transform=ComputeAngularEdgeFeatures(),
        )
        model = _make_model(input_dim=3, cond_dim=4, output_dim=1, n_noise_channels=1)
        template = ds[0]
        result = model.sample(template, n_steps=2)
        assert result.x.shape == (_N_NODES_A, 1)


# ---------------------------------------------------------------------------
# Conditional dataset — radial + node_concat (EXP-012)
# ---------------------------------------------------------------------------


def test_node_concat_train_loop() -> None:
    """1-epoch training — EXP-012: radial_norm + pos + per-node pressure concat."""
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_root(tmpdir, "c12")
        ds = EllipseConditionalDataset(
            root=str(root),
            feature_mode="radial_norm",
            cond_type="node_concat",
            n_samples=_N_SIMS,
            k_neighbors=2,
            global_dim=8,
            pre_transform=ComputeAngularEdgeFeatures(),
        )
        # p_cond=[cos θ, sin θ, p_node] for radial_norm node_concat
        assert ds[0].p_cond.shape == (_N_NODES_A, 3)
        loader = GraphDataLoader(ds, batch_size=8, val_split=0.2, seed=0)
        # input_dim=4 ([r_t, cos θ, sin θ, p_node]), output_dim=1, n_noise_channels=1
        model = _make_model(input_dim=4, output_dim=1, n_noise_channels=1)

        for batch in loader.train_loader():
            loss = model.compute_loss(batch)
            assert loss.ndim == 0 and loss.item() >= 0
            break


def test_node_concat_sample() -> None:
    """sample() with radial_norm + node_concat produces (N, 1) output (EXP-012)."""
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_root(tmpdir, "c12s")
        ds = EllipseConditionalDataset(
            root=str(root),
            feature_mode="radial_norm",
            cond_type="node_concat",
            n_samples=10,
            k_neighbors=2,
            global_dim=8,
            pre_transform=ComputeAngularEdgeFeatures(),
        )
        model = _make_model(input_dim=4, output_dim=1, n_noise_channels=1)
        template = ds[0]
        result = model.sample(template, n_steps=2)
        assert result.x.shape == (_N_NODES_A, 1)
