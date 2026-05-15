"""
Tests for ellipse conditioning: EllipseConditionalDataset (radial mode),
ScoreNetwork cond_dim/output_dim, and GraphDiffusionModel n_noise_channels.

Covers EXP-011 (radial + global_summary) and EXP-012 (radial + node_concat).
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from context import graph_diffusion  # noqa: F401
from torch_geometric.data import Data

from graph_diffusion.building_blocks.noise_schedule import NoiseSchedule
from graph_diffusion.data.ellipsedataset import (
    _H5_N_TIMESTEPS,
    EllipseConditionalDataset,
)
from graph_diffusion.data.transforms import ComputeAngularEdgeFeatures
from graph_diffusion.model.graph_diffusion_model import GraphDiffusionModel
from graph_diffusion.model.score_network import ScoreNetwork

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_N_SIMS = 8
_MAX_NODES = 8
_N_NODES_A = 6
_N_NODES_B = 8


def _write_fake_h5(path: Path) -> None:
    import h5py

    rng = np.random.default_rng(42)
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


def _make_cond_dataset(tmpdir: str, cond_type: str) -> EllipseConditionalDataset:
    root = Path(tmpdir) / f"cond_{cond_type}"
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True)
    _write_fake_h5(raw_dir / "pOnEllipseTrain.h5")
    return EllipseConditionalDataset(
        root=str(root),
        feature_mode="radial",
        cond_type=cond_type,
        n_samples=None,
        k_neighbors=2,
        global_dim=8,
        pre_transform=ComputeAngularEdgeFeatures(),
    )


# ---------------------------------------------------------------------------
# EllipseConditionalDataset — radial mode (EXP-011/012)
# ---------------------------------------------------------------------------


def test_conditional_dataset_invalid_cond_type() -> None:
    with pytest.raises(ValueError, match="cond_type"):
        EllipseConditionalDataset(root="/tmp/dummy", cond_type="attention")


def test_conditional_dataset_global_summary_shape() -> None:
    """EXP-011: global_summary produces x=(N,1) radial + cond=(1,4)."""
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = _make_cond_dataset(tmpdir, "global_summary")
        assert len(ds) == _N_SIMS
        graph = ds[0]
        assert graph.x.shape == (_N_NODES_A, 1)  # r
        assert graph.cond.shape == (1, 4)  # [p_mean, p_std, p_max, p_min]
        assert not hasattr(graph, "p_cond") or graph.p_cond is None


def test_conditional_dataset_node_concat_shape() -> None:
    """EXP-012: node_concat produces x=(N,1) radial + p_cond=(N,1)."""
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = _make_cond_dataset(tmpdir, "node_concat")
        assert len(ds) == _N_SIMS
        graph = ds[0]
        assert graph.x.shape == (_N_NODES_A, 1)  # r only
        assert graph.p_cond.shape == (_N_NODES_A, 1)  # per-node pressure
        assert not hasattr(graph, "cond") or graph.cond is None


# ---------------------------------------------------------------------------
# ScoreNetwork — cond_dim and output_dim
# ---------------------------------------------------------------------------


def _make_batched(n_nodes: int = 10, n_edges: int = 20, n_graphs: int = 2) -> Data:
    torch.manual_seed(0)
    batch = torch.zeros(n_nodes, dtype=torch.long)
    batch[n_nodes // 2 :] = 1
    return Data(
        x=torch.randn(n_nodes, 1),
        edge_index=torch.stack(
            [
                torch.randint(0, n_nodes, (n_edges,)),
                torch.randint(0, n_nodes, (n_edges,)),
            ]
        ),
        edge_attr=torch.randn(n_edges, 2),
        u=torch.zeros(n_graphs, 8),
        batch=batch,
    )


def test_score_network_with_cond_dim() -> None:
    """EXP-011: cond_proj injects global pressure summary into u."""
    torch.manual_seed(0)
    net = ScoreNetwork(
        node_dim=16,
        edge_dim=2,
        global_dim=8,
        time_embed_dim=32,
        n_layers=1,
        hidden_dims=[16],
        input_dim=1,
        cond_dim=4,
    )
    assert net.cond_proj is not None
    data = _make_batched()
    t = torch.zeros(2, dtype=torch.long)
    cond = torch.randn(2, 4)
    eps = net(data, t, cond=cond)
    assert eps.shape == (10, 1)


def test_score_network_without_cond() -> None:
    """cond=None does not raise and produces correct output shape."""
    torch.manual_seed(0)
    net = ScoreNetwork(
        node_dim=16,
        edge_dim=2,
        global_dim=8,
        time_embed_dim=32,
        n_layers=1,
        hidden_dims=[16],
        input_dim=1,
    )
    data = _make_batched()
    t = torch.zeros(2, dtype=torch.long)
    eps = net(data, t)
    assert eps.shape == (10, 1)


def test_score_network_output_dim() -> None:
    """EXP-012: output_dim=1 with input_dim=2 predicts noise for radial channel only."""
    torch.manual_seed(0)
    net = ScoreNetwork(
        node_dim=16,
        edge_dim=2,
        global_dim=8,
        time_embed_dim=32,
        n_layers=1,
        hidden_dims=[16],
        input_dim=2,
        output_dim=1,
    )
    assert net.output_decode is not None
    assert net.output_decode.out_features == 1
    data = _make_batched()
    data.x = torch.randn(10, 2)  # [r_t, p_cond]
    t = torch.zeros(2, dtype=torch.long)
    eps = net(data, t)
    assert eps.shape == (10, 1)


def test_score_network_output_dim_backward_compat() -> None:
    """Without output_dim, output shape defaults to input_dim."""
    torch.manual_seed(0)
    net = ScoreNetwork(
        node_dim=16,
        edge_dim=2,
        global_dim=8,
        time_embed_dim=32,
        n_layers=1,
        hidden_dims=[16],
        input_dim=1,
    )
    data = _make_batched()
    t = torch.zeros(2, dtype=torch.long)
    eps = net(data, t)
    assert eps.shape == (10, 1)


# ---------------------------------------------------------------------------
# GraphDiffusionModel — n_noise_channels (EXP-012)
# ---------------------------------------------------------------------------


def _make_model_with_n_noise(n_noise_channels: int) -> GraphDiffusionModel:
    net = ScoreNetwork(
        node_dim=16,
        edge_dim=2,
        global_dim=8,
        time_embed_dim=32,
        n_layers=1,
        hidden_dims=[16],
        input_dim=2,  # sees [r_t, p_cond]
        output_dim=n_noise_channels,
    )
    ns = NoiseSchedule(T=10, schedule_type="linear")
    return GraphDiffusionModel(
        score_network=net, noise_schedule=ns, n_noise_channels=n_noise_channels
    )


def test_compute_loss_n_noise_channels() -> None:
    """EXP-012: compute_loss noises only first channel, uses p_cond for concat."""
    torch.manual_seed(0)
    model = _make_model_with_n_noise(1)
    N, E, B = 10, 20, 2
    batch = torch.zeros(N, dtype=torch.long)
    batch[N // 2 :] = 1
    data = Data(
        x=torch.randn(N, 1),  # radial shape node features
        p_cond=torch.randn(N, 1),  # pressure conditioning
        edge_index=torch.stack([torch.randint(0, N, (E,)), torch.randint(0, N, (E,))]),
        edge_attr=torch.randn(E, 2),
        u=torch.zeros(B, 8),
        batch=batch,
    )
    loss = model.compute_loss(data)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_sample_n_noise_channels_output_shape() -> None:
    """EXP-012: sample() generates only n_noise_channels dims."""
    torch.manual_seed(0)
    model = _make_model_with_n_noise(1)
    N, E = 10, 20
    template = Data(
        x=torch.randn(N, 1),
        p_cond=torch.randn(N, 1),
        edge_index=torch.stack([torch.randint(0, N, (E,)), torch.randint(0, N, (E,))]),
        edge_attr=torch.randn(E, 2),
        u=torch.zeros(1, 8),
        pos=torch.randn(N, 2),
    )
    result = model.sample(template, n_steps=3)
    assert result.x.shape == (N, 1)
