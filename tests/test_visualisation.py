"""Unit tests for the graph_diffusion.visualisation package."""

from __future__ import annotations

import math
import shutil

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from graph_diffusion.building_blocks.noise_schedule import NoiseSchedule
from graph_diffusion.model.graph_diffusion_model import GraphDiffusionModel
from graph_diffusion.model.score_network import ScoreNetwork
from graph_diffusion.visualisation.plotting import (
    plot_conditioning_grid,
    plot_trajectory_filmstrip,
    write_trajectory_animation,
)
from graph_diffusion.visualisation.trajectory import (
    collect_forward,
    collect_reverse,
)


def _tiny_model() -> GraphDiffusionModel:
    """Build the smallest possible model for fast tests."""
    torch.manual_seed(0)
    sn = ScoreNetwork(
        node_dim=8,
        edge_dim=2,
        global_dim=4,
        time_embed_dim=8,
        n_layers=1,
        hidden_dims=[8, 8],
        input_dim=1,
        output_dim=1,
    )
    ns = NoiseSchedule(T=20, schedule_type="linear")
    return GraphDiffusionModel(score_network=sn, noise_schedule=ns)


def _tiny_template() -> Data:
    """8-node ring graph, radial features."""
    torch.manual_seed(0)
    n = 8
    theta = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    pos = torch.stack([theta.cos(), theta.sin()], dim=1)
    x = torch.ones(n, 1)
    edge_index = torch.stack(
        [
            torch.arange(n),
            (torch.arange(n) + 1) % n,
        ],
        dim=0,
    )
    edge_attr = torch.zeros(edge_index.size(1), 2)
    u = torch.zeros(1, 4)
    batch = torch.zeros(n, dtype=torch.long)
    return Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        u=u,
        batch=batch,
    )


def test_collect_forward_shapes() -> None:
    model = _tiny_model()
    template = _tiny_template()
    snapshots = collect_forward(model, template, snapshot_steps=[0, 5, 10, 19], seed=0)
    assert len(snapshots) == 4
    for snap in snapshots:
        assert snap.shape == (8, 1)
        assert snap.dtype == torch.float32
        assert torch.isfinite(snap).all()


def test_collect_forward_deterministic() -> None:
    model = _tiny_model()
    template = _tiny_template()
    a = collect_forward(model, template, snapshot_steps=[5, 15], seed=42)
    b = collect_forward(model, template, snapshot_steps=[5, 15], seed=42)
    for x, y in zip(a, b, strict=True):
        assert torch.allclose(x, y)


def test_collect_reverse_shapes() -> None:
    model = _tiny_model()
    template = _tiny_template()
    snapshots = collect_reverse(
        model,
        template,
        cond=None,
        snapshot_steps=[19, 10, 5, 0],
        seed=0,
    )
    assert len(snapshots) == 4
    for snap in snapshots:
        assert snap.shape == (8, 1)
        assert snap.dtype == torch.float32
        assert torch.isfinite(snap).all()


def test_collect_reverse_deterministic() -> None:
    model = _tiny_model()
    template = _tiny_template()
    a = collect_reverse(model, template, cond=None, snapshot_steps=[10, 0], seed=7)
    b = collect_reverse(model, template, cond=None, snapshot_steps=[10, 0], seed=7)
    for x, y in zip(a, b, strict=True):
        assert torch.allclose(x, y)


def test_plot_conditioning_grid_axes_count() -> None:
    n_targets = 4
    n_samples = 3
    n_cp_points = 50
    n_shape_nodes = 16

    rng = np.random.default_rng(0)
    target_cps = [rng.standard_normal(n_cp_points) for _ in range(n_targets)]
    head_pred_cps = [rng.standard_normal(n_cp_points) for _ in range(n_targets)]
    head_pred_stds = [
        np.abs(rng.standard_normal(n_cp_points)) for _ in range(n_targets)
    ]
    sample_shapes = [
        [rng.standard_normal((n_shape_nodes, 2)) for _ in range(n_samples)]
        for _ in range(n_targets)
    ]
    row_labels = [f"target {i}" for i in range(n_targets)]

    fig = plot_conditioning_grid(
        target_cps=target_cps,
        sample_shapes=sample_shapes,
        head_pred_cps=head_pred_cps,
        head_pred_stds=head_pred_stds,
        row_labels=row_labels,
    )
    expected_axes = n_targets * (2 + n_samples)
    assert len(fig.axes) == expected_axes


def test_plot_trajectory_filmstrip_axes_count() -> None:
    n_frames = 6
    n_nodes = 16
    rng = np.random.default_rng(1)
    forward = [rng.standard_normal((n_nodes, 2)) for _ in range(n_frames)]
    reverse = [rng.standard_normal((n_nodes, 2)) for _ in range(n_frames)]
    timesteps = [0, 5, 20, 50, 100, 199]
    target_cp = rng.standard_normal(50)

    fig = plot_trajectory_filmstrip(
        forward_snapshots=forward,
        reverse_snapshots=reverse,
        timesteps=timesteps,
        target_cp=target_cp,
    )
    # 2 rows × n_frames shape axes + 1 Cp panel.
    assert len(fig.axes) == 2 * n_frames + 1


def test_write_trajectory_animation_writes_gif(tmp_path) -> None:
    n_frames = 5
    n_nodes = 12
    rng = np.random.default_rng(2)
    frames = [rng.standard_normal((n_nodes, 2)) for _ in range(n_frames)]
    target_cp = rng.standard_normal(40)
    out_gif = tmp_path / "test_trajectory.gif"
    write_trajectory_animation(
        reverse_snapshots=frames,
        target_cp=target_cp,
        out_path_mp4=None,
        out_path_gif=out_gif,
        fps=10,
    )
    assert out_gif.exists()
    assert out_gif.stat().st_size > 1024


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_write_trajectory_animation_writes_mp4(tmp_path) -> None:
    n_frames = 5
    n_nodes = 12
    rng = np.random.default_rng(3)
    frames = [rng.standard_normal((n_nodes, 2)) for _ in range(n_frames)]
    target_cp = rng.standard_normal(40)
    out_mp4 = tmp_path / "test_trajectory.mp4"
    write_trajectory_animation(
        reverse_snapshots=frames,
        target_cp=target_cp,
        out_path_mp4=out_mp4,
        out_path_gif=None,
        fps=10,
    )
    assert out_mp4.exists()
    assert out_mp4.stat().st_size > 1024
