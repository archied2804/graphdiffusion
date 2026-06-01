"""
Tests for graph_diffusion.data.pOnEllipse
==========================================

Unit tests for DatasetUrl, DatasetDownloader, pOnEllipseDataset, and
ComputeArcLengthEdgeFeatures.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from context import graph_diffusion  # noqa: F401

from graph_diffusion.data.pOnEllipse import (
    _H5_N_TIMESTEPS,
    DatasetDownloader,
    DatasetUrl,
    pOnEllipseDataset,
)
from graph_diffusion.data.transforms import (
    ComputeAngularEdgeFeatures,
    ComputeArcLengthEdgeFeatures,
)

# ---------------------------------------------------------------------------
# Helpers — fake HDF5 matching the real pOnEllipse layout
# ---------------------------------------------------------------------------

_N_SIMS = 6
_MAX_NODES = 8
_N_NODES_A = 6
_N_NODES_B = 8


def _fake_ellipse_nodes(
    n_nodes: int, re: float, aoa: float, rng: np.random.Generator
) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, n_nodes, endpoint=False)
    a = rng.uniform(1.0, 2.0)
    b = rng.uniform(0.5, 1.5)
    x = (a * np.cos(theta)).astype(np.float32)
    y = (b * np.sin(theta)).astype(np.float32)
    re_col = np.full(n_nodes, re, dtype=np.float32)
    aoa_col = np.full(n_nodes, aoa, dtype=np.float32)
    p = rng.standard_normal((n_nodes, _H5_N_TIMESTEPS)).astype(np.float32)
    return np.concatenate(
        [x[:, None], y[:, None], re_col[:, None], aoa_col[:, None], p], axis=1
    )


def _write_fake_h5(path: Path) -> None:
    import h5py

    rng = np.random.default_rng(42)
    n_feat = 4 + _H5_N_TIMESTEPS
    data_arr = np.full((_N_SIMS, _MAX_NODES, n_feat), np.nan, dtype=np.float32)
    for i in range(_N_SIMS):
        n_nodes = _N_NODES_A if i % 2 == 0 else _N_NODES_B
        rows = _fake_ellipse_nodes(n_nodes, re=500.0 + i, aoa=5.0 + 0.1 * i, rng=rng)
        data_arr[i, :n_nodes, :] = rows
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=data_arr)


def _make_dataset_root(tmpdir: str) -> tuple[Path, Path]:
    root = Path(tmpdir) / "dataset"
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True)
    _write_fake_h5(raw_dir / "pOnEllipseTrain.h5")
    return root, raw_dir  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# DatasetUrl
# ---------------------------------------------------------------------------


def test_dataset_url_contains_repo() -> None:
    for url in DatasetUrl:
        assert "mariolinov/Ellipse" in url.value
        assert url.value.startswith("https://")


def test_dataset_url_filenames() -> None:
    assert DatasetUrl.TRAIN_H5.value.endswith("pOnEllipseTrain.h5")
    assert DatasetUrl.TEST_H5.value.endswith("pOnEllipseTest.h5")
    assert DatasetUrl.TIME_TRAIN_NPY.value.endswith("TimeEllipseTrain.npy")
    assert DatasetUrl.AOA10_H5.value.endswith("pOnEllipseAoA10.h5")


# ---------------------------------------------------------------------------
# pOnEllipseDataset.variant
# ---------------------------------------------------------------------------


def test_dataset_variant_default_picks_train_h5() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root, _ = _make_dataset_root(tmpdir)
        ds = pOnEllipseDataset(
            root=str(root),
            feature_mode="radial",
            split="train",
            n_samples=2,
            k_neighbors=2,
            pre_transform=ComputeAngularEdgeFeatures(),
        )
        assert ds.variant == "default"
        assert ds.raw_file_names == ["pOnEllipseTrain.h5"]


def test_dataset_variant_aoa10_picks_aoa_h5() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "dataset"
        raw_dir = root / "raw"
        raw_dir.mkdir(parents=True)
        _write_fake_h5(raw_dir / "pOnEllipseAoA10.h5")
        ds = pOnEllipseDataset(
            root=str(root),
            feature_mode="radial",
            split="train",
            variant="aoa10",
            n_samples=2,
            k_neighbors=2,
            pre_transform=ComputeAngularEdgeFeatures(),
        )
        assert ds.variant == "aoa10"
        assert ds.raw_file_names == ["pOnEllipseAoA10.h5"]


def test_dataset_invalid_variant_raises() -> None:
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        pytest.raises(ValueError, match="variant"),
    ):
        pOnEllipseDataset(
            root=tmpdir,
            feature_mode="radial",
            split="train",
            variant="aoa20",
        )


def test_dataset_aoa10_processed_cache_differs_from_default() -> None:
    # If the processed cache file name doesn't reflect the variant, switching
    # variants would silently return the wrong cached graphs.
    with (
        tempfile.TemporaryDirectory() as tmpdir_a,
        tempfile.TemporaryDirectory() as tmpdir_b,
    ):
        root_a, _ = _make_dataset_root(tmpdir_a)
        root_b = Path(tmpdir_b) / "dataset"
        raw_b = root_b / "raw"
        raw_b.mkdir(parents=True)
        _write_fake_h5(raw_b / "pOnEllipseAoA10.h5")
        ds_default = pOnEllipseDataset(
            root=str(root_a),
            feature_mode="radial",
            split="train",
            n_samples=1,
            k_neighbors=2,
            pre_transform=ComputeAngularEdgeFeatures(),
        )
        ds_aoa = pOnEllipseDataset(
            root=str(root_b),
            feature_mode="radial",
            split="train",
            variant="aoa10",
            n_samples=1,
            k_neighbors=2,
            pre_transform=ComputeAngularEdgeFeatures(),
        )
        assert ds_default.processed_file_names != ds_aoa.processed_file_names


# ---------------------------------------------------------------------------
# DatasetDownloader
# ---------------------------------------------------------------------------


def test_downloader_local_path() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        dl = DatasetDownloader(root=tmpdir)
        p = dl.local_path(DatasetUrl.TRAIN_H5)
        assert p.name == "pOnEllipseTrain.h5"
        assert p.parent == Path(tmpdir)


def test_downloader_is_downloaded_false() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        dl = DatasetDownloader(root=tmpdir)
        assert not dl.is_downloaded(DatasetUrl.TRAIN_H5)


def test_downloader_is_downloaded_after_touch() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        dl = DatasetDownloader(root=tmpdir)
        dl.local_path(DatasetUrl.TRAIN_H5).touch()
        assert dl.is_downloaded(DatasetUrl.TRAIN_H5)


def test_downloader_creates_root_dir() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        nested = Path(tmpdir) / "a" / "b" / "c"
        DatasetDownloader(root=str(nested))
        assert nested.exists()


# ---------------------------------------------------------------------------
# ComputeArcLengthEdgeFeatures
# ---------------------------------------------------------------------------


def test_arc_length_features_shape() -> None:
    torch.manual_seed(0)
    from torch_geometric.data import Data

    N, E = 16, 32
    theta = torch.linspace(0, 2 * torch.pi, N + 1)[:-1]
    pos = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    edge_index = torch.stack([torch.randint(0, N, (E,)), torch.randint(0, N, (E,))])
    data = Data(pos=pos, edge_index=edge_index)
    out = ComputeArcLengthEdgeFeatures()(data)
    assert out.edge_attr.shape == (E, 2)


def test_arc_length_features_unit_values() -> None:
    torch.manual_seed(0)
    from torch_geometric.data import Data

    N = 32
    theta = torch.linspace(0, 2 * torch.pi, N + 1)[:-1]
    pos = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    src = torch.arange(N)
    dst = (src + 1) % N
    edge_index = torch.stack([src, dst])
    data = Data(pos=pos, edge_index=edge_index)
    out = ComputeArcLengthEdgeFeatures()(data)
    assert out.edge_attr[:, 0].abs().max() <= 1.0 + 1e-5
    assert out.edge_attr[:, 1].abs().max() <= 1.0 + 1e-5


def test_arc_length_raises_no_pos() -> None:
    from torch_geometric.data import Data

    data = Data(edge_index=torch.zeros(2, 4, dtype=torch.long))
    with pytest.raises(ValueError, match="pos"):
        ComputeArcLengthEdgeFeatures()(data)


def test_arc_length_raises_no_edge_index() -> None:
    from torch_geometric.data import Data

    data = Data(pos=torch.randn(8, 2))
    with pytest.raises(ValueError, match="edge_index"):
        ComputeArcLengthEdgeFeatures()(data)


# ---------------------------------------------------------------------------
# pOnEllipseDataset — validation
# ---------------------------------------------------------------------------


def test_invalid_feature_mode() -> None:
    with pytest.raises(ValueError, match="feature_mode"):
        pOnEllipseDataset(root="/tmp/dummy", feature_mode="polar")


def test_invalid_split() -> None:
    with pytest.raises(ValueError, match="split"):
        pOnEllipseDataset(root="/tmp/dummy", split="val")


def test_invalid_k_neighbors() -> None:
    with pytest.raises(ValueError, match="k_neighbors"):
        pOnEllipseDataset(root="/tmp/dummy", k_neighbors=0)


def test_normalised_requires_global_dim_2() -> None:
    with pytest.raises(ValueError, match="global_dim"):
        pOnEllipseDataset(root="/tmp/dummy", feature_mode="normalised", global_dim=1)


# ---------------------------------------------------------------------------
# pOnEllipseDataset — feature modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode,expected_x_dim",
    [
        ("radial", 1),
        ("radial_norm", 1),
        ("cartesian", 2),
        ("normalised", 1),
    ],
)
def test_feature_modes(mode: str, expected_x_dim: int) -> None:
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        root, _ = _make_dataset_root(tmpdir)
        pre_transform = (
            ComputeArcLengthEdgeFeatures()
            if mode == "cartesian"
            else ComputeAngularEdgeFeatures()
        )
        ds = pOnEllipseDataset(
            root=str(root),
            feature_mode=mode,
            n_samples=5,
            k_neighbors=2,
            global_dim=8,
            pre_transform=pre_transform,
        )
        graph = ds[0]
        assert graph.x.shape == (
            _N_NODES_A,
            expected_x_dim,
        ), f"mode={mode}: expected x dim {expected_x_dim}, got {graph.x.shape}"
        assert graph.edge_attr.shape[1] == 2
        assert graph.u.shape == (1, 8)


def test_normalised_stores_ab() -> None:
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        root, _ = _make_dataset_root(tmpdir)
        ds = pOnEllipseDataset(
            root=str(root),
            feature_mode="normalised",
            n_samples=5,
            k_neighbors=2,
            global_dim=8,
            pre_transform=ComputeAngularEdgeFeatures(),
        )
        graph = ds[0]
        assert graph.u[0, 0] > 0, "semi-axis a should be positive"
        assert graph.u[0, 1] > 0, "semi-axis b should be positive"
        assert graph.u[0, 2:].abs().sum() == 0, "remaining u dims should be zero"


def test_radial_norm_stores_r_scale_and_p_cond() -> None:
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        root, _ = _make_dataset_root(tmpdir)
        ds = pOnEllipseDataset(
            root=str(root),
            feature_mode="radial_norm",
            n_samples=5,
            k_neighbors=2,
            global_dim=8,
            pre_transform=ComputeAngularEdgeFeatures(),
        )
        graph = ds[0]
        assert graph.x.shape == (_N_NODES_A, 1)
        assert hasattr(graph, "r_scale") and graph.r_scale.item() > 0
        mean_r_norm = graph.x[:, 0].mean().item()
        assert abs(mean_r_norm - 1.0) < 1e-5
        assert hasattr(graph, "p_cond")
        assert graph.p_cond.shape == (_N_NODES_A, 2)
        norms = (graph.p_cond**2).sum(dim=-1).sqrt()
        assert torch.allclose(norms, torch.ones(_N_NODES_A), atol=1e-5)
