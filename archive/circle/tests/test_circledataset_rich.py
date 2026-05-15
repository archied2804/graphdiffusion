"""Tests for UnitCircleDataset curvature and arc-length features (EXP-006)."""

import numpy as np
import pytest
import torch

from graph_diffusion.data.circledataset import UnitCircleDataset


@pytest.fixture()
def dataset_rich(tmp_path: "pytest.fixture") -> UnitCircleDataset:
    return UnitCircleDataset(
        root=str(tmp_path),
        n_graphs=4,
        n_nodes=32,
        n_fourier_modes=3,
        amplitude_scale=0.10,
        r_min=0.5,
        r_max=1.5,
        k_neighbors=2,
        global_dim=8,
        include_curvature=True,
        include_arc_length=True,
        seed=42,
    )


@pytest.fixture()
def dataset_base(tmp_path: "pytest.fixture") -> UnitCircleDataset:
    return UnitCircleDataset(
        root=str(tmp_path / "base"),
        n_graphs=4,
        n_nodes=32,
        n_fourier_modes=3,
        amplitude_scale=0.10,
        r_min=0.5,
        r_max=1.5,
        k_neighbors=2,
        global_dim=8,
        seed=42,
    )


def test_rich_feature_shape(dataset_rich: UnitCircleDataset) -> None:
    data = dataset_rich[0]
    assert data.x.shape == (32, 3), f"Expected (32, 3), got {data.x.shape}"


def test_base_feature_shape(dataset_base: UnitCircleDataset) -> None:
    data = dataset_base[0]
    assert data.x.shape == (32, 1), f"Expected (32, 1), got {data.x.shape}"


def test_radius_column_same_in_both(
    dataset_rich: UnitCircleDataset,
    dataset_base: UnitCircleDataset,
) -> None:
    """The r column (col 0) must be identical whether or not extras are added."""
    r_rich = dataset_rich[0].x[:, 0]
    r_base = dataset_base[0].x[:, 0]
    assert torch.allclose(
        r_rich, r_base, atol=1e-6
    ), "Radius column differs between rich and base datasets"


def test_curvature_column_finite(dataset_rich: UnitCircleDataset) -> None:
    kappa = dataset_rich[0].x[:, 1]
    assert torch.isfinite(kappa).all(), "Curvature contains non-finite values"


def test_curvature_column_non_negative(dataset_rich: UnitCircleDataset) -> None:
    kappa = dataset_rich[0].x[:, 1]
    assert (kappa >= 0).all(), f"Curvature should be non-negative, min={kappa.min()}"


def test_arc_length_column_range(dataset_rich: UnitCircleDataset) -> None:
    """Normalised arc-length fraction should lie in [0, 1]."""
    s = dataset_rich[0].x[:, 2]
    assert (s >= 0.0).all() and (
        s <= 1.0
    ).all(), f"Arc-length fraction out of [0,1]: min={s.min():.4f}, max={s.max():.4f}"


def test_arc_length_starts_at_zero(dataset_rich: UnitCircleDataset) -> None:
    s = dataset_rich[0].x[:, 2]
    assert (
        abs(s[0].item()) < 1e-5
    ), f"Arc-length fraction at node 0 should be 0, got {s[0].item()}"


def test_arc_length_monotone(dataset_rich: UnitCircleDataset) -> None:
    s = dataset_rich[0].x[:, 2].numpy()
    diffs = np.diff(s)
    assert (
        diffs >= 0
    ).all(), "Arc-length fraction should be monotonically non-decreasing"


def test_include_curvature_only(tmp_path: "pytest.fixture") -> None:
    ds = UnitCircleDataset(
        root=str(tmp_path / "curv"),
        n_graphs=2,
        n_nodes=16,
        n_fourier_modes=2,
        amplitude_scale=0.1,
        include_curvature=True,
        include_arc_length=False,
        seed=0,
    )
    assert ds[0].x.shape == (16, 2)


def test_include_arc_length_only(tmp_path: "pytest.fixture") -> None:
    ds = UnitCircleDataset(
        root=str(tmp_path / "arc"),
        n_graphs=2,
        n_nodes=16,
        n_fourier_modes=2,
        amplitude_scale=0.1,
        include_curvature=False,
        include_arc_length=True,
        seed=0,
    )
    assert ds[0].x.shape == (16, 2)


def test_unit_circle_curvature_approx_one(tmp_path: "pytest.fixture") -> None:
    """A near-perfect circle (very small amplitude) should have κ ≈ 1 everywhere."""
    ds = UnitCircleDataset(
        root=str(tmp_path / "unit"),
        n_graphs=1,
        n_nodes=64,
        n_fourier_modes=1,
        amplitude_scale=0.001,  # nearly perfect circle
        include_curvature=True,
        seed=0,
    )
    kappa = ds[0].x[:, 1].numpy()
    assert np.allclose(
        kappa, 1.0, atol=0.05
    ), f"Near-circle curvature should be ~1.0, got mean={kappa.mean():.4f}"
