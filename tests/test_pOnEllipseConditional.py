"""
Tests for graph_diffusion.data.pOnEllipseConditional
=====================================================

Unit tests for the DCT encoding helper and pOnEllipseConditionalDataset.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from context import graph_diffusion  # noqa: F401

from graph_diffusion.data.pOnEllipse import _H5_N_TIMESTEPS
from graph_diffusion.data.pOnEllipseConditional import (
    dct_ii,
    dct_ii_inverse,
    pOnEllipseConditionalDataset,
)
from graph_diffusion.data.transforms import ComputeAngularEdgeFeatures

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


def _make_dataset_root(tmpdir: str) -> Path:
    root = Path(tmpdir) / "dataset"
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True)
    _write_fake_h5(raw_dir / "pOnEllipseTrain.h5")
    return root


# ---------------------------------------------------------------------------
# dct_ii
# ---------------------------------------------------------------------------


def test_dct_ii_output_shape() -> None:
    sig = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    out = dct_ii(sig, k_modes=8)
    assert out.shape == (8,)
    assert out.dtype == np.float32


def test_dct_ii_dc_mode_is_mean_proportional() -> None:
    sig = np.ones(32, dtype=np.float32) * 0.5
    out = dct_ii(sig, k_modes=8)
    # k=0 coefficient under our normalisation is sqrt(1/N) * sum = sqrt(N)*mean.
    expected = np.sqrt(32) * 0.5
    assert abs(out[0] - expected) < 1e-5
    # Higher modes should be zero for a constant signal.
    assert np.allclose(out[1:], 0.0, atol=1e-5)


def test_dct_ii_basis_orthogonality() -> None:
    # Pure cosine k=2 should give a single non-zero coefficient at index 2.
    n = 64
    k_target = 2
    n_idx = np.arange(n, dtype=np.float32)
    sig = np.cos(np.pi * (2.0 * n_idx + 1.0) * k_target / (2.0 * n)).astype(np.float32)
    out = dct_ii(sig, k_modes=8)
    assert abs(out[k_target]) > 0.5
    mask = np.ones(8, dtype=bool)
    mask[k_target] = False
    assert np.max(np.abs(out[mask])) < 1e-4


# ---------------------------------------------------------------------------
# dct_ii_inverse
# ---------------------------------------------------------------------------


def test_dct_ii_inverse_output_shape() -> None:
    coeffs = np.zeros(8, dtype=np.float32)
    out = dct_ii_inverse(coeffs, n_samples=64)
    assert out.shape == (64,)
    assert out.dtype == np.float32


def test_dct_ii_inverse_constant_recovery() -> None:
    # k=0 coefficient = sqrt(N)*mean for our normalisation; inverse with only
    # the DC mode should reproduce a constant signal of that mean.
    n = 32
    mean_val = 0.5
    coeffs = np.zeros(8, dtype=np.float32)
    coeffs[0] = np.sqrt(n) * mean_val
    out = dct_ii_inverse(coeffs, n_samples=n)
    assert np.allclose(out, mean_val, atol=1e-5)


def test_dct_ii_round_trip_on_band_limited_signal() -> None:
    # Signal that is exactly representable in K=8 DCT modes should round-trip.
    n = 64
    k = 8
    coeffs_in = np.array([1.0, 0.3, -0.4, 0.05, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    signal = dct_ii_inverse(coeffs_in, n_samples=n)
    coeffs_out = dct_ii(signal, k_modes=k)
    assert np.allclose(coeffs_in, coeffs_out, atol=1e-5)


# ---------------------------------------------------------------------------
# pOnEllipseConditionalDataset — validation
# ---------------------------------------------------------------------------


def test_invalid_cond_mode() -> None:
    with pytest.raises(ValueError, match="cond_mode"):
        pOnEllipseConditionalDataset(root="/tmp/dummy", cond_mode="spectral")


def test_invalid_k_modes() -> None:
    with pytest.raises(ValueError, match="k_modes"):
        pOnEllipseConditionalDataset(root="/tmp/dummy", k_modes=0)


# ---------------------------------------------------------------------------
# pOnEllipseConditionalDataset — attached conditioning
# ---------------------------------------------------------------------------


def test_fourier_mode_attaches_cond() -> None:
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_dataset_root(tmpdir)
        ds = pOnEllipseConditionalDataset(
            root=str(root),
            cond_mode="fourier",
            k_modes=8,
            feature_mode="radial_norm",
            n_samples=5,
            k_neighbors=2,
            global_dim=8,
            pre_transform=ComputeAngularEdgeFeatures(),
        )
        graph = ds[0]
        assert hasattr(graph, "cond")
        assert graph.cond.shape == (1, 8)
        # Parent class' p_cond (angular positional) should still be present.
        assert hasattr(graph, "p_cond")
        assert graph.p_cond.shape[1] == 2
        # Fourier mode does NOT attach cp_nodal
        assert not hasattr(graph, "cp_nodal") or graph.cp_nodal is None


def test_nodal_mode_attaches_cp_nodal() -> None:
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_dataset_root(tmpdir)
        ds = pOnEllipseConditionalDataset(
            root=str(root),
            cond_mode="nodal",
            k_modes=8,
            feature_mode="radial_norm",
            n_samples=5,
            k_neighbors=2,
            global_dim=8,
            pre_transform=ComputeAngularEdgeFeatures(),
        )
        graph = ds[0]
        assert hasattr(graph, "cond")
        assert graph.cond.shape == (1, 8)
        assert hasattr(graph, "cp_nodal")
        assert graph.cp_nodal.shape[1] == 1
        # One Cp value per surface node
        assert graph.cp_nodal.shape[0] == graph.x.shape[0]


def test_cond_differs_across_graphs() -> None:
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_dataset_root(tmpdir)
        ds = pOnEllipseConditionalDataset(
            root=str(root),
            cond_mode="fourier",
            k_modes=8,
            feature_mode="radial_norm",
            n_samples=5,
            k_neighbors=2,
            global_dim=8,
            pre_transform=ComputeAngularEdgeFeatures(),
        )
        cond0 = ds[0].cond
        cond1 = ds[1].cond
        # Different simulations should produce different DCT encodings
        assert not torch.allclose(cond0, cond1)
