"""
Tests for the ``fourier_dual`` cond mode on pOnEllipseConditionalDataset.

The dual mode splits the boundary into upper (``y - y.mean() >= 0``) and
lower halves, interpolates each half's Cp onto a shared uniform x/c
grid, then DCT-II truncates each to K modes. The output cond vector is
``(1, 2K)`` with upper modes first.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
from context import graph_diffusion  # noqa: F401

from graph_diffusion.data.pOnEllipse import _H5_N_TIMESTEPS
from graph_diffusion.data.pOnEllipseConditional import (
    N_CP_GRID_DEFAULT,
    pOnEllipseConditionalDataset,
)
from graph_diffusion.data.transforms import ComputeAngularEdgeFeatures


def _fake_asymmetric_nodes(
    n_nodes: int,
    upper_cp_amp: float,
    lower_cp_amp: float,
    re: float = 500.0,
    aoa: float = 3.0,
) -> np.ndarray:
    """Chord-parallel ellipse with deliberately asymmetric per-node Cp.

    Upper-surface nodes get a Cp(x/c) of amplitude ``upper_cp_amp`` and
    lower-surface nodes get ``lower_cp_amp`` — so when the two halves
    DCT-truncated independently the cond vectors must differ.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n_nodes, endpoint=False)
    a, b = 1.0, 0.5
    x = a * np.cos(theta) + 2.0  # chord-parallel x in [1, 3]
    y = b * np.sin(theta) + 1.0
    x_c = (x - x.min()) / (x.max() - x.min())  # x/c in [0, 1]
    # Upper = y - mean > 0, lower otherwise. Build a synthetic Cp:
    #   upper: amp * sin(pi * x/c)  (a positive bump)
    #   lower: amp * sin(pi * x/c)  (also a bump, but with different amplitude)
    is_upper = (y - y.mean()) >= 0
    cp_per_node = np.where(
        is_upper,
        upper_cp_amp * np.sin(np.pi * x_c),
        lower_cp_amp * np.sin(np.pi * x_c),
    ).astype(np.float32)
    # Broadcast Cp into 101 unsteady timesteps so the mean equals cp_per_node.
    p = np.tile(cp_per_node[:, None], (1, _H5_N_TIMESTEPS))
    re_col = np.full(n_nodes, re, dtype=np.float32)
    aoa_col = np.full(n_nodes, aoa, dtype=np.float32)
    return np.concatenate(
        [
            x.astype(np.float32)[:, None],
            y.astype(np.float32)[:, None],
            re_col[:, None],
            aoa_col[:, None],
            p.astype(np.float32),
        ],
        axis=1,
    )


def _write_fake_h5_asymmetric(path: Path) -> None:
    n_sims, max_nodes = 3, 16
    n_feat = 4 + _H5_N_TIMESTEPS
    data_arr = np.full((n_sims, max_nodes, n_feat), np.nan, dtype=np.float32)
    # Three sims with different upper/lower asymmetry.
    for i, (upper, lower) in enumerate([(1.0, 0.2), (0.5, 0.5), (-0.3, 0.7)]):
        data_arr[i, :max_nodes, :] = _fake_asymmetric_nodes(
            max_nodes, upper_cp_amp=upper, lower_cp_amp=lower
        )
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=data_arr)


def _make_root(tmpdir: str) -> Path:
    root = Path(tmpdir) / "dataset"
    raw = root / "raw"
    raw.mkdir(parents=True)
    _write_fake_h5_asymmetric(raw / "pOnEllipseTrain.h5")
    return root


def test_fourier_dual_cond_shape_is_2k() -> None:  # noqa: N802
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_root(tmpdir)
        ds = pOnEllipseConditionalDataset(
            root=str(root),
            cond_mode="fourier_dual",
            k_modes=8,
            feature_mode="radial",
            split="train",
            k_neighbors=2,
            global_dim=4,
            pre_transform=ComputeAngularEdgeFeatures(),
        )
        assert len(ds) == 3
        assert ds[0].cond.shape == (1, 16)


def test_fourier_dual_upper_modes_differ_from_lower_when_cp_asymmetric() -> None:
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_root(tmpdir)
        ds = pOnEllipseConditionalDataset(
            root=str(root),
            cond_mode="fourier_dual",
            k_modes=8,
            feature_mode="radial",
            split="train",
            k_neighbors=2,
            global_dim=4,
            pre_transform=ComputeAngularEdgeFeatures(),
        )
        # Sim 0: upper=1.0, lower=0.2 — strongly asymmetric.
        cond0 = ds[0].cond.squeeze(0).numpy()
        upper0, lower0 = cond0[:8], cond0[8:]
        # Sim 1: upper=lower=0.5 — symmetric.
        cond1 = ds[1].cond.squeeze(0).numpy()
        upper1, lower1 = cond1[:8], cond1[8:]

        asym_gap = float(np.abs(upper0 - lower0).max())
        sym_gap = float(np.abs(upper1 - lower1).max())
        # Asymmetric gap should be much bigger than the residual interpolation
        # noise gap in the symmetric case.
        assert asym_gap > 0.5
        assert sym_gap < 0.1
        assert asym_gap > 10.0 * sym_gap


def test_fourier_mode_regression_still_k() -> None:  # noqa: N802
    """Existing fourier (single-curve) mode must still produce (1, K)."""
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_root(tmpdir)
        ds = pOnEllipseConditionalDataset(
            root=str(root),
            cond_mode="fourier",
            k_modes=8,
            feature_mode="radial",
            split="train",
            k_neighbors=2,
            global_dim=4,
            pre_transform=ComputeAngularEdgeFeatures(),
        )
        assert ds[0].cond.shape == (1, 8)


def test_n_cp_grid_default_is_reasonable() -> None:  # noqa: N802
    # 64 is a sensible interp grid (Nyquist comfortably covers K=8 modes).
    assert N_CP_GRID_DEFAULT >= 16
    assert N_CP_GRID_DEFAULT <= 256


def test_conditional_cache_name_includes_variant() -> None:
    # Regression: the default and aoa10 variants share every cache-name
    # field except `variant`. If the variant is dropped, the two caches
    # collide and the wrong data is silently loaded (happened once).
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_root(tmpdir)
        ds_default = pOnEllipseConditionalDataset(
            root=str(root),
            cond_mode="fourier_dual",
            k_modes=8,
            feature_mode="radial",
            split="train",
            k_neighbors=2,
            global_dim=4,
            variant="default",
            pre_transform=ComputeAngularEdgeFeatures(),
        )
        default_name = ds_default.processed_file_names[0]
        # The aoa10 cache name must differ (we don't instantiate it — that
        # would need the aoa10 file — just check the naming contract).
        ds_default.variant = "aoa10"
        aoa10_name = ds_default.processed_file_names[0]
        assert default_name != aoa10_name
        assert "aoa10" in aoa10_name
