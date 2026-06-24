"""Tests for graph_diffusion.data.cp_manifold."""

import numpy as np
from context import graph_diffusion  # noqa: F401

from graph_diffusion.data.cp_manifold import (
    CpPcaBasis,
    cp_pca_basis,
    dataset_cp_curves,
    dataset_cp_pca,
    pc_ladder_targets,
)


def _fake_raw(n_sims: int = 20, max_nodes: int = 16) -> np.ndarray:
    raw = np.full((n_sims, max_nodes, 105), np.nan, dtype=np.float32)
    grid = np.linspace(0.0, 1.0, 12)
    for s in range(n_sims):
        raw[s, :12, 0] = grid - 0.5  # x in [-0.5, 0.5]
        raw[s, :12, 1] = 0.0
        raw[s, :12, 2] = 700.0
        raw[s, :12, 3] = 5.5
        # Cp curve = base + scaled mode, so PCA has a clear leading direction.
        base = np.cos(np.pi * grid)
        raw[s, :12, 4:105] = (base + 0.3 * s / n_sims * grid)[:, None]
    return raw


def test_dataset_cp_curves_shape():
    raw = _fake_raw()
    curves = dataset_cp_curves(raw, n_grid=64)
    assert curves.shape == (20, 64)
    assert np.isfinite(curves).all()


def test_cp_pca_basis_orthonormal_components():
    raw = _fake_raw()
    curves = dataset_cp_curves(raw, n_grid=64)
    basis = cp_pca_basis(curves, n_components=2)
    assert isinstance(basis, CpPcaBasis)
    assert basis.mean.shape == (64,)
    assert basis.components.shape == (2, 64)
    assert basis.proj_std.shape == (2,)
    # Components orthonormal.
    gram = basis.components @ basis.components.T
    assert np.allclose(gram, np.eye(2), atol=1e-5)
    # Projection of the mean curve is ~0.
    assert np.allclose(basis.project(basis.mean), 0.0, atol=1e-5)


def test_dataset_cp_pca_grid_matches():
    raw = _fake_raw()
    grid, basis = dataset_cp_pca(raw, n_grid=32, n_components=2)
    assert grid.shape == (32,)
    assert basis.components.shape == (2, 32)


def test_pc_ladder_targets_count_and_names():
    raw = _fake_raw()
    grid, basis = dataset_cp_pca(raw, n_grid=64, n_components=2)
    targets = pc_ladder_targets(basis, amplitudes=(1.0, 2.0, 3.0, 4.0), n_modes=2)
    assert len(targets) == 2 * 4 * 2  # modes x amps x signs = 16
    names = [t["name"] for t in targets]
    assert len(set(names)) == 16  # unique
    assert "pc1_p1sig" in names and "pc2_m4sig" in names
    for t in targets:
        assert t["cp_target"].shape == (64,)


def test_pc_ladder_amplitude_increases_deviation():
    raw = _fake_raw()
    _, basis = dataset_cp_pca(raw, n_grid=64, n_components=2)
    targets = pc_ladder_targets(basis, amplitudes=(1.0, 4.0), n_modes=1)
    by_name = {t["name"]: t for t in targets}
    dev1 = np.linalg.norm(by_name["pc1_p1sig"]["cp_target"] - basis.mean)
    dev4 = np.linalg.norm(by_name["pc1_p4sig"]["cp_target"] - basis.mean)
    assert dev4 > dev1  # larger sigma-amplitude => farther from the mean
