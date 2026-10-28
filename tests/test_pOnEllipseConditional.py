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
    quadrature_weights,
    uniform_positions,
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


def test_dct_ii_ortho_dc_mode_is_mean_proportional() -> None:
    sig = np.ones(32, dtype=np.float32) * 0.5
    out = dct_ii(sig, k_modes=8, norm="ortho")
    # Legacy k=0 coefficient is sqrt(1/N) * sum = sqrt(N)*mean — the sqrt(N)
    # dependence that P1-D01 identified as the source of the scaling.
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


def _smooth_signal(s: np.ndarray) -> np.ndarray:
    """A smooth test curve, evaluated at arbitrary positions."""
    sig = 1.0 - 4.0 * np.sin(np.pi * s) ** 2 + 0.3 * np.cos(6 * np.pi * s)
    return sig.astype(np.float32)


# --- quadrature weights ----------------------------------------------------


def test_quadrature_weights_sum_to_one() -> None:
    for n in (7, 48, 96):
        w = quadrature_weights(uniform_positions(n))
        assert abs(float(w.sum()) - 1.0) < 1e-12


def test_quadrature_weights_reduce_to_uniform() -> None:
    """For DCT-II's own midpoints the weights must be exactly ``1/N``."""
    n = 64
    w = quadrature_weights(uniform_positions(n))
    assert np.max(np.abs(w - 1.0 / n)) < 1e-12


def test_quadrature_weights_reject_unsorted() -> None:
    with pytest.raises(ValueError, match="sorted ascending"):
        quadrature_weights(np.array([0.1, 0.9, 0.4]))


def test_quadrature_weights_reject_out_of_range() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        quadrature_weights(np.array([0.1, 0.5, 1.7]))


def test_quadrature_weights_handle_duplicate_positions() -> None:
    """Upper and lower surfaces share chord stations after collapsing."""
    s = np.array([0.0, 0.25, 0.25, 0.5, 0.75, 0.75, 1.0])
    w = quadrature_weights(s)
    assert abs(float(w.sum()) - 1.0) < 1e-12
    assert np.all(w >= 0.0)


# --- forward convention: the invariance properties -------------------------


def test_dct_ii_recovers_analytic_coefficients() -> None:
    """``f(s) = cos(3*pi*s)`` must give ``c_3 = 1`` and zero elsewhere."""
    s = uniform_positions(256)
    out = dct_ii(np.cos(3.0 * np.pi * s).astype(np.float32), k_modes=8)
    expected = np.zeros(8, dtype=np.float32)
    expected[3] = 1.0
    assert np.allclose(out, expected, atol=1e-4)


def test_dct_ii_dc_mode_is_the_mean() -> None:
    """Under the forward convention ``c_0`` is the plain mean, free of N."""
    for n in (32, 97, 200):
        out = dct_ii(np.full(n, 0.5, dtype=np.float32), k_modes=8)
        assert abs(float(out[0]) - 0.5) < 1e-5
        assert np.allclose(out[1:], 0.0, atol=1e-5)


def test_dct_ii_invariant_to_sample_count() -> None:
    """P1-D01's falsified hypothesis, now satisfied by construction."""
    torch.manual_seed(0)
    ref = dct_ii(_smooth_signal(uniform_positions(96)), k_modes=8)
    for n in (48, 64, 80, 128, 192, 384):
        got = dct_ii(_smooth_signal(uniform_positions(n)), k_modes=8)
        rel = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
        assert rel < 0.01, f"N={n}: relative deviation {rel}"


def test_dct_ii_invariant_to_node_spacing() -> None:
    """The same curve sampled on very different grids gives the same modes."""
    torch.manual_seed(0)
    uni = uniform_positions(96)
    ref = dct_ii(_smooth_signal(uni), k_modes=8, positions=uni)
    grids = {
        "le_te_clustered": 0.5 * (1.0 - np.cos(np.pi * uni)),
        "mid_clustered": np.clip(uni + 0.12 * np.sin(2.0 * np.pi * uni), 0.0, 1.0),
    }
    for name, s in grids.items():
        s = np.sort(s)
        got = dct_ii(_smooth_signal(s), k_modes=8, positions=s)
        rel = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
        assert rel < 0.02, f"{name}: relative deviation {rel}"


def test_dct_ii_ignoring_positions_is_materially_wrong() -> None:
    """Guards the reason `positions` exists.

    Passing clustered samples without their positions treats them as evenly
    spaced, over-weighting wherever nodes bunch up. That error is large, so the
    dataset must keep forwarding positions.
    """
    uni = uniform_positions(96)
    ref = dct_ii(_smooth_signal(uni), k_modes=8, positions=uni)
    clustered = 0.5 * (1.0 - np.cos(np.pi * uni))
    naive = dct_ii(_smooth_signal(clustered), k_modes=8)
    rel = float(np.linalg.norm(naive - ref) / np.linalg.norm(ref))
    assert rel > 0.1, f"expected a large error, got {rel}"


# --- legacy convention -----------------------------------------------------


def test_dct_ii_ortho_still_scales_as_sqrt_n() -> None:
    """The legacy path must keep its exact old behaviour for reproduction.

    ``norm="ortho"`` reproduces checkpoints trained before 2026-08-05, so its
    sqrt(N) scaling is a property to preserve, not a bug to fix here.
    """
    torch.manual_seed(0)
    ref = dct_ii(_smooth_signal(uniform_positions(96)), k_modes=8, norm="ortho")
    for n in (48, 64, 128, 192):
        got = dct_ii(_smooth_signal(uniform_positions(n)), k_modes=8, norm="ortho")
        ratio = float(np.linalg.norm(got) / np.linalg.norm(ref))
        assert abs(ratio - np.sqrt(n / 96)) < 0.01, f"N={n}: ratio {ratio}"


def test_dct_ii_ortho_ignores_positions() -> None:
    """The legacy basis has no notion of sample position."""
    s = np.sort(np.linspace(0.0, 1.0, 64) ** 2)
    a = dct_ii(_smooth_signal(s), k_modes=8, norm="ortho")
    b = dct_ii(_smooth_signal(s), k_modes=8, positions=s, norm="ortho")
    assert np.allclose(a, b)


def test_dct_ii_rejects_unknown_norm() -> None:
    with pytest.raises(ValueError, match="norm must be one of"):
        dct_ii(np.ones(16, dtype=np.float32), k_modes=4, norm="unitary")


# ---------------------------------------------------------------------------
# dct_ii_inverse
# ---------------------------------------------------------------------------


def test_dct_ii_inverse_output_shape() -> None:
    coeffs = np.zeros(8, dtype=np.float32)
    out = dct_ii_inverse(coeffs, n_samples=64)
    assert out.shape == (64,)
    assert out.dtype == np.float32


def test_dct_ii_inverse_constant_recovery() -> None:
    # Under the forward convention c_0 is the mean itself, so a DC-only
    # coefficient vector reconstructs a constant signal of that value.
    mean_val = 0.5
    coeffs = np.zeros(8, dtype=np.float32)
    coeffs[0] = mean_val
    out = dct_ii_inverse(coeffs, n_samples=32)
    assert np.allclose(out, mean_val, atol=1e-5)


def test_dct_ii_inverse_ortho_constant_recovery() -> None:
    n = 32
    mean_val = 0.5
    coeffs = np.zeros(8, dtype=np.float32)
    coeffs[0] = np.sqrt(n) * mean_val
    out = dct_ii_inverse(coeffs, n_samples=n, norm="ortho")
    assert np.allclose(out, mean_val, atol=1e-5)


def test_dct_ii_inverse_at_arbitrary_positions() -> None:
    """Coefficients describe a continuous curve, so any grid can be sampled.

    This is what makes transferring a descriptor between discretisations well
    posed: encode at one node count, decode at another.
    """
    torch.manual_seed(0)
    s_in = uniform_positions(128)
    coeffs = dct_ii(_smooth_signal(s_in), k_modes=16, positions=s_in)

    rng = np.random.default_rng(1)
    s_out = np.sort(rng.uniform(0.0, 1.0, 53))
    rec = dct_ii_inverse(coeffs, positions=s_out)
    assert rec.shape == (53,)
    assert np.max(np.abs(rec - _smooth_signal(s_out))) < 1e-3


def test_dct_ii_inverse_requires_a_grid() -> None:
    with pytest.raises(ValueError, match="n_samples or positions"):
        dct_ii_inverse(np.zeros(4, dtype=np.float32))


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
