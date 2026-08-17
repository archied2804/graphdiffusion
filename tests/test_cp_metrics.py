"""Tests for graph_diffusion.postprocessing.cp_metrics.

The grading metric is the single point every reported number in Paper 1 passes
through, so the analytic invariants are asserted directly: exact recovery under
an affine relation, affine invariance of rho, and resampling invariance across
node counts.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from graph_diffusion.postprocessing.cp_metrics import (
    DATASET_P_STAR_TO_CP,
    cp_agreement,
    dead_probe_mask,
    polygon_validity,
    read_cp_csv,
    read_force_csv,
    read_reference_cp,
    split_branches,
)

torch.manual_seed(0)


def _ellipse_traversal(
    n: int = 96, thickness: float = 0.3, asymmetry: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesise an ordered boundary traversal with a plausible Cp curve.

    Args:
        n (int): Node count.
        thickness (float): Semi-minor axis.
        asymmetry (float): Upper/lower Cp offset; 0 gives a symmetric body.

    Returns:
        tuple: ``(xy, x_over_c, cp)`` in traversal order.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = 0.5 * np.cos(theta)
    y = thickness * np.sin(theta)
    x_over_c = (x - x.min()) / (x.max() - x.min())
    # A stagnation-to-suction-to-recovery shape, plus optional surface offset.
    cp = 1.0 - 4.0 * np.sin(np.pi * x_over_c) ** 2 + asymmetry * np.sign(y)
    return np.column_stack([x, y]), x_over_c, cp


# ---------------------------------------------------------------------------
# Analytic invariants
# ---------------------------------------------------------------------------


def test_exact_affine_relation_recovers_a_and_b() -> None:
    """Under Cp_meas = a*Cp_ref + b exactly, rho == 1 and (a, b) are recovered."""
    _, x, cp = _ellipse_traversal()
    a_true, b_true = 1.74, -0.3
    res = cp_agreement(x, cp, x, a_true * cp + b_true)
    assert res.rho == pytest.approx(1.0, abs=1e-9)
    assert res.a == pytest.approx(a_true, abs=1e-9)
    assert res.b == pytest.approx(b_true, abs=1e-9)
    assert res.affine_rms == pytest.approx(0.0, abs=1e-9)


def test_rho_is_invariant_to_positive_affine_rescaling() -> None:
    """rho must not move when the measured curve is affinely rescaled."""
    _, x, cp = _ellipse_traversal()
    rng = np.random.default_rng(0)
    meas = 1.5 * cp + 0.2 + 0.05 * rng.standard_normal(cp.size)
    base = cp_agreement(x, cp, x, meas).rho
    for alpha, beta in ((2.0, 0.0), (0.5, 3.0), (10.0, -7.0)):
        assert cp_agreement(x, cp, x, alpha * meas + beta).rho == pytest.approx(
            base, abs=1e-9
        )


def test_sign_flip_gives_negative_unit_correlation() -> None:
    _, x, cp = _ellipse_traversal()
    assert cp_agreement(x, cp, x, -cp).rho == pytest.approx(-1.0, abs=1e-9)


def test_resampling_invariance_across_node_counts() -> None:
    """The same body at 52, 96 and 100 nodes must grade identically.

    The corpus spans 52-96 nodes and the thick evaluation set reaches 100, so a
    metric that depended on node count would confound discretisation with
    performance.
    """
    results = []
    for n in (52, 96, 100):
        _, x, cp = _ellipse_traversal(n=n)
        results.append(cp_agreement(x, cp, x, 1.74 * cp - 0.3))
    for r in results[1:]:
        assert r.rho == pytest.approx(results[0].rho, abs=1e-6)
        assert r.a == pytest.approx(results[0].a, abs=1e-3)


def test_grades_curves_with_different_node_counts_against_each_other() -> None:
    """Target and measurement may have different node counts."""
    _, x_ref, cp_ref = _ellipse_traversal(n=60)
    _, x_meas, cp_meas = _ellipse_traversal(n=100)
    res = cp_agreement(x_ref, cp_ref, x_meas, 1.74 * cp_meas)
    assert res.rho > 0.999
    assert res.a == pytest.approx(1.74, abs=0.02)


# ---------------------------------------------------------------------------
# Surface splitting -- the lift-blindness issue
# ---------------------------------------------------------------------------


def test_split_branches_covers_every_node_once_per_arc() -> None:
    _, x, cp = _ellipse_traversal(n=40)
    br = split_branches(x, cp)
    assert br.x_a.size + br.x_b.size >= x.size
    assert np.all(np.diff(br.x_a) >= 0)
    assert np.all(np.diff(br.x_b) >= 0)


def test_symmetric_body_has_near_zero_asymmetry() -> None:
    _, x, cp = _ellipse_traversal(asymmetry=0.0)
    res = cp_agreement(x, cp, x, cp)
    assert res.asymmetry_ref < 1e-6


def test_collapsed_mode_hides_asymmetry_that_split_mode_sees() -> None:
    """A lifting body differs on each surface; collapsing averages that away.

    Measured on real AoA10 data: 11% of the Cp range mean, 37% max. This is why
    "split" is the default and the DCT conditioning, which is built on a
    collapsed curve, is structurally blind to lift.
    """
    _, x, cp_sym = _ellipse_traversal(asymmetry=0.0)
    _, _, cp_asym = _ellipse_traversal(asymmetry=0.4)

    split = cp_agreement(x, cp_sym, x, cp_asym, mode="split")
    collapsed = cp_agreement(x, cp_sym, x, cp_asym, mode="collapsed")

    assert split.asymmetry_meas > 0.5
    # Split mode registers the mismatch; collapsed mode largely cannot.
    assert split.affine_rms > collapsed.affine_rms


# ---------------------------------------------------------------------------
# Residual structure
# ---------------------------------------------------------------------------


def test_runs_test_flags_structured_residual() -> None:
    """A quadratic misfit gives a structured residual; noise does not."""
    _, x, cp = _ellipse_traversal()
    rng = np.random.default_rng(1)

    noisy = 1.74 * cp - 0.3 + 0.02 * rng.standard_normal(cp.size)
    structured = 1.74 * cp - 0.3 + 0.3 * (x - 0.5) ** 2

    z_noise = abs(cp_agreement(x, cp, x, noisy).residual_runs_z)
    z_struct = abs(cp_agreement(x, cp, x, structured).residual_runs_z)
    assert z_struct > z_noise


def test_degenerate_constant_reference_does_not_crash() -> None:
    _, x, cp = _ellipse_traversal()
    res = cp_agreement(x, np.ones_like(cp), x, cp)
    assert math.isnan(res.rho) or abs(res.rho) <= 1.0


# ---------------------------------------------------------------------------
# Validation of inputs
# ---------------------------------------------------------------------------


def test_rejects_unknown_mode_and_bad_grid() -> None:
    _, x, cp = _ellipse_traversal()
    with pytest.raises(ValueError, match="mode must be"):
        cp_agreement(x, cp, x, cp, mode="nope")
    with pytest.raises(ValueError, match="n_grid must be"):
        cp_agreement(x, cp, x, cp, n_grid=2)


def test_rejects_mismatched_array_lengths() -> None:
    _, x, cp = _ellipse_traversal()
    with pytest.raises(ValueError, match="same length"):
        cp_agreement(x[:-1], cp, x, cp)


def test_split_branches_rejects_tiny_input() -> None:
    with pytest.raises(ValueError, match=">= 4"):
        split_branches(np.array([0.0, 1.0]), np.array([0.0, 1.0]))


# ---------------------------------------------------------------------------
# Geometry validity
# ---------------------------------------------------------------------------


def test_valid_ellipse_passes() -> None:
    xy, _, _ = _ellipse_traversal()
    v = polygon_validity(xy)
    assert v.is_valid and v.is_simple
    assert v.n_self_intersections == 0
    assert v.min_spacing_ratio > 0.5


def test_self_intersecting_polygon_is_rejected() -> None:
    """A bow-tie is closed but not simple."""
    xy = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    v = polygon_validity(xy)
    assert not v.is_simple
    assert not v.is_valid
    assert v.n_self_intersections >= 1


def test_collapsed_node_spacing_is_rejected() -> None:
    xy, _, _ = _ellipse_traversal(n=40)
    xy[5] = xy[4] + 1e-9  # two coincident nodes
    v = polygon_validity(xy)
    assert v.min_spacing_ratio < 0.1
    assert not v.is_valid


def test_polygon_validity_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        polygon_validity(np.zeros((2, 2)))


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def test_read_cp_and_force_csv(tmp_path: Path) -> None:
    cp_csv = tmp_path / "ellipse2d_cp.csv"
    cp_csv.write_text("x_over_c,Cp,Cp_rms\n1.0,-0.9,0.7\n0.5,-1.2,0.6\n0.0,1.0,0.1\n")
    x, cp, rms = read_cp_csv(cp_csv)
    assert x.size == 3
    assert cp[2] == pytest.approx(1.0)
    assert rms[0] == pytest.approx(0.7)

    f_csv = tmp_path / "ellipse2d_force.csv"
    f_csv.write_text(
        "Cd_mean,Cd_std,Cl_mean,Cl_std,n_samples,t_start,t_end\n"
        "1.105,1.476,0.00034,0.00739,120,1.983,3.983\n"
    )
    fc = read_force_csv(f_csv)
    assert fc.cd == pytest.approx(1.105)
    assert fc.cl == pytest.approx(0.00034)
    assert fc.n_samples == 120


def test_read_force_csv_rejects_empty(tmp_path: Path) -> None:
    p = tmp_path / "empty.csv"
    p.write_text("Cd_mean,Cd_std,Cl_mean,Cl_std,n_samples,t_start,t_end\n")
    with pytest.raises(ValueError, match="no data row"):
        read_force_csv(p)


# ---------------------------------------------------------------------------
# Dataset p* -> Cp conversion
# ---------------------------------------------------------------------------


def test_dataset_scale_is_two() -> None:
    """The dataset stores p/(rho U^2); Cp uses 0.5 rho U^2, so Cp = 2 p*."""
    assert DATASET_P_STAR_TO_CP == 2.0


def test_read_reference_cp_applies_the_factor(tmp_path) -> None:
    p = tmp_path / "ref.csv"
    p.write_text("x_over_c,cp_ref\n0.0,0.5\n0.5,-0.25\n1.0,0.1\n")
    x, cp = read_reference_cp(p)
    assert np.allclose(cp, [1.0, -0.5, 0.2])
    x_raw, cp_raw, _ = read_cp_csv(p)
    assert np.allclose(cp, cp_raw * 2.0)


def test_read_reference_cp_scale_one_reproduces_legacy(tmp_path) -> None:
    """scale=1.0 must reproduce a pre-2026-08-07 comparison exactly."""
    p = tmp_path / "ref.csv"
    p.write_text("x_over_c,cp_ref\n0.0,0.5\n1.0,0.1\n")
    _, cp = read_reference_cp(p, scale=1.0)
    _, cp_raw, _ = read_cp_csv(p)
    assert np.allclose(cp, cp_raw)


def test_factor_of_two_removes_the_historical_slope() -> None:
    """A curve differing only by the units convention must grade at a = 1.

    This is the regression guarding the Year-1 finding: grading solver Cp
    against unscaled dataset p* yields a ~ 2, and against scaled p* yields
    a ~ 1.
    """
    torch.manual_seed(0)
    x = np.linspace(0.0, 1.0, 96)
    p_star = 0.5 - 1.5 * np.sin(np.pi * x) ** 2  # dataset units
    cp_solver = 2.0 * p_star  # same curve, Cp units

    unscaled = cp_agreement(x, p_star, x, cp_solver, mode="collapsed")
    scaled = cp_agreement(
        x, p_star * DATASET_P_STAR_TO_CP, x, cp_solver, mode="collapsed"
    )
    assert unscaled.a == pytest.approx(2.0, abs=1e-6)
    assert scaled.a == pytest.approx(1.0, abs=1e-6)
    assert unscaled.rho == pytest.approx(scaled.rho, abs=1e-12)


def test_dead_probe_mask_flags_solid_cell_samples() -> None:
    """Exactly-zero time-RMS and non-finite Cp are the two failure signatures."""
    cp = np.array([1.0, 0.06, -0.5, math.nan, 0.06])
    rms = np.array([0.03, 0.0, 0.05, 0.02, 0.0])
    mask = dead_probe_mask(cp, rms)
    assert mask.dtype == bool
    assert mask.tolist() == [False, True, False, True, True]


def test_dead_probe_mask_keeps_reference_curves_without_rms() -> None:
    """A dataset reference has no RMS column and must not be filtered away."""
    cp = np.array([1.0, 0.5, -0.5])
    rms = np.full(3, math.nan)
    assert not dead_probe_mask(cp, rms).any()


def test_read_cp_csv_drops_dead_probes_by_default(tmp_path: Path) -> None:
    p = tmp_path / "ellipse2d_cp.csv"
    p.write_text(
        "x_over_c,Cp,Cp_rms\n"
        "0.0,1.0,0.03\n"
        "0.25,0.06,0.0\n"  # probe landed in the body
        "0.5,-0.5,0.05\n"
    )
    x, cp, rms = read_cp_csv(p)
    assert x.size == 2
    assert np.allclose(x, [0.0, 0.5])
    assert np.allclose(cp, [1.0, -0.5])
    assert np.allclose(rms, [0.03, 0.05])

    x_raw, cp_raw, _ = read_cp_csv(p, drop_dead=False)
    assert x_raw.size == 3
    assert np.allclose(cp_raw, [1.0, 0.06, -0.5])
