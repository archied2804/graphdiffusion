"""Tests for graph_diffusion.visualisation.sweep_results."""

import matplotlib

matplotlib.use("Agg")  # headless backend, set before any pyplot import

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from context import graph_diffusion  # noqa: E402, F401

from graph_diffusion.visualisation.sweep_results import cp_match_metrics  # noqa: E402


def test_identical_curves_perfect_match():
    x = np.linspace(0.0, 1.0, 50)
    cp = np.cos(np.pi * x)
    m = cp_match_metrics(x, cp, x, cp)
    assert m["corr"] == pytest.approx(1.0, abs=1e-6)
    assert m["affine_a"] == pytest.approx(1.0, abs=1e-6)
    assert m["rms"] == pytest.approx(0.0, abs=1e-6)


def test_affine_slope_recovered_on_nonmonotonic_measured():
    # Target monotonic in x; measured nodes trace a non-monotonic path but
    # carry cp = 2*target. The interp-onto-measured direction must recover a=2.
    xt = np.linspace(0.0, 1.0, 64)
    cpt = np.cos(np.pi * xt)
    xm = np.r_[np.linspace(0.0, 1.0, 32), np.linspace(1.0, 0.0, 32)]  # non-monotonic
    cpm = 2.0 * np.cos(np.pi * xm)
    m = cp_match_metrics(xt, cpt, xm, cpm)
    assert m["corr"] == pytest.approx(1.0, abs=1e-3)
    assert m["affine_a"] == pytest.approx(2.0, rel=1e-2)


import matplotlib.figure  # noqa: E402

from graph_diffusion.visualisation.sweep_results import (  # noqa: E402
    SampleRecord,
    plot_sample_gallery,
)


def _fake_samples(n: int = 4) -> list:
    rng = np.random.default_rng(0)
    out = []
    xt = np.linspace(0.0, 1.0, 64)
    th = np.linspace(0.0, 2 * np.pi, 40, endpoint=False)
    for i in range(n):
        poly = np.stack([0.5 * np.cos(th), (0.25 + 0.02 * i) * np.sin(th)], axis=1)
        cpt = np.cos(np.pi * xt) + 0.1 * i
        out.append(
            SampleRecord(
                name=f"pc1_p{i + 1}sig",
                re=754.0,
                h=5.5,
                ood_dist=float(i),
                polygon=poly,
                target_x=xt,
                target_cp=cpt,
                meas_x=xt,
                meas_cp=cpt + 0.05 * rng.standard_normal(64),
            )
        )
    return out


def test_sample_gallery_smoke():
    fig = plot_sample_gallery(_fake_samples(4), ncols=2)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) >= 4
    matplotlib.pyplot.close("all")
