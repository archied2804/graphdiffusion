"""Tests for graph_diffusion.data.design_space."""

import numpy as np
import pytest
from context import graph_diffusion  # noqa: F401

from graph_diffusion.data.design_space import make_case_name, sample_design_space


def test_make_case_name_format():
    assert make_case_name(500.0, 5.3, 0) == "re0500_h53_idx0000"
    assert make_case_name(1000.0, 6.0, 4849) == "re1000_h60_idx4849"


def test_make_case_name_sorts_by_re_then_h():
    names = [
        make_case_name(1000.0, 5.0, 1),
        make_case_name(500.0, 6.0, 2),
        make_case_name(500.0, 5.0, 3),
    ]
    assert sorted(names)[0].startswith("re0500_h50")


def _fake_raw_grid() -> np.ndarray:
    # 36 sims spanning Re in [500,1000], H in [5,6].
    res = np.repeat(np.linspace(500, 1000, 6), 6)
    hs = np.tile(np.linspace(5.0, 6.0, 6), 6)
    raw = np.full((36, 8, 105), np.nan, dtype=np.float32)
    for s in range(36):
        raw[s, :6, 0] = np.linspace(-0.5, 0.5, 6)
        raw[s, :6, 1] = np.linspace(-0.2, 0.2, 6)
        raw[s, :6, 2] = res[s]
        raw[s, :6, 3] = hs[s]
    return raw


def test_sample_returns_unique_in_range_indices():
    raw = _fake_raw_grid()
    idxs = sample_design_space(raw, n_re=3, n_h=2)
    assert len(idxs) == len(set(idxs))  # unique
    res = raw[idxs, 0, 2]
    hs = raw[idxs, 0, 3]
    assert res.min() >= 500.0 and res.max() <= 1000.0
    assert hs.min() >= 5.0 and hs.max() <= 6.0


def test_sample_includes_corners():
    raw = _fake_raw_grid()
    idxs = sample_design_space(raw, n_re=3, n_h=3)
    res = raw[idxs, 0, 2]
    hs = raw[idxs, 0, 3]
    pairs = set(
        zip(np.round(res).astype(int), np.round(hs * 10).astype(int), strict=False)
    )
    assert (500, 50) in pairs
    assert (1000, 60) in pairs
    assert (500, 60) in pairs
    assert (1000, 50) in pairs


def test_sample_rejects_small_grid():
    raw = _fake_raw_grid()
    with pytest.raises(ValueError):
        sample_design_space(raw, n_re=1, n_h=3)
