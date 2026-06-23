"""Tests for graph_diffusion.data.design_space."""

from context import graph_diffusion  # noqa: F401

from graph_diffusion.data.design_space import make_case_name


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
