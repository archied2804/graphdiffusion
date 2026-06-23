"""Tests for scripts/export_ellipse_for_openlb helpers."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from context import graph_diffusion  # noqa: F401

_spec = importlib.util.spec_from_file_location(
    "export_openlb",
    Path(__file__).resolve().parent.parent / "scripts" / "export_ellipse_for_openlb.py",
)
export_openlb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_openlb)


def test_verify_manifest_row_matches_raw():
    raw = np.full((2, 8, 105), np.nan, dtype=np.float32)
    raw[0, :6, 0] = np.linspace(-0.5, 0.5, 6)
    raw[0, :6, 1] = np.linspace(-0.2, 0.2, 6)
    raw[0, :6, 2] = 754.0  # Re
    raw[0, :6, 3] = 5.5  # H
    row = {"idx": 0, "Re": 754.0, "H": 5.5}
    export_openlb.verify_manifest_row(raw, row)  # should not raise


def test_verify_manifest_row_detects_mismatch():
    raw = np.full((1, 8, 105), np.nan, dtype=np.float32)
    raw[0, :6, 0] = np.linspace(-0.5, 0.5, 6)
    raw[0, :6, 1] = np.linspace(-0.2, 0.2, 6)
    raw[0, :6, 2] = 754.0
    raw[0, :6, 3] = 5.5
    bad = {"idx": 0, "Re": 754.0, "H": 8.0}  # wrong H
    with pytest.raises(ValueError, match="H mismatch"):
        export_openlb.verify_manifest_row(raw, bad)
