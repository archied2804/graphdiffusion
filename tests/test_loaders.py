"""
Tests for graph_diffusion.postprocessing.loaders
=================================================

Exercise the package-level helpers for building and loading
EXP-020-style models from a YAML config and an experiment directory.
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import torch
import yaml
from context import graph_diffusion  # noqa: F401

from graph_diffusion.data.pOnEllipse import _H5_N_TIMESTEPS
from graph_diffusion.model.graph_diffusion_model import GraphDiffusionModel
from graph_diffusion.postprocessing.loaders import (
    build_dataset,
    build_model,
    load_exp020,
)

# ---------------------------------------------------------------------------
# Tiny fake dataset (mirrors the test_pOnEllipseConditional pattern)
# ---------------------------------------------------------------------------


def _write_fake_h5(path: Path) -> None:
    rng = np.random.default_rng(7)
    n_sims = 4
    max_nodes = 8
    n_feat = 4 + _H5_N_TIMESTEPS
    data_arr = np.full((n_sims, max_nodes, n_feat), np.nan, dtype=np.float32)
    for i in range(n_sims):
        theta = np.linspace(0.0, 2.0 * np.pi, max_nodes, endpoint=False)
        a, b = 1.0 + 0.1 * i, 0.7 + 0.05 * i
        x = (a * np.cos(theta)).astype(np.float32)
        y = (b * np.sin(theta)).astype(np.float32)
        re = np.full(max_nodes, 500.0 + i, dtype=np.float32)
        aoa = np.full(max_nodes, 5.0, dtype=np.float32)
        p = rng.standard_normal((max_nodes, _H5_N_TIMESTEPS)).astype(np.float32)
        rows = np.concatenate(
            [x[:, None], y[:, None], re[:, None], aoa[:, None], p], axis=1
        )
        data_arr[i, :max_nodes, :] = rows
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=data_arr)


def _make_dataset_root(tmpdir: str) -> Path:
    root = Path(tmpdir) / "dataset"
    raw = root / "raw"
    raw.mkdir(parents=True)
    _write_fake_h5(raw / "pOnEllipseTrain.h5")
    return root


def _minimal_config(root: Path) -> dict[str, Any]:
    """Smallest config valid for build_model + build_dataset."""
    return {
        "ellipse_dataset": {
            "root": str(root),
            "split": "train",
            "feature_mode": "radial_norm",
            "cond_mode": "fourier",
            "k_modes": 8,
            "n_samples": None,
            "k_neighbors": 2,
            "global_dim": 4,
        },
        "noise_schedule": {
            "T": 10,
            "schedule_type": "cosine",
            "beta_start": 1.0e-4,
            "beta_end": 0.02,
        },
        "score_network": {
            "node_dim": 8,
            "edge_dim": 2,
            "global_dim": 4,
            "time_embed_dim": 8,
            "n_layers": 2,
            "hidden_dims": [8, 8],
            "input_dim": 3,
            "output_dim": 1,
            "cond_dim": 8,
            "p_uncond": 0.15,
        },
        "pressure_head": {
            "in_dim": 3,
            "out_dim": 8,
            "node_hidden": [8],
            "global_hidden": [8],
            "node_embed_dim": 8,
        },
        "mlp": {
            "activation": "silu",
            "layer_norm": True,
            "residual": True,
        },
        "model": {
            "n_noise_channels": 1,
            "lambda_pressure": 0.1,
        },
    }


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------


def test_build_model_returns_graph_diffusion_model() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_dataset_root(tmpdir)
        config = _minimal_config(root)
        model = build_model(config, device="cpu")
        assert isinstance(model, GraphDiffusionModel)
        assert model.noise_schedule.T == 10
        assert model.pressure_head is not None
        assert model.lambda_pressure == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# build_dataset
# ---------------------------------------------------------------------------


def test_build_dataset_attaches_cond_with_configured_k_modes() -> None:
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_dataset_root(tmpdir)
        config = _minimal_config(root)
        ds = build_dataset(config)
        assert len(ds) > 0
        sample = ds[0]
        assert hasattr(sample, "cond")
        assert sample.cond.shape == (1, 8)


# ---------------------------------------------------------------------------
# load_exp020
# ---------------------------------------------------------------------------


def test_load_exp020_round_trips_state_dict() -> None:
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_dataset_root(tmpdir)
        config = _minimal_config(root)
        config_path = Path(tmpdir) / "cfg.yaml"
        with open(config_path, "w") as fh:
            yaml.safe_dump(config, fh)

        # Build a reference model, save its state to a fake experiment dir.
        ref_model = build_model(config, device="cpu")
        exp_dir = Path(tmpdir) / "exp"
        exp_dir.mkdir()
        torch.save(
            {"model_state_dict": ref_model.state_dict(), "epoch": 0},
            exp_dir / "checkpoint_best.pt",
        )

        # Load via the helper and compare weights.
        loaded_model, loaded_ds, loaded_cfg = load_exp020(
            experiment_dir=str(exp_dir),
            config_path=str(config_path),
            device="cpu",
        )
        assert isinstance(loaded_model, GraphDiffusionModel)
        assert loaded_cfg["noise_schedule"]["T"] == 10
        assert len(loaded_ds) > 0

        # Every parameter must match exactly (we just saved it).
        ref_state = ref_model.state_dict()
        loaded_state = loaded_model.state_dict()
        assert set(ref_state.keys()) == set(loaded_state.keys())
        for k in ref_state:
            assert torch.equal(ref_state[k], loaded_state[k])


def test_load_exp020_respects_custom_checkpoint_filename() -> None:
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_dataset_root(tmpdir)
        config = _minimal_config(root)
        config_path = Path(tmpdir) / "cfg.yaml"
        with open(config_path, "w") as fh:
            yaml.safe_dump(config, fh)

        ref_model = build_model(config, device="cpu")

        # Move one parameter so the two checkpoints diverge meaningfully.
        ref_model_modified = copy.deepcopy(ref_model)
        with torch.no_grad():
            for p in ref_model_modified.parameters():
                p.add_(1.0)
                break

        exp_dir = Path(tmpdir) / "exp"
        exp_dir.mkdir()
        torch.save(
            {"model_state_dict": ref_model.state_dict(), "epoch": 0},
            exp_dir / "checkpoint_best.pt",
        )
        torch.save(
            {"model_state_dict": ref_model_modified.state_dict(), "epoch": 1},
            exp_dir / "checkpoint_custom.pt",
        )

        loaded_default, _, _ = load_exp020(
            experiment_dir=str(exp_dir),
            config_path=str(config_path),
            device="cpu",
        )
        loaded_custom, _, _ = load_exp020(
            experiment_dir=str(exp_dir),
            config_path=str(config_path),
            device="cpu",
            checkpoint_name="checkpoint_custom.pt",
        )

        # Pick a parameter that we know differs.
        default_p = next(iter(loaded_default.parameters()))
        custom_p = next(iter(loaded_custom.parameters()))
        assert not torch.equal(default_p, custom_p)
