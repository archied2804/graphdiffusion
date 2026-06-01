"""
Tests for graph_diffusion.postprocessing.inference
====================================================

Smoke + determinism tests for the package-level shape sampler used by
the interactive Cp notebook and ``scripts/postprocess_exp020.py``.
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
from context import graph_diffusion  # noqa: F401

from graph_diffusion.data.pOnEllipse import _H5_N_TIMESTEPS
from graph_diffusion.postprocessing.inference import sample_shapes_from_cond
from graph_diffusion.postprocessing.loaders import (
    build_dataset,
    build_model,
)

# ---------------------------------------------------------------------------
# Fake dataset / minimal config (same idiom as test_loaders.py).
# ---------------------------------------------------------------------------


def _write_fake_h5(path: Path) -> None:
    rng = np.random.default_rng(11)
    n_sims = 3
    max_nodes = 8
    n_feat = 4 + _H5_N_TIMESTEPS
    data_arr = np.full((n_sims, max_nodes, n_feat), np.nan, dtype=np.float32)
    for i in range(n_sims):
        theta = np.linspace(0.0, 2.0 * np.pi, max_nodes, endpoint=False)
        x = (1.0 + 0.1 * i) * np.cos(theta)
        y = (0.7 + 0.05 * i) * np.sin(theta)
        re = np.full(max_nodes, 500.0 + i, dtype=np.float32)
        aoa = np.full(max_nodes, 5.0, dtype=np.float32)
        p = rng.standard_normal((max_nodes, _H5_N_TIMESTEPS)).astype(np.float32)
        rows = np.concatenate(
            [
                x[:, None].astype(np.float32),
                y[:, None].astype(np.float32),
                re[:, None],
                aoa[:, None],
                p,
            ],
            axis=1,
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


def _minimal_config(root: Path) -> dict:
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
            "T": 6,
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
        "mlp": {"activation": "silu", "layer_norm": True, "residual": True},
        "model": {"n_noise_channels": 1, "lambda_pressure": 0.1},
    }


def _setup(tmpdir: str):
    torch.manual_seed(0)
    root = _make_dataset_root(tmpdir)
    config = _minimal_config(root)
    model = build_model(config, device="cpu")
    model.eval()
    dataset = build_dataset(config)
    template = copy.copy(dataset[0])
    return model, dataset, template


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sample_shapes_from_cond_output_shapes() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model, dataset, template = _setup(tmpdir)
        cond_vec = dataset[0].cond.squeeze(0)
        n_nodes = template.pos.shape[0]
        n_samples = 3

        radii, head_modes = sample_shapes_from_cond(
            model=model,
            template=template,
            cond_vec=cond_vec,
            n_samples=n_samples,
            guidance_scale=3.0,
            device="cpu",
            clamp_range=(0.5, 2.0),
            seed=0,
        )

        assert radii.shape == (n_samples, n_nodes)
        assert head_modes.shape == (n_samples, cond_vec.shape[0])
        assert radii.dtype == np.float32
        assert head_modes.dtype == np.float32


def test_sample_shapes_from_cond_is_deterministic_under_fixed_seed() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model, dataset, template = _setup(tmpdir)
        cond_vec = dataset[0].cond.squeeze(0)

        radii_a, modes_a = sample_shapes_from_cond(
            model,
            template,
            cond_vec,
            n_samples=2,
            guidance_scale=3.0,
            device="cpu",
            seed=42,
        )
        radii_b, modes_b = sample_shapes_from_cond(
            model,
            template,
            cond_vec,
            n_samples=2,
            guidance_scale=3.0,
            device="cpu",
            seed=42,
        )

        np.testing.assert_allclose(radii_a, radii_b)
        np.testing.assert_allclose(modes_a, modes_b)


def test_sample_shapes_from_cond_seed_changes_output() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model, dataset, template = _setup(tmpdir)
        cond_vec = dataset[0].cond.squeeze(0)

        radii_a, _ = sample_shapes_from_cond(
            model,
            template,
            cond_vec,
            n_samples=1,
            guidance_scale=3.0,
            device="cpu",
            seed=0,
        )
        radii_b, _ = sample_shapes_from_cond(
            model,
            template,
            cond_vec,
            n_samples=1,
            guidance_scale=3.0,
            device="cpu",
            seed=1,
        )

        # Different seeds → different reverse-diffusion noise → different output.
        assert not np.allclose(radii_a, radii_b)
