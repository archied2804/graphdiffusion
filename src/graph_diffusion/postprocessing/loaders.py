"""
graph_diffusion.postprocessing.loaders
========================================

Checkpoint, config, and model-construction utilities for post-processing
trained diffusion models. Provides the package-level entry points used by
``scripts/postprocess_exp020.py`` and the interactive Cp notebook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
import yaml  # type: ignore[import-untyped]

from graph_diffusion.building_blocks.noise_schedule import NoiseSchedule
from graph_diffusion.data.pOnEllipseConditional import pOnEllipseConditionalDataset
from graph_diffusion.data.transforms import (
    ComputeAngularEdgeFeatures,
    ComputeArcLengthEdgeFeatures,
)
from graph_diffusion.model.graph_diffusion_model import GraphDiffusionModel
from graph_diffusion.model.pressure_head import PressurePredictionHead
from graph_diffusion.model.score_network import ScoreNetwork


def load_checkpoint(file_path: str) -> dict[str, Any]:
    """Load a checkpoint file and print summary info.

    Args:
        file_path: Path to the ``.pt`` checkpoint file.

    Returns:
        Checkpoint dictionary.
    """
    checkpoint = cast(
        dict[str, Any],
        torch.load(file_path, map_location=torch.device("cpu"), weights_only=False),
    )
    print(f"Loaded checkpoint from: {file_path}")
    epoch = checkpoint.get("epoch", checkpoint.get("epochs", "?"))
    lr = checkpoint.get("lr", "?")
    print(f"Epoch: {epoch}, Learning Rate: {lr}")
    return checkpoint


def read_tensorboard_logs(
    log_dir: str,
) -> dict[str, list[tuple[int, float]]]:
    """Read TensorBoard logs and extract scalar data.

    Args:
        log_dir: Path to the TensorBoard events directory.

    Returns:
        Dictionary mapping tag names to lists of (step, value) tuples.
    """
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    scalar_data: dict[str, list[tuple[int, float]]] = {}
    for tag in event_acc.Tags()["scalars"]:
        scalar_data[tag] = [
            (scalar.step, scalar.value) for scalar in event_acc.Scalars(tag)
        ]

    print(f"Extracted tags from TensorBoard logs: {list(scalar_data.keys())}")
    return scalar_data


def load_config(path: str) -> dict[str, Any]:
    """Read a YAML config file and return it as a dict.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed configuration.
    """
    with open(path) as fh:  # noqa: PTH123
        return cast(dict[str, Any], yaml.safe_load(fh))


def build_dataset(config: dict[str, Any]) -> pOnEllipseConditionalDataset:
    """Instantiate the conditional ellipse dataset from a parsed config.

    Args:
        config: Parsed YAML config — must contain an ``ellipse_dataset``
            section matching the EXP-020 schema.

    Returns:
        The configured :class:`pOnEllipseConditionalDataset`.
    """
    ds_cfg = config["ellipse_dataset"]
    feature_mode = ds_cfg.get("feature_mode", "radial_norm")
    pre_transform = (
        ComputeArcLengthEdgeFeatures()
        if feature_mode == "cartesian"
        else ComputeAngularEdgeFeatures()
    )
    return pOnEllipseConditionalDataset(
        root=ds_cfg.get("root", "data/ellipse"),
        cond_mode=ds_cfg.get("cond_mode", "fourier"),
        k_modes=ds_cfg.get("k_modes", 8),
        feature_mode=feature_mode,
        split=ds_cfg.get("split", "train"),
        n_samples=ds_cfg.get("n_samples", None),
        k_neighbors=ds_cfg.get("k_neighbors", 6),
        global_dim=ds_cfg.get("global_dim", 8),
        variant=ds_cfg.get("variant", "default"),
        pre_transform=pre_transform,
    )


def build_model(config: dict[str, Any], device: str) -> GraphDiffusionModel:
    """Build an EXP-020-style :class:`GraphDiffusionModel` from a config.

    Mirrors the construction used at training time so that checkpoints
    saved by ``train.py`` can be loaded back without surgery.

    Args:
        config: Parsed YAML config — must contain ``noise_schedule``,
            ``score_network``, ``pressure_head``, ``mlp``, and ``model``
            sections matching the EXP-020 schema.
        device: Torch device string (e.g. ``"cpu"`` or ``"cuda"``).

    Returns:
        The configured (untrained) :class:`GraphDiffusionModel`, placed
        on ``device``.
    """
    ns_cfg = config["noise_schedule"]
    schedule = NoiseSchedule(
        T=ns_cfg["T"],
        schedule_type=ns_cfg.get("schedule_type", "cosine"),
        beta_start=ns_cfg.get("beta_start", 1.0e-4),
        beta_end=ns_cfg.get("beta_end", 0.02),
    )
    sn_cfg = config["score_network"]
    mlp_cfg = config["mlp"]
    sn = ScoreNetwork(
        node_dim=sn_cfg["node_dim"],
        edge_dim=sn_cfg["edge_dim"],
        global_dim=sn_cfg["global_dim"],
        time_embed_dim=sn_cfg["time_embed_dim"],
        n_layers=sn_cfg["n_layers"],
        hidden_dims=sn_cfg.get("hidden_dims", [64, 64]),
        activation=mlp_cfg.get("activation", "silu"),
        layer_norm=mlp_cfg.get("layer_norm", True),
        residual=mlp_cfg.get("residual", True),
        input_dim=sn_cfg.get("input_dim", None),
        cond_dim=sn_cfg.get("cond_dim", None),
        p_uncond=float(sn_cfg.get("p_uncond", 0.0)),
        output_dim=sn_cfg.get("output_dim", None),
    )
    ph_cfg = config["pressure_head"]
    head = PressurePredictionHead(
        in_dim=ph_cfg["in_dim"],
        out_dim=ph_cfg["out_dim"],
        node_hidden=ph_cfg.get("node_hidden", [64, 64]),
        global_hidden=ph_cfg.get("global_hidden", [64, 64]),
        node_embed_dim=ph_cfg.get("node_embed_dim", 64),
        activation=mlp_cfg.get("activation", "silu"),
        layer_norm=mlp_cfg.get("layer_norm", True),
    )
    model_cfg = config.get("model", {})
    min_snr_gamma_cfg = model_cfg.get("min_snr_gamma", None)
    return GraphDiffusionModel(
        score_network=sn,
        noise_schedule=schedule,
        n_noise_channels=model_cfg.get("n_noise_channels", None),
        pressure_head=head,
        lambda_pressure=float(model_cfg.get("lambda_pressure", 0.0)),
        min_snr_gamma=(
            float(min_snr_gamma_cfg) if min_snr_gamma_cfg is not None else None
        ),
        prediction_type=str(model_cfg.get("prediction_type", "epsilon")),
    ).to(device)


def load_exp020(
    experiment_dir: str,
    config_path: str,
    device: str,
    checkpoint_name: str = "checkpoint_best.pt",
) -> tuple[GraphDiffusionModel, pOnEllipseConditionalDataset, dict[str, Any]]:
    """Load an EXP-020-style trained model, its dataset, and its config.

    Convenience wrapper that ties together :func:`load_config`,
    :func:`build_model`, :func:`build_dataset`, and
    :func:`load_checkpoint`. The model is returned in ``eval`` mode with
    weights loaded from ``<experiment_dir>/<checkpoint_name>``.

    Args:
        experiment_dir: Directory containing the checkpoint file.
        config_path: Path to the YAML config used at training time.
        device: Torch device string (e.g. ``"cpu"`` or ``"cuda"``).
        checkpoint_name: Filename of the checkpoint inside
            ``experiment_dir``. Defaults to ``"checkpoint_best.pt"``;
            pass ``"checkpoint_ema.pt"`` to load EMA weights.

    Returns:
        Tuple ``(model, dataset, config)``.
    """
    config = load_config(config_path)
    model = build_model(config, device=device)
    checkpoint = load_checkpoint(str(Path(experiment_dir) / checkpoint_name))
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    dataset = build_dataset(config)
    return model, dataset, config
