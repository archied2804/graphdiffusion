"""
graph_diffusion.postprocessing.loaders
========================================

Checkpoint and TensorBoard log loading utilities for post-processing
trained diffusion models.
"""

from __future__ import annotations

import torch


def load_checkpoint(file_path: str) -> dict:
    """Load a checkpoint file and print summary info.

    Args:
        file_path: Path to the ``.pt`` checkpoint file.

    Returns:
        Checkpoint dictionary.
    """
    checkpoint = torch.load(
        file_path, map_location=torch.device("cpu"), weights_only=False
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
