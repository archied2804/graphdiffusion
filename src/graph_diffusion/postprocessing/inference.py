"""
graph_diffusion.postprocessing.inference
==========================================

Package-level helpers for running a trained conditional diffusion model
in inference mode. Used by ``scripts/postprocess_exp020.py`` and by the
interactive Cp adjuster notebook.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch_geometric.data import Data

from graph_diffusion.model.graph_diffusion_model import GraphDiffusionModel


def sample_shapes_from_cond(
    model: GraphDiffusionModel,
    template: Data,
    cond_vec: torch.Tensor,
    n_samples: int,
    guidance_scale: float,
    device: str,
    clamp_range: tuple[float, float] = (0.5, 2.0),
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the reverse diffusion ``n_samples`` times against a cond vector.

    The function batches per-sample seeding so two calls with the same
    ``seed`` produce identical outputs — important for the interactive
    notebook to give reproducible drags.

    Args:
        model: Trained :class:`GraphDiffusionModel` (already on
            ``device``, in ``eval`` mode is assumed but not enforced).
        template: PyG :class:`~torch_geometric.data.Data` object whose
            topology (``pos``, ``edge_index``, ``edge_attr``, ``batch``)
            is reused for every sample. Its ``x`` is ignored.
        cond_vec: Conditioning vector of shape ``(K,)`` (the DCT modes).
        n_samples: Number of independent samples to draw.
        guidance_scale: Classifier-free-guidance scale ``w``. Pass
            ``1.0`` to disable CFG.
        device: Torch device string (e.g. ``"cpu"`` or ``"cuda"``).
        clamp_range: Per-step clamp applied to the radial feature
            during sampling. Defaults to ``(0.5, 2.0)``.
        seed: Base seed. Sample ``i`` uses ``torch.manual_seed(seed + i)``.
            Pass ``None`` to skip seeding entirely.

    Returns:
        Tuple ``(radii, head_pred_modes)`` of NumPy arrays:

        * ``radii`` — shape ``(n_samples, N_nodes)``, the raw radial
          channel of the generated graphs.
        * ``head_pred_modes`` — shape ``(n_samples, K)``, the
          pressure-head DCT-mode prediction for each generated shape.
          Useful for measuring how well a sample matches its target.
    """
    template_with_cond = copy.copy(template)
    template_with_cond.cond = cond_vec.unsqueeze(0).to(device)

    pos = template_with_cond.pos
    n_nodes = pos.shape[0]
    k_modes = int(cond_vec.shape[0])

    radii = np.empty((n_samples, n_nodes), dtype=np.float32)
    head_modes = np.empty((n_samples, k_modes), dtype=np.float32)

    pressure_head = model.pressure_head
    batch_vec = torch.zeros(n_nodes, dtype=torch.long, device=device)

    for i in range(n_samples):
        if seed is not None:
            torch.manual_seed(seed + i)
        out = model.sample(
            template_with_cond,
            clamp_range=clamp_range,
            guidance_scale=guidance_scale,
        )
        radii[i] = out.x[:, 0].detach().cpu().numpy()
        if pressure_head is not None:
            with torch.no_grad():
                pred = pressure_head(out.x, pos, batch_vec)
            head_modes[i] = pred[0].detach().cpu().numpy()
    return radii, head_modes


def radial_to_xy(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Convert ``(N,)`` radii and ``(N,)`` thetas to ``(N, 2)`` Cartesian.

    Sorting by theta gives a proper boundary ordering for ``plt.plot``
    to draw a closed curve; the raw dataset order is not chordwise.

    Args:
        r: Radial coordinates, shape ``(N,)``.
        theta: Polar angles in radians, shape ``(N,)``.

    Returns:
        Cartesian coordinates of shape ``(N, 2)``, sorted by ``theta``.
    """
    order = np.argsort(theta)
    r_sorted = r[order]
    theta_sorted = theta[order]
    return np.stack(
        [r_sorted * np.cos(theta_sorted), r_sorted * np.sin(theta_sorted)],
        axis=1,
    )


def template_thetas(template: Data) -> np.ndarray:
    """Recover the ``(N,)`` theta vector from ``template.pos = (cos t, sin t)``.

    Args:
        template: PyG :class:`~torch_geometric.data.Data` object with
            ``pos`` storing per-node ``(cos θ, sin θ)``.

    Returns:
        Per-node theta in radians, shape ``(N,)``.
    """
    pos = template.pos.detach().cpu().numpy()
    theta: np.ndarray = np.arctan2(pos[:, 1], pos[:, 0])
    return theta
