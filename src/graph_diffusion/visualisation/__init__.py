"""
graph_diffusion.visualisation
==============================

Plotting and trajectory helpers for diffusion experiments. The package
is dependency-light: it consumes a trained ``GraphDiffusionModel`` and
a PyG ``Data`` template through their public APIs, never reaching into
model internals.
"""

from __future__ import annotations

from graph_diffusion.visualisation.plotting import (
    plot_conditioning_grid,
    plot_trajectory_filmstrip,
    write_trajectory_animation,
)
from graph_diffusion.visualisation.trajectory import (
    collect_forward,
    collect_reverse,
)

__all__ = [
    "collect_forward",
    "collect_reverse",
    "plot_conditioning_grid",
    "plot_trajectory_filmstrip",
    "write_trajectory_animation",
]
