"""
graph_diffusion.visualisation
==============================

Plotting and trajectory helpers for diffusion experiments. The package
is dependency-light: it consumes a trained ``GraphDiffusionModel`` and
a PyG ``Data`` template through their public APIs, never reaching into
model internals.
"""

from __future__ import annotations

from graph_diffusion.visualisation.dataset_overview import (
    plot_cp_envelope,
    plot_cp_pca,
    plot_design_space_coverage,
    plot_design_space_drag,
    plot_drag_vs_re,
    plot_shape_gallery,
    pressure_drag_coefficient,
    sim_summary,
)
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
    "plot_cp_envelope",
    "plot_cp_pca",
    "plot_design_space_coverage",
    "plot_design_space_drag",
    "plot_drag_vs_re",
    "plot_shape_gallery",
    "plot_trajectory_filmstrip",
    "pressure_drag_coefficient",
    "sim_summary",
    "write_trajectory_animation",
]
