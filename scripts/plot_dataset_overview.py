"""
Render the full dataset-overview figure suite from the pOnEllipse HDF5.

Writes one PNG + one PDF per figure into ``--out-dir``. Read-only on the
dataset; no model is loaded. Final figure selection for the report is made
by hand from the rendered set.

Usage:
    python scripts/plot_dataset_overview.py \
        --h5 data/ellipse/raw/pOnEllipseTrain.h5 \
        --out-dir outputs/dataset_overview \
        --max-sims 800
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from graph_diffusion.data.pOnEllipse import _load_h5_raw  # noqa: E402
from graph_diffusion.visualisation.dataset_overview import (  # noqa: E402
    plot_cp_envelope,
    plot_cp_pca,
    plot_design_space_coverage,
    plot_design_space_drag,
    plot_drag_vs_re,
    plot_shape_gallery,
)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h5",
        type=Path,
        default=Path("data/ellipse/raw/pOnEllipseTrain.h5"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/dataset_overview"),
    )
    parser.add_argument(
        "--max-sims",
        type=int,
        default=800,
        help="Cap the number of sims plotted (subsampled evenly) for speed.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw = _load_h5_raw(args.h5)
    n = raw.shape[0]
    step = max(1, n // args.max_sims)
    indices = list(range(0, n, step))
    # Evaluation-sample overlay: reuse the same even subsample's corners.
    eval_indices = [indices[0], indices[len(indices) // 2], indices[-1]]

    figures = {
        "design_space_drag": plot_design_space_drag(raw, indices),
        "cp_envelope_by_re": plot_cp_envelope(raw, indices, colour_by="re"),
        "cp_envelope_by_b": plot_cp_envelope(raw, indices, colour_by="b"),
        "design_space_coverage": plot_design_space_coverage(raw, indices, eval_indices),
        "shape_gallery": plot_shape_gallery(raw, indices[:9]),
        "drag_vs_re": plot_drag_vs_re(raw, indices),
        "cp_pca_by_re": plot_cp_pca(raw, indices, colour_by="re"),
    }
    for name, fig in figures.items():
        for ext in ("png", "pdf"):
            fig.savefig(args.out_dir / f"{name}.{ext}", dpi=200, bbox_inches="tight")
        print(f"wrote {name}.png / .pdf")


if __name__ == "__main__":
    main()
