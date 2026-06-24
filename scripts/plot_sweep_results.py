"""Render the sweep-results figure suite for a validated sample set.

Loads a shape set's manifest, each run's OpenLB-measured Cp, the target/
reference Cp, and the body polygon, then writes five figures (gallery,
design-space context, Cp-manifold context, Cp-envelope overlay, results-vs-
OOD summary) to outputs/sweep_results/<set>/. Read-only on the run outputs.

Usage:
    python scripts/plot_sweep_results.py --set eval_grid
    python scripts/plot_sweep_results.py --set ood_systematic
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

from graph_diffusion.data.pOnEllipse import _load_h5_raw  # noqa: E402
from graph_diffusion.visualisation.sweep_results import (  # noqa: E402
    SampleRecord,
    plot_cp_envelope_overlay,
    plot_cp_manifold_context,
    plot_design_space_context,
    plot_results_summary,
    plot_sample_gallery,
)

_CFD = Path("cfd/ellipse2d")


def _read_xy(dat_path: Path) -> np.ndarray:
    """Read an ``x y`` polygon .dat (skipping ``#`` comment lines)."""
    pts = []
    for line in dat_path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        x, y = line.split()
        pts.append((float(x), float(y)))
    return np.array(pts)


def _read_cd(force_path: Path) -> float | None:
    """Read Cd_mean (first column) from an OpenLB force CSV, or None if absent."""
    if not force_path.is_file():
        return None
    with force_path.open() as fh:
        reader = csv.reader(fh)
        next(reader)  # header: Cd_mean,Cd_std,Cl_mean,Cl_std,...
        row = next(reader, None)
    return float(row[0]) if row else None


def _read_cp(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a 2-column Cp CSV (x_over_c, Cp); returns (x, cp)."""
    x, cp = [], []
    with csv_path.open() as fh:
        reader = csv.reader(fh)
        next(reader)  # header
        for row in reader:
            x.append(float(row[0]))
            cp.append(float(row[1]))
    return np.array(x), np.array(cp)


def _load_ood_systematic() -> list[SampleRecord]:
    shapes = _CFD / "shapes" / "ood_systematic"
    records = []
    with (shapes / "manifest.csv").open() as fh:
        for row in csv.DictReader(fh):
            name = row["name"]
            mx, mcp = _read_cp(_CFD / "runs" / name / "tmp" / "ellipse2d_cp.csv")
            tx, tcp = _read_cp(shapes / f"{name}_target_cp.csv")
            records.append(
                SampleRecord(
                    name=name,
                    re=float(row["Re"]),
                    h=float(row["H"]),
                    ood_dist=float(row["ood_dist"]),
                    polygon=_read_xy(shapes / f"{name}.dat"),
                    target_x=tx,
                    target_cp=tcp,
                    meas_x=mx,
                    meas_cp=mcp,
                    cd=_read_cd(_CFD / "runs" / name / "tmp" / "ellipse2d_force.csv"),
                )
            )
    return records


def _load_eval_grid() -> list[SampleRecord]:
    shapes = _CFD / "shapes" / "eval_grid"
    records = []
    with (shapes / "manifest.csv").open() as fh:
        for row in csv.DictReader(fh):
            name = row["name"]
            idx = row["idx"]
            mx, mcp = _read_cp(_CFD / "runs" / name / "tmp" / "ellipse2d_cp.csv")
            tx, tcp = _read_cp(shapes / f"sim_{idx}_ref_cp.csv")
            records.append(
                SampleRecord(
                    name=name,
                    re=float(row["Re"]),
                    h=float(row["H"]),
                    ood_dist=None,
                    polygon=_read_xy(shapes / f"sim_{idx}.dat"),
                    target_x=tx,
                    target_cp=tcp,
                    meas_x=mx,
                    meas_cp=mcp,
                    cd=_read_cd(_CFD / "runs" / name / "tmp" / "ellipse2d_force.csv"),
                )
            )
    return records


_LOADERS = {"ood_systematic": _load_ood_systematic, "eval_grid": _load_eval_grid}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", required=True, choices=sorted(_LOADERS))
    parser.add_argument(
        "--h5", type=Path, default=Path("data/ellipse/raw/pOnEllipseTrain.h5")
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or Path("outputs/sweep_results") / args.set
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = _LOADERS[args.set]()
    raw = _load_h5_raw(args.h5)

    figures = {
        "gallery": plot_sample_gallery(samples),
        "design_space_context": plot_design_space_context(raw, samples),
        "cp_manifold_context": plot_cp_manifold_context(raw, samples),
        "cp_envelope_overlay": plot_cp_envelope_overlay(raw, samples),
    }
    if any(s.ood_dist is not None for s in samples):
        figures["results_summary"] = plot_results_summary(samples)

    for name, fig in figures.items():
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"{name}.{ext}", dpi=200, bbox_inches="tight")
        print(f"wrote {name}.png / .pdf")


if __name__ == "__main__":
    main()
