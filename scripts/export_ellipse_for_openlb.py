"""Export pOnEllipse HDF5 simulations to OpenLB ``ellipse2d`` inputs.

For each selected CFD simulation this writes:

- ``sim_<idx>.dat`` — the ellipse surface boundary points as an ordered,
  angularly-sorted (CCW) closed polygon, one ``x y`` per line. This is the
  body input consumed by ``cfd/ellipse2d`` via ``IndicatorSDF2D``. The raw
  HDF5 node order is mesh-generation order, NOT geometric order, so points
  are sorted by angle about the centroid to form a simple polygon.
- ``sim_<idx>_ref_cp.csv`` — the dataset's reference surface pressure curve,
  ``x_over_c,cp_ref``, the per-node mean over the 101 unsteady timesteps,
  sorted by chordwise position. This is what the OpenLB-measured Cp(x/c) is
  validated against (shape match; absolute normalisation reconciled
  separately — see ``cfd/ellipse2d/README.md``).
- ``manifest.csv`` — one row per exported sim: ``idx,Re,H,a,b,n_nodes``.

The reference Cp's absolute scale is left as stored in the HDF5; the
baseline gate is about curve shape and the Re-reference-length convention.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from graph_diffusion.data.pOnEllipse import (
    _H5_COL_H,
    _H5_COL_P_START,
    _H5_COL_RE,
    _H5_COL_X,
    _H5_COL_Y,
    _H5_N_TIMESTEPS,
    _load_h5_raw,
    _node_count,
)


def _ordered_polygon(xy: np.ndarray) -> np.ndarray:
    """Sort 2-D boundary points into a CCW simple polygon.

    The ellipse surface is convex, so an angular sort about the centroid
    yields a non-self-intersecting boundary in counter-clockwise order.

    Args:
        xy (np.ndarray): Boundary points, shape ``(N, 2)``.

    Returns:
        np.ndarray: The same points reordered CCW, shape ``(N, 2)``.
    """
    centroid = xy.mean(axis=0)
    rel = xy - centroid
    theta = np.arctan2(rel[:, 1], rel[:, 0])
    return xy[np.argsort(theta)]


def _reference_cp(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the dataset reference Cp(x/c) for one simulation.

    Args:
        data (np.ndarray): Valid node rows, shape ``(N, 105)``.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(x_over_c, cp_ref)``, each shape
            ``(N,)``, sorted ascending by chordwise position.
    """
    xc = data[:, _H5_COL_X]
    chord = xc.max() - xc.min()
    x_over_c = (xc - xc.min()) / chord
    cp_mean = data[:, _H5_COL_P_START : _H5_COL_P_START + _H5_N_TIMESTEPS].mean(axis=1)
    order = np.argsort(x_over_c)
    return x_over_c[order], cp_mean[order]


def export_sim(raw: np.ndarray, idx: int, out_dir: Path) -> dict[str, float]:
    """Export one simulation's body polygon and reference Cp.

    Args:
        raw (np.ndarray): Full HDF5 array, shape ``(N_sims, MAX_NODES, 105)``.
        idx (int): Simulation index to export.
        out_dir (Path): Directory to write outputs into.

    Returns:
        dict[str, float]: Manifest row for this simulation.
    """
    sample = raw[idx]
    n_nodes = _node_count(sample)
    data = sample[:n_nodes]

    re = float(data[0, _H5_COL_RE])
    h = float(data[0, _H5_COL_H])
    xy = np.stack([data[:, _H5_COL_X], data[:, _H5_COL_Y]], axis=1)
    a = float((xy[:, 0].max() - xy[:, 0].min()) / 2.0)
    b = float((xy[:, 1].max() - xy[:, 1].min()) / 2.0)

    poly = _ordered_polygon(xy)
    dat_path = out_dir / f"sim_{idx}.dat"
    with dat_path.open("w") as fh:
        fh.write(
            f"# pOnEllipse sim {idx}: Re={re:.4g} H={h:.4g} "
            f"a={a:.4g} b={b:.4g} n={n_nodes} (CCW polygon)\n"
        )
        fh.write("# x y\n")
        for x, y in poly:
            fh.write(f"{x:.10f} {y:.10f}\n")

    x_over_c, cp_ref = _reference_cp(data)
    cp_path = out_dir / f"sim_{idx}_ref_cp.csv"
    with cp_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["x_over_c", "cp_ref"])
        for xc, cp in zip(x_over_c, cp_ref, strict=True):
            writer.writerow([f"{xc:.6f}", f"{cp:.6f}"])

    return {"idx": idx, "Re": re, "H": h, "a": a, "b": b, "n_nodes": n_nodes}


def verify_manifest_row(raw: np.ndarray, row: dict[str, float]) -> None:
    """Assert a manifest row's Re and H equal the HDF5 columns for that sim.

    Args:
        raw (np.ndarray): Full HDF5 array.
        row (dict[str, float]): A manifest row with ``idx``, ``Re``, ``H``.

    Raises:
        ValueError: If the manifest Re or H differs from the HDF5 value.
    """
    idx = int(row["idx"])
    re_h5 = float(raw[idx, 0, _H5_COL_RE])
    h_h5 = float(raw[idx, 0, _H5_COL_H])
    if abs(re_h5 - float(row["Re"])) > 1e-3:
        raise ValueError(
            f"Re mismatch for sim {idx}: manifest {row['Re']} vs h5 {re_h5}"
        )
    if abs(h_h5 - float(row["H"])) > 1e-3:
        raise ValueError(f"H mismatch for sim {idx}: manifest {row['H']} vs h5 {h_h5}")


def select_indices(raw: np.ndarray, n: int) -> list[int]:
    """Pick ``n`` simulations spanning the Reynolds-number range.

    Args:
        raw (np.ndarray): Full HDF5 array.
        n (int): Number of simulations to select.

    Returns:
        list[int]: Selected simulation indices, ordered by Re.
    """
    re = raw[:, 0, _H5_COL_RE]
    order = np.argsort(re)
    picks = np.linspace(0, len(order) - 1, n).round().astype(int)
    return [int(order[p]) for p in picks]


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h5",
        type=Path,
        default=Path("data/ellipse/raw/pOnEllipseTrain.h5"),
        help="Path to the pOnEllipse HDF5 file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("cfd/ellipse2d/shapes/baseline"),
        help="Directory to write .dat / ref_cp.csv / manifest.csv into.",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=None,
        help="Explicit sim indices to export. Overrides --n-sims.",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=3,
        help="Number of sims spanning the Re range (when --indices unset).",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = _load_h5_raw(args.h5)
    indices = args.indices if args.indices else select_indices(raw, args.n_sims)

    rows = [export_sim(raw, idx, args.out_dir) for idx in indices]

    for row in rows:
        verify_manifest_row(raw, row)

    manifest = args.out_dir / "manifest.csv"
    with manifest.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["idx", "Re", "H", "a", "b", "n_nodes"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Exported {len(rows)} sims to {args.out_dir}")
    for row in rows:
        print(
            f"  sim {row['idx']:>4}: Re={row['Re']:.4g} H={row['H']:.4g} "
            f"a={row['a']:.3g} b={row['b']:.3g} n={row['n_nodes']}"
        )


if __name__ == "__main__":
    main()
