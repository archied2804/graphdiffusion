"""Resample a generated boundary polygon to test the node-density hypothesis.

EXP-042 magnitude-creep / boundary-discretisation study. A diffusion-generated
shape is exported with a fixed (low) node count, so its OpenLB body indicator is
a coarse polygon. The hypothesis is that a *higher-fidelity* boundary — more
nodes interpolated along the **same** geometry — rounds the facets the lattice
sees, narrowing spurious separation and lowering the drag (and possibly the Cp
magnitude creep, where the generated OOD shape over-produces pressure ~2x the
in-distribution normalisation).

We resample with a **periodic cubic spline** in normalised arc length. This
matters: linear subdivision only inserts collinear points the signed-distance
indicator ignores, so it cannot change the body the lattice resolves. A smooth
periodic spline genuinely rounds the polygon as ``n`` rises, which is what
"higher boundary fidelity" means here.

Generates a sweep into ``cfd/ellipse2d/shapes/node_sweep/``: raw spline at
several node counts plus a Fourier-smoothed variant (roughness control), each as
an OpenLB ``x y`` ``.dat`` with a manifest.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline


def _read_dat(path: Path) -> np.ndarray:
    """Read an OpenLB ``# x y`` boundary ``.dat`` into an ``(N, 2)`` array.

    Args:
        path (Path): Point file with ``#`` comment lines and ``x y`` rows.

    Returns:
        np.ndarray: Boundary points ``(N, 2)`` in file order.
    """
    pts = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            x, y = line.split()
            pts.append((float(x), float(y)))
    return np.array(pts)


def _order_ccw(xy: np.ndarray) -> np.ndarray:
    """Order boundary points counter-clockwise by angle about the centroid.

    Args:
        xy (np.ndarray): Boundary points ``(N, 2)``.

    Returns:
        np.ndarray: The same points ordered CCW.
    """
    centroid = xy.mean(axis=0)
    rel = xy - centroid
    theta = np.arctan2(rel[:, 1], rel[:, 0])
    return xy[np.argsort(theta)]


def fourier_smooth(xy: np.ndarray, keep_modes: int) -> np.ndarray:
    """Low-pass a closed boundary by truncating the Fourier series of r(theta).

    Args:
        xy (np.ndarray): Boundary points ``(N, 2)``, ordered by angle.
        keep_modes (int): Number of low radial Fourier modes to retain.

    Returns:
        np.ndarray: Smoothed boundary points ``(N, 2)``.
    """
    centroid = xy.mean(axis=0)
    rel = xy - centroid
    theta = np.arctan2(rel[:, 1], rel[:, 0])
    radius = np.hypot(rel[:, 0], rel[:, 1])
    order = np.argsort(theta)
    coeffs = np.fft.rfft(radius[order])
    coeffs[keep_modes + 1 :] = 0.0
    r_smooth = np.fft.irfft(coeffs, n=len(order))
    th = theta[order]
    return np.column_stack(
        [centroid[0] + r_smooth * np.cos(th), centroid[1] + r_smooth * np.sin(th)]
    )


def spline_resample(xy: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed boundary to ``n`` points via a periodic cubic spline.

    Parametrises the (CCW-ordered) polygon by cumulative chord length, fits a
    periodic cubic spline to ``x(s)`` and ``y(s)``, and samples ``n`` points
    uniformly in ``s``. As ``n`` rises the body becomes a smooth curve sampled
    densely, rounding the original facets.

    Args:
        xy (np.ndarray): Boundary points ``(N, 2)``.
        n (int): Number of output points.

    Returns:
        np.ndarray: Resampled boundary points ``(n, 2)``.
    """
    ccw = _order_ccw(xy)
    closed = np.vstack([ccw, ccw[0]])
    seg = np.hypot(np.diff(closed[:, 0]), np.diff(closed[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    s_norm = s / total
    cs_x = CubicSpline(s_norm, closed[:, 0], bc_type="periodic")
    cs_y = CubicSpline(s_norm, closed[:, 1], bc_type="periodic")
    s_new = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.column_stack([cs_x(s_new), cs_y(s_new)])


def _write_dat(path: Path, xy: np.ndarray, header: str) -> None:
    """Write boundary points as an OpenLB ``x y`` ``.dat``.

    Args:
        path (Path): Output path.
        xy (np.ndarray): Boundary points ``(N, 2)``.
        header (str): One-line ``#`` description.
    """
    with path.open("w") as fh:
        fh.write(f"# {header}\n# x y\n")
        for x, y in xy:
            fh.write(f"{x:.10f} {y:.10f}\n")


def main() -> None:
    """CLI entry point: build the node-density sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("cfd/ellipse2d/shapes/generated/synth_ood_raw.dat"),
        help="Generated boundary .dat to resample.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("cfd/ellipse2d/shapes/node_sweep"),
        help="Output directory for the sweep .dat files and manifest.",
    )
    parser.add_argument(
        "--counts",
        type=int,
        nargs="+",
        default=[120, 240, 480],
        help="Node counts to spline-resample to (raw).",
    )
    parser.add_argument(
        "--smooth-n",
        type=int,
        default=240,
        help="Node count for the Fourier-smoothed roughness control.",
    )
    parser.add_argument(
        "--smooth-modes", type=int, default=6, help="Fourier modes kept when smoothing."
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    src = _read_dat(args.source)

    rows = []
    # Original polygon, verbatim (the n=60 baseline; rounds nothing).
    _write_dat(
        args.out_dir / "orig.dat", _order_ccw(src), f"node_sweep orig ({len(src)} pts)"
    )
    rows.append({"tag": "orig", "n_nodes": len(src), "kind": "raw", "modes": ""})

    for n in args.counts:
        xy = spline_resample(src, n)
        tag = f"raw_n{n}"
        _write_dat(args.out_dir / f"{tag}.dat", xy, f"node_sweep raw spline n={n}")
        rows.append({"tag": tag, "n_nodes": n, "kind": "raw_spline", "modes": ""})

    # Roughness control: smooth first, then resample to the same density.
    sm = spline_resample(fourier_smooth(src, args.smooth_modes), args.smooth_n)
    tag = f"smooth_n{args.smooth_n}"
    hdr = (
        f"node_sweep Fourier-smoothed ({args.smooth_modes} modes) "
        f"spline n={args.smooth_n}"
    )
    _write_dat(args.out_dir / f"{tag}.dat", sm, hdr)
    rows.append(
        {
            "tag": tag,
            "n_nodes": args.smooth_n,
            "kind": "smooth_spline",
            "modes": args.smooth_modes,
        }
    )

    with (args.out_dir / "manifest.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["tag", "n_nodes", "kind", "modes"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} node-sweep shapes to {args.out_dir}")
    for row in rows:
        print(f"  {row['tag']:>12}: n={row['n_nodes']} ({row['kind']})")


if __name__ == "__main__":
    main()
