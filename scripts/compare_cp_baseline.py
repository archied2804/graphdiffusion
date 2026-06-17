"""Compare OpenLB-measured surface Cp against the pOnEllipse dataset reference.

Overlays the OpenLB time-averaged Cp(x/c) (``ellipse2d_cp.csv``) on the
dataset's stored reference Cp (``sim_<idx>_ref_cp.csv``) for one or more
simulations, and reports a shape-match error. Because the dataset Cp and the
OpenLB Cp may differ by an additive/multiplicative normalisation (the Re
reference-length convention, gauge offset), the comparison reports both the
raw RMS difference and the RMS after a least-squares affine fit (Cp_olb ≈
a·Cp_ref + b) — the affine-residual isolates *shape* agreement from a constant
normalisation offset that can be reconciled separately.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_cp(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read an ``x_over_c,Cp[,...]`` CSV into sorted arrays.

    Args:
        path (Path): CSV file with a header and at least two columns.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(x_over_c, cp)`` sorted by x/c.
    """
    xs, cps = [], []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader)  # header
        for row in reader:
            if len(row) < 2:
                continue
            xs.append(float(row[0]))
            cps.append(float(row[1]))
    x = np.array(xs)
    cp = np.array(cps)
    order = np.argsort(x)
    return x[order], cp[order]


def compare_one(
    ref_csv: Path, olb_csv: Path, idx: str, ax: plt.Axes
) -> dict[str, float]:
    """Overlay one sim's reference vs OpenLB Cp and compute match metrics.

    Args:
        ref_csv (Path): Dataset reference Cp CSV.
        olb_csv (Path): OpenLB-measured Cp CSV.
        idx (str): Simulation identifier for labelling.
        ax (plt.Axes): Axis to plot onto.

    Returns:
        dict[str, float]: Match metrics for this simulation.
    """
    xr, cpr = _read_cp(ref_csv)
    xo, cpo = _read_cp(olb_csv)

    # Interpolate OpenLB onto the reference x/c grid for a like-for-like diff.
    cpo_on_ref = np.interp(xr, xo, cpo)

    raw_rms = float(np.sqrt(np.mean((cpo_on_ref - cpr) ** 2)))
    # Affine fit cpo ≈ a*cpr + b (isolates shape from normalisation/offset).
    design = np.vstack([cpr, np.ones_like(cpr)]).T
    (a, b), *_ = np.linalg.lstsq(design, cpo_on_ref, rcond=None)
    affine_rms = float(np.sqrt(np.mean((cpo_on_ref - (a * cpr + b)) ** 2)))
    corr = float(np.corrcoef(cpr, cpo_on_ref)[0, 1])

    ax.plot(xr, cpr, "k-", lw=2, label="dataset ref")
    ax.plot(xo, cpo, "C1.-", ms=4, label="OpenLB")
    ax.invert_yaxis()  # aerodynamic convention
    ax.set_title(
        f"sim {idx}\nr_rms={raw_rms:.3f} affine_rms={affine_rms:.3f} ρ={corr:.3f}"
    )
    ax.set_xlabel("x/c")
    ax.set_ylabel("Cp")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    return {
        "idx": idx,
        "raw_rms": raw_rms,
        "affine_rms": affine_rms,
        "affine_a": float(a),
        "affine_b": float(b),
        "corr": corr,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("cfd/ellipse2d/shapes/baseline"),
        help="Directory with sim_<idx>_ref_cp.csv files.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("cfd/ellipse2d/runs"),
        help="Directory with per-sim run subdirs holding tmp/ellipse2d_cp.csv.",
    )
    parser.add_argument(
        "--sims",
        nargs="+",
        required=True,
        help="Sim identifiers, e.g. 0 2850 5700 (matches sim_<id> run dirs).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("cfd/ellipse2d/runs/cp_comparison.png"),
        help="Output figure path.",
    )
    args = parser.parse_args()

    n = len(args.sims)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    rows = []
    for ax, sim in zip(axes[0], args.sims, strict=True):
        ref = args.baseline_dir / f"sim_{sim}_ref_cp.csv"
        olb = args.runs_dir / f"sim_{sim}" / "tmp" / "ellipse2d_cp.csv"
        if not olb.exists():
            ax.set_title(f"sim {sim}: no OpenLB cp.csv")
            continue
        rows.append(compare_one(ref, olb, sim, ax))

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")
    print(f"{'sim':>6} {'raw_rms':>9} {'affine_rms':>11} {'a':>7} {'b':>8} {'corr':>7}")
    for r in rows:
        print(
            f"{r['idx']:>6} {r['raw_rms']:>9.4f} {r['affine_rms']:>11.4f} "
            f"{r['affine_a']:>7.3f} {r['affine_b']:>8.3f} {r['corr']:>7.4f}"
        )


if __name__ == "__main__":
    main()
