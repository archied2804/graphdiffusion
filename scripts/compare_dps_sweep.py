"""Compare OpenLB-measured Cp against the shared target across DPS weights.

EXP-041 DPS-sweep analysis. Every weight conditions on the *same* synthetic
out-of-distribution (OOD) Cp target; only the Diffusion-Posterior-Sampling
guidance strength differs. This asks whether stronger posterior guidance lifts
the **true** (CFD-measured) Cp fidelity OOD, or merely chases the model's own
self-trained pressure head.

The OpenLB surface Cp traces the full body boundary, so its x/c is *not*
monotonic (it folds over the top and bottom surfaces). The target Cp(x/c) is a
single monotonic curve, so we interpolate the **target onto the measured nodes**
(robust) rather than the reverse. Because dataset and OpenLB Cp differ by the
EXP-040 affine normalisation (Cp_olb ≈ a·Cp_ref + b, a≈1.78), we report the
correlation (affine-invariant shape fidelity) and the post-affine RMS.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_cp(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read an ``x_over_c,Cp[,...]`` CSV into arrays (unsorted).

    Args:
        path (Path): CSV with a header and at least two columns.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(x_over_c, cp)`` in file order.
    """
    xs, cps = [], []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            xs.append(float(row[0]))
            cps.append(float(row[1]))
    return np.array(xs), np.array(cps)


def compare_one(
    target_csv: Path, olb_csv: Path, label: str, ax: plt.Axes
) -> dict[str, float]:
    """Overlay one DPS weight's measured Cp on the shared target and score it.

    Args:
        target_csv (Path): Shared target Cp(x/c) curve (monotonic in x/c).
        olb_csv (Path): OpenLB-measured surface Cp for this weight.
        label (str): DPS weight label for the panel title.
        ax (plt.Axes): Axis to draw onto.

    Returns:
        dict[str, float]: Fidelity metrics for this weight.
    """
    xt, cpt = _read_cp(target_csv)
    order = np.argsort(xt)
    xt, cpt = xt[order], cpt[order]
    xo, cpo = _read_cp(olb_csv)

    # Interpolate the (monotonic) target onto the measured surface nodes.
    cpt_on_nodes = np.interp(xo, xt, cpt)

    corr = float(np.corrcoef(cpt_on_nodes, cpo)[0, 1])
    # Affine fit cpo ≈ a*cpt + b (EXP-040 normalisation); residual = shape error.
    design = np.vstack([cpt_on_nodes, np.ones_like(cpt_on_nodes)]).T
    (a, b), *_ = np.linalg.lstsq(design, cpo, rcond=None)
    affine_rms = float(np.sqrt(np.mean((cpo - (a * cpt_on_nodes + b)) ** 2)))

    ax.plot(xt, cpt, "k-", lw=2, label="target")
    ax.plot(xo, cpo, "C1.", ms=5, label="OpenLB")
    # Target rescaled by the affine fit, for visual shape comparison.
    xt_fine = np.linspace(0.0, 1.0, 200)
    ax.plot(
        xt_fine, a * np.interp(xt_fine, xt, cpt) + b, "C0--", lw=1, label="a·target+b"
    )
    ax.invert_yaxis()
    ax.set_title(f"{label}\nρ={corr:.3f}  affine_rms={affine_rms:.3f}  a={a:.2f}")
    ax.set_xlabel("x/c")
    ax.set_ylabel("Cp")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    return {
        "label": label,
        "corr": corr,
        "affine_rms": affine_rms,
        "affine_a": float(a),
        "affine_b": float(b),
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path("cfd/ellipse2d/shapes/dps_sweep"),
        help="Directory with synth_target_cp.csv and manifest.csv.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("cfd/ellipse2d/runs"),
        help="Directory with dps_<tag>/tmp/ellipse2d_cp.csv run outputs.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/assets/EXP-041_dps_sweep.png"),
        help="Output overlay figure path.",
    )
    args = parser.parse_args()

    target = args.sweep_dir / "synth_target_cp.csv"
    manifest = args.sweep_dir / "manifest.csv"
    weights = []
    with manifest.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            weights.append((float(row["dps"]), row["tag"]))

    n = len(weights)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
    rows = []
    for ax, (dps, tag) in zip(axes[0], weights, strict=True):
        olb = args.runs_dir / f"dps_{tag}" / "tmp" / "ellipse2d_cp.csv"
        if not olb.exists():
            ax.set_title(f"dps={dps:g}: no cp.csv")
            continue
        m = compare_one(target, olb, f"dps={dps:g}", ax)
        m["dps"] = dps
        rows.append(m)

    fig.suptitle("EXP-041 DPS sweep: measured Cp vs synthetic OOD target", y=1.02)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")
    print(f"{'dps':>6} {'corr':>8} {'affine_rms':>11} {'a':>7} {'b':>8}")
    for r in rows:
        print(
            f"{r['dps']:>6.2g} {r['corr']:>8.4f} {r['affine_rms']:>11.4f} "
            f"{r['affine_a']:>7.3f} {r['affine_b']:>8.3f}"
        )


if __name__ == "__main__":
    main()
