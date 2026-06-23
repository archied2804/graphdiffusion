"""Analyse the EXP-041 OOD-target spread: fidelity vs out-of-distribution-ness.

For each synthetic OOD target (a different perturbation of the dataset-mean Cp),
read the OpenLB-measured surface Cp and score it against the target — correlation
(shape fidelity) and affine slope ``a`` (magnitude creep) — then plot both
against the target's OOD distance to the training set. This turns the single-OOD
result of EXP-041 into a trend: does fidelity degrade, and the creep grow, as
targets move further out of distribution?

Emits a two-panel ``corr`` / ``a`` vs OOD-distance figure and a Cp-overlay grid.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_cp(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read an ``x_over_c,Cp[,...]`` CSV into arrays (file order).

    Args:
        path (Path): CSV with header and >= 2 columns.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(x_over_c, cp)``.
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


def _metrics(target_csv: Path, olb_csv: Path) -> tuple[float, float, np.ndarray]:
    """Correlation, affine slope, and the (xo, cpo, target-on-nodes) overlay.

    Args:
        target_csv (Path): Monotonic target Cp(x/c).
        olb_csv (Path): OpenLB-measured surface Cp.

    Returns:
        tuple[float, float, np.ndarray]: ``(corr, a, stack)`` where ``stack`` is
            ``(xt, cpt, xo, cpo)`` packed for plotting.
    """
    xt, cpt = _read_cp(target_csv)
    o = np.argsort(xt)
    xt, cpt = xt[o], cpt[o]
    xo, cpo = _read_cp(olb_csv)
    cpt_n = np.interp(xo, xt, cpt)
    corr = float(np.corrcoef(cpt_n, cpo)[0, 1])
    design = np.vstack([cpt_n, np.ones_like(cpt_n)]).T
    (a, b), *_ = np.linalg.lstsq(design, cpo, rcond=None)
    return corr, float(a), (xt, cpt, xo, cpo, float(a), float(b))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-dir", type=Path, default=Path("cfd/ellipse2d/shapes/ood_sweep")
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("cfd/ellipse2d/runs"))
    parser.add_argument(
        "--out", type=Path, default=Path("docs/assets/EXP-041_ood_spread.png")
    )
    parser.add_argument(
        "--out-overlay",
        type=Path,
        default=Path("docs/assets/EXP-041_ood_cp_overlays.png"),
    )
    args = parser.parse_args()

    with (args.sweep_dir / "manifest.csv").open() as fh:
        targets = list(csv.DictReader(fh))

    rows, overlays = [], []
    for t in targets:
        tag = t["tag"]
        cp = args.runs_dir / f"ood_{tag}" / "tmp" / "ellipse2d_cp.csv"
        force = args.runs_dir / f"ood_{tag}" / "tmp" / "ellipse2d_force.csv"
        if not cp.exists():
            print(f"  skip {tag}: no cp.csv")
            continue
        corr, a, ov = _metrics(args.sweep_dir / f"ood_{tag}_target_cp.csv", cp)
        cd = float("nan")
        if force.exists():
            with force.open() as fh:
                cd = float(next(csv.DictReader(fh))["Cd_mean"])
        rows.append(
            {"tag": tag, "ood": float(t["ood_dist"]), "corr": corr, "a": a, "cd": cd}
        )
        overlays.append((tag, ov))

    rows.sort(key=lambda r: r["ood"])

    # Panel figure: corr and a vs OOD distance.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ood = [r["ood"] for r in rows]
    ax1.axhspan(0.94, 0.97, color="C2", alpha=0.12, label="in-dist band (~0.96)")
    ax1.scatter(ood, [r["corr"] for r in rows], s=80, color="C3", zorder=5)
    for r in rows:
        ax1.annotate(
            r["tag"],
            (r["ood"], r["corr"]),
            (0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )
    ax1.set_xlabel("OOD distance (min L2 to training cond)")
    ax1.set_ylabel("measured-vs-target Cp correlation ρ")
    ax1.set_title("Fidelity vs OOD-ness")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.axhline(1.78, ls="--", c="0.5", lw=1, label="validated baseline 1.78")
    ax2.axhspan(2.4, 3.0, color="C0", alpha=0.12, label="in-dist generated band")
    ax2.scatter(ood, [r["a"] for r in rows], s=80, color="C3", zorder=5)
    for r in rows:
        ax2.annotate(
            r["tag"],
            (r["ood"], r["a"]),
            (0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )
    ax2.set_xlabel("OOD distance (min L2 to training cond)")
    ax2.set_ylabel("affine slope a (Cp magnitude)")
    ax2.set_title("Magnitude creep vs OOD-ness")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.suptitle(
        "EXP-041: out-of-distribution spread (5 synthetic targets, Re 754)", y=1.02
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")

    # Overlay grid: measured vs target Cp per OOD shape.
    n = len(overlays)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.4), squeeze=False)
    for ax, (tag, (xt, cpt, xo, cpo, a, b)) in zip(axes[0], overlays, strict=True):
        ax.plot(xt, cpt, "k-", lw=2, label="target")
        ax.plot(xo, cpo, "C1.", ms=4, label="OpenLB")
        xf = np.linspace(0, 1, 200)
        ax.plot(xf, a * np.interp(xf, xt, cpt) + b, "C0--", lw=1, label="a·target+b")
        ax.invert_yaxis()
        ax.set_title(tag, fontsize=9)
        ax.set_xlabel("x/c")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("Cp")
    axes[0][-1].legend(fontsize=7)
    fig.suptitle("EXP-041: measured vs target Cp across OOD targets", y=1.03)
    fig.tight_layout()
    fig.savefig(args.out_overlay, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out_overlay}")

    print(f"\n{'tag':>10} {'ood_dist':>9} {'corr':>7} {'a':>7} {'Cd':>7}")
    for r in rows:
        print(
            f"{r['tag']:>10} {r['ood']:>9.3f} {r['corr']:>7.3f} "
            f"{r['a']:>7.3f} {r['cd']:>7.3f}"
        )


if __name__ == "__main__":
    main()
