"""Analyse the EXP-042 node-density / magnitude-creep sweep.

For each OpenLB run of the same generated geometry at a different boundary node
count (and the smoothed / higher-lattice-resolution controls), extract the
time-averaged drag ``Cd`` and the surface-``Cp`` fidelity against the shared
synthetic OOD target (correlation and affine slope ``a``, where ``a`` measures
the Cp-magnitude creep relative to the in-distribution ``a ≈ 1.78``).

Tests the hypothesis that a higher-fidelity boundary (more interpolated nodes)
lowers spurious separation, dropping ``Cd`` and pulling ``a`` toward 1.78, then
plateauing once the polygon is finer than the lattice. Emits a summary table and
a two-panel ``Cd`` / ``a`` vs node-count figure.
"""

from __future__ import annotations

import argparse
import csv
import re
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


def _cp_metrics(target_csv: Path, olb_csv: Path) -> tuple[float, float]:
    """Correlation and affine slope of measured Cp vs the target curve.

    Args:
        target_csv (Path): Monotonic target Cp(x/c).
        olb_csv (Path): OpenLB-measured surface Cp.

    Returns:
        tuple[float, float]: ``(corr, affine_a)``.
    """
    xt, cpt = _read_cp(target_csv)
    order = np.argsort(xt)
    xt, cpt = xt[order], cpt[order]
    xo, cpo = _read_cp(olb_csv)
    cpt_on_nodes = np.interp(xo, xt, cpt)
    corr = float(np.corrcoef(cpt_on_nodes, cpo)[0, 1])
    design = np.vstack([cpt_on_nodes, np.ones_like(cpt_on_nodes)]).T
    (a, _), *_ = np.linalg.lstsq(design, cpo, rcond=None)
    return corr, float(a)


def _read_force(path: Path) -> tuple[float, float]:
    """Read time-averaged ``(Cd_mean, Cd_std)`` from a force CSV.

    Args:
        path (Path): ``ellipse2d_force.csv`` (header + one summary row).

    Returns:
        tuple[float, float]: ``(Cd_mean, Cd_std)``.
    """
    with path.open() as fh:
        reader = csv.DictReader(fh)
        row = next(reader)
    return float(row["Cd_mean"]), float(row["Cd_std"])


def _node_count(run_dir: Path) -> int:
    """Recover the loaded boundary node count from a run log.

    Args:
        run_dir (Path): Per-run directory containing ``run.log``.

    Returns:
        int: Number of polygon points loaded (0 if not found).
    """
    log = run_dir / "run.log"
    if not log.exists():
        return 0
    for line in log.read_text().splitlines():
        m = re.search(r"loaded (\d+) points", line)
        if m:
            return int(m.group(1))
    return 0


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("cfd/ellipse2d/runs"),
        help="Directory holding node_*_res* run subdirs.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("cfd/ellipse2d/shapes/dps_sweep/synth_target_cp.csv"),
        help="Shared synthetic OOD target Cp curve.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/assets/EXP-042_node_sweep.png"),
        help="Output figure path.",
    )
    args = parser.parse_args()

    run_dirs = sorted(args.runs_dir.glob("node_*_res*"))
    rows = []
    for d in run_dirs:
        name = d.name[len("node_") :]
        tag, _, res = name.rpartition("_res")
        cp_csv = d / "tmp" / "ellipse2d_cp.csv"
        force_csv = d / "tmp" / "ellipse2d_force.csv"
        if not (cp_csv.exists() and force_csv.exists()):
            print(f"  skip {d.name}: missing outputs")
            continue
        cd, cd_std = _read_force(force_csv)
        corr, a = _cp_metrics(args.target, cp_csv)
        rows.append(
            {
                "tag": tag,
                "res": int(res),
                "n": _node_count(d),
                "cd": cd,
                "cd_std": cd_std,
                "corr": corr,
                "a": a,
                "smooth": tag.startswith("smooth"),
            }
        )

    # Plot the RES-160 raw series against node count; mark the controls.
    raw = sorted(
        (r for r in rows if not r["smooth"] and r["res"] == 160), key=lambda r: r["n"]
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    if raw:
        ns = [r["n"] for r in raw]
        ax1.errorbar(
            ns, [r["cd"] for r in raw], yerr=[r["cd_std"] for r in raw], marker="o"
        )
        ax1.set_xlabel("boundary node count")
        ax1.set_ylabel("time-averaged Cd")
        ax1.set_title("Drag vs boundary fidelity (RES 160 raw)")
        ax1.grid(alpha=0.3)
        ax2.plot(ns, [r["a"] for r in raw], "o-", label="raw")
        ax2.axhline(1.78, ls="--", c="grey", label="in-dist a≈1.78")
        ax2.set_xlabel("boundary node count")
        ax2.set_ylabel("affine slope a (Cp magnitude)")
        ax2.set_title("Cp magnitude creep vs fidelity")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
    # Overlay controls.
    for r in rows:
        if r["smooth"]:
            ax1.scatter([r["n"]], [r["cd"]], marker="s", c="C2", zorder=5)
            ax2.scatter([r["n"]], [r["a"]], marker="s", c="C2", zorder=5)
        if r["res"] != 160:
            ax1.scatter([r["n"]], [r["cd"]], marker="^", c="C3", zorder=5)
            ax2.scatter([r["n"]], [r["a"]], marker="^", c="C3", zorder=5)
    fig.suptitle("EXP-042 node-density sweep (synth OOD shape)", y=1.0)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")

    print(
        f"{'tag':>14} {'res':>4} {'n':>5} {'Cd':>8} {'Cd_std':>8} {'corr':>7} {'a':>7}"
    )
    for r in sorted(rows, key=lambda r: (r["smooth"], r["res"], r["n"])):
        print(
            f"{r['tag']:>14} {r['res']:>4} {r['n']:>5} {r['cd']:>8.3f} "
            f"{r['cd_std']:>8.3f} {r['corr']:>7.3f} {r['a']:>7.3f}"
        )


if __name__ == "__main__":
    main()
