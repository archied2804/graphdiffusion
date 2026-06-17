"""Analyse the EXP-042 follow-up: roughness-vs-resolution and in-dist creep.

Two questions:

* **Q2 — does under-resolving the boundary mask the surface roughness?** The
  same synthetic geometry is run rough (``raw_n240``) and Fourier-smoothed
  (``smooth_n240``) across a ladder of lattice resolutions. If a coarse lattice
  cannot resolve the surface wobble, the rough and smooth drags coincide; as the
  resolution rises and the lattice begins to resolve the wobble, the rough-minus-
  smooth ``Cd`` gap should open up. Left panel plots ``Cd`` vs RES for both;
  right panel plots the gap.

* **Q1 — does the Cp-magnitude creep appear in-distribution?** Reports the
  in-distribution shape's affine slope ``a`` across node count / resolution to
  confirm it is a model property, not a meshing or OOD-target artefact.

Pulls runs from both the original EXP-042 sweep (``node_*``) and the follow-up
(``roughres_*``, ``nodeind_*``).
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_cp(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read an ``x_over_c,Cp[,...]`` CSV into arrays.

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


def _affine_a(target_csv: Path, olb_csv: Path) -> tuple[float, float]:
    """Correlation and affine slope of measured Cp vs target.

    Args:
        target_csv (Path): Monotonic target Cp(x/c).
        olb_csv (Path): OpenLB-measured surface Cp.

    Returns:
        tuple[float, float]: ``(corr, a)``.
    """
    xt, cpt = _read_cp(target_csv)
    o = np.argsort(xt)
    xt, cpt = xt[o], cpt[o]
    xo, cpo = _read_cp(olb_csv)
    cpt_n = np.interp(xo, xt, cpt)
    corr = float(np.corrcoef(cpt_n, cpo)[0, 1])
    design = np.vstack([cpt_n, np.ones_like(cpt_n)]).T
    (a, _), *_ = np.linalg.lstsq(design, cpo, rcond=None)
    return corr, float(a)


def _cd(path: Path) -> tuple[float, float]:
    """Read ``(Cd_mean, Cd_std)`` from a force CSV.

    Args:
        path (Path): ``ellipse2d_force.csv``.

    Returns:
        tuple[float, float]: ``(Cd_mean, Cd_std)``.
    """
    with path.open() as fh:
        row = next(csv.DictReader(fh))
    return float(row["Cd_mean"]), float(row["Cd_std"])


def _res(name: str) -> int:
    """Parse the trailing ``_res<N>`` lattice resolution from a run dir name.

    Args:
        name (str): Run directory name.

    Returns:
        int: Lattice resolution (0 if absent).
    """
    m = re.search(r"_res(\d+)$", name)
    return int(m.group(1)) if m else 0


def _collect(runs_dir: Path, kind: str) -> list[dict]:
    """Collect synth roughness runs of one kind across resolutions.

    Args:
        runs_dir (Path): Directory of run subdirs.
        kind (str): ``"raw"`` or ``"smooth"``.

    Returns:
        list[dict]: ``{res, cd, cd_std}`` sorted by resolution.
    """
    out = []
    for d in runs_dir.glob(f"*{kind}_n240_res*"):
        if d.name.startswith("nodeind"):
            continue
        f = d / "tmp" / "ellipse2d_force.csv"
        if not f.exists():
            continue
        cd, std = _cd(f)
        out.append({"res": _res(d.name), "cd": cd, "cd_std": std})
    return sorted(out, key=lambda r: r["res"])


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("cfd/ellipse2d/runs"))
    parser.add_argument(
        "--synth-target",
        type=Path,
        default=Path("cfd/ellipse2d/shapes/dps_sweep/synth_target_cp.csv"),
    )
    parser.add_argument(
        "--indist-target",
        type=Path,
        default=Path("cfd/ellipse2d/shapes/generated/indist_4849_target_cp.csv"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("docs/assets/EXP-042_roughness_resolution.png")
    )
    args = parser.parse_args()

    raw = _collect(args.runs_dir, "raw")
    smooth = _collect(args.runs_dir, "smooth")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    if raw:
        ax1.errorbar(
            [r["res"] for r in raw],
            [r["cd"] for r in raw],
            yerr=[r["cd_std"] for r in raw],
            marker="o",
            label="raw (rough)",
        )
    if smooth:
        ax1.errorbar(
            [r["res"] for r in smooth],
            [r["cd"] for r in smooth],
            yerr=[r["cd_std"] for r in smooth],
            marker="s",
            label="Fourier-smooth",
        )
    ax1.set_xlabel("lattice resolution RES")
    ax1.set_ylabel("time-averaged Cd")
    ax1.set_title("Drag vs lattice resolution (synth, n=240)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Rough-minus-smooth gap at shared resolutions.
    sm = {r["res"]: r["cd"] for r in smooth}
    shared = [(r["res"], r["cd"] - sm[r["res"]]) for r in raw if r["res"] in sm]
    if shared:
        ax2.plot([s[0] for s in shared], [s[1] for s in shared], "o-", c="C3")
    ax2.axhline(0.0, ls="--", c="grey")
    ax2.set_xlabel("lattice resolution RES")
    ax2.set_ylabel("Cd(raw) − Cd(smooth)")
    ax2.set_title("Roughness drag signature vs resolution")
    ax2.grid(alpha=0.3)
    fig.suptitle("EXP-042 follow-up: does under-resolution mask roughness?", y=1.0)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")

    print("\nQ2 roughness vs resolution (synth, n=240):")
    print(f"{'RES':>5} {'Cd_raw':>8} {'Cd_smooth':>10} {'gap':>8}")
    for r in raw:
        g = r["cd"] - sm[r["res"]] if r["res"] in sm else float("nan")
        s = sm.get(r["res"], float("nan"))
        print(f"{r['res']:>5} {r['cd']:>8.3f} {s:>10.3f} {g:>8.3f}")

    print("\nQ1 in-distribution creep (gen_4849):")
    print(f"{'run':>26} {'Cd':>8} {'corr':>7} {'a':>7}")
    for d in sorted(args.runs_dir.glob("nodeind_*")):
        f = d / "tmp" / "ellipse2d_force.csv"
        cp = d / "tmp" / "ellipse2d_cp.csv"
        if not (f.exists() and cp.exists()):
            continue
        cd, _ = _cd(f)
        corr, a = _affine_a(args.indist_target, cp)
        print(f"{d.name:>26} {cd:>8.3f} {corr:>7.3f} {a:>7.3f}")


if __name__ == "__main__":
    main()
