"""Generate explanatory figures for the EXP-04x physical-validation series.

Produces four documentation figures from the OpenLB run outputs already on disk,
each embedded in the corresponding experiment log:

* ``EXP-040_domain_schematic`` — the lattice-Boltzmann domain and boundary-
  condition layout (inlet, free-stream walls, body, sponge zone, pressure
  outlet).
* ``EXP-040_stability_trace`` — avg-density and drag time-traces *before* vs
  *after* the sponge/rest-init acoustic fix (Re 1000 case), showing the ±15%
  density "breathing" collapse to a quiescent regime.
* ``EXP-041_correlation_bars`` — inverse-design correlation per target,
  raw vs Fourier-smoothed, grouped in-distribution vs out-of-distribution.
* ``EXP-042_creep_ladder`` — the affine slope ``a`` (Cp-magnitude) across the
  whole series: validated dataset shapes (~1.78) → in-distribution generated
  (~2.5-3.0) → out-of-distribution generated (~3.5).

The canonical sponge-fixed baseline runs live in ``gpu_sponge_*`` / ``sponge_smoke``
(the ``sim_*`` dirs are stale pre-sponge runs), so the run dirs are mapped
explicitly below rather than by the default naming.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

RUNS = Path("cfd/ellipse2d/runs")
BASE = Path("cfd/ellipse2d/shapes/baseline")
GEN = Path("cfd/ellipse2d/shapes/generated")
ASSETS = Path("docs/assets")

# Canonical sponge-fixed baseline runs (NOT the stale sim_* dirs).
BASELINE = {
    500: ("sponge_smoke", "sim_0"),
    754: ("gpu_sponge_2850", "sim_2850"),
    1000: ("gpu_sponge_5700", "sim_5700"),
}


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


def _corr_a(target_csv: Path, olb_csv: Path) -> tuple[float, float]:
    """Correlation and affine slope of measured Cp vs a target/reference curve.

    Args:
        target_csv (Path): Monotonic target/reference Cp(x/c).
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


def _trace(log: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse (physT, avgRho) and (t, Cd) time-traces from an OpenLB run log.

    Args:
        log (Path): ``run.log`` with LatticeStatistics and getResults lines.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: ``(t_rho, rho, cd)`` where
            ``cd`` is aligned to the LatticeStatistics times by nearest getResults.
    """
    t_rho, rho, t_cd, cd = [], [], [], []
    rho_re = re.compile(r"physT=([\d.eE+-]+);.*avgRho=([\d.eE+-]+)")
    cd_re = re.compile(r"getResults\] t=([\d.eE+-]+); Cd=([\d.eE+-]+)")
    for line in log.read_text().splitlines():
        m = rho_re.search(line)
        if m:
            t_rho.append(float(m.group(1)))
            rho.append(float(m.group(2)))
            continue
        m = cd_re.search(line)
        if m:
            t_cd.append(float(m.group(1)))
            cd.append(float(m.group(2)))
    t_rho, rho = np.array(t_rho), np.array(rho)
    t_cd, cd = np.array(t_cd), np.array(cd)
    cd_on_rho = np.interp(t_rho, t_cd, cd) if len(t_cd) else np.full_like(t_rho, np.nan)
    return t_rho, rho, cd_on_rho


def domain_schematic() -> None:
    """Draw the LBM domain and boundary-condition layout (to-scale-ish)."""
    # Geometry (chords): upstream 6, body 1, downstream 17 -> length 24; H=5.3.
    length, height, sponge_x0 = 24.0, 5.3, 20.0
    body_cx, body_a, body_b = 6.5, 0.5, 0.255
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.add_patch(mpatches.Rectangle((0, 0), length, height, fc="#eef4fb", ec="none"))
    ax.add_patch(
        mpatches.Rectangle(
            (sponge_x0, 0), length - sponge_x0, height, fc="#fde9d9", ec="none"
        )
    )
    ax.add_patch(
        mpatches.Ellipse((body_cx, height / 2), 2 * body_a, 2 * body_b, fc="0.4")
    )
    # Boundary labels.
    ax.annotate(
        "velocity\ninlet\n(u=1)",
        (0, height / 2),
        (-2.7, height / 2),
        va="center",
        ha="center",
        fontsize=9,
        color="C0",
    )
    ax.annotate(
        "pressure\noutlet\n(p=0)",
        (length, height / 2),
        (length + 2.6, height / 2),
        va="center",
        ha="center",
        fontsize=9,
        color="C3",
    )
    ax.text(
        length / 2,
        height + 0.35,
        "free-stream Dirichlet  (u=1, v=0)",
        ha="center",
        fontsize=9,
        color="C2",
    )
    ax.text(
        length / 2,
        -1.15,
        "free-stream Dirichlet  (u=1, v=0)",
        ha="center",
        fontsize=9,
        color="C2",
    )
    ax.text(
        sponge_x0 + (length - sponge_x0) / 2,
        height + 0.35,
        "sponge (mat 6)",
        ha="center",
        fontsize=9,
        color="#b5651d",
    )
    ax.annotate(
        "Bouzidi\nno-slip body",
        (body_cx, height / 2),
        (body_cx, height / 2 + 1.4),
        ha="center",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="0.3"),
    )
    ax.annotate(
        "",
        (0.2, -0.2),
        (length - 0.2, -0.2),
        arrowprops=dict(arrowstyle="<->", color="0.5"),
    )
    ax.text(
        length / 2,
        -0.45,
        "24 chords (6 up · body · 17 down)",
        ha="center",
        fontsize=8,
        color="0.4",
    )
    ax.set_xlim(-5, length + 5)
    ax.set_ylim(-1.6, height + 0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("EXP-040: 2D OpenLB domain & boundary conditions", fontsize=11)
    fig.tight_layout()
    fig.savefig(ASSETS / "EXP-040_domain_schematic.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote EXP-040_domain_schematic.png")


def stability_trace() -> None:
    """Before/after acoustic-fix avgRho and Cd traces (Re 1000)."""
    before = _trace(RUNS / "gpu_5700" / "run.log")
    after = _trace(RUNS / "gpu_sponge_5700" / "run.log")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for (t, rho, cd), lbl, c in [
        (before, "no sponge (breathing)", "C3"),
        (after, "sponge + rest-init", "C2"),
    ]:
        ax1.plot(t, rho, c, lw=1, label=lbl)
        ax2.plot(t, cd, c, lw=0.8, label=lbl)
    ax1.axhline(1.0, ls=":", c="0.6")
    ax1.set_xlabel("physical time")
    ax1.set_ylabel("avg lattice density ⟨ρ⟩")
    ax1.set_title("Acoustic breathing → quiescent")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.set_xlabel("physical time")
    ax2.set_ylabel("drag coefficient Cd")
    ax2.set_title("Cd trace (the old 'noise' was acoustics)")
    ax2.set_ylim(-2, 6)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.suptitle(
        "EXP-040: sponge + rest-init kills the domain acoustic resonance (Re 1000)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "EXP-040_stability_trace.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote EXP-040_stability_trace.png")


def correlation_bars() -> None:
    """Inverse-design correlation per target, raw vs smoothed."""
    targets = [
        ("#4849\n(in-dist)", "indist_4849", "gen_4849"),
        ("#4066\n(in-dist)", "indist_4066", "gen_4066"),
        ("synthetic\n(OOD)", "synth_ood", "gen_synth"),
    ]
    raw, smooth = [], []
    for _, tgt, run in targets:
        t = GEN / f"{tgt}_target_cp.csv"
        raw.append(_corr_a(t, RUNS / f"{run}_raw" / "tmp" / "ellipse2d_cp.csv")[0])
        smooth.append(
            _corr_a(t, RUNS / f"{run}_smooth" / "tmp" / "ellipse2d_cp.csv")[0]
        )
    x = np.arange(len(targets))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - w / 2, raw, w, label="raw", color="C0")
    ax.bar(x + w / 2, smooth, w, label="Fourier-smoothed", color="C1")
    for xi, (r, s) in enumerate(zip(raw, smooth, strict=True)):
        ax.text(xi - w / 2, r + 0.01, f"{r:.3f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, s + 0.01, f"{s:.3f}", ha="center", fontsize=8)
    ax.axhline(0.95, ls="--", c="0.5", lw=1, label="in-dist target ≈0.95")
    ax.set_xticks(x)
    ax.set_xticklabels([t[0] for t in targets])
    ax.set_ylabel("measured-vs-target Cp correlation ρ")
    ax.set_ylim(0.7, 1.0)
    ax.set_title("EXP-041: inverse-design fidelity (independent OpenLB grade)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "EXP-041_correlation_bars.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote EXP-041_correlation_bars.png")


def creep_ladder() -> None:
    """Affine slope a across the series: dataset shapes → in-dist gen → OOD gen."""
    pts = []  # (label, a, group)
    for re_, (run, refid) in BASELINE.items():
        _, a = _corr_a(
            BASE / f"{refid}_ref_cp.csv", RUNS / run / "tmp" / "ellipse2d_cp.csv"
        )
        pts.append((f"Re{re_}", a, 0))
    for lbl, tgt, run in [
        ("#4849", "indist_4849", "gen_4849_raw"),
        ("#4066", "indist_4066", "gen_4066_raw"),
    ]:
        _, a = _corr_a(
            GEN / f"{tgt}_target_cp.csv", RUNS / run / "tmp" / "ellipse2d_cp.csv"
        )
        pts.append((lbl, a, 1))
    _, a = _corr_a(
        GEN / "synth_ood_target_cp.csv",
        RUNS / "gen_synth_raw" / "tmp" / "ellipse2d_cp.csv",
    )
    pts.append(("synthetic", a, 2))

    colors = {0: "C2", 1: "C0", 2: "C3"}
    names = {
        0: "dataset shapes\n(validated)",
        1: "in-dist generated",
        2: "OOD generated",
    }
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for g in (0, 1, 2):
        xs = [i for i, p in enumerate(pts) if p[2] == g]
        ys = [pts[i][1] for i in xs]
        ax.scatter(xs, ys, s=90, color=colors[g], label=names[g], zorder=5)
        for i in xs:
            ax.annotate(
                f"{pts[i][1]:.2f}",
                (i, pts[i][1]),
                (0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    ax.axhline(1.78, ls="--", c="0.5", lw=1)
    ax.text(
        len(pts) - 0.5,
        1.78,
        "validated baseline a≈1.78",
        va="bottom",
        ha="right",
        fontsize=8,
        color="0.4",
    )
    ax.set_xticks(range(len(pts)))
    ax.set_xticklabels([p[0] for p in pts])
    ax.set_ylabel("affine slope a  (measured Cp / target Cp)")
    ax.set_title(
        "EXP-042: Cp-magnitude creep is model-side — grows from the dataset baseline"
    )
    ax.set_ylim(1.3, 3.8)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "EXP-042_creep_ladder.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote EXP-042_creep_ladder.png")


def main() -> None:
    """Generate all four EXP-04x documentation figures."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    domain_schematic()
    stability_trace()
    correlation_bars()
    creep_ladder()


if __name__ == "__main__":
    main()
