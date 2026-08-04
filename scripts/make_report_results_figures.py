"""Regenerate the Results-section (§4) report figures in one consistent style.

Single source of truth for the figures embedded in the year-1 report Results
section, so they share fonts, sizes, palette, and (deliberately) carry no
redundant in-plot titles --- the LaTeX/Markdown caption is the title. All data
is read from the OpenLB run outputs already on disk; no model or solver is
re-run. Writes PNG + PDF to ``docs/assets/`` under the names the report embeds.

Run from the repository root::

    python scripts/make_report_results_figures.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for sibling scripts
from _report_style import NAVY, ORANGE, PALETTE, apply_house_style  # noqa: E402

from graph_diffusion.data.pOnEllipse import _load_h5_raw  # noqa: E402
from graph_diffusion.visualisation import sweep_results as sr  # noqa: E402

CFD = Path("cfd/ellipse2d")
RUNS = CFD / "runs"
ASSETS = Path("docs/assets")
H5 = Path("data/ellipse/raw/pOnEllipseTrain.h5")

# Validated in-distribution solver normalisation (mean affine slope over the
# 15-case eval_grid; see §4.3). Used as the reference line on creep plots.
BASELINE_A = 1.74

_REP_COLORS = [PALETTE["navy"], PALETTE["green"], PALETTE["red"]]


def _save(fig: plt.Figure, stem: str) -> None:
    """Write a figure to ``docs/assets/<stem>`` as PNG and PDF."""
    fig.savefig(ASSETS / f"{stem}.png")
    fig.savefig(ASSETS / f"{stem}.pdf")
    plt.close(fig)
    print(f"wrote {stem}.png/.pdf")


def _read_cp(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read ``x_over_c,Cp[,...]`` in file order."""
    xs, cps = [], []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                xs.append(float(row[0]))
                cps.append(float(row[1]))
    return np.array(xs), np.array(cps)


def _read_cp_rms(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read ``x_over_c,Cp,Cp_rms`` in file order."""
    xs, cps, rms = [], [], []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader)
        for row in reader:
            if len(row) >= 3:
                xs.append(float(row[0]))
                cps.append(float(row[1]))
                rms.append(float(row[2]))
    return np.array(xs), np.array(cps), np.array(rms)


def _read_ref(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a reference/target Cp, returned strictly increasing in x/c."""
    x, cp = _read_cp(path)
    order = np.argsort(x, kind="stable")
    x, cp = x[order], cp[order]
    keep = np.concatenate(([True], np.diff(x) > 0))
    return x[keep], cp[keep]


def _read_cd(force_path: Path) -> tuple[float, float]:
    """Read ``(Cd_mean, Cd_std)`` from an OpenLB force CSV."""
    with force_path.open() as fh:
        next(fh)
        row = next(csv.reader(fh))
    return float(row[0]), float(row[1])


def _match(
    target_x: np.ndarray,
    target_cp: np.ndarray,
    meas_x: np.ndarray,
    meas_cp: np.ndarray,
) -> dict[str, float]:
    """Corrected Cp-match metrics (target interpolated onto measured nodes)."""
    return sr.cp_match_metrics(target_x, target_cp, meas_x, meas_cp)


# ---------------------------------------------------------------------------
# §4.3 figures
# ---------------------------------------------------------------------------
# Dataset's documented velocity normalisation ("u / 1.8"); the single physical
# constant the solver Cp differs from the dataset reference by.
_DATASET_CP_SCALE = 1.8


def cp_validation(scaling: str = "fit") -> None:
    """Measured vs dataset Cp for three known ellipses (corrected channel H).

    The OpenLB surface Cp differs from the dataset reference essentially by the
    single multiplicative normalisation ``meas ~ a*ref`` where ``a`` is the
    dataset's documented ~1.8 velocity scaling, not a solver error. The Pearson
    ``rho`` is affine-invariant, so the scaling choice below changes only what
    the eye sees, never the reported correlation.

    Args:
        scaling (str): How to map the measured Cp before overlaying it:

            * ``"fit"`` (default): per-case least-squares inverse ``(meas - b)/a``
              (``EXP-043_cp_validation``). Tightest overlay, but ``a``/``b`` are
              fit to the very curves shown.
            * ``"fixed"``: divide every case by the single documented constant
              ``a = 1.8``, ``b = 0`` (``EXP-043_cp_validation_fixed18``). Uses the
              physical normalisation, not a per-curve fit -- the most defensible
              overlay; residual misalignment is honest.
            * ``"none"``: raw measured Cp in native solver units
              (``EXP-043_cp_validation_raw``), a factor ~1.8 off the reference.
    """
    if scaling not in {"fit", "fixed", "none"}:
        raise ValueError(f"scaling must be 'fit', 'fixed' or 'none'; got {scaling!r}")
    cases = [
        ("re0503_h55_idx0059", 59, 503),
        ("re0750_h55_idx2824", 2824, 750),
        ("re0998_h55_idx5645", 5645, 998),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for (name, idx, re), c in zip(cases, _REP_COLORS, strict=True):
        mx, mcp = _read_cp(RUNS / name / "tmp" / "ellipse2d_cp.csv")
        tx, tcp = _read_ref(CFD / "shapes" / "eval_grid" / f"sim_{idx}_ref_cp.csv")
        m = _match(tx, tcp, mx, mcp)
        if scaling == "fit":
            plotted = (mcp - m["affine_b"]) / m["affine_a"]
        elif scaling == "fixed":
            plotted = mcp / _DATASET_CP_SCALE
        else:
            plotted = mcp
        ax.scatter(mx, plotted, s=10, color=c, alpha=0.6, edgecolors="none")
        ax.plot(
            tx,
            tcp,
            "-",
            color=c,
            lw=1.8,
            label=rf"$Re={re}$:  $\rho={m['corr']:.3f}$, $a={m['affine_a']:.2f}$",
        )
    meas_label = {
        "fit": "OpenLB measured (per-case affine, surface nodes)",
        "fixed": rf"OpenLB measured ($\div {_DATASET_CP_SCALE:g}$, surface nodes)",
        "none": "OpenLB measured (raw, surface nodes)",
    }[scaling]
    ax.scatter([], [], s=16, color="0.35", label=meas_label)
    ax.plot([], [], "-", color="0.35", lw=1.8, label="dataset reference")
    ax.set_xlabel("$x/c$")
    ax.set_ylabel("$C_p$")
    ax.invert_yaxis()
    ax.legend(loc="lower center", ncol=2)
    fig.tight_layout()
    stem = {
        "fit": "EXP-043_cp_validation",
        "fixed": "EXP-043_cp_validation_fixed18",
        "none": "EXP-043_cp_validation_raw",
    }[scaling]
    _save(fig, stem)


def cp_stability() -> None:
    """Per-node time-averaged Cp with ±temporal-RMS bars (Re=750, H=5.5)."""
    name = "re0750_h55_idx2824"
    x, cp, rms = _read_cp_rms(RUNS / name / "tmp" / "ellipse2d_cp.csv")
    cd_mean, cd_std = _read_cd(RUNS / name / "tmp" / "ellipse2d_force.csv")
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.errorbar(
        x,
        cp,
        yerr=rms,
        fmt="o",
        ms=3.5,
        color=NAVY,
        ecolor="#8fb0cf",
        elinewidth=1.0,
        capsize=1.5,
        label=r"per-node $C_p$ $\pm$ temporal RMS",
    )
    ax.set_xlabel("$x/c$")
    ax.set_ylabel("$C_p$")
    ax.invert_yaxis()
    ax.annotate(
        rf"node RMS $\leq {rms.max():.2f}$,   $C_d = {cd_mean:.2f} \pm {cd_std:.2f}$",
        xy=(0.5, 0.04),
        xycoords="axes fraction",
        ha="center",
        fontsize=9,
        color="0.3",
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, "EXP-043_cp_stability")


# ---------------------------------------------------------------------------
# §4.4 figure
# ---------------------------------------------------------------------------
def cp_inverse_design() -> None:
    """In-distribution generated Cp (measured nodes) vs conditioning target."""
    gen = CFD / "shapes" / "generated"
    cases = [
        ("#4849", "indist_4849", "gen_4849_raw", PALETTE["navy"]),
        ("#4066", "indist_4066", "gen_4066_raw", PALETTE["green"]),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for label, tgt, run, c in cases:
        tx, tcp = _read_ref(gen / f"{tgt}_target_cp.csv")
        mx, mcp = _read_cp(RUNS / run / "tmp" / "ellipse2d_cp.csv")
        m = _match(tx, tcp, mx, mcp)
        ax.scatter(mx, mcp, s=10, color=c, alpha=0.6, edgecolors="none")
        ax.plot(
            tx,
            tcp,
            "-",
            color=c,
            lw=1.8,
            label=rf"target {label}:  $\rho={m['corr']:.3f}$",
        )
    ax.scatter([], [], s=16, color="0.35", label="generated shape (OpenLB measured)")
    ax.plot([], [], "-", color="0.35", lw=1.8, label="conditioning target")
    ax.set_xlabel("$x/c$")
    ax.set_ylabel("$C_p$")
    ax.invert_yaxis()
    ax.legend(loc="lower center", ncol=2)
    fig.tight_layout()
    _save(fig, "EXP-041_cp_inverse_design")


# ---------------------------------------------------------------------------
# §4.5 figures
# ---------------------------------------------------------------------------
def creep_ladder() -> None:
    """Affine slope a across dataset → in-dist generated → OOD generated."""
    base_dir = CFD / "shapes" / "baseline"
    gen = CFD / "shapes" / "generated"
    baseline = {500: "sim_0", 754: "sim_2850", 1000: "sim_5700"}
    baseline_runs = {
        500: "sponge_smoke",
        754: "gpu_sponge_2850",
        1000: "gpu_sponge_5700",
    }

    pts: list[tuple[str, float, int]] = []
    for re, sim in baseline.items():
        tx, tcp = _read_ref(base_dir / f"{sim}_ref_cp.csv")
        mx, mcp = _read_cp(RUNS / baseline_runs[re] / "tmp" / "ellipse2d_cp.csv")
        pts.append((f"$Re${re}", _match(tx, tcp, mx, mcp)["affine_a"], 0))
    for label, tgt, run in [
        ("#4849", "indist_4849", "gen_4849_raw"),
        ("#4066", "indist_4066", "gen_4066_raw"),
    ]:
        tx, tcp = _read_ref(gen / f"{tgt}_target_cp.csv")
        mx, mcp = _read_cp(RUNS / run / "tmp" / "ellipse2d_cp.csv")
        pts.append((label, _match(tx, tcp, mx, mcp)["affine_a"], 1))
    tx, tcp = _read_ref(gen / "synth_ood_target_cp.csv")
    mx, mcp = _read_cp(RUNS / "gen_synth_raw" / "tmp" / "ellipse2d_cp.csv")
    pts.append(("synthetic", _match(tx, tcp, mx, mcp)["affine_a"], 2))

    colors = {0: PALETTE["green"], 1: PALETTE["navy"], 2: PALETTE["red"]}
    names = {
        0: "dataset shapes (validated)",
        1: "in-distribution generated",
        2: "out-of-distribution generated",
    }
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for g in (0, 1, 2):
        xs = [i for i, p in enumerate(pts) if p[2] == g]
        ys = [pts[i][1] for i in xs]
        ax.scatter(xs, ys, s=80, color=colors[g], label=names[g], zorder=5)
        for i in xs:
            ax.annotate(
                f"{pts[i][1]:.2f}",
                (i, pts[i][1]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )
    ax.axhline(BASELINE_A, ls="--", color="0.5", lw=1)
    ax.annotate(
        f"validated solver baseline $a={BASELINE_A:.2f}$",
        xy=(len(pts) - 1, BASELINE_A),
        xytext=(0, -14),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color="0.4",
    )
    ax.set_xticks(range(len(pts)))
    ax.set_xticklabels([p[0] for p in pts])
    ax.set_ylabel(r"affine slope $a$  (measured $C_p$ / target $C_p$)")
    ax.legend(loc="upper left")
    ax.grid(axis="y")
    fig.tight_layout()
    _save(fig, "EXP-042_creep_ladder")


def dps_sweep() -> None:
    """Measured Cp vs the shared OOD target across DPS guidance weights."""
    sweep = CFD / "shapes" / "dps_sweep"
    target_x, target_cp = _read_ref(sweep / "synth_target_cp.csv")
    weights = []
    with (sweep / "manifest.csv").open() as fh:
        for row in csv.DictReader(fh):
            weights.append((float(row["dps"]), row["tag"]))
    n = len(weights)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.8), squeeze=False, sharey=True)
    for ax, (dps, tag) in zip(axes[0], weights, strict=True):
        mx, mcp = _read_cp(RUNS / f"dps_{tag}" / "tmp" / "ellipse2d_cp.csv")
        m = _match(target_x, target_cp, mx, mcp)
        ax.plot(target_x, target_cp, "-", color="0.35", lw=1.6, label="target")
        ax.scatter(
            mx, mcp, s=10, color=ORANGE, alpha=0.7, edgecolors="none", label="OpenLB"
        )
        ax.invert_yaxis()
        ax.set_xlabel("$x/c$")
        ax.annotate(
            rf"DPS $={dps:g}$" "\n" rf"$\rho={m['corr']:.3f}$, $a={m['affine_a']:.2f}$",
            xy=(0.5, 0.93),
            xycoords="axes fraction",
            ha="center",
            va="top",
            fontsize=8.5,
        )
    axes[0][0].set_ylabel("$C_p$")
    axes[0][-1].legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "EXP-041_dps_sweep")


# ---------------------------------------------------------------------------
# §4.3 / §4.5 figures via the sweep_results library (titles off)
# ---------------------------------------------------------------------------
def sweep_library_figures() -> None:
    """Coverage / manifold / results-summary figures, restyled, titles off."""
    import plot_sweep_results as psr  # sibling script; uses CFD run outputs

    raw = _load_h5_raw(H5)
    eval_recs = psr._load_eval_grid()
    ood_recs = psr._load_ood_systematic()

    _save(
        sr.plot_design_space_context(raw, eval_recs, title=False),
        "EXP-043_eval_grid_coverage",
    )
    _save(
        sr.plot_results_summary(ood_recs, title=False, baseline_a=BASELINE_A),
        "EXP-043_ood_results_summary",
    )
    _save(
        sr.plot_results_summary(
            ood_recs, title=False, baseline_a=BASELINE_A, stacked=True
        ),
        "EXP-043_ood_results_summary_stacked",
    )
    _save(
        sr.plot_cp_manifold_context(raw, ood_recs, title=False),
        "EXP-043_ood_cp_manifold",
    )


def main() -> None:
    """Regenerate every Results-section figure in the house style."""
    apply_house_style()
    ASSETS.mkdir(parents=True, exist_ok=True)
    cp_validation()
    cp_validation(scaling="fixed")
    cp_validation(scaling="none")
    cp_stability()
    cp_inverse_design()
    creep_ladder()
    dps_sweep()
    sweep_library_figures()
    print("done.")


if __name__ == "__main__":
    main()
