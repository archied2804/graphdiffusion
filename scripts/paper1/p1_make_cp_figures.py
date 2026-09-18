"""
Surface-pressure comparisons for the Paper 1 reference network (P1-REF), P1-G01.

For each held-out target, three curves, none with a fitted offset:

  target geometry   the target's own body through the same solver -- the truth,
                    measured by the same instrument as the model, so every
                    shared systematic cancels
  generated         the model's shape for that target through the same solver,
                    mean of the five P1-REF seeds with their min-max as a band
  dataset           the dataset's stored curve converted to aerodynamic Cp by
                    the sourced factor of two (``DATASET_P_STAR_TO_CP``,
                    P1-F07) -- a unit conversion, not the Year-1 fitted 1.74

The bodies are symmetric at zero incidence, so each solver curve is split into
its two surfaces and averaged onto a common chordwise grid.

  fig11  six targets, chosen by rank of drag error at fixed quantiles
  fig12  all twenty targets (appendix)
  fig13  three targets -- best, median and worst by drag error
  fig14  summary of all twenty: where along the chord the pressure error sits,
         and generated against target drag
  fig15  fig13's three targets, one generated sample each (seed 0) rather than
         the five-seed mean -- what a single generation gives
  fig16  one target Cp given to three independently trained networks (P1-REF
         seeds 0-2), one per panel -- training variability, not sampling noise

Usage:
    python scripts/paper1/p1_make_cp_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from graph_diffusion.data.pOnEllipse import (  # noqa: E402
    _H5_COL_X,
    _H5_COL_Y,
    _load_h5_raw,
    _node_count,
)
from graph_diffusion.postprocessing.cp_metrics import (  # noqa: E402
    read_cp_csv,
    read_force_csv,
    read_reference_cp,
    split_branches,
)

_OUT = _ROOT / "outputs" / "paper1" / "figures"
_G01 = _ROOT / "outputs" / "paper1" / "P1-G01_holdout-inverse-design"
_SH = _ROOT / "cfd" / "ellipse2d" / "shapes" / "g01_holdout"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"

GRID = np.linspace(0.0, 1.0, 201)
SIGMA_CD = 0.04742  # frozen by P1-F04, PROTOCOL section 6
_RAW = _ROOT / "data" / "ellipse" / "raw" / "pOnEllipseInDist.h5"
SEEDS = range(5)
QUANTILES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

STYLE = {
    "font.size": 9,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "axes.grid": False,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
}


def collapsed(run: Path) -> np.ndarray:
    """Solver Cp averaged over both surfaces, on the common chordwise grid.

    Args:
        run (Path): Run directory holding ``tmp/ellipse2d_cp.csv``.

    Returns:
        np.ndarray: Cp on ``GRID``.
    """
    x, cp, _ = read_cp_csv(run / "tmp" / "ellipse2d_cp.csv")
    br = split_branches(x, cp)
    a = np.interp(GRID, br.x_a, br.cp_a)
    b = np.interp(GRID, br.x_b, br.cp_b)
    return 0.5 * (a + b)


def curves(label: str) -> dict[str, np.ndarray]:
    """All three curves for one target.

    Args:
        label (str): G01 target label, e.g. ``"holdout_t020"``.

    Returns:
        dict[str, np.ndarray]: ``truth``, ``gen`` (seeds x grid) and the
            dataset curve as ``ds_x``, ``ds_cp``.
    """
    truth = collapsed(_G01 / "truth" / f"truth_{label}")
    gen = np.stack([collapsed(_G01 / "solver" / f"{label}_s{k}") for k in SEEDS])
    ds_x, ds_cp = read_reference_cp(_SH / f"{label}_ref_cp.csv")
    return {"truth": truth, "gen": gen, "ds_x": ds_x, "ds_cp": ds_cp}


def panel(ax: plt.Axes, label: str, cd_err: float, legend: bool) -> None:
    """Draw one target's comparison.

    Args:
        ax (plt.Axes): Target axes.
        label (str): G01 target label.
        cd_err (float): Mean relative drag error for the panel tag.
        legend (bool): Attach legend labels to the artists.
    """
    c = curves(label)
    ax.fill_between(
        GRID, c["gen"].min(0), c["gen"].max(0), color=ORANGE, alpha=0.22, lw=0, zorder=1
    )
    ax.plot(
        c["ds_x"],
        c["ds_cp"],
        color=BLUE,
        lw=1.1,
        ls=(0, (4, 2)),
        zorder=2,
        label="dataset, $C_p = 2p/\\rho U^2$" if legend else None,
    )
    ax.plot(
        GRID,
        c["truth"],
        color=INK,
        lw=1.2,
        zorder=3,
        label="target geometry, same solver" if legend else None,
    )
    ax.plot(
        GRID,
        c["gen"].mean(0),
        color=ORANGE,
        lw=1.3,
        zorder=4,
        label="generated, mean of 5 seeds" if legend else None,
    )
    ax.set_xlim(0, 1)
    ax.text(
        0.97,
        0.06,
        f"#{label[-3:]}  ·  $\\Delta C_d$ {100 * cd_err:+.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.4,
        color=MUTED,
    )


def per_target() -> list[tuple[str, float]]:
    """Held-out targets with their mean relative drag error.

    Returns:
        list[tuple[str, float]]: ``(label, cd_err)`` sorted by ``|cd_err|``.
    """
    g = json.loads((_G01 / "grading.json").read_text())
    rows = [(lab, float(v["Cd_rel_err"])) for lab, v in g["per_target"].items()]
    return sorted(rows, key=lambda r: abs(r[1]))


def fig11() -> None:
    """Six targets spanning best to worst drag error."""
    rows = per_target()
    picks = [rows[round(q * (len(rows) - 1))] for q in QUANTILES]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.1), sharex=True, sharey=True)
        for i, (ax, (lab, err)) in enumerate(zip(axes.flat, picks, strict=True)):
            panel(ax, lab, err, legend=i == 0)
        axes.flat[0].invert_yaxis()  # shared y -- invert once, suction up
        for ax in axes[1]:
            ax.set_xlabel("$x/c$")
        for ax in axes[:, 0]:
            ax.set_ylabel("$C_p$")
        fig.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=3,
            frameon=False,
            handlelength=2.4,
            columnspacing=1.4,
        )
        fig.tight_layout(h_pad=0.8, w_pad=0.6)
        for ext in ("png", "pdf"):
            fig.savefig(_OUT / f"fig11_cp_holdout.{ext}", bbox_inches="tight")
        plt.close(fig)
    print("  fig11_cp_holdout.png / .pdf")


def fig12() -> None:
    """All twenty held-out targets, for the appendix."""
    rows = per_target()
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(5, 4, figsize=(6.6, 8.2), sharex=True, sharey=True)
        for i, (ax, (lab, err)) in enumerate(zip(axes.flat, rows, strict=True)):
            panel(ax, lab, err, legend=i == 0)
        axes.flat[0].invert_yaxis()  # shared y -- invert once, suction up
        for ax in axes[-1]:
            ax.set_xlabel("$x/c$")
        for ax in axes[:, 0]:
            ax.set_ylabel("$C_p$")
        fig.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=3,
            frameon=False,
            handlelength=2.4,
            columnspacing=1.4,
        )
        fig.tight_layout(h_pad=0.5, w_pad=0.4)
        for ext in ("png", "pdf"):
            fig.savefig(_OUT / f"fig12_cp_holdout_all.{ext}", bbox_inches="tight")
        plt.close(fig)
    print("  fig12_cp_holdout_all.png / .pdf")


def fig13() -> None:
    """Three targets: best, median and worst by drag error."""
    rows = per_target()
    picks = [rows[round(q * (len(rows) - 1))] for q in (0.0, 0.5, 1.0)]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.5), sharex=True, sharey=True)
        for i, (ax, (lab, err)) in enumerate(zip(axes, picks, strict=True)):
            panel(ax, lab, err, legend=i == 0)
            ax.set_xlabel("$x/c$")
        axes[0].invert_yaxis()  # shared y -- invert once, suction up
        axes[0].set_ylabel("$C_p$")
        fig.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 0.97),
            ncol=3,
            frameon=False,
            handlelength=2.4,
            columnspacing=1.4,
        )
        fig.tight_layout(w_pad=0.6)
        for ext in ("png", "pdf"):
            fig.savefig(_OUT / f"fig13_cp_three_examples.{ext}", bbox_inches="tight")
        plt.close(fig)
    print("  fig13_cp_three_examples.png / .pdf")


def fig14() -> None:
    """All twenty targets in one figure: chordwise error and drag parity."""
    rows = per_target()
    raw = _load_h5_raw(_RAW)

    d_gen, d_ds, cd_t, cd_g, cd_lo, cd_hi, thick = [], [], [], [], [], [], []
    for lab, _err in rows:
        c = curves(lab)
        d_gen.append(c["gen"].mean(0) - c["truth"])
        d_ds.append(np.interp(GRID, c["ds_x"], c["ds_cp"]) - c["truth"])
        cd_t.append(
            read_force_csv(
                _G01 / "truth" / f"truth_{lab}" / "tmp" / "ellipse2d_force.csv"
            ).cd
        )
        g = [
            read_force_csv(
                _G01 / "solver" / f"{lab}_s{k}" / "tmp" / "ellipse2d_force.csv"
            ).cd
            for k in SEEDS
        ]
        cd_g.append(np.mean(g))
        cd_lo.append(min(g))
        cd_hi.append(max(g))
        sm = raw[int(lab[-3:])][: _node_count(raw[int(lab[-3:])])]
        thick.append(np.ptp(sm[:, _H5_COL_Y]) / np.ptp(sm[:, _H5_COL_X]))

    d_gen_a, d_ds_a = np.asarray(d_gen), np.asarray(d_ds)
    cd_t_a, cd_g_a = np.asarray(cd_t), np.asarray(cd_g)
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list(
        "thickness", ["#d3e3f7", "#0f4a99"]
    )

    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(6.8, 2.9), gridspec_kw={"width_ratios": [1.35, 1.0]}
        )

        # (a) where along the chord the pressure error sits ----------------
        for d in d_gen_a:
            ax1.plot(GRID, d, color=ORANGE, lw=0.5, alpha=0.28, zorder=1)
        q25, q50, q75 = np.percentile(d_gen_a, [25, 50, 75], axis=0)
        ax1.fill_between(GRID, q25, q75, color=ORANGE, alpha=0.30, lw=0, zorder=2)
        ax1.plot(GRID, q50, color=ORANGE, lw=1.6, zorder=4, label="generated − target")
        ax1.plot(
            GRID,
            np.median(d_ds_a, axis=0),
            color=BLUE,
            lw=1.3,
            ls=(0, (4, 2)),
            zorder=3,
            label="dataset − target",
        )
        ax1.axhline(0, color=MUTED, lw=0.7, zorder=0)
        ax1.set_xlim(0, 1)
        ax1.set_xlabel("$x/c$")
        ax1.set_ylabel("$\\Delta C_p$ against target geometry")
        ax1.legend(frameon=False, loc="lower right", handlelength=2.2)
        ax1.text(
            -0.2, 1.02, "(a)", transform=ax1.transAxes, fontsize=9.5, fontweight="bold"
        )

        # (b) generated against target drag ----------------------------------
        lo = min(cd_t_a.min(), np.min(cd_lo)) - 0.04
        hi = max(cd_t_a.max(), np.max(cd_hi)) + 0.04
        xs = np.array([lo, hi])
        ax2.fill_between(
            xs, xs - 2 * SIGMA_CD, xs + 2 * SIGMA_CD, color="#ecece8", lw=0, zorder=0
        )
        ax2.plot(xs, xs, color=MUTED, lw=0.8, ls="--", zorder=1)
        ax2.errorbar(
            cd_t_a,
            cd_g_a,
            yerr=[cd_g_a - np.asarray(cd_lo), np.asarray(cd_hi) - cd_g_a],
            fmt="none",
            ecolor="#9a9a95",
            elinewidth=0.7,
            capsize=1.4,
            zorder=2,
        )
        sc = ax2.scatter(
            cd_t_a,
            cd_g_a,
            c=thick,
            cmap=ramp,
            s=30,
            edgecolor=INK,
            linewidth=0.6,
            zorder=3,
        )
        ax2.set_xlim(lo, hi)
        ax2.set_ylim(lo, hi)
        ax2.set_aspect("equal")
        ax2.set_xlabel("$C_d$, target geometry")
        ax2.set_ylabel("$C_d$, generated")
        cb = fig.colorbar(sc, ax=ax2, fraction=0.05, pad=0.03)
        cb.set_label("target $b/a$", fontsize=8.5)
        cb.ax.tick_params(labelsize=7.5)
        cb.outline.set_linewidth(0.5)
        ax2.text(
            -0.28, 1.02, "(b)", transform=ax2.transAxes, fontsize=9.5, fontweight="bold"
        )

        fig.tight_layout(w_pad=2.0)
        for ext in ("png", "pdf"):
            fig.savefig(_OUT / f"fig14_cp_summary.{ext}", bbox_inches="tight")
        plt.close(fig)
    print("  fig14_cp_summary.png / .pdf")


def fig15(seed: int = 0) -> None:
    """fig13's three targets with a single generated sample, not the seed mean.

    The seed is fixed by rule rather than chosen per target, so the figure is
    not a selection of flattering samples. The targets are fig13's, kept so the
    two figures pair; each panel's drag error is that sample's own.

    Args:
        seed (int): The P1-REF seed shown for every target.
    """
    rows = per_target()
    picks = [rows[round(q * (len(rows) - 1))][0] for q in (0.0, 0.5, 1.0)]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.5), sharex=True, sharey=True)
        for i, (ax, lab) in enumerate(zip(axes, picks, strict=True)):
            c = curves(lab)
            run = _G01 / "solver" / f"{lab}_s{seed}"
            cd_t = read_force_csv(
                _G01 / "truth" / f"truth_{lab}" / "tmp" / "ellipse2d_force.csv"
            ).cd
            cd_s = read_force_csv(run / "tmp" / "ellipse2d_force.csv").cd
            first = i == 0
            ax.plot(
                c["ds_x"],
                c["ds_cp"],
                color=BLUE,
                lw=1.1,
                ls=(0, (4, 2)),
                zorder=2,
                label="dataset, $C_p = 2p/\\rho U^2$" if first else None,
            )
            ax.plot(
                GRID,
                c["truth"],
                color=INK,
                lw=1.2,
                zorder=3,
                label="target geometry, same solver" if first else None,
            )
            ax.plot(
                GRID,
                collapsed(run),
                color=ORANGE,
                lw=1.3,
                zorder=4,
                label=f"generated, network seed {seed}" if first else None,
            )
            ax.set_xlim(0, 1)
            ax.set_xlabel("$x/c$")
            ax.text(
                0.97,
                0.06,
                f"#{lab[-3:]}  ·  $\\Delta C_d$ {100 * (cd_s / cd_t - 1):+.1f}%",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.4,
                color=MUTED,
            )
        axes[0].invert_yaxis()  # shared y -- invert once, suction up
        axes[0].set_ylabel("$C_p$")
        fig.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 0.97),
            ncol=3,
            frameon=False,
            handlelength=2.4,
            columnspacing=1.4,
        )
        fig.tight_layout(w_pad=0.6)
        for ext in ("png", "pdf"):
            fig.savefig(
                _OUT / f"fig15_cp_three_examples_single_sample.{ext}",
                bbox_inches="tight",
            )
        plt.close(fig)
    print("  fig15_cp_three_examples_single_sample.png / .pdf")


def fig16(seeds: tuple[int, ...] = (0, 1, 2)) -> None:
    """One target Cp, three independently trained networks -- one per panel.

    Every panel is asked for the same pressure curve. Each P1-REF seed is a
    separately TRAINED network, and the sampler fixes its noise seed at 0, so
    the spread across panels is training variability, not the stochasticity of
    repeated sampling from one network. The target is fig13's
    median by drag error, and the seeds are the first three, both fixed by rule
    so the panels are not a selection of flattering samples.

    Args:
        seeds (tuple[int, ...]): P1-REF seeds, one per panel.
    """
    rows = per_target()
    lab = rows[round(0.5 * (len(rows) - 1))][0]
    c = curves(lab)
    cd_t = read_force_csv(
        _G01 / "truth" / f"truth_{lab}" / "tmp" / "ellipse2d_force.csv"
    ).cd
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            1, len(seeds), figsize=(6.6, 2.5), sharex=True, sharey=True
        )
        for i, (ax, seed) in enumerate(zip(axes, seeds, strict=True)):
            run = _G01 / "solver" / f"{lab}_s{seed}"
            cd_s = read_force_csv(run / "tmp" / "ellipse2d_force.csv").cd
            first = i == 0
            ax.plot(
                c["ds_x"],
                c["ds_cp"],
                color=BLUE,
                lw=1.1,
                ls=(0, (4, 2)),
                zorder=2,
                label="dataset, $C_p = 2p/\\rho U^2$" if first else None,
            )
            ax.plot(
                GRID,
                c["truth"],
                color=INK,
                lw=1.2,
                zorder=3,
                label="target geometry, same solver" if first else None,
            )
            ax.plot(
                GRID,
                collapsed(run),
                color=ORANGE,
                lw=1.3,
                zorder=4,
                label="generated, one network" if first else None,
            )
            ax.set_xlim(0, 1)
            ax.set_xlabel("$x/c$")
            ax.text(
                0.97,
                0.06,
                f"network seed {seed}  ·  "
                f"$\\Delta C_d$ {100 * (cd_s / cd_t - 1):+.1f}%",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.4,
                color=MUTED,
            )
        axes[0].invert_yaxis()  # shared y -- invert once, suction up
        axes[0].set_ylabel("$C_p$")
        fig.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 0.97),
            ncol=3,
            frameon=False,
            handlelength=2.4,
            columnspacing=1.4,
        )
        fig.tight_layout(w_pad=0.6)
        for ext in ("png", "pdf"):
            fig.savefig(
                _OUT / f"fig16_cp_same_target_three_samples.{ext}",
                bbox_inches="tight",
            )
        plt.close(fig)
    print(f"  fig16_cp_same_target_three_samples.png / .pdf   (target #{lab[-3:]})")


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    print(f"\nCp figures -> {_OUT.relative_to(_ROOT)}\n")
    fig11()
    fig12()
    fig13()
    fig14()
    fig15()
    fig16()
    print()


if __name__ == "__main__":
    main()
