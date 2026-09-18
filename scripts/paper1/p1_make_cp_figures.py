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

from graph_diffusion.postprocessing.cp_metrics import (  # noqa: E402
    read_cp_csv,
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


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    print(f"\nCp figures -> {_OUT.relative_to(_ROOT)}\n")
    fig11()
    fig12()
    print()


if __name__ == "__main__":
    main()
