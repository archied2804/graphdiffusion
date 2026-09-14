"""
Figures for the C1 framing decision.

Three questions, one figure each:

  fig1  Can the instrument tell a one-mode PCA reconstruction from the real
        shape?  (P1-B03 section 10)
  fig2  Does the model beat nearest-neighbour retrieval on held-out targets?
        (P1-G01 section 3 against P1-B01)
  fig3  What does a rho value actually mean on this corpus?
        (P1-F04 section 4)

Palette: dataviz categorical slots 1 and 2, validated all-pairs in light mode.
The noise floor is drawn as a shaded band everywhere a difference is claimed,
because on this corpus no difference means anything without it.

Usage:
    python scripts/paper1/p1_make_framing_figures.py
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))


_OUT = _ROOT / "outputs" / "paper1" / "figures"
_B03 = _ROOT / "outputs" / "paper1" / "P1-B03_basis-reachability"
_B03_SH = _ROOT / "cfd" / "ellipse2d" / "shapes" / "b03_recon"
_B01 = _ROOT / "outputs" / "paper1" / "P1-B01_retrieval-baseline"
_G01 = _ROOT / "outputs" / "paper1" / "P1-G01_holdout-inverse-design"

BLUE = "#2a78d6"  # reference / retrieval
ORANGE = "#eb6834"  # test / model
INK = "#0b0b0b"
MUTED = "#52514e"
BAND = "#9a9a95"

SIGMA_RHO = 0.00108
RHO_CHANCE = 0.9954

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": "#e6e6e2",
        "grid.linewidth": 0.6,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 200,
    }
)


def read_body(p: Path) -> np.ndarray:
    """Read a solver body file as an (N, 2) point array.

    Args:
        p (Path): The ``.dat`` file.

    Returns:
        np.ndarray: Points, shape ``(N, 2)``.
    """
    return np.asarray(
        [
            tuple(map(float, ln.split()))
            for ln in p.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
    )


def fig1() -> None:
    """One mode is inside the noise floor -- geometry and graded difference."""
    man = [
        {**r, "max_dev_norm": float(r["max_dev_norm"])}
        for r in csv.DictReader((_B03_SH / "manifest.csv").open())
    ]
    labels = sorted({r["label"] for r in man})
    tol = json.loads((_B03 / "solver_tolerance.json").read_text())

    fig = plt.figure(figsize=(7.2, 4.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.15], hspace=0.55, wspace=0.35)

    # --- top: the shapes themselves -----------------------------------
    for i, lab in enumerate(labels):
        ax = fig.add_subplot(gs[0, i])
        o = read_body(_B03_SH / f"{lab}_orig.dat")
        r = read_body(_B03_SH / f"{lab}_pca01.dat")
        for arr, c, lw, z in ((o, BLUE, 2.0, 2), (r, ORANGE, 1.2, 3)):
            xy = np.vstack([arr, arr[:1]])
            ax.plot(xy[:, 0], xy[:, 1], color=c, lw=lw, zorder=z)
        dev = float(
            next(
                r["max_dev_norm"]
                for r in man
                if r["label"] == lab and r["variant"] == "pca01"
            )
        )
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(lab.split("_")[0], fontsize=7.5, color=MUTED, pad=2)
        ax.text(
            0.5,
            -0.06,
            f"max gap {100 * dev:.2f}% of radius",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=6.8,
            color=MUTED,
        )
    fig.text(
        0.5,
        0.955,
        "One PCA mode is inside the solver's measurement uncertainty",
        ha="center",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.915,
        "Top: held-out shape (blue) with its 1-mode reconstruction (orange) overlaid.  "
        "Bottom: the graded difference, against the frozen noise floor.",
        ha="center",
        fontsize=7.6,
        color=MUTED,
    )

    # --- bottom: the graded difference vs the floor --------------------
    ax = fig.add_subplot(gs[1, :])
    levels = ["roundtrip", "pca01", "pca02", "pca03", "pca10"]
    names = {
        "roundtrip": "resample\nonly",
        "pca01": "M = 1",
        "pca02": "M = 2",
        "pca03": "M = 3",
        "pca10": "M = 10",
    }
    runs = tol["runs"]
    ax.axhspan(0, 2 * SIGMA_RHO, color=BAND, alpha=0.20, zorder=0)
    ax.axhline(2 * SIGMA_RHO, color=BAND, lw=1.0, ls="--", zorder=1)
    for j, lv in enumerate(levels):
        vals = [abs(r["d_rho"]) for r in runs if r["variant"] == lv]
        ax.scatter(
            [j] * len(vals),
            vals,
            s=34,
            color=ORANGE,
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([names[v] for v in levels])
    ax.set_xlim(-0.5, len(levels) - 0.5)
    ax.set_ylim(0, max(2.6 * SIGMA_RHO, 0.0017))
    ax.set_ylabel(r"|change in $\rho$| vs the true shape")
    ax.text(
        len(levels) - 0.55,
        2 * SIGMA_RHO * 0.46,
        "below this line the instrument\ncannot tell them apart  (2$\\sigma$)",
        fontsize=7.2,
        color=MUTED,
        ha="right",
        va="center",
    )
    ax.set_title(
        "Every truncation level, every shape, sits inside the floor",
        fontsize=8.5,
        loc="left",
        pad=6,
    )
    fig.savefig(_OUT / "fig1_one_mode_is_enough.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig1_one_mode_is_enough.png")


def fig2() -> None:
    """Retrieval beats the model on held-out targets -- paired."""
    g = json.loads((_G01 / "grading.json").read_text())
    b = json.loads((_B01 / "retrieval_grading.json").read_text())
    ret = {int(r["target_idx"]): r["rho"] for r in b["runs"] if r["set"] == "indist"}
    pairs = []
    for lab, v in g["per_target"].items():
        ti = int(lab.split("_t")[1])
        if ti in ret:
            pairs.append((lab, ret[ti], v["rho"]))
    pairs.sort(key=lambda p: p[1])
    rx = np.array([p[1] for p in pairs])
    my = np.array([p[2] for p in pairs])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.5))
    fig.suptitle(
        "A lookup table beats the model on 18 of 20 held-out targets",
        fontsize=10.5,
        fontweight="bold",
        y=1.02,
    )

    lo = min(rx.min(), my.min()) - 0.001
    hi = 1.0002
    ax1.plot([lo, hi], [lo, hi], color=MUTED, lw=1.0, ls="--", zorder=1)
    ax1.fill_between(
        [lo, hi],
        [lo - 2 * SIGMA_RHO, hi - 2 * SIGMA_RHO],
        [lo + 2 * SIGMA_RHO, hi + 2 * SIGMA_RHO],
        color=BAND,
        alpha=0.20,
        zorder=0,
        lw=0,
    )
    ax1.scatter(rx, my, s=38, color=ORANGE, edgecolor="white", linewidth=1.0, zorder=3)
    ax1.axvline(RHO_CHANCE, color=BLUE, lw=1.0, ls=":", zorder=2)
    ax1.axhline(RHO_CHANCE, color=BLUE, lw=1.0, ls=":", zorder=2)
    ax1.text(
        RHO_CHANCE,
        lo + 0.0008,
        " chance",
        color=BLUE,
        fontsize=7,
        va="bottom",
        ha="left",
    )
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.set_aspect("equal")
    ax1.set_xlabel(r"retrieval  $\rho$")
    ax1.set_ylabel(r"model  $\rho$")
    ax1.text(
        0.04,
        0.93,
        "below the line =\nretrieval wins",
        transform=ax1.transAxes,
        fontsize=7.4,
        color=MUTED,
        va="top",
    )
    ax1.set_title("Paired, same targets", fontsize=8.5, loc="left", pad=6)

    y = np.arange(len(pairs))
    for i, (_lab, r_, m_) in enumerate(pairs):
        ax2.plot([r_, m_], [i, i], color=BAND, lw=1.4, zorder=1)
    ax2.scatter(
        rx,
        y,
        s=26,
        color=BLUE,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
        label="retrieval",
    )
    ax2.scatter(
        my,
        y,
        s=26,
        color=ORANGE,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
        label="model",
    )
    ax2.axvline(RHO_CHANCE, color=MUTED, lw=1.0, ls=":", zorder=2)
    ax2.text(
        RHO_CHANCE,
        len(pairs) - 0.2,
        " chance level 0.9954",
        color=MUTED,
        fontsize=7,
        va="top",
        ha="left",
    )
    ax2.set_yticks([])
    ax2.set_ylim(-1, len(pairs))
    ax2.set_xlabel(r"$\rho$   (held-out target, mean over 5 seeds)")
    ax2.set_title("Per target, sorted", fontsize=8.5, loc="left", pad=6)
    ax2.legend(frameon=False, fontsize=7.4, loc="lower left")
    fig.savefig(_OUT / "fig2_retrieval_beats_model.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig2_retrieval_beats_model.png")


def fig3() -> None:
    """What a rho value means on this corpus -- the usable range."""
    g = json.loads((_G01 / "grading.json").read_text())
    model = float(np.median([v["rho"] for v in g["per_target"].values()]))
    marks = [
        ("unrelated dataset shapes\n(chance)", RHO_CHANCE, BLUE),
        ("corpus-mean predictor\n(ignores its input)", 0.9969, BLUE),
        ("the model", model, ORANGE),
        ("solver grading the\ncorrect shape", 0.99896, MUTED),
        ("retrieval", 0.99905, BLUE),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    lo, hi = 0.9945, 1.0002
    ax.axhline(0, color=MUTED, lw=1.2)
    ax.plot([1.0, 1.0], [-0.14, 0.14], color=INK, lw=1.4)
    ax.text(1.0, 0.20, "perfect", ha="center", fontsize=7.6, color=INK)
    for i, (lab, v, c) in enumerate(marks):
        up = i % 2 == 0
        ax.plot([v, v], [0, 0.10 if up else -0.10], color=c, lw=1.2)
        ax.scatter([v], [0], s=46, color=c, edgecolor="white", linewidth=1.2, zorder=4)
        ax.text(
            v,
            0.155 if up else -0.155,
            f"{lab}\n{v:.5f}",
            ha="center",
            va="bottom" if up else "top",
            fontsize=7.2,
            color=c,
            linespacing=1.35,
        )
    ax.annotate(
        "",
        xy=(RHO_CHANCE, -0.36),
        xytext=(1.0, -0.36),
        arrowprops={"arrowstyle": "<->", "color": MUTED, "lw": 1.0},
    )
    ax.text(
        (RHO_CHANCE + 1.0) / 2,
        -0.43,
        f"the entire usable range is {1 - RHO_CHANCE:.4f}  "
        f"= {(1 - RHO_CHANCE) / SIGMA_RHO:.0f}$\\sigma$",
        ha="center",
        va="top",
        fontsize=7.6,
        color=MUTED,
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.62, 0.42)
    ax.set_yticks([])
    ax.grid(False)
    for sp in ("left", "right", "top", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.set_xticks([0.995, 0.996, 0.997, 0.998, 0.999, 1.000])
    ax.set_xlabel(r"Pearson $\rho$ against the target C$_p$ curve")
    ax.set_title(
        r"$\rho$ has almost no room to vary on this corpus",
        fontsize=10.5,
        fontweight="bold",
        loc="left",
        pad=8,
    )
    fig.savefig(_OUT / "fig3_rho_has_no_range.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig3_rho_has_no_range.png")


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    print(f"\nFraming figures -> {_OUT.relative_to(_ROOT)}\n")
    fig1()
    fig2()
    fig3()
    print()


if __name__ == "__main__":
    main()
