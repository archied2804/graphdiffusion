"""
Figures for the model-side results of the Paper 1 suite.

Four figures, each carrying one claim:

  fig4  The model draws the same shape at every node count      (P1-D02 section 2)
  fig5  The model is steered by its conditioning                (P1-G01 section 4a)
  fig6  The sampling template's node spacing leaks shape        (P1-D02 section 3)
  fig7  Fabricated surface probes, before the fix               (P1-F04 section 2)

Written as PNG for viewing and PDF for the paper, which is LaTeX.

Colour: dataviz categorical slots 1 (blue) and 2 (orange), validated all-pairs
in light mode. Text is set in ink, never in a series colour; identity is carried
by markers and direct labels. Figure 4's four targets share one neutral colour
because each is compared with itself across node counts, not with the others.

Usage:
    python scripts/paper1/p1_make_model_figures.py
"""

import csv
import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from p1_b01_retrieval_index import descriptor  # noqa: E402

from graph_diffusion.data.pOnEllipse import (  # noqa: E402
    _H5_COL_X,
    _H5_COL_Y,
    _load_h5_raw,
    _node_count,
)
from graph_diffusion.postprocessing.cp_metrics import (  # noqa: E402
    dead_probe_mask,
    read_cp_csv,
)

_OUT = _ROOT / "outputs" / "paper1" / "figures"
_RAW = _ROOT / "data" / "ellipse" / "raw"
_D02 = _ROOT / "cfd" / "ellipse2d" / "shapes" / "d02_nodes"
_G01 = _ROOT / "cfd" / "ellipse2d" / "shapes" / "g01_holdout"
_TARGETS = _ROOT / "data" / "splits" / "paper1_eval_targets.json"
_F03_RUN = (
    _ROOT
    / "outputs"
    / "paper1"
    / "P1-F03_solver-qualification"
    / "envelope"
    / "re0997_h60_idx5639"
)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
FAINT = "#8f8e89"
BAND = "#b9b8b2"

TRAIN_MIN, TRAIN_MAX = 52, 96
G01_TEMPLATE_ASPECT = 0.511

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
        "savefig.dpi": 220,
        "pdf.fonttype": 42,
    }
)


def read_body(p: Path) -> np.ndarray:
    """Read a solver body file as an ``(N, 2)`` array.

    Args:
        p (Path): The ``.dat`` file.

    Returns:
        np.ndarray: Points.
    """
    return np.asarray(
        [
            tuple(map(float, ln.split()))
            for ln in p.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
    )


def aspect(xy: np.ndarray) -> float:
    """Thickness ratio ``b/a`` of a body.

    Args:
        xy (np.ndarray): Boundary points, shape ``(N, 2)``.

    Returns:
        float: Height over chord.
    """
    return float((xy[:, 1].max() - xy[:, 1].min()) / (xy[:, 0].max() - xy[:, 0].min()))


def profile(xy: np.ndarray, n_grid: int = 128) -> np.ndarray:
    """Resample a body to a common ``r(theta)/mean(r)`` grid.

    Args:
        xy (np.ndarray): Boundary points, shape ``(N, 2)``.
        n_grid (int): Output grid size.

    Returns:
        np.ndarray: Normalised radius, shape ``(n_grid,)``.
    """
    c = xy - xy.mean(axis=0)
    r = np.hypot(c[:, 0], c[:, 1])
    r = r / r.mean()
    th = np.mod(np.arctan2(c[:, 1], c[:, 0]), 2 * np.pi)
    o = np.argsort(th)
    return np.interp(
        np.linspace(0.0, 2.0 * np.pi, n_grid, endpoint=False),
        np.concatenate([th[o] - 2 * np.pi, th[o], th[o] + 2 * np.pi]),
        np.concatenate([r[o], r[o], r[o]]),
    )


def true_aspect(raw: np.ndarray, idx: int) -> float:
    """Aspect ratio of a dataset target.

    Args:
        raw (np.ndarray): HDF5 array.
        idx (int): Row index.

    Returns:
        float: ``b/a``.
    """
    s = raw[idx][: _node_count(raw[idx])]
    return aspect(np.stack([s[:, _H5_COL_X], s[:, _H5_COL_Y]], axis=1))


def save(fig: plt.Figure, stem: str) -> None:
    """Write PNG and PDF.

    Args:
        fig (plt.Figure): The figure.
        stem (str): File stem, without extension.
    """
    for ext in ("png", "pdf"):
        fig.savefig(_OUT / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  {stem}.png / .pdf")


def fig4() -> None:
    """Aspect ratio against node count -- the shape does not move."""
    man = list(csv.DictReader((_D02 / "manifest.csv").open()))
    labels = sorted({m["label"] for m in man})
    ladder = sorted({int(m["n_nodes"]) for m in man})
    raw = _load_h5_raw(_RAW / "pOnEllipseInDist.h5")

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.axvspan(TRAIN_MIN, TRAIN_MAX, color=BLUE, alpha=0.10, lw=0, zorder=0)
    ax.text(
        np.sqrt(TRAIN_MIN * TRAIN_MAX),
        0.905,
        "trained range\n52–96 nodes",
        ha="center",
        va="top",
        fontsize=7.2,
        color=MUTED,
    )

    worst = 0.0
    ends: list[tuple[float, str]] = []
    for lab in labels:
        mean, lo, hi = [], [], []
        for n in ladder:
            a = [
                aspect(read_body(_D02 / f"{m['tag']}.dat"))
                for m in man
                if m["label"] == lab and int(m["n_nodes"]) == n
            ]
            mean.append(np.mean(a))
            lo.append(min(a))
            hi.append(max(a))
        mean_a = np.asarray(mean)
        ref = float(
            mean_a[
                [ladder.index(n) for n in ladder if TRAIN_MIN <= n <= TRAIN_MAX]
            ].mean()
        )
        worst = max(worst, float(np.abs(mean_a / ref - 1).max()))
        ax.fill_between(ladder, lo, hi, color=BAND, alpha=0.45, lw=0, zorder=1)
        ax.plot(ladder, mean_a, color=INK, lw=1.4, zorder=3)
        ax.scatter(
            ladder, mean_a, s=20, color=INK, edgecolor="white", linewidth=0.8, zorder=4
        )
        t = int(man[[m["label"] for m in man].index(lab)]["target_idx"])
        ends.append(
            (
                float(mean_a[-1]),
                f"target {lab[1:]}  (true b/a {true_aspect(raw, t):.2f})",
            )
        )

    # Direct labels at the right end, pushed apart where lines nearly coincide.
    ends.sort()
    placed: list[float] = []
    for y, txt in ends:
        y_lab = max(y, placed[-1] + 0.016) if placed else y
        placed.append(y_lab)
        ax.annotate(
            txt,
            xy=(ladder[-1], y),
            xytext=(ladder[-1] * 1.10, y_lab),
            va="center",
            ha="left",
            fontsize=7,
            color=MUTED,
            arrowprops={"arrowstyle": "-", "color": BAND, "lw": 0.7},
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(ladder)
    ax.set_xticklabels([str(n) for n in ladder])
    ax.minorticks_off()
    ax.set_xlim(ladder[0] / 1.12, ladder[-1] * 1.12)
    ax.set_ylim(0.62, 0.91)
    ax.set_xlabel("boundary nodes in the generated shape")
    ax.set_ylabel("aspect ratio  b/a")
    ax.set_title(
        "The model draws the same shape at every node count",
        fontsize=10.5,
        fontweight="bold",
        loc="left",
        pad=16,
    )
    ax.text(
        0.0,
        1.015,
        f"Four held-out targets, 3 seeds each (band = seed range).  "
        f"All 144 shapes valid; worst drift from the in-range mean "
        f"{100 * worst:.1f}% across a 16x range.",
        transform=ax.transAxes,
        fontsize=7.4,
        color=MUTED,
        va="bottom",
    )
    save(fig, "fig4_node_count_invariance")


def fig5() -> None:
    """Target-Cp distance against generated-shape distance."""
    man = list(csv.DictReader((_G01 / "manifest.csv").open()))
    hold = [m for m in man if m["is_training_sample"] == "0"]
    labels = sorted({m["label"] for m in hold})
    raw = _load_h5_raw(_RAW / "pOnEllipseInDist.h5")

    shapes = {
        lab: np.stack(
            [
                profile(read_body(_G01 / f"{m['tag']}.dat"))
                for m in hold
                if m["label"] == lab
            ]
        )
        for lab in labels
    }
    conds = {
        lab: descriptor(
            raw[int(next(m for m in hold if m["label"] == lab)["target_idx"])],
            "forward",
        )[0]
        for lab in labels
    }
    mean = {lab: shapes[lab].mean(axis=0) for lab in labels}

    dc, ds = [], []
    for i, j in itertools.combinations(labels, 2):
        dc.append(float(np.linalg.norm(conds[i] - conds[j])))
        ds.append(float(np.linalg.norm(mean[i] - mean[j])))
    dc_a, ds_a = np.asarray(dc), np.asarray(ds)
    r = float(np.corrcoef(dc_a, ds_a)[0, 1])

    within = [
        float(np.linalg.norm(a - b))
        for lab in labels
        for a, b in itertools.combinations(shapes[lab], 2)
    ]
    w_lo, w_md, w_hi = np.percentile(within, [25, 50, 75])

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.axhspan(w_lo, w_hi, color=BAND, alpha=0.35, lw=0, zorder=0)
    ax.axhline(w_md, color=FAINT, lw=1.0, ls="--", zorder=1)
    ax.text(
        dc_a.max() * 0.99,
        w_hi,
        "two samples of the SAME target\n(seed noise, interquartile)",
        ha="right",
        va="bottom",
        fontsize=7,
        color=MUTED,
    )
    ax.scatter(dc_a, ds_a, s=18, color=BLUE, edgecolor="white", linewidth=0.6, zorder=3)
    k = np.polyfit(dc_a, ds_a, 1)
    xs = np.linspace(dc_a.min(), dc_a.max(), 2)
    ax.plot(xs, np.polyval(k, xs), color=INK, lw=1.2, zorder=4)
    ax.set_xlim(0, dc_a.max() * 1.04)
    ax.set_ylim(0, ds_a.max() * 1.08)
    ax.set_xlabel("distance between two REQUESTED pressure curves  (K=8 DCT)")
    ax.set_ylabel(
        "distance between the two GENERATED shapes\n(each the mean of 5 seeds)"
    )
    ax.set_title(
        "Ask for different pressure, get a different shape",
        fontsize=10.5,
        fontweight="bold",
        loc="left",
        pad=16,
    )
    ax.text(
        0.0,
        1.015,
        f"All {len(ds)} pairs of G01's 20 held-out targets.  r = {r:.3f}",
        transform=ax.transAxes,
        fontsize=7.4,
        color=MUTED,
        va="bottom",
    )
    save(fig, "fig5_conditioning_response")


def fig6() -> None:
    """Generated against true aspect ratio, under two different templates."""
    meta = json.loads(_TARGETS.read_text())
    tset = meta["sets"]["indist"]["target_indices"]
    raw = _load_h5_raw(_RAW / "pOnEllipseInDist.h5")

    tg, gg = [], []
    for t in tset:
        tg.append(true_aspect(raw, t))
        gg.append(
            np.mean(
                [
                    aspect(read_body(_G01 / f"holdout_t{t:03d}_s{k}.dat"))
                    for k in range(5)
                ]
            )
        )
    tg_a, gg_a = np.asarray(tg), np.asarray(gg)

    man = list(csv.DictReader((_D02 / "manifest.csv").open()))
    td, dd = [], []
    for lab in sorted({m["label"] for m in man}):
        rows = [
            m
            for m in man
            if m["label"] == lab and TRAIN_MIN <= int(m["n_nodes"]) <= TRAIN_MAX
        ]
        td.append(true_aspect(raw, int(rows[0]["target_idx"])))
        dd.append(np.mean([aspect(read_body(_D02 / f"{m['tag']}.dat")) for m in rows]))
    td_a, dd_a = np.asarray(td), np.asarray(dd)

    lo, hi = 0.48, 0.88
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.0, ls="--", zorder=1)
    ax.text(0.83, 0.855, "perfect", fontsize=7, color=MUTED, rotation=38, ha="center")

    ax.axhline(G01_TEMPLATE_ASPECT, color=ORANGE, lw=0.9, ls=":", zorder=1)
    ax.text(
        hi - 0.005,
        G01_TEMPLATE_ASPECT - 0.004,
        "G01 template's own shape (b/a 0.511)",
        ha="right",
        va="top",
        fontsize=7,
        color=MUTED,
    )
    ax.annotate(
        "",
        xy=(0.50, hi - 0.004),
        xytext=(0.50, hi - 0.06),
        arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.0},
    )
    ax.text(
        0.508,
        hi - 0.03,
        "D02 template: evenly spaced = a circle\n(b/a 1.0, off the top)",
        fontsize=7,
        color=MUTED,
        va="center",
        ha="left",
    )

    ax.scatter(
        tg_a,
        gg_a,
        s=34,
        color=ORANGE,
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
        label="G01: sampled on dataset[0]'s node spacing",
    )
    k = np.polyfit(tg_a, gg_a, 1)
    xs = np.array([tg_a.min(), tg_a.max()])
    ax.plot(xs, np.polyval(k, xs), color=ORANGE, lw=1.3, zorder=2)
    ax.text(
        tg_a.max() + 0.006,
        np.polyval(k, tg_a.max()),
        f"slope {k[0]:.2f}",
        fontsize=7,
        color=MUTED,
        va="center",
    )
    ax.scatter(
        td_a,
        dd_a,
        s=46,
        marker="D",
        color=BLUE,
        edgecolor="white",
        linewidth=0.9,
        zorder=4,
        label="D02: sampled on evenly spaced nodes",
    )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("TRUE aspect ratio of the target")
    ax.set_ylabel("aspect ratio the model GENERATED")
    ax.legend(frameon=False, fontsize=7.2, loc="upper left", bbox_to_anchor=(0.0, 0.80))
    ax.set_title(
        "The sampling template leaks its own shape",
        fontsize=10.5,
        fontweight="bold",
        loc="left",
        pad=16,
    )
    ax.text(
        0.0,
        1.015,
        "Same model, same targets. Each template pulls shapes toward itself.",
        transform=ax.transAxes,
        fontsize=7.4,
        color=MUTED,
        va="bottom",
    )
    save(fig, "fig6_template_bias")


def fig7() -> None:
    """A pre-fix surface-pressure curve with its fabricated samples."""
    x, cp, rms = read_cp_csv(_F03_RUN / "tmp" / "ellipse2d_cp.csv", drop_dead=False)
    dead = dead_probe_mask(cp, rms)
    fake = float(cp[dead][0]) if dead.any() else float("nan")

    # The body is symmetric at zero incidence, so upper- and lower-surface
    # samples coincide in (x/c, Cp). Average them into one guide curve.
    xl = np.round(x[~dead], 4)
    ux = np.unique(xl)
    guide = np.array([cp[~dead][xl == u].mean() for u in ux])

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    ax.plot(ux, guide, color=BAND, lw=1.2, zorder=1)
    bars = list(np.unique(np.round(x[dead], 4)))
    for i, xd in enumerate(bars):
        truth = float(np.interp(xd, ux, guide))
        ax.plot([xd, xd], [truth, fake], color=ORANGE, lw=1.0, alpha=0.7, zorder=2)
        # Put the label on the left if another bar sits close on the right.
        crowded = i + 1 < len(bars) and bars[i + 1] - xd < 0.13
        ax.text(
            xd - 0.012 if crowded else xd + 0.012,
            0.5 * (truth + fake),
            f"error {fake - truth:+.2f}",
            fontsize=6.8,
            color=MUTED,
            va="center",
            ha="right" if crowded else "left",
        )
    ax.scatter(
        x[~dead],
        cp[~dead],
        s=20,
        color=BLUE,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
        label=f"real samples ({int((~dead).sum())})",
    )
    ax.scatter(
        x[dead],
        cp[dead],
        s=52,
        marker="X",
        color=ORANGE,
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
        label=f"fabricated: probe inside the body ({int(dead.sum())})",
    )
    ax.axhline(fake, color=ORANGE, lw=0.7, ls=":", zorder=1)
    ax.text(
        0.36,
        fake + 0.05,
        f"all {int(dead.sum())} read exactly {fake:+.6f} = -Cp_inf from this run's log",
        fontsize=6.9,
        color=MUTED,
        va="top",
        ha="left",
    )
    ax.invert_yaxis()
    ax.set_xlabel("x / c")
    ax.set_ylabel("Cp   (suction up)")
    ax.legend(frameon=False, fontsize=7.2, loc="lower right")
    ax.set_title(
        "Before the fix, the solver fabricated surface pressure",
        fontsize=10.5,
        fontweight="bold",
        loc="left",
        pad=16,
    )
    ax.text(
        0.0,
        1.015,
        f"P1-F03 envelope case, Re 997: {int(dead.sum())} of {x.size} samples "
        f"({dead.mean():.0%}). Symmetric body, so upper and lower surfaces "
        "overlap: each cross is two samples.",
        transform=ax.transAxes,
        fontsize=7.4,
        color=MUTED,
        va="bottom",
    )
    save(fig, "fig7_dead_probes")


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    print(f"\nModel figures -> {_OUT.relative_to(_ROOT)}\n")
    fig4()
    fig5()
    fig6()
    fig7()
    print()


if __name__ == "__main__":
    main()
