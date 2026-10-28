"""
Figures on the flow itself: shedding onset, solver-grid convergence, and the
flow's response to boundary node count.

Every solver run was pruned to its surface data on completion, so no flow
FIELDS survive. What does survive in each run is the Cd/Cl time history in
``run.log`` (101 samples over t = 0-50), from which the shedding frequency
(Strouhal number, from upward zero crossings of Cl) and the shedding amplitude
(Cl rms) are recovered. No new runs.

  fig8   Shedding onset is uncontrolled, and differs by shape source.
         The averaging window is fixed at t = 25-50 but shedding begins
         anywhere from t ~ 12 to t ~ 47. Smooth, symmetric dataset shapes wait
         for round-off to break the symmetry; generated shapes carry per-node
         jitter and shed early.
  fig9   Solver-grid convergence, P1-F03 resolution sweep (fp32 production).
         Hollow markers: shedding still ramping up inside the window.
  fig10  Drag, shedding frequency and shedding amplitude against boundary
         node count, P1-D02, at fixed solver resolution.

Report style: no in-figure titles or notes; captions carry the explanation.

Usage:
    python scripts/paper1/p1_make_flow_figures.py
"""

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_OUT = _ROOT / "outputs" / "paper1" / "figures"
_P1 = _ROOT / "outputs" / "paper1"
_D02_SH = _ROOT / "cfd" / "ellipse2d" / "shapes" / "d02_nodes"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
FAINT = "#8f8e89"
WINDOW = "#e8edf4"

T_AVG_START = 25.0
ONSET_CL = 0.05
GROWING = 1.5
TRAIN_MIN, TRAIN_MAX = 52, 96

_PAT = re.compile(
    r"\[getResults\] t=([-\d.e+]+); Cd=([-\d.e+]+); Cl=([-\d.e+]+) \[(\w+)\]"
)

STYLE = {
    "font.size": 9,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
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


def history(run: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cd/Cl time history from a run log.

    Args:
        run (Path): Run directory holding ``run.log``.

    Returns:
        tuple: ``(t, cd, cl, phase)`` arrays; ``phase`` is ``"transient"`` or
            ``"averaging"``.
    """
    t, cd, cl, ph = [], [], [], []
    for line in (run / "run.log").read_text(errors="ignore").splitlines():
        m = _PAT.search(line)
        if m:
            t.append(float(m[1]))
            cd.append(float(m[2]))
            cl.append(float(m[3]))
            ph.append(m[4])
    return np.array(t), np.array(cd), np.array(cl), np.array(ph)


def characterise(run: Path) -> dict[str, float]:
    """Onset, stationarity, drag, shedding frequency and amplitude of one run.

    Args:
        run (Path): Run directory.

    Returns:
        dict[str, float]: ``onset``, ``growth``, ``Cd``, ``St``, ``Cl_rms``.
    """
    t, cd, cl, ph = history(run)
    hit = np.abs(cl) > ONSET_CL
    onset = float(t[np.argmax(hit)]) if hit.any() else float("inf")
    a = ph == "averaging"
    ta, ca = t[a], cl[a]
    mid = 0.5 * (ta.min() + ta.max())
    first, second = ca[ta < mid].std(), ca[ta >= mid].std()
    y = ca - ca.mean()
    ups = [
        ta[i] - y[i] * (ta[i + 1] - ta[i]) / (y[i + 1] - y[i])
        for i in range(len(y) - 1)
        if y[i] < 0 <= y[i + 1]
    ]
    st = 1.0 / float(np.mean(np.diff(ups))) if len(ups) >= 3 else float("nan")
    return {
        "onset": onset,
        "growth": float(second / first) if first else float("inf"),
        "Cd": float(cd[a].mean()),
        "St": st,
        "Cl_rms": float(ca.std()),
    }


def save(fig: plt.Figure, stem: str) -> None:
    """Write PNG and PDF.

    Args:
        fig (plt.Figure): The figure.
        stem (str): File stem.
    """
    for ext in ("png", "pdf"):
        fig.savefig(_OUT / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  {stem}.png / .pdf")


def fig8() -> None:
    """Shedding onset: example histories, and onset times by shape source."""
    examples = [
        (
            "Generated shape (D02)",
            _P1 / "P1-D02_node-count-transfer/solver/t020_n096_s0",
        ),
        (
            "Dataset shape, Re 751",
            _P1 / "P1-F03_solver-qualification/resolution/re0751_h50_idx2833_res150",
        ),
        (
            "Dataset shape, Re 501",
            _P1 / "P1-F03_solver-qualification/resolution/re0501_h50_idx0018_res100",
        ),
        (
            "Dataset shape, Re 501, fp64",
            _P1 / "P1-F06_schafer-turek/f03_fp64_sweep/re0501_h50_idx0018_res100",
        ),
    ]
    groups = [
        ("F04 noise floor", "P1-F04_noise-floor", "dataset"),
        ("F03 envelope", "P1-F03_solver-qualification/envelope", "dataset"),
        ("B03 reconstructions", "P1-B03_basis-reachability/solver", "dataset"),
        ("B01 retrieval", "P1-B01_retrieval-baseline/solver", "dataset"),
        ("G01 targets", "P1-G01_holdout-inverse-design/truth", "dataset"),
        ("G01 generated", "P1-G01_holdout-inverse-design/solver", "generated"),
        ("D02 generated", "P1-D02_node-count-transfer/solver", "generated"),
    ]

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(6.9, 4.3))
        gs = fig.add_gridspec(
            len(examples), 2, width_ratios=[1.3, 1.0], wspace=0.72, hspace=0.62
        )
        for i, (lab, run) in enumerate(examples):
            ax = fig.add_subplot(gs[i, 0])
            t, _, cl, _ = history(run)
            ax.axvspan(T_AVG_START, 50, color=WINDOW, lw=0, zorder=0)
            ax.plot(t, cl, color=INK, lw=0.8, zorder=2)
            on = characterise(run)["onset"]
            if np.isfinite(on):
                ax.axvline(on, color=ORANGE, lw=1.0, zorder=1)
            ax.set_xlim(0, 50)
            ax.set_ylim(-0.9, 0.9)
            ax.set_yticks([-0.5, 0, 0.5])
            ax.set_title(lab, loc="left", fontsize=7.8, color=MUTED, pad=2)
            if i < len(examples) - 1:
                ax.tick_params(labelbottom=False)
            if i == len(examples) // 2 - 1:
                ax.set_ylabel("$C_l$", labelpad=2)
        ax.set_xlabel("Time, $tU/c$")

        ax = fig.add_subplot(gs[:, 1])
        rng = np.random.default_rng(0)
        for k, (_lab, rel, src) in enumerate(groups):
            on = []
            for d in sorted((_P1 / rel).glob("*/")):
                if (d / "run.log").exists():
                    v = characterise(d)["onset"]
                    if np.isfinite(v):
                        on.append(v)
            y = len(groups) - 1 - k + rng.uniform(-0.22, 0.22, len(on))
            ax.scatter(
                on,
                y,
                s=7,
                color=BLUE if src == "dataset" else ORANGE,
                alpha=0.55,
                lw=0,
                zorder=2,
            )
            ax.plot(
                [np.median(on)] * 2,
                [y.mean() - 0.32, y.mean() + 0.32],
                color=INK,
                lw=1.4,
                zorder=3,
            )
        ax.axvline(T_AVG_START, color=MUTED, lw=0.9, ls="--", zorder=1)
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels([g[0] for g in reversed(groups)])
        ax.set_xlim(0, 50)
        ax.set_xlabel("Shedding onset, $tU/c$")
        ax.scatter([], [], s=14, color=BLUE, label="dataset shape")
        ax.scatter([], [], s=14, color=ORANGE, label="generated shape")
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=2,
            frameon=False,
            handletextpad=0.3,
            columnspacing=1.0,
        )
        save(fig, "fig8_shedding_onset")


def fig9() -> None:
    """Solver-grid convergence of drag and shedding frequency."""
    cases = [
        ("Re 501", "re0501_h50_idx0018", 100, "o"),
        ("Re 751", "re0751_h50_idx2833", 150, "s"),
        ("Re 998", "re0998_h50_idx5642", 200, "^"),
    ]
    root = _P1 / "P1-F03_solver-qualification" / "resolution"
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7))
        for lab, case, rule, mk in cases:
            runs = sorted(
                root.glob(f"{case}_res*"), key=lambda p: int(p.name.split("res")[1])
            )
            res = np.array([int(p.name.split("res")[1]) for p in runs]) / rule
            ch = [characterise(p) for p in runs]
            cd = np.array([c["Cd"] for c in ch])
            st = np.array([c["St"] for c in ch])
            still = np.array([c["growth"] > GROWING for c in ch])
            for ax, y in ((axes[0], 100 * (cd / cd[-1] - 1)), (axes[1], st)):
                ax.plot(res, y, color=INK, lw=0.9, zorder=2)
                ax.plot(
                    res[~still],
                    y[~still],
                    ls="none",
                    marker=mk,
                    ms=5,
                    mfc=INK,
                    mec=INK,
                    zorder=3,
                    label=lab if ax is axes[0] else None,
                )
                ax.plot(
                    res[still],
                    y[still],
                    ls="none",
                    marker=mk,
                    ms=5,
                    mfc="white",
                    mec=INK,
                    mew=0.9,
                    zorder=3,
                )
        for ax in axes:
            ax.axvspan(0.95, 1.05, color=WINDOW, lw=0, zorder=0)
            ax.set_xticks([0.5, 1.0, 1.5, 2.0])
            ax.set_xticklabels(["0.5×", "1×", "1.5×", "2×"])
            ax.set_xlabel("Resolution / production rule")
        axes[0].axhline(0, color=FAINT, lw=0.7, zorder=1)
        axes[0].set_ylabel("$C_d$ vs finest (%)")
        axes[1].set_ylabel("Strouhal number, $fc/U$")
        axes[0].plot(
            [],
            [],
            ls="none",
            marker="o",
            ms=5,
            mfc="white",
            mec=INK,
            mew=0.9,
            label="still ramping up",
        )
        fig.legend(
            loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=False
        )
        fig.tight_layout(w_pad=2.2)
        save(fig, "fig9_grid_convergence")


def target_markers(labels: list[str]) -> dict[str, str]:
    """Marker per D02 target, assigned exactly as in fig4.

    fig4 orders targets by mean generated aspect ratio, thickest first, and
    assigns o, s, ^, D in that order. Re-deriving the order here keeps each
    target on one symbol in every figure of the report.

    Args:
        labels (list[str]): D02 target labels, e.g. ``"t039"``.

    Returns:
        dict[str, str]: Label to matplotlib marker.
    """
    man = list(csv.DictReader((_D02_SH / "manifest.csv").open()))

    def mean_aspect(lab: str) -> float:
        vals = []
        for m in man:
            if m["label"] != lab:
                continue
            xy = np.asarray(
                [
                    tuple(map(float, ln.split()))
                    for ln in (_D02_SH / f"{m['tag']}.dat").read_text().splitlines()
                    if ln.strip() and not ln.startswith("#")
                ]
            )
            vals.append(np.ptp(xy[:, 1]) / np.ptp(xy[:, 0]))
        return float(np.mean(vals))

    order = sorted(labels, key=lambda lab: -mean_aspect(lab))
    return dict(zip(order, ("o", "s", "^", "D"), strict=True))


def fig10() -> None:
    """Drag, shedding frequency and amplitude against boundary node count."""
    man = list(csv.DictReader((_D02_SH / "manifest.csv").open()))
    root = _P1 / "P1-D02_node-count-transfer" / "solver"
    labels = sorted({m["label"] for m in man})
    ladder = sorted({int(m["n_nodes"]) for m in man})
    ch = {m["tag"]: characterise(root / m["tag"]) for m in man}

    ref = {
        lab: np.mean(
            [
                ch[m["tag"]]["Cd"]
                for m in man
                if m["label"] == lab and TRAIN_MIN <= int(m["n_nodes"]) <= TRAIN_MAX
            ]
        )
        for lab in labels
    }
    mk = target_markers(labels)
    thick_first = list(mk)
    markers = tuple(mk.values())
    nudge = (0.965, 0.988, 1.012, 1.035)
    panels = [
        (
            "$C_d$ vs in-range mean (%)",
            lambda m: 100 * (ch[m["tag"]]["Cd"] / ref[m["label"]] - 1),
        ),
        ("Strouhal number, $fc/U$", lambda m: ch[m["tag"]]["St"]),
        ("$C_l$ rms", lambda m: ch[m["tag"]]["Cl_rms"]),
    ]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.5))
        for ax, (ylab, fn) in zip(axes, panels, strict=True):
            ax.axvspan(TRAIN_MIN, TRAIN_MAX, color=WINDOW, lw=0, zorder=0)
            for i, lab in enumerate(thick_first):
                mean, lo, hi = [], [], []
                for n in ladder:
                    v = [
                        fn(m)
                        for m in man
                        if m["label"] == lab and int(m["n_nodes"]) == n
                    ]
                    mean.append(np.nanmean(v))
                    lo.append(np.nanmin(v))
                    hi.append(np.nanmax(v))
                mean_a, lo_a, hi_a = map(np.asarray, (mean, lo, hi))
                xs = np.asarray(ladder, dtype=float) * nudge[i]
                ax.errorbar(
                    xs,
                    mean_a,
                    yerr=[mean_a - lo_a, hi_a - mean_a],
                    fmt="none",
                    ecolor=FAINT,
                    elinewidth=0.7,
                    capsize=1.4,
                    zorder=2,
                )
                ax.plot(xs, mean_a, color=INK, lw=0.8, zorder=3)
                ax.plot(
                    xs,
                    mean_a,
                    ls="none",
                    marker=markers[i],
                    ms=3.8,
                    mfc="white",
                    mec=INK,
                    mew=0.8,
                    zorder=4,
                    label=f"Target {lab[1:]}" if ax is axes[0] else None,
                )
            ax.set_xscale("log", base=2)
            ax.set_xticks([24, 48, 96, 192, 384])
            ax.set_xticklabels(["24", "48", "96", "192", "384"])
            ax.minorticks_off()
            ax.set_xlabel("Boundary nodes, $N$")
            ax.set_ylabel(ylab)
        axes[0].axhline(0, color=FAINT, lw=0.7, zorder=1)
        fig.legend(
            loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=False
        )
        fig.tight_layout(w_pad=1.6)
        save(fig, "fig10_flow_vs_node_count")


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    print(f"\nFlow figures -> {_OUT.relative_to(_ROOT)}\n")
    fig8()
    fig9()
    fig10()
    print()


if __name__ == "__main__":
    main()
