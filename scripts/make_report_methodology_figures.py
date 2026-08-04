"""Generate the four explanatory methodology figures for the first-year report.

All four are grounded in real project data (reference Cp curves and exported
boundary polygons under ``cfd/ellipse2d/shapes/``) and use the model's own
DCT-II encoding from ``graph_diffusion.data.pOnEllipseConditional``:

* ``report_conditioning_pipeline`` — the pressure-conditioning data flow:
  target ``Cp(x/c)`` -> DCT-II (first K=8) -> context vector ``c`` into the
  score model, with the jointly-trained DeepSets head mapping the denoised
  shape back to a predicted spectrum (the auxiliary geometry->pressure
  coupling).
* ``report_diffusion_process`` — the forward/reverse diffusion on a real
  boundary's radial field ``r(theta)``: a clean shape dissolved to noise by the
  cosine-schedule forward marginal and recovered by the reverse process.
* ``report_dct_encoding`` — a real reference ``Cp`` sequence and its K=8 DCT-II
  reconstruction (with a coarser K=3 for contrast) plus the coefficient stems,
  motivating the fixed-size conditioning.
* ``report_roughness_metric`` — a real raw (wobbly) versus Fourier-smoothed
  generated boundary and their per-node second-difference of ``r(theta)``, the
  surface-roughness diagnostic, with the mean roughness annotated.

Run from the repo root:  ``python scripts/make_report_methodology_figures.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for sibling scripts
from _report_style import apply_house_style  # noqa: E402

from graph_diffusion.data.pOnEllipseConditional import (  # noqa: E402
    dct_ii,
    dct_ii_inverse,
)

SHAPES = Path("cfd/ellipse2d/shapes")
ASSETS = Path("docs/assets")

# Shared palette (consistent with report_architecture.png).
C_INPUT, C_INPUT_E = "#e8eef5", "#5b7fa6"
C_ENCODE, C_ENCODE_E = "#fdf0e2", "#d08a3e"
C_GLOBAL, C_GLOBAL_E = "#f3e9f3", "#9a4f8a"
C_CORE, C_CORE_E = "#e9f3ec", "#4f9a6a"
C_OUT, C_OUT_E = "#f5e9e9", "#b05050"
TXT = "#1f2a33"
RAW_C, SMOOTH_C = "#b05050", "#4f9a6a"


# --------------------------------------------------------------------------- #
# Data loading helpers
# --------------------------------------------------------------------------- #
def _load_xy(path: Path) -> np.ndarray:
    """Load an ``x y`` boundary polygon, skipping ``#`` comment lines.

    Args:
        path (Path): A ``.dat`` file with two whitespace-separated columns.

    Returns:
        np.ndarray: Array of shape ``(N, 2)``.
    """
    pts = []
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        pts.append((float(parts[0]), float(parts[1])))
    return np.asarray(pts, dtype=float)


def _load_cp(path: Path) -> np.ndarray:
    """Load the second column (Cp) of an ``x_over_c,cp`` CSV in file order."""
    rows = path.read_text().splitlines()[1:]
    return np.asarray([float(r.split(",")[1]) for r in rows if r.strip()], dtype=float)


def _radial(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a boundary polygon to a theta-sorted radial field.

    Args:
        xy (np.ndarray): Boundary points, shape ``(N, 2)``.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(theta, r)`` sorted by angle about the
        centroid.
    """
    c = xy.mean(axis=0)
    d = xy - c
    theta = np.arctan2(d[:, 1], d[:, 0])
    r = np.hypot(d[:, 0], d[:, 1])
    order = np.argsort(theta)
    return theta[order], r[order]


# --------------------------------------------------------------------------- #
# Shared box/arrow primitives (shared with the architecture figure style)
# --------------------------------------------------------------------------- #
def _box(ax, xy, w, h, text, face, edge, fs=10.5, bold=False):
    """Draw a rounded labelled box; return its centre."""
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=1.4,
            facecolor=face,
            edgecolor=edge,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=TXT,
        weight="bold" if bold else "normal",
        zorder=5,
    )
    return x + w / 2, y + h / 2


def _arrow(ax, start, end, color="#3a4750", lw=1.6, style="-|>"):
    """Draw an annotation arrow."""
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=2, shrinkB=2),
        zorder=4,
    )


# --------------------------------------------------------------------------- #
# Figure A — conditioning pipeline
# --------------------------------------------------------------------------- #
def _mini_curve(ax, x0, y0, w, h, ys, title, color, pad=0.16):
    """Draw a framed mini line-plot of ``ys`` in data coordinates."""
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor="white", edgecolor=color, lw=1.4))
    xs = np.linspace(x0 + pad * w, x0 + (1 - pad) * w, len(ys))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    yy = y0 + pad * h + (ys - ymin) / (ymax - ymin + 1e-9) * (1 - 2 * pad) * h
    ax.plot(xs, yy, color=color, lw=1.5, zorder=5)
    ax.text(
        x0 + w / 2,
        y0 + h + 0.28,
        title,
        ha="center",
        va="bottom",
        fontsize=9,
        color=TXT,
    )


def _mini_stem(ax, x0, y0, w, h, vals, title, color, pad=0.18):
    """Draw a framed mini stem-plot of ``vals`` in data coordinates."""
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor="white", edgecolor=color, lw=1.4))
    base = y0 + h * 0.5
    xs = np.linspace(x0 + pad * w, x0 + (1 - pad) * w, len(vals))
    vmax = float(np.max(np.abs(vals))) + 1e-9
    ax.plot([x0 + pad * w, x0 + (1 - pad) * w], [base, base], color="#9aa6ad", lw=0.7)
    for xx, v in zip(xs, vals):
        yy = base + (v / vmax) * (0.5 - pad) * h
        ax.plot([xx, xx], [base, yy], color=color, lw=1.3, zorder=5)
        ax.plot(xx, yy, "o", color=color, ms=3.2, zorder=6)
    ax.text(
        x0 + w / 2,
        y0 + h + 0.28,
        title,
        ha="center",
        va="bottom",
        fontsize=9,
        color=TXT,
    )


def fig_conditioning_pipeline() -> None:
    """Schematic of the pressure-conditioning data flow (all in data coords)."""
    cp = _load_cp(SHAPES / "baseline" / "sim_0_ref_cp.csv")
    coeffs = dct_ii(cp, 8)

    fig, ax = plt.subplots(figsize=(13.0, 5.0))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 9)
    ax.axis("off")

    bw, bh = 3.8, 2.0
    ytop, ybot = 5.6, 1.3
    cx = [2.4, 7.2, 12.0, 16.8, 21.6]  # column centres

    # top row: Cp -> DCT -> context c -> score model -> generated shape
    _mini_curve(ax, cx[0] - bw / 2, ytop, bw, bh, cp, "target $C_p(x/c)$", C_INPUT_E)
    _box(
        ax,
        (cx[1] - bw / 2, ytop),
        bw,
        bh,
        "DCT-II\n(first $K{=}8$)",
        C_ENCODE,
        C_ENCODE_E,
    )
    _mini_stem(
        ax,
        cx[2] - bw / 2,
        ytop,
        bw,
        bh,
        coeffs,
        "context $c\\in\\mathbb{R}^8$",
        C_GLOBAL_E,
    )
    _box(
        ax,
        (cx[3] - bw / 2, ytop),
        bw,
        bh,
        "score model\n$\\varepsilon_\\theta$",
        C_CORE,
        C_CORE_E,
        10.5,
        bold=True,
    )
    _box(
        ax,
        (cx[4] - bw / 2, ytop),
        bw,
        bh,
        "generated\nshape $\\hat{\\mathbf{x}}_0$",
        C_INPUT,
        C_INPUT_E,
        9.8,
    )

    # bottom row (returns right->left): x0 -> DeepSets head -> predicted spectrum
    _box(
        ax,
        (cx[3] - bw / 2, ybot),
        bw,
        bh,
        "DeepSets head\n$h(\\hat{\\mathbf{x}}_0,\\mathrm{pos})$",
        "#eef2ef",
        C_CORE_E,
        9.5,
    )
    _mini_stem(
        ax,
        cx[2] - bw / 2,
        ybot,
        bw,
        bh,
        coeffs * 1.05,
        "predicted $\\hat{c}$",
        C_CORE_E,
    )

    # top arrows (left -> right)
    for a, b in zip(cx[:-1], cx[1:]):
        _arrow(ax, (a + bw / 2, ytop + bh / 2), (b - bw / 2, ytop + bh / 2))
    # generated shape down into the head
    _arrow(ax, (cx[4], ytop), (cx[3] + bw / 2, ybot + bh / 2), color=C_CORE_E)
    # head -> predicted spectrum (right -> left)
    _arrow(
        ax,
        (cx[3] - bw / 2, ybot + bh / 2),
        (cx[2] + bw / 2, ybot + bh / 2),
        color=C_CORE_E,
    )
    # auxiliary loss: vertical dashed compare between c (top) and c-hat (bottom)
    ax.annotate(
        "",
        xy=(cx[2], ybot + bh + 0.05),
        xytext=(cx[2], ytop - 0.05),
        arrowprops=dict(
            arrowstyle="<->", color="#9a4f8a", lw=1.6, linestyle=(0, (4, 2))
        ),
    )
    ax.text(
        cx[2] + 0.4,
        (ytop + ybot + bh) / 2,
        "auxiliary loss\n$\\|\\hat{c}-c\\|^2$",
        ha="left",
        va="center",
        fontsize=8.8,
        color="#7a3a6a",
        style="italic",
    )

    handles = [
        mpatches.Patch(facecolor=C_INPUT, edgecolor=C_INPUT_E, label="inputs / shape"),
        mpatches.Patch(facecolor=C_ENCODE, edgecolor=C_ENCODE_E, label="DCT encoder"),
        mpatches.Patch(
            facecolor=C_GLOBAL, edgecolor=C_GLOBAL_E, label="context vector"
        ),
        mpatches.Patch(facecolor=C_CORE, edgecolor=C_CORE_E, label="model / aux head"),
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=4,
        frameon=False,
        fontsize=8.6,
    )
    out = ASSETS / "report_conditioning_pipeline.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------- #
# Figure B — forward/reverse diffusion on r(theta)
# --------------------------------------------------------------------------- #
def _cosine_alpha_bar(t_frac: np.ndarray, s: float = 0.008) -> np.ndarray:
    """Nichol & Dhariwal cosine schedule cumulative product alpha-bar."""
    f = np.cos((t_frac + s) / (1 + s) * np.pi / 2) ** 2
    f0 = np.cos((0.0 + s) / (1 + s) * np.pi / 2) ** 2
    return f / f0


def fig_diffusion_process() -> None:
    """Forward/reverse diffusion illustrated on a real boundary's r(theta)."""
    xy = _load_xy(SHAPES / "generated" / "indist_4849_smooth.dat")
    theta, r = _radial(xy)
    r0 = (r - r.mean()) / r.std()  # standardise the radial field as the x_0 target

    rng = np.random.default_rng(0)
    eps = rng.standard_normal(r0.shape)
    fracs = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    abar = _cosine_alpha_bar(fracs)

    fig, axes = plt.subplots(1, 6, figsize=(13.5, 3.1), subplot_kw={"aspect": "equal"})
    for ax, frac, ab in zip(axes, fracs, abar):
        xt = np.sqrt(ab) * r0 + np.sqrt(1 - ab) * eps
        rr = xt * r.std() + r.mean()  # back to radius for plotting
        rr = np.clip(rr, 0.05, None)
        xs = rr * np.cos(theta)
        ys = rr * np.sin(theta)
        xs = np.append(xs, xs[0])
        ys = np.append(ys, ys[0])
        shade = plt.cm.viridis(0.15 + 0.7 * frac)
        ax.plot(xs, ys, "-", color=shade, lw=1.5)
        ax.fill(xs, ys, color=shade, alpha=0.12)
        ax.set_title(f"$t/T={frac:.1f}$", fontsize=10, color=TXT)
        ax.set_xticks([])
        ax.set_yticks([])
        lim = 2.2
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        for sp in ax.spines.values():
            sp.set_visible(False)

    fig.text(
        0.5,
        1.02,
        "forward  $q(x_t\\mid x_0)$: add noise  $\\longrightarrow$",
        ha="center",
        fontsize=11,
        color="#5b7fa6",
    )
    fig.text(
        0.5,
        -0.06,
        "$\\longleftarrow$  reverse  $p_\\theta(x_{t-1}\\mid x_t)$: denoise",
        ha="center",
        fontsize=11,
        color="#b05050",
    )
    out = ASSETS / "report_diffusion_process.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------- #
# Figure C — DCT encoding of a real Cp
# --------------------------------------------------------------------------- #
def fig_dct_encoding() -> None:
    """Real Cp sequence and its K=8 DCT-II reconstruction."""
    cp = _load_cp(SHAPES / "baseline" / "sim_0_ref_cp.csv")
    n = cp.shape[0]
    rec8 = dct_ii_inverse(dct_ii(cp, 8), n)
    rec3 = dct_ii_inverse(dct_ii(cp, 3), n)
    coeffs = dct_ii(cp, 8)

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.5, 3.9), width_ratios=[1.7, 1])
    idx = np.arange(n)
    axl.plot(idx, cp, color="#9aa6ad", lw=2.4, label="reference $C_p$ (N nodes)")
    axl.plot(idx, rec8, color=C_ENCODE_E, lw=1.8, label="$K{=}8$ reconstruction")
    axl.plot(
        idx, rec3, color=C_INPUT_E, lw=1.4, ls="--", label="$K{=}3$ reconstruction"
    )
    axl.set_xlabel("surface node index (around the boundary)", fontsize=9.5)
    axl.set_ylabel("$C_p$", fontsize=10)
    axl.legend(fontsize=8.6, frameon=False)
    axl.grid(alpha=0.25)

    axr.stem(np.arange(8), coeffs, linefmt=C_ENCODE_E, markerfmt="o", basefmt=" ")
    axr.axhline(0, color="#999", lw=0.7)
    axr.set_xlabel("DCT-II mode $k$", fontsize=9.5)
    axr.set_ylabel(r"coefficient (context vector $c\in\mathbb{R}^8$)", fontsize=9.5)
    axr.set_xticks(range(8))
    axr.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    out = ASSETS / "report_dct_encoding.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------- #
# Figure D — roughness metric on real raw vs smoothed boundaries
# --------------------------------------------------------------------------- #
def _roughness(r: np.ndarray) -> tuple[np.ndarray, float]:
    """Second difference of a periodic radial field and its mean magnitude."""
    d2 = r - 2 * np.roll(r, 1) + np.roll(r, 2)
    return d2, float(np.mean(np.abs(d2)))


def fig_roughness_metric() -> None:
    """Real raw vs Fourier-smoothed boundary and their second-difference."""
    raw = _load_xy(SHAPES / "generated" / "indist_4849_raw.dat")
    smooth = _load_xy(SHAPES / "generated" / "indist_4849_smooth.dat")
    th_r, r_raw = _radial(raw)
    th_s, r_sm = _radial(smooth)
    d2_raw, rough_raw = _roughness(r_raw)
    d2_sm, rough_sm = _roughness(r_sm)

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.5, 4.1), width_ratios=[1, 1.25])

    # radial profiles r(theta) — where the high-frequency wobble is visible
    axl.plot(th_r, r_raw, color=RAW_C, lw=1.3, label="raw (wobbly)")
    axl.plot(th_s, r_sm, color=SMOOTH_C, lw=1.8, label="Fourier-smoothed")
    axl.set_xlabel("angle $\\theta$ (rad)", fontsize=9.5)
    axl.set_ylabel("$r(\\theta)$", fontsize=10)
    axl.legend(fontsize=8.8, frameon=False, loc="upper right")
    axl.set_title(
        "Radial profile $r(\\theta)$: raw vs smoothed", fontsize=10.5, color=TXT
    )
    axl.grid(alpha=0.25)

    # second difference
    axr.plot(
        np.abs(d2_raw),
        color=RAW_C,
        lw=1.4,
        label=f"raw:   mean $|\\Delta^2 r|$ = {rough_raw:.4f}",
    )
    axr.plot(
        np.abs(d2_sm),
        color=SMOOTH_C,
        lw=1.4,
        label=f"smooth: mean $|\\Delta^2 r|$ = {rough_sm:.4f}",
    )
    axr.set_xlabel("surface node index", fontsize=9.5)
    axr.set_ylabel("$|\\Delta^2 r(\\theta)|$", fontsize=10)
    axr.set_title("Second-difference roughness diagnostic", fontsize=10.5, color=TXT)
    axr.legend(fontsize=8.8, frameon=False)
    axr.grid(alpha=0.25)

    fig.tight_layout()
    out = ASSETS / "report_roughness_metric.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    """Render and save the methodology figures in the report house style."""
    apply_house_style()
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig_conditioning_pipeline()
    fig_diffusion_process()
    fig_dct_encoding()


if __name__ == "__main__":
    main()
