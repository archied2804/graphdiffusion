"""Generate the score-network architecture figure for the first-year report.

Produces ``docs/assets/report_architecture.png`` — a schematic of the conditional
graph-network DDPM score model used for inverse design:

* inputs: the noisy boundary ``x_t = r(theta)`` on a ring graph ``G_t``, the diffusion
  timestep ``t``, and the target pressure ``Cp(x/c)``;
* the target is DCT-encoded (first K=8 type-II coefficients) into a fixed-size context;
* the sinusoidal time embedding and the projected conditioning are *summed once* into the
  per-graph global attribute ``u`` (with a learnable null token for classifier-free
  guidance), not concatenated onto every node;
* ``u`` propagates through ``N`` stacked Battaglia GN blocks (EdgeModel phi^e ->
  NodeModel phi^v -> GlobalModel phi^u) along the boundary ring edges;
* an output MLP predicts the injected noise ``eps``;
* a jointly-trained DeepSets pressure head maps the denoised shape back to a predicted
  DCT spectrum, supplying the geometry->pressure coupling gradient.

The layout mirrors ``ScoreNetwork.forward`` in
``src/graph_diffusion/model/score_network.py``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ASSETS = Path("docs/assets")

# Palette (muted, print-friendly).
C_INPUT = "#e8eef5"
C_INPUT_E = "#5b7fa6"
C_ENCODE = "#fdf0e2"
C_ENCODE_E = "#d08a3e"
C_CORE = "#e9f3ec"
C_CORE_E = "#4f9a6a"
C_GLOBAL = "#f3e9f3"
C_GLOBAL_E = "#9a4f8a"
C_OUT = "#f5e9e9"
C_OUT_E = "#b05050"
TXT = "#1f2a33"


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    w: float,
    h: float,
    text: str,
    face: str,
    edge: str,
    fontsize: float = 10.5,
    bold: bool = False,
) -> tuple[float, float]:
    """Draw a rounded box with centred text; return its centre.

    Args:
        ax (plt.Axes): Target axes.
        xy (tuple[float, float]): Bottom-left corner ``(x, y)``.
        w (float): Width.
        h (float): Height.
        text (str): Box label (LaTeX-style maths permitted).
        face (str): Fill colour.
        edge (str): Edge colour.
        fontsize (float): Text size. Defaults to ``10.5``.
        bold (bool): Bold text. Defaults to ``False``.

    Returns:
        tuple[float, float]: Box centre ``(cx, cy)``.
    """
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.4,
        facecolor=face,
        edgecolor=edge,
        mutation_aspect=1.0,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=TXT,
        weight="bold" if bold else "normal",
        zorder=5,
    )
    return x + w / 2, y + h / 2


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#3a4750",
    style: str = "-|>",
    lw: float = 1.6,
    connection: str = "arc3,rad=0.0",
) -> None:
    """Draw an annotation arrow between two points."""
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle=style,
            color=color,
            lw=lw,
            connectionstyle=connection,
            shrinkA=2,
            shrinkB=2,
        ),
        zorder=4,
    )


def _ring_glyph(ax: plt.Axes, cx: float, cy: float, r: float) -> None:
    """Draw a small wobbly closed boundary on a ring graph (the x_t glyph)."""
    theta = np.linspace(0, 2 * np.pi, 13)
    rad = r * (1 + 0.18 * np.sin(3 * theta + 0.6) + 0.10 * np.cos(5 * theta))
    xs = cx + rad * np.cos(theta)
    ys = cy + rad * np.sin(theta)
    ax.plot(xs, ys, "-", color=C_INPUT_E, lw=1.3, zorder=6)
    ax.plot(xs, ys, "o", color=C_INPUT_E, ms=3.0, zorder=6)


def main() -> None:
    """Render and save the architecture figure."""
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 12.5)
    ax.axis("off")

    # ----- Inputs (left column) -----
    _box(ax, (0.4, 8.3), 4.2, 2.6, "", C_INPUT, C_INPUT_E)
    _ring_glyph(ax, 1.7, 9.6, 0.85)
    ax.text(
        3.55,
        9.6,
        "noisy\nboundary\n$\\mathbf{x}_t=r(\\theta)$\non ring $G_t$",
        ha="center",
        va="center",
        fontsize=9.5,
        color=TXT,
    )
    in_t = _box(ax, (0.4, 5.6), 4.2, 1.5, "timestep $t$", C_INPUT, C_INPUT_E)
    in_cp = _box(
        ax, (0.4, 2.7), 4.2, 1.7, "target\n$C_p(x/c)$", C_INPUT, C_INPUT_E
    )

    # ----- Encoders -----
    enc_dct = _box(
        ax, (6.0, 2.75), 3.6, 1.6, "DCT\n(first $K{=}8$)", C_ENCODE, C_ENCODE_E
    )
    enc_time = _box(
        ax,
        (6.0, 5.55),
        3.6,
        1.6,
        "sinusoidal\ntime embed",
        C_ENCODE,
        C_ENCODE_E,
    )

    # ----- Global attribute u (sum junction) -----
    u_c = _box(
        ax,
        (11.0, 4.35),
        3.5,
        2.6,
        "global $\\mathbf{u}$\n$\\leftarrow\\,\\mathbf{u}+$ time $+$ cond",
        C_GLOBAL,
        C_GLOBAL_E,
        bold=True,
    )
    ax.text(
        12.75,
        3.95,
        "(CFG null token\nfor cond dropout)",
        ha="center",
        va="center",
        fontsize=7.8,
        color="#6b4d66",
        style="italic",
    )

    # ----- GN core stack -----
    _box(ax, (15.6, 3.5), 5.2, 5.4, "", C_CORE, C_CORE_E)
    ax.text(
        18.2,
        8.35,
        "GN block  $\\times\\,N$",
        ha="center",
        va="center",
        fontsize=11,
        color=TXT,
        weight="bold",
    )
    _box(ax, (16.1, 6.7), 4.2, 1.2, "EdgeModel  $\\phi^e$", "#ffffff", C_CORE_E, 9.8)
    _box(ax, (16.1, 5.35), 4.2, 1.2, "NodeModel  $\\phi^v$", "#ffffff", C_CORE_E, 9.8)
    _box(ax, (16.1, 4.0), 4.2, 1.2, "GlobalModel  $\\phi^u$", "#ffffff", C_CORE_E, 9.8)
    _arrow(ax, (18.2, 6.7), (18.2, 6.55))
    _arrow(ax, (18.2, 5.35), (18.2, 5.2))
    # recurrence hint
    ax.annotate(
        "",
        xy=(20.55, 6.9),
        xytext=(20.55, 4.3),
        arrowprops=dict(
            arrowstyle="-|>",
            color=C_CORE_E,
            lw=1.3,
            connectionstyle="arc3,rad=-0.55",
        ),
    )

    # ----- Output -----
    out_mlp = _box(ax, (21.9, 6.6), 2.7, 1.5, "output\nMLP", C_OUT, C_OUT_E)
    out_eps = _box(
        ax,
        (21.9, 4.0),
        2.7,
        1.6,
        "$\\hat{\\boldsymbol{\\varepsilon}}$\npred. noise",
        C_OUT,
        C_OUT_E,
        bold=True,
    )

    # ----- Pressure head branch (auxiliary) -----
    head = _box(
        ax,
        (15.6, 0.5),
        5.2,
        1.9,
        "DeepSets pressure head $h(\\hat{\\mathbf{x}}_0,\\,\\mathrm{pos})$",
        "#eef2ef",
        C_CORE_E,
        9.8,
    )
    aux = _box(
        ax,
        (21.9, 0.6),
        2.7,
        1.7,
        "pred. DCT\nspectrum",
        "#eef2ef",
        C_CORE_E,
        9.0,
    )

    # ----- Arrows -----
    _arrow(ax, (4.6, 9.6), (15.6, 7.4))  # noisy graph -> core
    _arrow(ax, (in_t[0] + 2.1, in_t[1]), (6.0, 6.35))  # t -> time embed
    _arrow(ax, (in_cp[0] + 2.1, in_cp[1]), (6.0, 3.55))  # Cp -> DCT
    _arrow(ax, (9.6, 6.35), (11.0, 6.0))  # time embed -> u
    _arrow(ax, (9.6, 3.55), (11.0, 5.2))  # DCT cond -> u
    _arrow(ax, (14.5, 5.65), (15.6, 5.9))  # u -> core
    _arrow(ax, (20.8, 7.3), (21.9, 7.35))  # core -> output MLP
    _arrow(ax, (out_mlp[0], 6.6), (out_eps[0], 5.6))  # MLP -> eps
    # aux branch: denoised shape -> head -> spectrum
    _arrow(ax, (18.2, 3.5), (18.2, 2.4), color=C_CORE_E, style="-|>")
    ax.text(
        18.9, 2.95, "$\\hat{\\mathbf{x}}_0$", ha="left", va="center", fontsize=9.5, color=TXT
    )
    _arrow(ax, (20.8, 1.45), (21.9, 1.45), color=C_CORE_E)
    ax.text(
        23.25,
        2.55,
        "aux. loss\n$\\to C_p$ coupling",
        ha="center",
        va="bottom",
        fontsize=7.8,
        color="#3a6b50",
        style="italic",
    )

    # ----- Legend / caption strip -----
    handles = [
        mpatches.Patch(facecolor=C_INPUT, edgecolor=C_INPUT_E, label="inputs"),
        mpatches.Patch(facecolor=C_ENCODE, edgecolor=C_ENCODE_E, label="encoders"),
        mpatches.Patch(
            facecolor=C_GLOBAL, edgecolor=C_GLOBAL_E, label="global attribute $u$"
        ),
        mpatches.Patch(facecolor=C_CORE, edgecolor=C_CORE_E, label="GN core / aux head"),
        mpatches.Patch(facecolor=C_OUT, edgecolor=C_OUT_E, label="noise prediction"),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.005, 0.045),
        ncol=5,
        frameon=False,
        fontsize=8.5,
        handlelength=1.4,
        columnspacing=1.3,
    )

    ax.set_title(
        "Conditional graph-network DDPM score model $\\varepsilon_\\theta(\\mathbf{x}_t,\\,t,\\,c)$",
        fontsize=12.5,
        color=TXT,
        pad=8,
    )

    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / "report_architecture.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
