"""Render the LBM ground-truth setup as a slide-ready PNG table.

A single house-styled table summarising the OpenLB ``ellipse2d`` case as it was
ACTUALLY RUN for the DSFD 2026 talk (not the source-file defaults): every
validation run scaled resolution with Re to hold Mach ~ 0.13 fixed. Numbers are
taken from the UnitConverter printouts and Timer summaries in
``cfd/ellipse2d/runs/*/run.log`` (e.g. Re501->N100->Ma0.130->60s;
Re998->N200->Ma0.130->~7min) and from the case headers.

Output: ``docs/assets/EXP-043_lbm_setup_table.png`` (+ .pdf), sized to drop
straight onto a slide. Run from the repo root::

    python scripts/make_lbm_setup_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for sibling modules
from _report_style import NAVY, apply_house_style  # noqa: E402

ASSETS = Path("docs/assets")

# (Quantity, Value, Why this choice) — verified against runs/*/run.log.
ROWS: list[tuple[str, str, str]] = [
    ("Method", "D2Q9, single-relaxation BGK",
     "simplest collision that is adequate in the laminar regime"),
    ("Reynolds number", "500 – 1000",
     "matches the pOnEllipse dataset; laminar shedding, St $\\approx$ 0.2"),
    ("Mach number", "$\\approx$ 0.13  (held fixed)",
     "weakly-compressible limit; error $O(\\mathrm{Ma}^2)\\approx$ 1.7%"),
    ("Relaxation  $\\tau$ / $\\omega$", "0.545 / 1.835",
     "fixed; $\\tau-0.5=0.045$ stability margin, $\\omega<2$"),
    ("Resolution", "N $\\approx$ 0.2$\\cdot$Re cells/chord (100$\\to$200)",
     "scaled with Re to pin Ma $\\approx$ 0.13 as $\\nu$ drops"),
    ("Knudsen number", "$\\approx 3\\times10^{-4}$",
     "$\\ll$ 1 $\\Rightarrow$ continuum / Navier-Stokes limit valid"),
    ("Wall model", "Bouzidi curved no-slip",
     "sub-grid wall distance; no staircasing on arbitrary polygons"),
    ("Turbulence model", "none (DNS-resolved)",
     "2D laminar, Re $\\leq$ 1000 — no LES needed"),
    ("Domain", "24 chords (6$\\cdot$body$\\cdot$17) + sponge outlet",
     "open free-stream box; sponge kills acoustic reflections"),
    ("Time-averaging", "$\\geq$10 shedding periods after washout",
     "unsteady shedding $\\Rightarrow$ mean $C_d$, $C_l$, surface $C_p$"),
    ("Cost (1 GPU, $\\approx$1500 MLUPs)", "60 s (Re500) $\\to$ 7.5 min (Re1000)",
     "per shape — cheap enough to sit in the design loop"),
    ("Validation", "surface $C_p$ vs dataset, $\\rho$ = 0.97 – 0.999",
     "calibrates the solver as an independent ground truth"),
]

COL_HEADERS = ("Quantity", "Value (as run)", "Why this choice")
COL_WIDTHS = (0.24, 0.34, 0.42)


def make_table() -> None:
    """Render ROWS as a house-styled table PNG/PDF in docs/assets."""
    apply_house_style()

    n_rows = len(ROWS)
    fig_h = 0.46 * (n_rows + 1) + 0.3
    fig, ax = plt.subplots(figsize=(12.0, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=[list(r) for r in ROWS],
        colLabels=COL_HEADERS,
        colWidths=list(COL_WIDTHS),
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.55)

    n_cols = len(COL_HEADERS)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(1.2)
        cell.PAD = 0.03
        if row == 0:  # header
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", weight="bold")
            cell.set_height(cell.get_height() * 1.05)
        else:
            cell.set_facecolor("#eef2f7" if row % 2 else "#ffffff")
            if col == 0:
                cell.set_text_props(weight="bold", color=NAVY)
        # keep right edge of last column clean
        if col == n_cols - 1:
            cell.set_linewidth(1.2)

    fig.tight_layout(pad=0.4)
    ASSETS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = ASSETS / f"EXP-043_lbm_setup_table.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    make_table()
