"""Shared house style for first-year-report figures.

A single matplotlib style so every figure in the report shares fonts, sizes,
grid, and colour palette. Report figures rely on the LaTeX/Markdown *caption*
for their description, so in-plot titles are dropped (callers pass
``title=False`` where supported); axis labels use mathtext so $C_p$, $x/c$,
$Re$, $H$ render consistently without a LaTeX toolchain.

Usage::

    from _report_style import apply_house_style, PALETTE
    apply_house_style()
"""

from __future__ import annotations

from cycler import cycler

# Colour palette. Ordering matters: it maps onto matplotlib's ``C0..C4`` so the
# existing ``C0/C1/C2/C3`` choices in the plotting code keep their meaning
# (C0 primary/navy, C1 measured/orange, C2 drag/green, C3 magnitude/red).
NAVY = "#1f4e79"
ORANGE = "#e08214"
GREEN = "#2e7d32"
RED = "#c0392b"
PURPLE = "#6a3d9a"
GREY = "#5a5a5a"

PALETTE = {
    "navy": NAVY,
    "orange": ORANGE,
    "green": GREEN,
    "red": RED,
    "purple": PURPLE,
    "grey": GREY,
}

_RC = {
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.6,
    "legend.frameon": False,
    "axes.prop_cycle": cycler(color=[NAVY, ORANGE, GREEN, RED, PURPLE]),
}


def apply_house_style() -> None:
    """Apply the report house style to the global matplotlib rcParams."""
    import matplotlib

    matplotlib.rcParams.update(_RC)
