"""Render a compact 8-panel OOD gallery sized to fit one A4 report page.

The full ``ood_systematic`` gallery (16 panels, ``EXP-043_ood_gallery.png``)
is too tall to embed cleanly. This selects a representative subset of 8 cases
--- the positive OOD ladders along both principal components (pc1 and pc2 at
sigma = 1..4) --- and lays them out as a 4-row x 2-col grid at an A4-friendly
figure size. Left column is the pc1 ladder, right column the pc2 ladder, with
OOD distance increasing down the page. Reuses the shared ``plot_sample_gallery``
renderer and the on-disk loader from ``plot_sweep_results.py``.

Run from the repository root::

    python scripts/make_report_ood_gallery.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_sweep_results import _load_ood_systematic  # noqa: E402

from graph_diffusion.visualisation.sweep_results import (  # noqa: E402
    SampleRecord,
    plot_sample_gallery,
)

ASSETS = Path("docs/assets")

# Left column = pc1 ladder, right column = pc2 ladder; sigma increases down the
# page. plot_sample_gallery fills row-major with ncols=2, so interleave the two
# ladders to land pc1 in the left column and pc2 in the right.
_SUBSET = [
    "pc1_p1sig",
    "pc2_p1sig",
    "pc1_p2sig",
    "pc2_p2sig",
    "pc1_p3sig",
    "pc2_p3sig",
    "pc1_p4sig",
    "pc2_p4sig",
]


def _select(records: list[SampleRecord], names: list[str]) -> list[SampleRecord]:
    """Return the requested records in ``names`` order (raises on any missing)."""
    by_name = {r.name: r for r in records}
    return [by_name[n] for n in names]


def main() -> None:
    """Build the 8-panel A4 gallery and write PNG + PDF to ``docs/assets/``."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    samples = _select(_load_ood_systematic(), _SUBSET)

    # A4 usable text area with typical margins is ~6.3 x 9.7 in; keep a little
    # headroom below for the caption.
    fig = plot_sample_gallery(samples, ncols=2, figsize=(6.3, 8.6))

    for ext in ("png", "pdf"):
        fig.savefig(ASSETS / f"EXP-043_ood_gallery_a4.{ext}", dpi=200)
    print("wrote EXP-043_ood_gallery_a4.png / .pdf")


if __name__ == "__main__":
    main()
