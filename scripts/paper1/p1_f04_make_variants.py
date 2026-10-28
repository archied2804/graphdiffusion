"""
P1-F04 -- generate the node-phase variants of a body polygon.

One of F04's nuisance factors is "node resampling phase: 3 rotations of the same
polygon". That means the *same body* discretised with the node distribution
started at a different point — not a rotated body. Rolling the point list by a
fraction of a period leaves the geometry identical to floating point while
moving every node along the surface, so any change in the graded result is pure
discretisation noise.

Writing the roll as a fraction of the node count keeps the shift comparable
across shapes with different node counts (52-96 in this corpus).

Usage:
    python scripts/paper1/p1_f04_make_variants.py
"""

import shutil
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "cfd" / "ellipse2d" / "shapes" / "eval_grid"
_OUT = _ROOT / "cfd" / "ellipse2d" / "shapes" / "f04_variants"

# Three shapes spanning the envelope: low-Re thin, mid, high-Re thick.
SHAPES = {
    "lowre_thin": ("re0502_h60_idx0031", 31, 502.0, 5.989999771118164),
    "mid": ("re0750_h55_idx2824", 2824, 750.0, 5.5),
    "highre_thick": ("re0880_h60_idx4018", 4018, 880.0, 6.0),
}

PHASE_FRACTIONS = (0.0, 1.0 / 3.0, 2.0 / 3.0)


def read_dat(path: Path) -> tuple[list[str], np.ndarray]:
    header, pts = [], []
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            header.append(line)
            continue
        x, y = line.split()
        pts.append((float(x), float(y)))
    return header, np.array(pts)


def main() -> None:
    if not _SRC.exists():
        sys.exit(f"missing {_SRC}")
    _OUT.mkdir(parents=True, exist_ok=True)

    print(f"\nP1-F04 node-phase variants -> {_OUT.relative_to(_ROOT)}\n")
    for label, (name, idx, _re, _h) in SHAPES.items():
        src = _SRC / f"sim_{idx}.dat"
        if not src.exists():
            print(f"  MISSING {src.name}")
            continue
        header, pts = read_dat(src)
        n = len(pts)
        for k, frac in enumerate(PHASE_FRACTIONS):
            shift = int(round(frac * n))
            rolled = np.roll(pts, shift, axis=0)
            out = _OUT / f"{label}_phase{k}.dat"
            with out.open("w") as f:
                for h in header:
                    f.write(h + "\n")
                f.write(f"# P1-F04 node-phase variant: rolled {shift}/{n} nodes\n")
                for x, y in rolled:
                    f.write(f"{x:.10f} {y:.10f}\n")
            # The geometry must be bit-identical: only the ordering changed.
            assert np.allclose(np.sort(rolled, axis=0), np.sort(pts, axis=0))
        # Reference copy under the label, for the non-phase factors.
        shutil.copy(src, _OUT / f"{label}.dat")
        print(f"  {label:<14} {name:<24} n={n:<3} phases {PHASE_FRACTIONS}")

    print("\n  geometry verified identical under every roll\n")


if __name__ == "__main__":
    main()
