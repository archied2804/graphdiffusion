"""
P1-F06 -- generate the Schaefer-Turek cylinder as an ellipse2d body polygon.

The ellipse2d case takes its geometry from a CCW point list, so the benchmark
cylinder is expressed as a regular polygon of diameter 0.1. That is deliberate:
running the benchmark geometry through the *same* loader, mesher, Bouzidi
treatment and force integration as every other experiment is what makes L1 a
test of this case setup rather than of a separate code path.

Benchmark geometry, Schaefer & Turek (1996), configurations 2D-1 and 2D-2:

    channel   2.2 x 0.41
    cylinder  diameter D = 0.1, centre (0.2, 0.2)

The centre is 0.005 *below* the channel centreline (0.205). That asymmetry is
deliberate in the benchmark -- a perfectly symmetric configuration will not shed
in 2D-2 -- and is reproduced with the BODY_Y_OFFSET case parameter.

Polygon discretisation is a real approximation: an n-gon inscribed in a circle
of radius r has area deficit 1 - (n/2pi)sin(2pi/n), which the script reports so
the deviation is stated rather than assumed. 200 points gives < 0.02%.

Usage:
    python scripts/paper1/p1_f06_make_cylinder.py [--n-points 200]
"""

import argparse
import math
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_OUT = _ROOT / "cfd" / "ellipse2d" / "shapes" / "schafer_turek"

DIAMETER = 0.1
CHANNEL_HEIGHT = 0.41
CENTRE_X = 0.2
CENTRE_Y = 0.2


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-points", type=int, default=200)
    args = p.parse_args()
    n = args.n_points

    _OUT.mkdir(parents=True, exist_ok=True)
    r = DIAMETER / 2.0

    # CCW, matching the pOnEllipse convention. The polygon is written centred on
    # the origin; placement in the domain is the case's job (UPSTREAM_CHORDS and
    # BODY_Y_OFFSET), exactly as for every dataset shape.
    pts = [
        (r * math.cos(2.0 * math.pi * i / n), r * math.sin(2.0 * math.pi * i / n))
        for i in range(n)
    ]

    area_poly = 0.5 * n * r * r * math.sin(2.0 * math.pi / n)
    area_circ = math.pi * r * r
    deficit = 1.0 - area_poly / area_circ

    path = _OUT / "cylinder_d0p1.dat"
    with path.open("w") as f:
        f.write(
            f"# Schaefer-Turek cylinder, D={DIAMETER}, {n}-gon "
            f"(area deficit {deficit * 100:.4f}%), CCW\n"
        )
        f.write("# x y\n")
        for x, y in pts:
            f.write(f"{x:.10f} {y:.10f}\n")

    # The case centres the body at physLengthY/2; the benchmark wants CENTRE_Y.
    y_offset = CENTRE_Y - CHANNEL_HEIGHT / 2.0

    print(f"\nwrote {path.relative_to(_ROOT)}  ({n} points)")
    print(f"  polygon area deficit vs true circle : {deficit * 100:.4f}%")
    print(f"  chord (x-extent)                    : {2 * r:.6f}")
    print("\ncase parameters for Schaefer-Turek:")
    print(f"  --CHANNEL_HEIGHT   {CHANNEL_HEIGHT}")
    print(
        f"  --BODY_Y_OFFSET    {y_offset:+.4f}   "
        f"(centre {CENTRE_Y} vs centreline {CHANNEL_HEIGHT / 2})"
    )
    print(
        f"  --UPSTREAM_CHORDS  {(CENTRE_X - r) / DIAMETER:.2f}   "
        f"(inlet to leading edge {CENTRE_X - r:.2f})"
    )
    print(
        f"  --DOWNSTREAM_CHORDS {(2.2 - CENTRE_X - r) / DIAMETER:.2f}   "
        f"(trailing edge to outlet {2.2 - CENTRE_X - r:.2f})"
    )
    print("  --INLET_PROFILE    1")
    print("  --WALL_NOSLIP      1")
    print("  --SPONGE_CHORDS    0")
    print("  2D-1: --REYNOLDS 20  --PHYS_CHAR_VELOCITY 0.2")
    print("  2D-2: --REYNOLDS 100 --PHYS_CHAR_VELOCITY 1.0\n")


if __name__ == "__main__":
    main()
