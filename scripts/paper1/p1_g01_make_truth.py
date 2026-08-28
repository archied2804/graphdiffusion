"""
P1-G01 -- export the target geometries themselves, for the reference Cd.

The spec calls ``Cd`` generated-versus-target **"the engineering quantity -- a
designer asks for drag, not pressure correlation"**, and P1-F04 since showed it
is also the only metric with real dynamic range. Computing it needs the
target's own drag, and the dataset stores Cp, not forces.

So each target geometry is run through the **same** solver at its **own** Re and
H. Both sides of the comparison then pass through one instrument, which means
F07's constant 9.35% offset and every other shared systematic cancels -- a
dataset-side drag estimate would not have that property.

These runs are reused by P1-B01, whose retrieval bodies were graded against the
same targets.

Usage:
    python scripts/paper1/p1_g01_make_truth.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from export_ellipse_for_openlb import _ordered_polygon  # noqa: E402

from graph_diffusion.data.pOnEllipse import (  # noqa: E402
    _H5_COL_H,
    _H5_COL_RE,
    _H5_COL_X,
    _H5_COL_Y,
    _load_h5_raw,
    _node_count,
)

_RAW = _ROOT / "data" / "ellipse" / "raw"
_TARGETS = _ROOT / "data" / "splits" / "paper1_eval_targets.json"
_OUT = _ROOT / "cfd" / "ellipse2d" / "shapes" / "g01_truth"

YEAR1_TARGETS = (4849, 4066)


def main() -> None:
    meta = json.loads(_TARGETS.read_text())
    indist = meta["sets"]["indist"]["target_indices"]
    test_raw = _load_h5_raw(_RAW / "pOnEllipseInDist.h5")
    train_raw = _load_h5_raw(_RAW / "pOnEllipseTrain.h5")

    _OUT.mkdir(parents=True, exist_ok=True)
    print("\nP1-G01 -- target geometries for the reference Cd\n")

    jobs = [(f"holdout_t{t:03d}", test_raw, t, False) for t in indist]
    jobs += [(f"year1_train_{t}", train_raw, t, True) for t in YEAR1_TARGETS]

    rows: list[dict] = []
    for label, raw, idx, is_train in jobs:
        sample = raw[idx]
        data = sample[: _node_count(sample)]
        re = float(data[0, _H5_COL_RE])
        h = float(data[0, _H5_COL_H])
        xy = np.stack([data[:, _H5_COL_X], data[:, _H5_COL_Y]], axis=1)
        poly = _ordered_polygon(xy)
        tag = f"truth_{label}"
        with (_OUT / f"{tag}.dat").open("w") as fh:
            fh.write(
                f"# P1-G01 TARGET geometry {label}: Re={re:.4g} H={h:.4g} "
                f"n={len(poly)}\n# x y\n"
            )
            for x, y in poly:
                fh.write(f"{x:.10f} {y:.10f}\n")
        rows.append(
            {
                "tag": tag,
                "label": label,
                "seed": -1,
                "target_idx": idx,
                "is_training_sample": int(is_train),
                "Re": re,
                "H": h,
                "n_nodes_target": int(len(poly)),
                "n_nodes_generated": int(len(poly)),
            }
        )
        print(f"  {tag:<26} Re={re:>6.1f} H={h:.3g} n={len(poly)}")

    man = _OUT / "manifest.csv"
    with man.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  {len(rows)} bodies -> {_OUT.relative_to(_ROOT)}\n")


if __name__ == "__main__":
    main()
