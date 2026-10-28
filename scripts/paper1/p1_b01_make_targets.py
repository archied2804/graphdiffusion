"""
P1-B01 -- fix the evaluation target set and export the retrieved bodies.

Two jobs.

**Fix the targets.** [[P1-G01]] specifies "20 targets drawn uniformly at random
from the test partition, with a recorded selection seed", and says that draw
fixes the evaluation set for A04, C01-C03 and S01-S03 as well. G01 has not run,
and B01 cannot be compared against it unless both use the same targets -- so the
draw is made here, written to ``data/splits/paper1_eval_targets.json``, and G01
consumes it rather than re-drawing.

The draw excludes the **8 in-distribution test samples P1-F01 found duplicated in
training** (and the 4 in AoA10). A target that already sits in the training set
measures memorisation, not generalisation, and retrieval would score perfectly on
it by construction -- which would flatter the baseline rather than the model.

**Export the retrieved bodies.** For each target, ``p1_b01_retrieval_index.py``
already recorded the nearest training shape in K=8 DCT-Cp space -- the exact
vector the model is conditioned on, so retrieval is given precisely the
information the model gets. This writes that shape as a solver body, together
with the **target's** reference Cp curve.

The bodies are run at the **target's** Re and H, not the retrieved sample's. That
is the honest comparison the spec insists on: the retrieved geometry is asked to
perform at a condition it was not drawn from, exactly as a generated one is.

Usage:
    python scripts/paper1/p1_b01_make_targets.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export_ellipse_for_openlb import _ordered_polygon, _reference_cp  # noqa: E402

from graph_diffusion.data.pOnEllipse import (  # noqa: E402
    _H5_COL_H,
    _H5_COL_RE,
    _H5_COL_X,
    _H5_COL_Y,
    _load_h5_raw,
    _node_count,
)

_ROOT = Path(__file__).resolve().parents[2]
_RAW = _ROOT / "data" / "ellipse" / "raw"
_SPLITS = _ROOT / "data" / "splits" / "paper1_partition.json"
_TARGETS = _ROOT / "data" / "splits" / "paper1_eval_targets.json"
_INDEX = (
    _ROOT / "outputs" / "paper1" / "P1-B01_retrieval-baseline" / "retrieval_index.json"
)
_OUT = _ROOT / "cfd" / "ellipse2d" / "shapes" / "b01_retrieval"

#: Recorded per P1-G01 method step 1. Never change once the draw is published.
TARGET_SEED = 42
N_INDIST = 20
N_PER_OOD = 3
OOD_SETS = ("lowre", "highre", "thin", "thick", "aoa10")

_FILES = {
    "indist": "pOnEllipseInDist.h5",
    "lowre": "pOnEllipseLowRe.h5",
    "highre": "pOnEllipseHighRe.h5",
    "thin": "pOnEllipseThin.h5",
    "thick": "pOnEllipseThick.h5",
    "aoa10": "pOnEllipseAoA10.h5",
}


def leaked_indices(part: dict, name: str) -> set[int]:
    """Test-set rows P1-F01 found duplicated in the training partition.

    Args:
        part (dict): The frozen partition record.
        name (str): Evaluation set name.

    Returns:
        set[int]: Row indices to exclude from the draw.
    """
    rec = part.get("leakage", {}).get(name, {})
    return {int(e["test_idx"]) for e in rec.get("examples", [])}


def draw(n_total: int, n_take: int, exclude: set[int], seed: int) -> list[int]:
    """Draw uniformly at random from the clean rows of an evaluation set.

    Args:
        n_total (int): Number of rows in the file.
        n_take (int): How many to draw.
        exclude (set[int]): Rows to leave out (leaked into training).
        seed (int): Recorded selection seed.

    Returns:
        list[int]: Sorted drawn indices.
    """
    pool = np.array([i for i in range(n_total) if i not in exclude])
    rng = np.random.default_rng(seed)
    take = min(n_take, pool.size)
    return sorted(int(i) for i in rng.choice(pool, size=take, replace=False))


def write_body(path: Path, sample: np.ndarray, header: str) -> None:
    """Write one HDF5 sample's boundary as a CCW solver body.

    Args:
        path (Path): Destination ``.dat``.
        sample (np.ndarray): Raw HDF5 record.
        header (str): First comment line, without the leading ``#``.
    """
    data = sample[: _node_count(sample)]
    xy = np.stack([data[:, _H5_COL_X], data[:, _H5_COL_Y]], axis=1)
    poly = _ordered_polygon(xy)
    with path.open("w") as fh:
        fh.write(f"# {header}\n# x y\n")
        for x, y in poly:
            fh.write(f"{x:.10f} {y:.10f}\n")


def write_ref(path: Path, sample: np.ndarray) -> None:
    """Write one sample's dataset reference Cp curve.

    Args:
        path (Path): Destination CSV.
        sample (np.ndarray): Raw HDF5 record.
    """
    data = sample[: _node_count(sample)]
    x, cp = _reference_cp(data)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x_over_c", "cp_ref"])
        for a, b in zip(x, cp, strict=True):
            w.writerow([f"{a:.6f}", f"{b:.6f}"])


def main() -> None:
    part = json.loads(_SPLITS.read_text())
    index = json.loads(_INDEX.read_text())["corrected"]["sets"]
    train = _load_h5_raw(_RAW / "pOnEllipseTrain.h5")

    _OUT.mkdir(parents=True, exist_ok=True)
    print(f"\nP1-B01 -- evaluation targets and retrieved bodies (seed {TARGET_SEED})\n")

    targets: dict[str, dict] = {}
    rows: list[dict] = []

    for name in ("indist", *OOD_SETS):
        path = _RAW / _FILES[name]
        if not path.exists() or name not in index:
            print(f"  {name:<9} SKIP (missing file or index entry)")
            continue
        raw = _load_h5_raw(path)
        excl = leaked_indices(part, name)
        n_take = N_INDIST if name == "indist" else N_PER_OOD
        picks = draw(raw.shape[0], n_take, excl, TARGET_SEED)
        retrieved = index[name]["retrieved_train_index"]
        nn_dist = index[name]["per_target_nn"]

        targets[name] = {
            "file": _FILES[name],
            "n_available": int(raw.shape[0]),
            "excluded_leaked": sorted(excl),
            "target_indices": picks,
        }
        print(
            f"  {name:<9} {len(picks)} targets from "
            f"{raw.shape[0] - len(excl)} clean of {raw.shape[0]}"
            f"{f' (excluded {sorted(excl)})' if excl else ''}"
        )

        for t in picks:
            tgt = raw[t]
            tdata = tgt[: _node_count(tgt)]
            re = float(tdata[0, _H5_COL_RE])
            h = float(tdata[0, _H5_COL_H])
            r_idx = int(retrieved[t])
            tag = f"{name}_t{t:03d}"

            write_body(
                _OUT / f"{tag}_retrieved.dat",
                train[r_idx],
                f"P1-B01 {tag}: retrieved train #{r_idx}, run at TARGET "
                f"Re={re:.4g} H={h:.4g}",
            )
            write_ref(_OUT / f"{tag}_ref_cp.csv", tgt)
            rows.append(
                {
                    "tag": tag,
                    "set": name,
                    "target_idx": t,
                    "retrieved_train_idx": r_idx,
                    "nn_distance": float(nn_dist[t]),
                    "Re": re,
                    "H": h,
                    "n_nodes_target": int(_node_count(tgt)),
                    "n_nodes_retrieved": int(_node_count(train[r_idx])),
                }
            )

    man = _OUT / "manifest.csv"
    with man.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    _TARGETS.write_text(
        json.dumps(
            {
                "created": "P1-B01",
                "purpose": (
                    "Frozen evaluation target set. P1-G01 method step 1 draws "
                    "these; every later experiment comparing against G01 must "
                    "use the same list."
                ),
                "selection_seed": TARGET_SEED,
                "n_indist": N_INDIST,
                "n_per_ood": N_PER_OOD,
                "leaked_excluded": True,
                "sets": targets,
            },
            indent=2,
        )
    )
    print(f"\n  {len(rows)} bodies -> {_OUT.relative_to(_ROOT)}")
    print(f"  target set  -> {_TARGETS.relative_to(_ROOT)}\n")


if __name__ == "__main__":
    main()
