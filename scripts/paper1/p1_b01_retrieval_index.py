"""
P1-B01 (analysis half) -- retrieval index and the difficulty axis.

Two jobs, both pure analysis:

  1. Regenerate the P1-F01 out-of-distribution ladder under the CORRECTED
     conditioning encoder. [[P1-D01]] section 5 requires this: F01's published
     ladder was computed with the sqrt(N)-scaling, spacing-blind descriptor, and
     must not appear in a figure until it has been recomputed. Both conventions
     are reported side by side so the change is visible rather than silent.

  2. Build the nearest-neighbour retrieval index P1-B01 needs, and record for
     every evaluation target its nearest training shape and the distance to it.
     Per the B01 spec, that distance "is the single best summary of how hard
     each target actually was" and becomes the x-axis of the key figure in 4.3.

The solver half of B01 -- pushing retrieved shapes through the pipeline at the
TARGET's Re and H -- needs the GPU and is not done here. This script writes the
retrieval assignments it will consume, so that step becomes a lookup rather than
a recomputation.

Distances are Euclidean in the K=8 descriptor space, which is the space the
model is actually conditioned on. That is the like-for-like comparison the spec
asks for: retrieval is given exactly the information the model gets.

Usage:
    python scripts/paper1/p1_b01_retrieval_index.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from graph_diffusion.data.pOnEllipse import (  # noqa: E402
    _H5_COL_P_START,
    _H5_COL_X,
    _H5_N_TIMESTEPS,
    _load_h5_raw,
    _node_count,
)
from graph_diffusion.data.pOnEllipseConditional import dct_ii  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_RAW = _ROOT / "data" / "ellipse" / "raw"
_SPLITS = _ROOT / "data" / "splits" / "paper1_partition.json"
_OUT = _ROOT / "outputs" / "paper1" / "P1-B01_retrieval-baseline"

K_MODES = 8
EVAL_SETS = ("indist", "lowre", "highre", "thin", "thick", "aoa10")
_FILES = {
    "indist": "pOnEllipseInDist.h5",
    "lowre": "pOnEllipseLowRe.h5",
    "highre": "pOnEllipseHighRe.h5",
    "thin": "pOnEllipseThin.h5",
    "thick": "pOnEllipseThick.h5",
    "aoa10": "pOnEllipseAoA10.h5",
}


def descriptor(sample: np.ndarray, norm: str) -> tuple[np.ndarray, int]:
    """Conditioning descriptor for one simulation, as the dataset builds it.

    Mirrors ``pOnEllipseConditionalDataset._build_graphs``: average Cp over the
    101 stored timesteps, sort by chord fraction, then transform. Under
    ``norm="forward"`` the sorted chord fractions are passed as quadrature
    abscissae, exactly as production now does.

    Args:
        sample (np.ndarray): One raw HDF5 record.
        norm (str): ``"forward"`` (production) or ``"ortho"`` (legacy).

    Returns:
        tuple[np.ndarray, int]: ``(descriptor, node_count)``.
    """
    n_nodes = _node_count(sample)
    data = sample[:n_nodes]

    cp = data[:, _H5_COL_P_START : _H5_COL_P_START + _H5_N_TIMESTEPS]
    cp_mean = cp.mean(axis=-1).astype(np.float32)

    xc = data[:, _H5_COL_X].astype(np.float32)
    xc = xc - xc.mean()
    x_over_c = (xc - float(xc.min())) / max(float(xc.max() - xc.min()), 1e-10)

    order = np.argsort(x_over_c)
    return (
        dct_ii(cp_mean[order], K_MODES, positions=x_over_c[order], norm=norm),
        n_nodes,
    )


def encode(path: Path, norm: str) -> tuple[np.ndarray, np.ndarray]:
    raw = _load_h5_raw(path)
    out = [descriptor(raw[i], norm) for i in range(raw.shape[0])]
    return np.stack([d for d, _ in out]), np.array([n for _, n in out])


def nn_against(query: np.ndarray, ref: np.ndarray, exclude_self: bool) -> tuple:
    """Nearest-neighbour distance and index from each query row into ``ref``."""
    d = np.linalg.norm(query[:, None, :] - ref[None, :, :], axis=-1)
    if exclude_self:
        d[np.arange(len(query)), np.argmin(d, axis=1)] = np.inf
    return d.min(axis=1), d.argmin(axis=1)


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    part = json.loads(_SPLITS.read_text())
    train_h5 = _ROOT / part["files"]["train"]["path"]
    train_idx = np.array(sorted(part["train_indices"]))

    results: dict[str, object] = {"k_modes": K_MODES}

    for norm in ("ortho", "forward"):
        tr_all, tr_nodes = encode(train_h5, norm)
        tr = tr_all[train_idx]  # index only the TRAIN split, never val

        # Within-train NN sets the scale. Subsample for cost, as F01 did, but
        # against the full training set so the reference is not thinned.
        rng = np.random.default_rng(0)
        sub = rng.choice(len(tr), size=min(800, len(tr)), replace=False)
        within, _ = nn_against(tr[sub], tr, exclude_self=True)
        w_med = float(np.median(within))

        tag = "corrected" if norm == "forward" else "legacy"
        print(f"\n=== {tag} encoder (norm={norm}) ===")
        print(f"  within-train median NN distance = {w_med:.6f}")
        print(f"  {'set':<10}{'n':>5}{'med NN':>12}{'ratio':>9}{'nodes':>12}")

        rung: dict[str, dict] = {}
        for name in EVAL_SETS:
            p = _RAW / _FILES[name]
            if not p.exists():
                continue
            q, q_nodes = encode(p, norm)
            dist, idx = nn_against(q, tr, exclude_self=False)
            ratio = float(np.median(dist) / max(w_med, 1e-12))
            rung[name] = {
                "n": int(len(q)),
                "median_nn": float(np.median(dist)),
                "ratio_to_within_train": ratio,
                "per_target_nn": [float(x) for x in dist],
                # Map back to the ORIGINAL training row so the solver half can
                # load the retrieved geometry directly.
                "retrieved_train_index": [int(train_idx[i]) for i in idx],
                "node_min": int(q_nodes.min()),
                "node_max": int(q_nodes.max()),
            }
            print(
                f"  {name:<10}{len(q):>5}{np.median(dist):>12.6f}{ratio:>8.2f}x"
                f"{f'{q_nodes.min()}-{q_nodes.max()}':>12}"
            )

        order = sorted(rung, key=lambda k: rung[k]["ratio_to_within_train"])
        print(f"  ladder order: {' < '.join(order)}")
        results[tag] = {
            "within_train_median_nn": w_med,
            "sets": rung,
            "ladder_order": order,
            "train_node_min": int(tr_nodes[train_idx].min()),
            "train_node_max": int(tr_nodes[train_idx].max()),
        }

    # Did the correction reorder the ladder?
    lo = results["legacy"]["ladder_order"]  # type: ignore[index]
    hi = results["corrected"]["ladder_order"]  # type: ignore[index]
    results["ladder_order_changed"] = lo != hi
    print(f"\n  ladder order changed by the correction: {lo != hi}")
    if lo != hi:
        print(f"    legacy    : {' < '.join(lo)}")
        print(f"    corrected : {' < '.join(hi)}")

    (_OUT / "retrieval_index.json").write_text(json.dumps(results, indent=2))
    print(f"\n  -> {(_OUT / 'retrieval_index.json').relative_to(_ROOT)}\n")


if __name__ == "__main__":
    main()
