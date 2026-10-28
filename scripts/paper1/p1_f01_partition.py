"""
P1-F01 — data partition and leakage audit.

Downloads ``pOnEllipseTest.h5`` (never previously fetched), audits it against
``pOnEllipseTrain.h5`` for leakage, characterises the node-count and design-space
coverage of every partition, and freezes an explicit index partition to
``data/splits/paper1_partition.json``.

Every downstream Paper 1 experiment loads that file; nothing may re-derive a split
at runtime.  See ``DGN_Docs/paper1/specs/P1-F01_data-partition.md``.

Usage:
    python scripts/paper1/p1_f01_partition.py
    python scripts/paper1/p1_f01_partition.py --no-download   # audit local files only
"""

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from graph_diffusion.data.cp_manifold import dataset_cp_curves
from graph_diffusion.data.pOnEllipse import DatasetDownloader, DatasetUrl
from graph_diffusion.data.pOnEllipseConditional import dct_ii

# HDF5 column layout (mirrors graph_diffusion.data.pOnEllipse).
_COL_X, _COL_Y, _COL_RE, _COL_H = 0, 1, 2, 3

# Leakage tolerance from the spec: a test sample counts as leaked if a training
# sample exists within BOTH of these of it.
_LEAK_TOL_RE = 1.0
_LEAK_TOL_B = 1.0e-3

# Split parameters, fixed to match GraphDataLoader's historical behaviour so the
# train/val boundary is unchanged from EXP-022.
_VAL_SPLIT = 0.1
_SPLIT_SEED = 42

# The two targets the Year-1 report used as "in-distribution" evidence.
_YEAR1_TARGETS = (4849, 4066)

_ROOT = Path(__file__).resolve().parents[2]
_RAW = _ROOT / "data" / "ellipse" / "raw"
_OUT_JSON = _ROOT / "data" / "splits" / "paper1_partition.json"
_REPORT = _ROOT / "outputs" / "paper1" / "P1-F01_data-partition" / "audit.json"


def sha256(path: Path) -> str:
    """Return the SHA-256 of a file, streamed in 1 MiB blocks."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_raw(path: Path) -> np.ndarray:
    """Load the full ``data`` array from a pOnEllipse HDF5 file."""
    with h5py.File(path, "r") as f:
        return np.array(f["data"], dtype=np.float32)


def node_counts(raw: np.ndarray) -> np.ndarray:
    """Per-sample count of valid (non-NaN) boundary nodes."""
    x = raw[:, :, _COL_X]
    return (x == x).sum(axis=1).astype(int)


def design_coords(raw: np.ndarray) -> dict[str, np.ndarray]:
    """Extract per-sample design coordinates.

    ``a`` (semi-major) and ``b`` (semi-minor) are derived from the boundary
    extents rather than read from a column, so the same code works for every
    file regardless of which parameters were logged.

    Args:
        raw (np.ndarray): ``(N, MAX_NODES, 105)`` HDF5 array.

    Returns:
        dict[str, np.ndarray]: ``Re``, ``H``, ``a``, ``b``, ``n_nodes``.
    """
    n = node_counts(raw)
    a = np.empty(len(raw))
    b = np.empty(len(raw))
    for i in range(len(raw)):
        pts = raw[i, : n[i]]
        a[i] = 0.5 * float(pts[:, _COL_X].max() - pts[:, _COL_X].min())
        b[i] = 0.5 * float(pts[:, _COL_Y].max() - pts[:, _COL_Y].min())
    return {
        "Re": raw[:, 0, _COL_RE].astype(float),
        "H": raw[:, 0, _COL_H].astype(float),
        "a": a,
        "b": b,
        "n_nodes": n,
    }


def leak_audit(
    train: dict[str, np.ndarray], test: dict[str, np.ndarray]
) -> dict[str, object]:
    """Count test samples with a near-duplicate training sample.

    Applies the spec's tolerance: leaked iff some training sample satisfies
    ``|dRe| < 1.0`` AND ``|db| < 1e-3``.
    """
    leaked: list[dict[str, float]] = []
    for i in range(len(test["Re"])):
        d_re = np.abs(train["Re"] - test["Re"][i])
        d_b = np.abs(train["b"] - test["b"][i])
        hit = np.where((d_re < _LEAK_TOL_RE) & (d_b < _LEAK_TOL_B))[0]
        if hit.size:
            leaked.append(
                {
                    "test_idx": int(i),
                    "n_matches": int(hit.size),
                    "first_train_idx": int(hit[0]),
                }
            )
    return {"n_leaked": len(leaked), "examples": leaked[:10]}


def nn_distance(query: np.ndarray, ref: np.ndarray, exclude_self: bool) -> np.ndarray:
    """Nearest-neighbour Euclidean distance from each query row to ``ref``.

    Args:
        query (np.ndarray): ``(Nq, D)``.
        ref (np.ndarray): ``(Nr, D)``.
        exclude_self (bool): If ``True``, ignore the zero-distance self match
            (use when ``query`` and ``ref`` are the same set).

    Returns:
        np.ndarray: ``(Nq,)`` nearest distances.
    """
    out = np.empty(len(query))
    for i in range(len(query)):
        d = np.linalg.norm(ref - query[i], axis=1)
        if exclude_self:
            d[np.argmin(d)] = np.inf
        out[i] = d.min()
    return out


def standardise(arrays: list[np.ndarray]) -> list[np.ndarray]:
    """Z-score a list of arrays using statistics from the first."""
    mu, sd = arrays[0].mean(axis=0), arrays[0].std(axis=0) + 1e-12
    return [(a - mu) / sd for a in arrays]


def main() -> None:
    p = argparse.ArgumentParser(description="P1-F01 partition and leakage audit")
    p.add_argument("--no-download", action="store_true", help="Audit local files only")
    p.add_argument("--k-modes", type=int, default=8, help="DCT modes for Cp space")
    args = p.parse_args()

    _OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    _REPORT.parent.mkdir(parents=True, exist_ok=True)

    # --- 1. Acquire every published evaluation set ---------------------------
    # There is NO pOnEllipseTest.h5 upstream (the URL 404s and was never
    # fetched).  The publisher provides one in-distribution hold-out plus four
    # single-axis OOD sets, none of which had been downloaded.
    wanted = {
        "train": DatasetUrl.TRAIN_H5,
        "indist": DatasetUrl.INDIST_H5,
        "lowre": DatasetUrl.LOWRE_H5,
        "highre": DatasetUrl.HIGHRE_H5,
        "thin": DatasetUrl.THIN_H5,
        "thick": DatasetUrl.THICK_H5,
        "aoa10": DatasetUrl.AOA10_H5,
    }
    files: dict[str, Path] = {}
    for name, url in wanted.items():
        path = _RAW / url.value.rsplit("/", 1)[-1]
        if not path.exists() and not args.no_download:
            print(f"Downloading {path.name} ...")
            DatasetDownloader(root=str(_RAW)).download(url)
        if path.exists():
            files[name] = path
        else:
            print(f"  WARNING: {path.name} unavailable, skipping")
    raws = {k: load_raw(v) for k, v in files.items() if v.exists()}
    coords = {k: design_coords(v) for k, v in raws.items()}

    report: dict[str, object] = {"files": {}, "partitions": {}}
    for k, path in files.items():
        if k in raws:
            report["files"][k] = {  # type: ignore[index]
                "path": str(path.relative_to(_ROOT)),
                "sha256": sha256(path),
                "n_samples": int(raws[k].shape[0]),
                "array_shape": list(raws[k].shape),
            }

    # --- 2. Node-count characterisation -------------------------------------
    print("\n=== node counts (the array is PADDED to its max; counts vary) ===")
    for k, c in coords.items():
        hist = dict(sorted(collections.Counter(c["n_nodes"].tolist()).items()))
        report["partitions"][k] = {  # type: ignore[index]
            "n": int(len(c["Re"])),
            "node_count_histogram": {str(a): int(b) for a, b in hist.items()},
            "node_count_min": int(c["n_nodes"].min()),
            "node_count_max": int(c["n_nodes"].max()),
            "Re_range": [float(c["Re"].min()), float(c["Re"].max())],
            "H_range": [float(c["H"].min()), float(c["H"].max())],
            "b_range": [float(c["b"].min()), float(c["b"].max())],
        }
        print(
            f"  {k:6s} n={len(c['Re']):5d}  nodes {c['n_nodes'].min()}-"
            f"{c['n_nodes'].max()} over {len(hist)} distinct values"
        )
        print(f"         {hist}")

    # --- 3. Leakage audit ----------------------------------------------------
    print("\n=== leakage audit (each eval set vs train) ===")
    leaks: dict[str, object] = {}
    for k in coords:
        if k == "train":
            continue
        lk = leak_audit(coords["train"], coords[k])
        leaks[k] = lk
        print(
            f"  {k:7s} leaked: {lk['n_leaked']:3d} / {len(coords[k]['Re']):3d}"
            f"   (|dRe|<{_LEAK_TOL_RE}, |db|<{_LEAK_TOL_B})"
        )
    report["leakage"] = leaks

    # --- 4. Is 'held out' meaningful? cross-file vs within-train NN distance --
    print("\n=== nearest-neighbour distance: does 'held out' mean anything? ===")
    design_keys = ["Re", "H", "b"]
    tr_design = np.stack([coords["train"][k] for k in design_keys], axis=1)
    sets = {"train": tr_design}
    for k in coords:
        if k != "train":
            sets[k] = np.stack([coords[k][d] for d in design_keys], axis=1)
    std = standardise([sets["train"], *[v for k, v in sets.items() if k != "train"]])
    tr_z = std[0]
    others = dict(zip([k for k in sets if k != "train"], std[1:], strict=True))

    # Subsample train-vs-train for speed; the distribution is what matters.
    rng = np.random.default_rng(0)
    sub = rng.choice(len(tr_z), size=min(800, len(tr_z)), replace=False)
    within = nn_distance(tr_z[sub], tr_z, exclude_self=True)
    nn_report = {"within_train_median": float(np.median(within))}
    print(f"  within-train  median NN dist = {np.median(within):.4f}")
    for k, arr in others.items():
        d = nn_distance(arr, tr_z, exclude_self=False)
        nn_report[f"{k}_to_train_median"] = float(np.median(d))
        ratio = float(np.median(d) / max(np.median(within), 1e-12))
        nn_report[f"{k}_to_train_ratio"] = ratio
        print(
            f"  {k:6s}->train median NN dist = {np.median(d):.4f}"
            f"   ({ratio:.2f}x within-train)"
        )
    report["nn_distance"] = nn_report

    # --- 5. Same, in the K-mode DCT conditioning space -----------------------
    print("\n=== nearest-neighbour distance in DCT-Cp conditioning space ===")
    dct_sets = {}
    for k, raw in raws.items():
        curves = dataset_cp_curves(raw, n_grid=128)
        dct_sets[k] = np.stack([dct_ii(c, args.k_modes, norm="ortho") for c in curves])
    d_within = nn_distance(dct_sets["train"][sub], dct_sets["train"], True)
    dct_report = {"within_train_median": float(np.median(d_within))}
    print(f"  within-train  median = {np.median(d_within):.5f}")
    for k in dct_sets:
        if k == "train":
            continue
        d = nn_distance(dct_sets[k], dct_sets["train"], False)
        dct_report[f"{k}_to_train_median"] = float(np.median(d))
        dct_report[f"{k}_to_train_ratio"] = float(
            np.median(d) / max(np.median(d_within), 1e-12)
        )
        print(
            f"  {k:6s}->train median = {np.median(d):.5f}"
            f"   ({dct_report[f'{k}_to_train_ratio']:.2f}x within-train)"
        )
    report["nn_distance_dct"] = dct_report

    # --- 6. Freeze the partition --------------------------------------------
    n_total = len(coords["train"]["Re"])
    n_val = max(1, int(n_total * _VAL_SPLIT))
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(_SPLIT_SEED)
    tr_sub, va_sub = random_split(range(n_total), [n_train, n_val], generator=gen)
    train_idx = sorted(int(i) for i in tr_sub.indices)
    val_idx = sorted(int(i) for i in va_sub.indices)

    # --- 7. Confirm the Year-1 leak -----------------------------------------
    tr_set = set(train_idx)
    year1 = {str(t): ("TRAIN" if t in tr_set else "VAL") for t in _YEAR1_TARGETS}
    report["year1_targets"] = year1
    print("\n=== Year-1 'in-distribution' evaluation targets ===")
    for t, where in year1.items():
        print(f"  sim #{t}: {where}  <- reported as in-distribution generalisation")

    partition = {
        "created": "P1-F01",
        "split_seed": _SPLIT_SEED,
        "val_split": _VAL_SPLIT,
        "files": report["files"],
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_source": "pOnEllipseInDist.h5" if "indist" in raws else None,
        "n_test": int(len(coords["indist"]["Re"])) if "indist" in coords else 0,
        "ood_sources": {
            k: str(files[k].name) for k in coords if k not in ("train", "indist")
        },
        "n_ood": {
            k: int(len(coords[k]["Re"])) for k in coords if k not in ("train", "indist")
        },
        "year1_target_membership": year1,
        "leakage": report.get("leakage"),
    }
    _OUT_JSON.write_text(json.dumps(partition, indent=2))
    _REPORT.write_text(json.dumps(report, indent=2))

    print(
        f"\n  n_train={len(train_idx)}  n_val={len(val_idx)}"
        f"  n_test(indist)={partition['n_test']}"
    )
    print(f"  OOD sets: {partition['n_ood']}")
    print(f"  partition -> {_OUT_JSON.relative_to(_ROOT)}")
    print(f"  audit     -> {_REPORT.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
