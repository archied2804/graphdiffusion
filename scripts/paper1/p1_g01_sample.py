"""
P1-G01 -- generate held-out inverse-design shapes and export them for the solver.

The paper's headline experiment. For each of the 20 frozen evaluation targets,
every one of the five P1-REF seeds is asked for a shape matching that target's
Cp, and the result is written as a solver body to be run at the **target's** Re
and H.

Two things make this the first honest measurement of the claim:

  clean targets   The 20 come from ``paper1_eval_targets.json`` -- drawn from
                  the test partition with a recorded seed, and excluding the 8
                  rows P1-F01 found duplicated in training. The Year-1 headline
                  was measured on training samples.

  a real comparator  P1-B01 has already run retrieval on these exact targets,
                  so the comparison the B01 spec asks for is a lookup once these
                  runs land.

Spec step 4 also requires the two Year-1 targets (#4849, #4066) to be graded
with this model and **labelled explicitly as training samples**. Placing the
leaked and clean numbers side by side is the clearest possible statement that
the problem was found and corrected, so they are generated here too.

The conditioning vector is built from the target's own stored Cp by the same
path the dataset uses (K=8 DCT-II, ``norm="forward"``), so the model is given
exactly what it was trained to consume.

Usage:
    python scripts/paper1/p1_g01_sample.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_ellipse_for_openlb import _ordered_polygon, _reference_cp  # noqa: E402
from p1_b01_retrieval_index import descriptor  # noqa: E402
from postprocess_exp020 import (  # noqa: E402
    _resolve_clamp_range,
    sample_shapes_for_target,
    shape_template_from_dataset,
)

from graph_diffusion.data.pOnEllipse import (  # noqa: E402
    _H5_COL_H,
    _H5_COL_RE,
    _load_h5_raw,
    _node_count,
)
from graph_diffusion.postprocessing.loaders import (  # noqa: E402
    build_dataset,
    build_model,
    load_config,
)

_RAW = _ROOT / "data" / "ellipse" / "raw"
_TARGETS = _ROOT / "data" / "splits" / "paper1_eval_targets.json"
_CONFIG = _ROOT / "configs" / "paper1" / "P1-F05_reference.yaml"
_REF = _ROOT / "outputs" / "paper1" / "P1-F05_reference"
_OUT = _ROOT / "cfd" / "ellipse2d" / "shapes" / "g01_holdout"

#: P1-REF seeds. PROTOCOL section 4 -- every later experiment reuses these.
SEEDS = (0, 1, 2, 3, 4)

#: Spec step 4. These two are TRAINING samples and must be labelled as such
#: wherever they appear; they are the Year-1 headline's evaluation targets.
YEAR1_TARGETS = (4849, 4066)

#: The conditioning convention P1-D01 corrected to and P1-F05 was trained with.
COND_NORM = "forward"

#: Physical chord, restored after sampling.
#:
#: ``feature_mode="radial_norm"`` trains on ``r(theta)/mean(r)``, so the model
#: cannot predict absolute size and the sampler returns a body whose *mean
#: radius* is 1 -- a semi-major axis of 1.13-1.36 where the corpus is 0.50.
#: Left uncorrected the body enters the channel roughly 2.5x oversized, and
#: since CHANNEL_HEIGHT is passed in absolute units the blockage ratio is then
#: 2.5x wrong. Cp and Cd would both be badly off, and it would read as the model
#: failing at inverse design rather than the export dropping a constant.
#:
#: Every shape in the corpus has chord exactly 1.0 (verified over the training
#: file, std = 0). So this is a **corpus-wide constant**, not the target's own
#: size: supplying it restores information the representation discards by
#: design, and leaks nothing about the individual target.
CHORD = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def to_physical(xy: np.ndarray) -> np.ndarray:
    """Centre a sampled body and restore the corpus chord.

    Args:
        xy (np.ndarray): Sampled boundary in mean-radius-normalised units,
            shape ``(N, 2)``.

    Returns:
        np.ndarray: Same shape, centred on the origin with chord ``CHORD``.
    """
    xy = np.asarray(xy, dtype=float)
    xy = xy - xy.mean(axis=0)
    span = float(xy[:, 0].max() - xy[:, 0].min())
    return xy * (CHORD / span) if span > 0 else xy


def write_dat(path: Path, xy: np.ndarray, header: str) -> None:
    """Write a generated boundary as a CCW closed polygon at physical scale.

    Args:
        path (Path): Destination ``.dat``.
        xy (np.ndarray): Boundary points, shape ``(N, 2)``.
        header (str): First comment line, without the leading ``#``.
    """
    poly = _ordered_polygon(to_physical(xy))
    with path.open("w") as fh:
        fh.write(f"# {header}\n# x y\n")
        for x, y in poly:
            fh.write(f"{x:.10f} {y:.10f}\n")


def write_ref(path: Path, sample: np.ndarray) -> None:
    """Write a target's dataset reference Cp curve.

    Args:
        path (Path): Destination CSV.
        sample (np.ndarray): Raw HDF5 record for the target.
    """
    data = sample[: _node_count(sample)]
    x, cp = _reference_cp(data)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x_over_c", "cp_ref"])
        for a, b in zip(x, cp, strict=True):
            w.writerow([f"{a:.6f}", f"{b:.6f}"])


def main() -> None:
    targets_meta = json.loads(_TARGETS.read_text())
    indist = targets_meta["sets"]["indist"]["target_indices"]

    config = load_config(str(_CONFIG))
    dataset = build_dataset(config)
    clamp_range = _resolve_clamp_range(config)
    k_modes = dataset.k_modes
    guidance = float(config.get("sampling", {}).get("guidance_scale", 3.0))

    test_raw = _load_h5_raw(_RAW / "pOnEllipseInDist.h5")
    train_raw = _load_h5_raw(_RAW / "pOnEllipseTrain.h5")

    _OUT.mkdir(parents=True, exist_ok=True)
    print(f"\nP1-G01 -- held-out inverse design ({DEVICE})\n")
    print(f"  {len(indist)} held-out targets x {len(SEEDS)} P1-REF seeds")
    print(f"  plus {len(YEAR1_TARGETS)} Year-1 TRAINING targets, labelled\n")
    print(f"  guidance = {guidance}   clamp = {clamp_range}   K = {k_modes}")

    # (label, raw array, row index, is_training_sample)
    jobs: list[tuple[str, np.ndarray, int, bool]] = [
        (f"holdout_t{t:03d}", test_raw, t, False) for t in indist
    ]
    jobs += [(f"year1_train_{t}", train_raw, t, True) for t in YEAR1_TARGETS]

    # Reference curves and conditioning vectors, once per target.
    conds: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}
    for label, raw, idx, is_train in jobs:
        sample = raw[idx]
        cond, n_nodes = descriptor(sample, COND_NORM)
        conds[label] = cond
        data = sample[: _node_count(sample)]
        meta[label] = {
            "target_idx": idx,
            "is_training_sample": is_train,
            "Re": float(data[0, _H5_COL_RE]),
            "H": float(data[0, _H5_COL_H]),
            "n_nodes_target": int(n_nodes),
        }
        write_ref(_OUT / f"{label}_ref_cp.csv", sample)

    rows: list[dict] = []
    template = shape_template_from_dataset(dataset, DEVICE)

    for seed in SEEDS:
        ckpt = _REF / f"seed{seed}" / "checkpoint_best.pt"
        if not ckpt.exists():
            print(f"  seed{seed}: MISSING {ckpt}")
            continue
        model = build_model(config, DEVICE)
        state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state.get("model_state_dict", state))
        model.eval()
        print(f"\n  seed{seed}: {ckpt.relative_to(_ROOT)}")

        for label, _raw, _idx, is_train in jobs:
            cond_t = torch.tensor(conds[label], dtype=torch.float32)
            shapes, _ = sample_shapes_for_target(
                model,
                template,
                cond_t,
                n_samples=1,
                guidance_scale=guidance,
                device=DEVICE,
                clamp_range=clamp_range,
                k_modes=k_modes,
            )
            xy = shapes[0]
            m = meta[label]
            tag = f"{label}_s{seed}"
            flag = " TRAINING-SAMPLE" if is_train else ""
            write_dat(
                _OUT / f"{tag}.dat",
                xy,
                f"P1-G01 generated: target={label} seed={seed} "
                f"Re={m['Re']:.4g} H={m['H']:.4g}{flag}",
            )
            rows.append(
                {
                    "tag": tag,
                    "label": label,
                    "seed": seed,
                    "target_idx": m["target_idx"],
                    "is_training_sample": int(m["is_training_sample"]),
                    "Re": m["Re"],
                    "H": m["H"],
                    "n_nodes_target": m["n_nodes_target"],
                    "n_nodes_generated": int(len(xy)),
                }
            )
        print(f"    {len(jobs)} shapes")

    man = _OUT / "manifest.csv"
    with man.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_hold = sum(1 for r in rows if not r["is_training_sample"])
    print(
        f"\n  {len(rows)} bodies ({n_hold} held-out, "
        f"{len(rows) - n_hold} labelled training) -> {_OUT.relative_to(_ROOT)}"
    )
    print(f"  manifest -> {man.relative_to(_ROOT)}\n")


if __name__ == "__main__":
    main()
