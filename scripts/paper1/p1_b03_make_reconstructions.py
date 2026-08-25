"""
P1-B03 -- build solver-ready reconstructions of held-out shapes.

The analytic half of B03 measured PCA reconstruction error in units of
normalised radius. That is a *geometric* tolerance, and the spec is explicit
that explained variance is not a physically meaningful one:

    "Convert geometric error into a physical tolerance. Take 4 held-out shapes,
     reconstruct each at M = 3 and at M = 10, and push original and
     reconstructed through the solver. Measure the change in Cp."

This script writes the bodies. For each selected held-out shape it emits:

    <label>_orig.dat        the shape exactly as the dataset stores it
    <label>_roundtrip.dat   resampled to the 128-point theta grid and back,
                            with no truncation
    <label>_pca01.dat       PCA reconstruction, M = 1
    <label>_pca02.dat                            M = 2
    <label>_pca03.dat                            M = 3   (registered)
    <label>_pca10.dat                            M = 10  (registered)
    <label>_ref_cp.csv     the dataset's own surface pressure curve, so every
                           variant can be graded through the standard path

``roundtrip`` is the control the spec does not ask for and the comparison needs.
Reconstructing on a uniform theta grid and interpolating back to the body's own
node angles is itself lossy, so ``orig`` vs ``pca03`` would confound basis
truncation with resampling. ``roundtrip`` isolates it: ``roundtrip`` vs ``pcaNN``
is truncation alone, ``orig`` vs ``roundtrip`` is the resampling artefact, and if
the latter is not far below the F04 noise floor the measurement does not mean
what the spec assumes.

M = 1 and M = 2 are added for the same reason. The analytic half found the
manifold is effectively one-dimensional, so the whole reachability question now
lives below M = 3 -- the registered levels both sit in the flat part of the
curve. Running them costs nothing extra to set up.

The PCA basis is fitted on the **training** partition only, and the shapes are
drawn from the **val** partition. The test split is untouched.

Usage:
    python scripts/paper1/p1_b03_make_reconstructions.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from graph_diffusion.data.pOnEllipse import (  # noqa: E402
    _H5_COL_H,
    _H5_COL_RE,
    _H5_COL_X,
    _H5_COL_Y,
    _load_h5_raw,
    _node_count,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export_ellipse_for_openlb import _reference_cp  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from p1_b03_basis_reachability import (  # noqa: E402
    N_GRID,
    load_profiles,
    pca_fit,
    radial_profile,
)

_ROOT = Path(__file__).resolve().parents[2]
_RAW = _ROOT / "data" / "ellipse" / "raw"
_SPLITS = _ROOT / "data" / "splits" / "paper1_partition.json"
_OUT = _ROOT / "cfd" / "ellipse2d" / "shapes" / "b03_recon"

# Registered levels, plus the two below M = 3 that the 1-D finding demands.
M_LEVELS = (1, 2, 3, 10)

# A 2x2 factorial on (thickness, Reynolds), at these quantiles of each. Fixed
# before any Cp exists, so the four shapes span the envelope by rule, not by
# outcome.
#
# An earlier draft selected on b alone. b and Re are uncorrelated in the held-out
# set (r = -0.03), and the four picks landed at Re 810-989 by chance -- the top
# of the envelope, where F03 measured the largest Re drift. Stratifying on both
# removes that. No Cp had been computed when this was changed.
SELECT_Q_LOW = 0.15
SELECT_Q_HIGH = 0.85


def polygon_from_profile(
    theta: np.ndarray, r_norm: np.ndarray, r_mean: float, centre: np.ndarray
) -> np.ndarray:
    """Rebuild a closed point cloud from a normalised radial profile.

    Args:
        theta (np.ndarray): Node angles of the original body, shape ``(n,)``.
        r_norm (np.ndarray): Normalised radius at those angles, shape ``(n,)``.
        r_mean (float): The original body's mean radius, restoring physical scale.
        centre (np.ndarray): Original centroid, shape ``(2,)``.

    Returns:
        np.ndarray: Points ordered by angle, shape ``(n, 2)``.
    """
    r = r_norm * r_mean
    xy = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    return xy + centre


def profile_to_nodes(grid_profile: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Interpolate a 128-point uniform-theta profile onto arbitrary angles.

    Periodic by construction -- the grid is wrapped one period either side so
    nodes near ``theta = 0`` interpolate across the seam rather than clamping.

    Args:
        grid_profile (np.ndarray): Profile on the uniform grid, shape ``(N_GRID,)``.
        theta (np.ndarray): Target angles in ``[0, 2pi)``, shape ``(n,)``.

    Returns:
        np.ndarray: Profile at ``theta``, shape ``(n,)``.
    """
    grid = np.linspace(0.0, 2.0 * np.pi, N_GRID, endpoint=False)
    g_ext = np.concatenate([grid - 2.0 * np.pi, grid, grid + 2.0 * np.pi])
    p_ext = np.concatenate([grid_profile, grid_profile, grid_profile])
    return np.interp(theta, g_ext, p_ext)


def write_dat(path: Path, poly: np.ndarray, header: str) -> None:
    """Write a CCW closed polygon in the solver's body format.

    Args:
        path (Path): Destination ``.dat`` path.
        poly (np.ndarray): Points, shape ``(n, 2)``, already angle-ordered.
        header (str): First comment line, without the leading ``#``.
    """
    with path.open("w") as fh:
        fh.write(f"# {header}\n")
        fh.write("# x y\n")
        for x, y in poly:
            fh.write(f"{x:.10f} {y:.10f}\n")


def select_shapes(raw: np.ndarray, val_idx: np.ndarray) -> list[int]:
    """Pick four held-out shapes on a 2x2 (thickness, Reynolds) factorial.

    Each corner is the held-out sample closest to its target in the plane of
    quantile-normalised ``(b, Re)``, so both axes are spanned and neither is
    left to chance.

    Args:
        raw (np.ndarray): Full HDF5 array.
        val_idx (np.ndarray): Held-out indices from the frozen partition.

    Returns:
        list[int]: Four dataset indices, ordered thin/low-Re first.
    """
    bs, res = [], []
    for i in val_idx:
        s = raw[i][: _node_count(raw[i])]
        bs.append((s[:, _H5_COL_Y].max() - s[:, _H5_COL_Y].min()) / 2.0)
        res.append(float(s[0, _H5_COL_RE]))
    b_arr, re_arr = np.asarray(bs), np.asarray(res)

    # Rank-normalise both axes so the distance is scale-free.
    b_u = b_arr.argsort().argsort() / (len(b_arr) - 1)
    re_u = re_arr.argsort().argsort() / (len(re_arr) - 1)

    picks: list[int] = []
    for tb in (SELECT_Q_LOW, SELECT_Q_HIGH):
        for tre in (SELECT_Q_LOW, SELECT_Q_HIGH):
            d = np.hypot(b_u - tb, re_u - tre)
            d[np.isin(val_idx, picks)] = np.inf  # no repeats
            picks.append(int(val_idx[int(np.argmin(d))]))
    return picks


def main() -> None:
    part = json.loads(_SPLITS.read_text())
    train_idx = np.asarray(part["train_indices"])
    val_idx = np.asarray(part["val_indices"])

    raw = _load_h5_raw(_RAW / "pOnEllipseTrain.h5")
    print("\nP1-B03 -- solver-ready reconstructions\n")

    train = load_profiles(_RAW / "pOnEllipseTrain.h5", train_idx)
    mean, comps, _ = pca_fit(train)
    print(f"  PCA fitted on {train.shape[0]} training profiles, {N_GRID}-point grid")

    _OUT.mkdir(parents=True, exist_ok=True)
    picks = select_shapes(raw, val_idx)
    rows: list[dict[str, object]] = []

    for rank, idx in enumerate(picks):
        sample = raw[idx]
        n = _node_count(sample)
        data = sample[:n]
        re = float(data[0, _H5_COL_RE])
        h = float(data[0, _H5_COL_H])

        xy = np.stack([data[:, _H5_COL_X], data[:, _H5_COL_Y]], axis=1)
        centre = xy.mean(axis=0)
        rel = xy - centre
        r = np.hypot(rel[:, 0], rel[:, 1])
        r_mean = float(r.mean())
        theta = np.mod(np.arctan2(rel[:, 1], rel[:, 0]), 2.0 * np.pi)
        order = np.argsort(theta)
        theta, r = theta[order], r[order]
        b = float((xy[:, 1].max() - xy[:, 1].min()) / 2.0)

        label = f"s{rank}_idx{idx:04d}"
        print(f"\n  {label}: Re={re:.0f} H={h:.3g} b={b:.4f} n={n}")

        # The dataset's own Cp curve. These four samples are not in eval_grid,
        # so without this there is nothing to grade any of the variants against.
        x_over_c, cp_ref = _reference_cp(data)
        with (_OUT / f"{label}_ref_cp.csv").open("w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["x_over_c", "cp_ref"])
            for xc, cp in zip(x_over_c, cp_ref, strict=True):
                wr.writerow([f"{xc:.6f}", f"{cp:.6f}"])

        # --- original, exactly as stored -----------------------------------
        write_dat(
            _OUT / f"{label}_orig.dat",
            np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1) + centre,
            f"P1-B03 {label} ORIGINAL: Re={re:.4g} H={h:.4g} b={b:.4g} n={n}",
        )

        # --- the 128-grid profile this whole analysis is built on ----------
        prof = radial_profile(sample, n_grid=N_GRID)

        variants: dict[str, np.ndarray] = {"roundtrip": prof}
        for m in M_LEVELS:
            coef = (prof - mean) @ comps[:m].T
            variants[f"pca{m:02d}"] = mean + coef @ comps[:m]

        for name, gp in variants.items():
            r_norm = profile_to_nodes(gp, theta)
            poly = polygon_from_profile(theta, r_norm, r_mean, centre)
            dev = float(np.max(np.abs(r_norm - r / r_mean)))
            write_dat(
                _OUT / f"{label}_{name}.dat",
                poly,
                f"P1-B03 {label} {name.upper()}: Re={re:.4g} H={h:.4g} "
                f"n={n} max|dr|/rbar={dev:.3e}",
            )
            print(f"    {name:<10} max|dr|/rbar = {dev:.3e}")
            rows.append(
                {
                    "label": label,
                    "idx": idx,
                    "variant": name,
                    "Re": re,
                    "H": h,
                    "b": b,
                    "n_nodes": n,
                    "max_dev_norm": dev,
                }
            )
        rows.append(
            {
                "label": label,
                "idx": idx,
                "variant": "orig",
                "Re": re,
                "H": h,
                "b": b,
                "n_nodes": n,
                "max_dev_norm": 0.0,
            }
        )

    man = _OUT / "manifest.csv"
    with man.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  {len(rows)} bodies -> {_OUT.relative_to(_ROOT)}")
    print(f"  manifest -> {man.relative_to(_ROOT)}\n")


if __name__ == "__main__":
    main()
