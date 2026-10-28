"""
P1-B03 -- how many basis modes cover this dataset's shape manifold?

C1 asserts that a low-dimensional shape parameterisation bottlenecks the
reachable design space, and that per-node diffusion escapes it.  That premise
is testable analytically, without training anything: fit a basis on the
training partition and measure how well it reconstructs shapes it never saw.

Shapes are ``r(theta)`` on the boundary ring, matching the production
``feature_mode="radial_norm"`` representation (radius divided by its own mean,
so scale is removed and only form remains).  Node counts vary 52-96 across the
corpus, so every curve is resampled onto a common ``N_GRID``-point theta grid
before the basis is fitted -- PCA needs a fixed dimension.  That resampling is
in *angle*, which is the natural parameter here and is unaffected by the
P1-D01 normalisation finding.

Two bases are compared on the same curves:

  ``pca``  fitted on the training partition; the optimal linear basis for this
           dataset, so it is the strongest form of the C1 premise
  ``dct``  a fixed analytic basis; comparable to A03's truncation ladder

Reconstruction round-trips through ``dct_ii`` / ``dct_ii_inverse``, so the
sqrt(N) convention cancels and P1-D01 does not affect these numbers.

**Partial run.** Step 3 of the spec -- converting geometric error into a Cp
tolerance via 8 solver runs -- requires the P1-F04 noise floor and is deferred.

Usage:
    python scripts/paper1/p1_b03_basis_reachability.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from graph_diffusion.data.pOnEllipse import (  # noqa: E402
    _H5_COL_X,
    _H5_COL_Y,
    _load_h5_raw,
    _node_count,
)
from graph_diffusion.data.pOnEllipseConditional import (  # noqa: E402
    dct_ii,
    dct_ii_inverse,
)

_ROOT = Path(__file__).resolve().parents[2]
_RAW = _ROOT / "data" / "ellipse" / "raw"
_SPLITS = _ROOT / "data" / "splits" / "paper1_partition.json"
_OUT = _ROOT / "outputs" / "paper1" / "P1-B03_basis-reachability"

N_GRID = 128
M_LADDER = (1, 2, 3, 5, 10, 20, 50)


def radial_profile(
    sample: np.ndarray, n_grid: int = N_GRID, rotate_deg: float = 0.0
) -> np.ndarray:
    """Extract ``r(theta) / mean(r)`` resampled onto a uniform theta grid.

    Mirrors the production ``feature_mode="radial_norm"`` feature, then
    resamples in angle so curves of different node count share a dimension.

    Args:
        sample (np.ndarray): One raw HDF5 record, shape ``(n_pad, 105)``.
        n_grid (int): Number of uniform theta stations.
        rotate_deg (float): Rigid rotation applied to the point cloud before
            the profile is taken. Used to test whether an apparently
            unreachable shape family is only a rotation away from the basis.

    Returns:
        np.ndarray: Normalised radius on ``[0, 2pi)``, shape ``(n_grid,)``.
    """
    n_nodes = _node_count(sample)
    data = sample[:n_nodes]
    xc = data[:, _H5_COL_X].astype(np.float64)
    yc = data[:, _H5_COL_Y].astype(np.float64)
    xc -= xc.mean()
    yc -= yc.mean()

    if rotate_deg:
        t = np.radians(rotate_deg)
        xc, yc = xc * np.cos(t) - yc * np.sin(t), xc * np.sin(t) + yc * np.cos(t)

    r = np.sqrt(xc**2 + yc**2)
    r = r / r.mean()
    theta = np.mod(np.arctan2(yc, xc), 2.0 * np.pi)

    order = np.argsort(theta)
    th, rr = theta[order], r[order]
    # Wrap one period so the interpolation is periodic rather than clamped.
    th_ext = np.concatenate([th - 2.0 * np.pi, th, th + 2.0 * np.pi])
    rr_ext = np.concatenate([rr, rr, rr])
    grid = np.linspace(0.0, 2.0 * np.pi, n_grid, endpoint=False)
    return np.interp(grid, th_ext, rr_ext)


def load_profiles(
    path: Path, indices: np.ndarray | None = None, rotate_deg: float = 0.0
) -> np.ndarray:
    raw = _load_h5_raw(path)
    idx = range(raw.shape[0]) if indices is None else indices
    return np.stack([radial_profile(raw[i], rotate_deg=rotate_deg) for i in idx])


def principal_axis(sample: np.ndarray) -> tuple[float, float]:
    """Aspect ratio and major-axis angle of one shape's point cloud.

    Distinguishes "this family occupies new geometry" from "this family is the
    same geometry, rotated" -- a distinction the ``r(theta)`` representation
    cannot make on its own, because it is anchored to the body-fixed frame.

    Args:
        sample (np.ndarray): One raw HDF5 record.

    Returns:
        tuple[float, float]: ``(minor/major, angle_deg mod 180)``.
    """
    n_nodes = _node_count(sample)
    data = sample[:n_nodes]
    x = data[:, _H5_COL_X].astype(np.float64)
    y = data[:, _H5_COL_Y].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    w, v = np.linalg.eigh(np.cov(np.stack([x, y])))
    return float(np.sqrt(w[0] / w[1])), float(
        np.degrees(np.arctan2(v[1, 1], v[0, 1])) % 180
    )


def geometry_summary(path: Path, limit: int | None = None) -> dict[str, float]:
    """Aspect-ratio and orientation distribution of a shape set."""
    raw = _load_h5_raw(path)
    n = raw.shape[0] if limit is None else min(limit, raw.shape[0])
    g = np.array([principal_axis(raw[i]) for i in range(n)])
    return {
        "n": int(n),
        "aspect_min": float(g[:, 0].min()),
        "aspect_max": float(g[:, 0].max()),
        "aspect_median": float(np.median(g[:, 0])),
        "angle_median_deg": float(np.median(g[:, 1])),
        "angle_spread_deg": float(g[:, 1].max() - g[:, 1].min()),
    }


def pca_fit(train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit PCA on the training profiles.

    Returns:
        tuple: ``(mean, components, explained_variance_ratio)`` with components
            shape ``(n_modes, N_GRID)``, rows ordered by decreasing variance.
    """
    mu = train.mean(axis=0)
    centred = train - mu
    _, sv, vt = np.linalg.svd(centred, full_matrices=False)
    var = sv**2
    return mu, vt, var / var.sum()


def pca_reconstruct(
    x: np.ndarray, mu: np.ndarray, vt: np.ndarray, m: int
) -> np.ndarray:
    """Project onto the leading ``m`` PCA modes and reconstruct."""
    basis = vt[:m]
    return mu + (x - mu) @ basis.T @ basis


def dct_reconstruct(x: np.ndarray, m: int) -> np.ndarray:
    """Truncate to ``m`` DCT modes and reconstruct."""
    return np.stack(
        [dct_ii_inverse(dct_ii(row.astype(np.float32), m), x.shape[1]) for row in x]
    ).astype(np.float64)


def errors(truth: np.ndarray, recon: np.ndarray) -> dict[str, float]:
    """Reconstruction error summary, in units of normalised radius."""
    resid = recon - truth
    rms = np.sqrt((resid**2).mean(axis=1))
    return {
        "rms_median": float(np.median(rms)),
        "rms_p95": float(np.percentile(rms, 95)),
        "max_abs_median": float(np.median(np.abs(resid).max(axis=1))),
        "rel_l2_median": float(
            np.median(
                np.linalg.norm(resid, axis=1) / np.linalg.norm(truth - 1.0, axis=1)
            )
        ),
    }


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    part = json.loads(_SPLITS.read_text())
    train_h5 = _ROOT / part["files"]["train"]["path"]

    # The frozen partition stores explicit index lists; the val split is the
    # only held-out data available until the G-group experiments run.
    tr_idx = np.array(sorted(part["train_indices"]))
    va_idx = np.array(sorted(part["val_indices"]))

    train = load_profiles(train_h5, tr_idx)
    held = load_profiles(train_h5, va_idx)

    print("\nP1-B03 -- basis reachability")
    print(
        f"  train {train.shape[0]}  held-out {held.shape[0]}  "
        f"grid {N_GRID} theta stations\n"
    )

    mu, vt, evr = pca_fit(train)
    cum = np.cumsum(evr)
    n99 = int(np.searchsorted(cum, 0.99) + 1)
    n999 = int(np.searchsorted(cum, 0.999) + 1)
    print(
        f"  explained variance: 1 mode {evr[0]:.4%}, 2 {cum[1]:.4%}, "
        f"3 {cum[2]:.4%}, 5 {cum[4]:.4%}"
    )
    print(f"  modes to 99% variance = {n99}     to 99.9% = {n999}\n")

    results: dict[str, object] = {
        "n_train": int(train.shape[0]),
        "n_heldout": int(held.shape[0]),
        "n_grid": N_GRID,
        "explained_variance_ratio": [float(v) for v in evr[:50]],
        "modes_to_99pct_variance": n99,
        "modes_to_999pct_variance": n999,
        "note": "solver half (spec step 3) deferred -- requires P1-F04",
    }

    sets = {"heldout": held}
    aoa = _RAW / "pOnEllipseAoA10.h5"
    if aoa.exists():
        sets["aoa10"] = load_profiles(aoa)
    thick = _RAW / "pOnEllipseThick.h5"
    if thick.exists():
        sets["thick"] = load_profiles(thick)

    for name, arr in sets.items():
        rec: dict[str, dict] = {"pca": {}, "dct": {}}
        for m in M_LADDER:
            rec["pca"][str(m)] = errors(arr, pca_reconstruct(arr, mu, vt, m))
            rec["dct"][str(m)] = errors(arr, dct_reconstruct(arr, m))
        results[f"reconstruction_{name}"] = rec
        print(f"  {name}  (n={arr.shape[0]})   median RMS error in r/mean(r)")
        print(f"    {'M':>4}" + "".join(f"{m:>11}" for m in M_LADDER))
        for basis in ("pca", "dct"):
            row = "".join(
                f"{rec[basis][str(m)]['rms_median']:>11.2e}" for m in M_LADDER
            )
            print(f"    {basis:>4}" + row)
        print()

    # Geometry check: is an "unreachable" family new geometry, or a rotation?
    geo = {
        "train": geometry_summary(train_h5, 400),
        "aoa10": geometry_summary(aoa) if aoa.exists() else None,
        "thick": geometry_summary(thick) if thick.exists() else None,
    }
    results["geometry"] = geo
    print("  Geometry of each set (aspect ratio = minor/major):")
    for name, g in geo.items():
        if g is None:
            continue
        print(
            f"    {name:<8} b/a {g['aspect_min']:.3f}-{g['aspect_max']:.3f} "
            f"(med {g['aspect_median']:.3f})   major-axis angle "
            f"{g['angle_median_deg']:.1f} deg"
        )
    print()

    # The decisive control. If AoA10 is the same ellipse family rotated, then
    # de-rotating it should restore held-out-level reconstruction -- in which
    # case its apparent unreachability is a property of the representation,
    # not of the design space.
    if aoa.exists():
        h10 = results["reconstruction_heldout"]["pca"]["10"][  # type: ignore[index]
            "rms_median"
        ]
        rot: dict[str, float] = {}
        print("  Rotation control -- AoA10 reconstruction at M=10:")
        for deg in (0.0, 5.0, 10.0, 15.0):
            arr = load_profiles(aoa, rotate_deg=deg)
            e = errors(arr, pca_reconstruct(arr, mu, vt, 10))["rms_median"]
            rot[f"{deg:g}"] = e
            print(
                f"    de-rotated by {deg:>4.0f} deg   RMS {e:.3e}   "
                f"({e / h10:>6.0f}x held-out)"
            )
        results["aoa10_rotation_control"] = rot
        results["aoa10_reachable_after_derotation"] = bool(rot["10"] < 3.0 * h10)
        print()

    (_OUT / "reachability.json").write_text(json.dumps(results, indent=2))
    print(f"  -> {(_OUT / 'reachability.json').relative_to(_ROOT)}\n")


if __name__ == "__main__":
    main()
