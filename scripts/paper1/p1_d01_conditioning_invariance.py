"""
P1-D01 -- is the DCT conditioning genuinely node-count independent?

C2 claims the K-mode DCT-II descriptor of Cp(x/c) is source-agnostic across
discretisations.  Half of that claim is about the *conditioning* and is
supposedly true by construction; this script checks it.

The production encoder (``pOnEllipseConditional._build_graphs``) does::

    order = np.argsort(x_over_c)
    modes = dct_ii(cp_mean[order], K)

i.e. it sorts Cp by chord fraction and then DCTs the resulting *sequence*.
``dct_ii`` has no knowledge of x/c -- it treats sample ``n`` as sitting at
uniform spacing.  The boundary nodes are not uniform in x/c (they cluster at
the leading and trailing edges), so the descriptor is a transform of the curve
*as sampled*, not of the curve as a function of chord.

That distinction is exactly what determines whether the invariance holds, so
both readings of "resample to N points" are tested:

  ``rank``   resample the sorted sequence in normalised rank space.  This is
             what a mesher does: lay N nodes around the body with the same
             spacing law.  It is the physically honest discretisation change
             and the one D02 will actually exercise.
  ``chord``  resample onto a uniform x/c grid.  This is the encoding the
             descriptor would need in order to be genuinely geometry-referenced
             rather than sampling-referenced.

Usage:
    python scripts/paper1/p1_d01_conditioning_invariance.py
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
_OUT = _ROOT / "outputs" / "paper1" / "P1-D01_conditioning-invariance"

N_LADDER = (48, 64, 80, 96, 128, 192)
K_LADDER = (3, 8, 16, 32)
N_REF = 96
K_PROD = 8  # the production setting
N_CURVES = 200

# The reported result was produced under the orthonormal convention, which was
# production at the time. Pinned so this script keeps reproducing the published
# table after the default changed on 2026-08-05; the fix is verified in
# `verify_fix` below.
LEGACY = "ortho"


def extract_curve(sample: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the production Cp(x/c) extraction for one simulation.

    Args:
        sample (np.ndarray): One raw HDF5 record, shape ``(n_pad, 105)``.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(x_over_c, cp_ordered)``, both sorted
            by chord fraction and of length equal to the true node count.
    """
    n_nodes = _node_count(sample)
    data = sample[:n_nodes]

    cp = data[:, _H5_COL_P_START : _H5_COL_P_START + _H5_N_TIMESTEPS]
    cp_mean = cp.mean(axis=-1).astype(np.float32)

    xc = data[:, _H5_COL_X].astype(np.float32)
    xc = xc - xc.mean()
    x_min, x_max = float(xc.min()), float(xc.max())
    x_over_c = (xc - x_min) / max(x_max - x_min, 1e-10)

    order = np.argsort(x_over_c)
    return x_over_c[order], cp_mean[order]


def resample(
    x_over_c: np.ndarray, cp: np.ndarray, n_target: int, mode: str
) -> np.ndarray:
    """Resample a sorted Cp curve to ``n_target`` points.

    Args:
        x_over_c (np.ndarray): Sorted chord fractions, shape ``(N0,)``.
        cp (np.ndarray): Cp at those stations, shape ``(N0,)``.
        n_target (int): Desired sample count.
        mode (str): ``"rank"`` (uniform in node index, preserving the
            clustering law) or ``"chord"`` (uniform in x/c).

    Returns:
        np.ndarray: Resampled curve, shape ``(n_target,)``.
    """
    if mode == "rank":
        src = np.linspace(0.0, 1.0, cp.size, dtype=np.float64)
        dst = np.linspace(0.0, 1.0, n_target, dtype=np.float64)
        return np.interp(dst, src, cp).astype(np.float32)
    if mode == "chord":
        dst = np.linspace(0.0, 1.0, n_target, dtype=np.float64)
        return np.interp(dst, x_over_c, cp).astype(np.float32)
    raise ValueError(f"unknown mode '{mode}'")


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    """Relative L2 deviation of ``a`` from reference ``b``."""
    denom = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b) / max(denom, 1e-12))


def sweep(curves: list[tuple[np.ndarray, np.ndarray]], mode: str) -> dict:
    """Run the full N x K invariance sweep for one resampling mode."""
    out: dict[str, dict] = {}
    for k in K_LADDER:
        ref = np.stack(
            [dct_ii(resample(x, c, N_REF, mode), k, norm=LEGACY) for x, c in curves]
        )  # (n_curves, k)
        per_n: dict[str, dict] = {}
        for n in N_LADDER:
            got = np.stack(
                [dct_ii(resample(x, c, n, mode), k, norm=LEGACY) for x, c in curves]
            )
            overall = np.array(
                [rel_l2(got[i], ref[i]) for i in range(len(curves))], dtype=np.float64
            )
            # Per-mode: deviation of each coefficient, normalised by the RMS
            # magnitude of that coefficient across the corpus, so a large
            # absolute error on a near-zero mode is not flattered.
            scale = np.sqrt((ref**2).mean(axis=0)) + 1e-12
            per_mode = np.abs(got - ref).mean(axis=0) / scale
            per_n[str(n)] = {
                "rel_l2_median": float(np.median(overall)),
                "rel_l2_p95": float(np.percentile(overall, 95)),
                "rel_l2_max": float(overall.max()),
                "per_mode_rel": [float(v) for v in per_mode],
            }
        out[str(k)] = per_n
    return out


def diagnose_scaling(curves: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """Test whether the deviation is explained by a pure ``sqrt(N)`` factor.

    ``dct_ii`` applies the orthonormal DCT-II convention (``sqrt(2/N)``).  By
    Parseval that preserves the *vector* norm, and the norm of a sampled
    continuous function grows as ``sqrt(N)``.  So the descriptor is expected to
    scale as ``sqrt(N)`` rather than being invariant.  If that is the whole
    story, the measured deviation should equal ``|sqrt(N/96) - 1|`` and the
    ``1/N`` convention should restore invariance.

    Args:
        curves: ``(x_over_c, cp)`` pairs.

    Returns:
        dict: measured vs predicted deviation, and the deviation after
            renormalising to the ``1/N`` convention.
    """
    out: dict[str, dict] = {}
    ref = np.stack(
        [dct_ii(resample(x, c, N_REF, "rank"), K_PROD, norm=LEGACY) for x, c in curves]
    )
    ref_fix = ref / np.sqrt(N_REF)
    for n in N_LADDER:
        got = np.stack(
            [dct_ii(resample(x, c, n, "rank"), K_PROD, norm=LEGACY) for x, c in curves]
        )
        meas = float(np.median([rel_l2(got[i], ref[i]) for i in range(len(curves))]))
        fixed = got / np.sqrt(n)
        fixed_dev = float(
            np.median([rel_l2(fixed[i], ref_fix[i]) for i in range(len(curves))])
        )
        out[str(n)] = {
            "measured_rel_l2": meas,
            "sqrt_law_prediction": float(abs(np.sqrt(n / N_REF) - 1.0)),
            "unexplained_residual": float(meas - abs(np.sqrt(n / N_REF) - 1.0)),
            "rel_l2_under_1_over_n_convention": fixed_dev,
        }
    return out


def node_count_confound(path: Path) -> dict:
    """How much of the training conditioning magnitude is just node count?"""
    raw = _load_h5_raw(path)
    mags, counts = [], []
    for i in range(raw.shape[0]):
        _, cp = extract_curve(raw[i])
        mags.append(float(np.linalg.norm(dct_ii(cp, K_PROD, norm=LEGACY))))
        counts.append(cp.size)
    mag = np.array(mags)
    cnt = np.array(counts)
    resid = mag / np.sqrt(cnt)
    return {
        "n": int(cnt.size),
        "node_min": int(cnt.min()),
        "node_max": int(cnt.max()),
        "node_median": float(np.median(cnt)),
        "corr_mag_sqrtN": float(np.corrcoef(mag, np.sqrt(cnt))[0, 1]),
        "cv_magnitude": float(mag.std() / mag.mean()),
        "cv_magnitude_after_removing_N": float(resid.std() / resid.mean()),
    }


def verify_fix(curves: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """Re-run the production setting under the corrected encoder.

    Two independent errors are corrected together, so both are isolated:

      ``forward_uniform``   right normalisation, still assuming even spacing
      ``forward_positions`` right normalisation *and* true chordwise abscissae

    The second is what the dataset now does. Reported against the same N=96
    reference as the headline table, so the numbers are directly comparable.

    Args:
        curves: ``(x_over_c, cp)`` pairs.

    Returns:
        dict: median relative L2 deviation per node count, per variant.
    """
    out: dict[str, dict[str, float]] = {"forward_uniform": {}, "forward_positions": {}}

    def enc(x: np.ndarray, c: np.ndarray, n: int, with_pos: bool) -> np.ndarray:
        cp = resample(x, c, n, "chord" if with_pos else "rank")
        pos = np.linspace(0.0, 1.0, n) if with_pos else None
        return dct_ii(cp, K_PROD, positions=pos)

    for key, with_pos in (("forward_uniform", False), ("forward_positions", True)):
        ref = np.stack([enc(x, c, N_REF, with_pos) for x, c in curves])
        for n in N_LADDER:
            got = np.stack([enc(x, c, n, with_pos) for x, c in curves])
            out[key][str(n)] = float(
                np.median([rel_l2(got[i], ref[i]) for i in range(len(curves))])
            )
    return out


def _load(path: Path, indices: np.ndarray | None = None) -> list:
    raw = _load_h5_raw(path)
    idx = range(raw.shape[0]) if indices is None else indices
    return [extract_curve(raw[i]) for i in idx]


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    part = json.loads(_SPLITS.read_text())

    train_h5 = _ROOT / part["files"]["train"]["path"]
    raw_n = _load_h5_raw(train_h5).shape[0]
    stride = max(raw_n // N_CURVES, 1)
    train_idx = np.arange(0, raw_n, stride)[:N_CURVES]
    train_curves = _load(train_h5, train_idx)
    native = np.array([c.size for _, c in train_curves])

    print("\nP1-D01 -- conditioning invariance")
    print(
        f"  {len(train_curves)} training curves, native node counts "
        f"{native.min()}-{native.max()} (median {int(np.median(native))})\n"
    )

    results: dict[str, object] = {
        "n_curves": len(train_curves),
        "native_node_counts": {
            "min": int(native.min()),
            "max": int(native.max()),
            "median": float(np.median(native)),
        },
        "n_ladder": list(N_LADDER),
        "k_ladder": list(K_LADDER),
        "n_reference": N_REF,
    }

    for mode in ("rank", "chord"):
        res = sweep(train_curves, mode)
        results[f"train_{mode}"] = res
        print(f"  mode={mode:<6}  median relative L2 deviation vs N={N_REF}")
        print(f"    {'K':<4}" + "".join(f"{n:>10}" for n in N_LADDER))
        for k in K_LADDER:
            row = "".join(
                f"{res[str(k)][str(n)]['rel_l2_median']:>10.2%}" for n in N_LADDER
            )
            print(f"    {k:<4}" + row)
        print()

    # Real-data check: AoA10 is a separate simulation campaign at 80 nodes.
    aoa_path = _RAW / "pOnEllipseAoA10.h5"
    if aoa_path.exists():
        aoa_curves = _load(aoa_path)
        aoa_native = np.array([c.size for _, c in aoa_curves])
        print(
            f"  AoA10 real data: {len(aoa_curves)} curves, "
            f"{aoa_native.min()}-{aoa_native.max()} nodes"
        )
        for mode in ("rank", "chord"):
            res = sweep(aoa_curves, mode)
            results[f"aoa10_{mode}"] = res
            med = res[str(K_PROD)]
            worst = max(med[str(n)]["rel_l2_median"] for n in N_LADDER)
            print(
                f"    mode={mode:<6} K={K_PROD}: worst median deviation " f"{worst:.2%}"
            )
        results["aoa10_native_nodes"] = {
            "min": int(aoa_native.min()),
            "max": int(aoa_native.max()),
        }
        print()

    # Is the deviation just the sqrt(N) normalisation convention?
    diag = diagnose_scaling(train_curves)
    results["scaling_diagnosis"] = diag
    print("  Diagnosis -- is the deviation a pure sqrt(N) scaling?")
    print(
        f"    {'N':>5} {'measured':>10} {'sqrt-law':>10} {'residual':>10} "
        f"{'under 1/N':>11}"
    )
    for n in N_LADDER:
        d = diag[str(n)]
        print(
            f"    {n:>5} {d['measured_rel_l2']:>10.2%} "
            f"{d['sqrt_law_prediction']:>10.2%} "
            f"{d['unexplained_residual']:>10.2%} "
            f"{d['rel_l2_under_1_over_n_convention']:>11.2%}"
        )
    print()

    fix = verify_fix(train_curves)
    results["fix_verification"] = fix
    print("  Fix verification -- same sweep under the corrected encoder:")
    print(f"    {'N':>5} {'legacy':>10} {'forward':>10} {'+positions':>12}")
    for n in N_LADDER:
        print(
            f"    {n:>5} {diag[str(n)]['measured_rel_l2']:>10.2%} "
            f"{fix['forward_uniform'][str(n)]:>10.2%} "
            f"{fix['forward_positions'][str(n)]:>12.2%}"
        )
    print()

    conf = node_count_confound(train_h5)
    results["node_count_confound"] = conf
    print("  Node-count confound in the training conditioning vectors:")
    print(f"    corr(|cond|, sqrt(N))        = {conf['corr_mag_sqrtN']:.3f}")
    print(f"    CV of |cond|                 = {conf['cv_magnitude']:.2%}")
    print(
        f"    CV after dividing by sqrt(N) = "
        f"{conf['cv_magnitude_after_removing_N']:.2%}"
    )
    print()

    # Verdict against the pre-registered rule, on the production setting.
    prod = results["train_rank"][str(K_PROD)]  # type: ignore[index]
    worst_med = max(prod[str(n)]["rel_l2_median"] for n in N_LADDER)
    worst_p95 = max(prod[str(n)]["rel_l2_p95"] for n in N_LADDER)
    if worst_med < 0.01:
        verdict = "SUPPORTED"
    elif worst_med > 0.05:
        verdict = "FALSIFIED"
    else:
        verdict = "PARTIAL"
    results["verdict"] = verdict
    results["production_worst_median_rel_l2"] = worst_med
    results["production_worst_p95_rel_l2"] = worst_p95

    print(
        f"  Production setting (mode=rank, K={K_PROD}) across "
        f"N in {N_LADDER[0]}-{N_LADDER[-1]}:"
    )
    print(f"    worst median rel-L2 = {worst_med:.2%}  (pass <1%, fail >5%)")
    print(f"    worst p95    rel-L2 = {worst_p95:.2%}")
    print(f"\n  VERDICT: {verdict}\n")

    (_OUT / "invariance.json").write_text(json.dumps(results, indent=2))
    print(f"  -> {(_OUT / 'invariance.json').relative_to(_ROOT)}\n")


if __name__ == "__main__":
    main()
