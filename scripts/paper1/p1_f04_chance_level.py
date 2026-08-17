"""
P1-F04 companion -- what does a given rho actually mean?

The noise floor gives the smallest *detectable* difference. It says nothing
about what a rho value means, and for this dataset that is a live risk: every
ellipse Cp curve looks broadly alike, so rho may be near-saturated for any
plausible shape. A headline of 0.92 reads very differently against a chance
level of 0.5 than against one of 0.85.

Three reference distributions, all from data already on disk:

  pair        rho between random PAIRS of dataset samples
  vs_mean     rho between each sample and the dataset MEAN Cp curve
  vs_extreme  rho between each sample and the most dissimilar sample found

If the pair distribution sits at 0.85, the meaningful axis is 0.85 -> 1.0, and
every rho in the paper must be positioned on that axis rather than on 0 -> 1.

Curves are compared with the same ``cp_agreement`` used for every graded result,
so the numbers are directly comparable to the paper's. The dataset stores p*,
and both sides here are dataset curves, so the p*->Cp factor cancels and is not
applied -- rho is affine-invariant regardless.

Usage:
    python scripts/paper1/p1_f04_chance_level.py
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
from graph_diffusion.postprocessing.cp_metrics import cp_agreement  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_SPLITS = _ROOT / "data" / "splits" / "paper1_partition.json"
_OUT = _ROOT / "outputs" / "paper1" / "P1-F04_noise-floor"

N_SAMPLES = 300
N_PAIRS = 4000
RNG = np.random.default_rng(0)


def curve(sample: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted Cp(x/c) for one simulation, as the grading pipeline sees it."""
    n = _node_count(sample)
    d = sample[:n]
    cp = d[:, _H5_COL_P_START : _H5_COL_P_START + _H5_N_TIMESTEPS].mean(axis=1)
    xc = d[:, _H5_COL_X].astype(np.float64)
    xc -= xc.mean()
    x = (xc - xc.min()) / max(xc.max() - xc.min(), 1e-10)
    o = np.argsort(x)
    return x[o], cp[o].astype(np.float64)


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    part = json.loads(_SPLITS.read_text())
    raw = _load_h5_raw(_ROOT / part["files"]["train"]["path"])
    idx = RNG.choice(raw.shape[0], size=min(N_SAMPLES, raw.shape[0]), replace=False)
    curves = [curve(raw[i]) for i in idx]
    n = len(curves)
    print(f"\nP1-F04 companion -- chance level of rho ({n} training curves)\n")

    # --- random pairs ------------------------------------------------------
    pair = []
    for _ in range(N_PAIRS):
        i, j = RNG.choice(n, size=2, replace=False)
        xi, ci = curves[i]
        xj, cj = curves[j]
        pair.append(cp_agreement(xi, ci, xj, cj, mode="collapsed").rho)
    pair_a = np.array(pair)

    # --- against the corpus mean curve, on a common grid -------------------
    grid = np.linspace(0.0, 1.0, 128)
    resampled = np.stack([np.interp(grid, x, c) for x, c in curves])
    mean_curve = resampled.mean(axis=0)
    vs_mean = np.array(
        [cp_agreement(grid, mean_curve, x, c, mode="collapsed").rho for x, c in curves]
    )

    # --- against the most dissimilar curve found ---------------------------
    worst_i = int(np.argmin([np.corrcoef(r, mean_curve)[0, 1] for r in resampled]))
    xw, cw = curves[worst_i]
    vs_extreme = np.array(
        [cp_agreement(xw, cw, x, c, mode="collapsed").rho for x, c in curves]
    )

    out: dict[str, object] = {"n_curves": n, "n_pairs": int(N_PAIRS)}
    print(f"  {'reference':<14}{'median':>9}{'p5':>9}{'p95':>9}{'min':>9}")
    for label, arr in (
        ("random pair", pair_a),
        ("corpus mean", vs_mean),
        ("most extreme", vs_extreme),
    ):
        print(
            f"  {label:<14}{np.median(arr):>9.4f}{np.percentile(arr, 5):>9.4f}"
            f"{np.percentile(arr, 95):>9.4f}{arr.min():>9.4f}"
        )
        out[label.replace(" ", "_")] = {
            "median": float(np.median(arr)),
            "p5": float(np.percentile(arr, 5)),
            "p95": float(np.percentile(arr, 95)),
            "min": float(arr.min()),
        }

    chance = float(np.median(pair_a))
    print(
        f"\n  Chance level (median rho between two unrelated dataset shapes): "
        f"{chance:.4f}"
    )
    for headline in (0.86, 0.92, 0.95, 0.99):
        frac = (headline - chance) / (1.0 - chance)
        print(
            f"    a headline rho of {headline:.2f} sits {frac:>6.1%} of the way "
            f"from chance to perfect"
        )
    out["chance_level"] = chance
    (_OUT / "chance_level.json").write_text(json.dumps(out, indent=2))
    print(f"\n  -> {(_OUT / 'chance_level.json').relative_to(_ROOT)}\n")


if __name__ == "__main__":
    main()
