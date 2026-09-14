"""
P1-D02 -- grade node-count transfer.

Four questions, in the order the spec insists on. Geometry first: a shape that
is not closed and simple is not a design, whatever its Cp agreement, so the
validity rate is reported before any fidelity number.

  validity   usable designs vs N
  fidelity   Cd error and rho vs N, in-range and out-of-range reported apart
  roughness  the predicted failure mode at high N
  spectrum   the mechanism -- is the learned frequency content fixed in node
             INDEX or in arc length? If the peak sits at a constant index-space
             frequency across the ladder, the model has learned an index-space
             prior, which is a precise statement about what per-node diffusion
             on a fixed ring actually learns.

**The registered rule is stated in rho and is underpowered.** Seed-to-seed sd on
rho is 0.00542 = 5.0 sigma_rho, so a per-rung mean over 4 targets x 3 seeds has
SEM ~1.45 sigma -- a 2 sigma rule cannot be resolved against it, and five seeds
would only reach 1.12 sigma. Cd's seed sd is 0.58 sigma_Cd, giving SEM ~0.17
sigma at the same cost. Both are reported; Cd carries the verdict, per
[[PROTOCOL]] section 2.1.

Usage:
    python scripts/paper1/p1_d02_grade.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from graph_diffusion.postprocessing.cp_metrics import (  # noqa: E402
    cp_agreement,
    polygon_validity,
    read_cp_csv,
    read_force_csv,
    read_reference_cp,
)

_SHAPES = _ROOT / "cfd" / "ellipse2d" / "shapes" / "d02_nodes"
_WORK = _ROOT / "outputs" / "paper1" / "P1-D02_node-count-transfer" / "solver"
_TRUTH = _ROOT / "outputs" / "paper1" / "P1-G01_holdout-inverse-design" / "truth"
_OUT = _ROOT / "outputs" / "paper1" / "P1-D02_node-count-transfer"

SIGMA_RHO = 0.00108
SIGMA_CD = 0.04742
TRAIN_MIN, TRAIN_MAX = 52, 96


def profile(xy: np.ndarray, n_grid: int = 256) -> np.ndarray:
    """Resample a body to a common ``r(theta)/mean(r)`` grid.

    Args:
        xy (np.ndarray): Boundary points, shape ``(N, 2)``.
        n_grid (int): Output grid size.

    Returns:
        np.ndarray: Normalised radius on ``[0, 2pi)``, shape ``(n_grid,)``.
    """
    c = xy - xy.mean(axis=0)
    r = np.hypot(c[:, 0], c[:, 1])
    r = r / r.mean()
    th = np.mod(np.arctan2(c[:, 1], c[:, 0]), 2 * np.pi)
    o = np.argsort(th)
    return np.interp(
        np.linspace(0.0, 2.0 * np.pi, n_grid, endpoint=False),
        np.concatenate([th[o] - 2 * np.pi, th[o], th[o] + 2 * np.pi]),
        np.concatenate([r[o], r[o], r[o]]),
    )


def roughness(xy: np.ndarray) -> float:
    """Node-to-node radial variation, normalised -- the high-N failure mode.

    Args:
        xy (np.ndarray): Boundary points, shape ``(N, 2)``.

    Returns:
        float: RMS second difference of ``r`` around the ring, over mean ``r``.
    """
    c = xy - xy.mean(axis=0)
    r = np.hypot(c[:, 0], c[:, 1])
    th = np.mod(np.arctan2(c[:, 1], c[:, 0]), 2 * np.pi)
    r = r[np.argsort(th)]
    d2 = r - 0.5 * (np.roll(r, 1) + np.roll(r, -1))
    return float(np.sqrt(np.mean(d2**2)) / r.mean())


def peak_index_freq(xy: np.ndarray) -> float:
    """Dominant frequency of ``r`` in **node-index** units.

    Index-space, deliberately: if this is constant across the ladder the model
    has learned a prior in node index rather than in arc length.

    Args:
        xy (np.ndarray): Boundary points, shape ``(N, 2)``.

    Returns:
        float: Peak wavenumber as a fraction of the Nyquist index frequency.
    """
    c = xy - xy.mean(axis=0)
    r = np.hypot(c[:, 0], c[:, 1])
    th = np.mod(np.arctan2(c[:, 1], c[:, 0]), 2 * np.pi)
    r = r[np.argsort(th)]
    r = r - r.mean()
    if r.size < 8:
        return float("nan")
    amp = np.abs(np.fft.rfft(r))[1:]
    if not amp.size or not np.isfinite(amp).any():
        return float("nan")
    return float((np.argmax(amp) + 1) / (r.size / 2))


def read_body(p: Path) -> np.ndarray:
    """Read a solver body file as an ``(N, 2)`` array.

    Args:
        p (Path): The ``.dat`` file.

    Returns:
        np.ndarray: Points.
    """
    return np.asarray(
        [
            tuple(map(float, ln.split()))
            for ln in p.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
    )


def main() -> None:
    man = list(csv.DictReader((_SHAPES / "manifest.csv").open()))
    if not man:
        sys.exit("no manifest -- run scripts/paper1/p1_d02_sample.py first")

    truth: dict[int, float] = {}
    tman = _ROOT / "cfd" / "ellipse2d" / "shapes" / "g01_truth" / "manifest.csv"
    if tman.exists():
        for r in csv.DictReader(tman.open()):
            f = _TRUTH / r["tag"] / "tmp" / "ellipse2d_force.csv"
            if f.exists():
                truth[int(r["target_idx"])] = read_force_csv(f).cd

    print("\nP1-D02 -- node-count transfer\n")
    rows: list[dict] = []
    for m in man:
        body = _SHAPES / f"{m['tag']}.dat"
        xy = read_body(body)
        val = polygon_validity(xy)
        row = {
            **m,
            "n_nodes": int(m["n_nodes"]),
            "seed": int(m["seed"]),
            "target_idx": int(m["target_idx"]),
            "valid": int(bool(val.is_valid)),
            "roughness": roughness(xy),
            "peak_index_freq": peak_index_freq(xy),
            "completed": 0,
        }
        cp = _WORK / m["tag"] / "tmp" / "ellipse2d_cp.csv"
        ref = _SHAPES / f"{m['label']}_ref_cp.csv"
        if cp.exists() and ref.exists():
            x_m, cp_m, _ = read_cp_csv(cp)
            x_r, cp_r = read_reference_cp(ref)
            agr = cp_agreement(x_r, cp_r, x_m, cp_m, mode="collapsed")
            row |= {"completed": 1, "rho": agr.rho, "a": agr.a}
            force = _WORK / m["tag"] / "tmp" / "ellipse2d_force.csv"
            if force.exists():
                cd = read_force_csv(force).cd
                t = truth.get(row["target_idx"], float("nan"))
                row |= {
                    "Cd": cd,
                    "Cd_rel_err": (
                        (cd - t) / t if np.isfinite(t) and t else float("nan")
                    ),
                }
        rows.append(row)

    ladder = sorted({r["n_nodes"] for r in rows})
    print(
        f"  {'N':>5}{'zone':>7}{'valid':>8}{'runs':>7}{'Cd err':>11}"
        f"{'+/-':>9}{'rho':>10}{'rough':>10}{'f_peak':>9}"
    )
    per_n: dict[int, dict] = {}
    for n in ladder:
        sub = [r for r in rows if r["n_nodes"] == n]
        done = [r for r in sub if r["completed"]]
        zone = sub[0]["zone"]
        vr = float(np.mean([r["valid"] for r in sub]))
        rough = float(np.mean([r["roughness"] for r in sub]))
        fpk = float(np.nanmean([r["peak_index_freq"] for r in sub]))
        cde = np.array([r.get("Cd_rel_err", np.nan) for r in done], dtype=float)
        cde = cde[np.isfinite(cde)]
        rho = np.array([r.get("rho", np.nan) for r in done], dtype=float)
        rho = rho[np.isfinite(rho)]
        cd_m = float(cde.mean()) if cde.size else float("nan")
        cd_s = (
            float(1.96 * cde.std(ddof=1) / np.sqrt(cde.size))
            if cde.size > 1
            else float("nan")
        )
        rho_m = float(rho.mean()) if rho.size else float("nan")
        per_n[n] = {
            "zone": zone,
            "validity": vr,
            "n_runs": len(done),
            "Cd_rel_err": cd_m,
            "Cd_ci95": cd_s,
            "rho": rho_m,
            "roughness": rough,
            "peak_index_freq": fpk,
        }
        print(
            f"  {n:>5}{zone:>7}{vr:>7.0%}{len(done):>7}{cd_m:>11.4f}"
            f"{cd_s:>9.4f}{rho_m:>10.5f}{rough:>10.5f}{fpk:>9.3f}"
        )

    inr = [n for n in ladder if TRAIN_MIN <= n <= TRAIN_MAX]
    out = [n for n in ladder if n not in inr]
    print(f"\n  Trained range {TRAIN_MIN}-{TRAIN_MAX}: rungs {inr}")
    print(f"  Outside:                 rungs {out}\n")

    def span(ns: list[int], key: str) -> float:
        v = [per_n[n][key] for n in ns if np.isfinite(per_n[n][key])]
        return float(max(v) - min(v)) if v else float("nan")

    print("  IN-RANGE uniformity (the C2 claim):")
    print(
        f"    Cd error span   {span(inr, 'Cd_rel_err'):.4f} "
        f"= {span(inr, 'Cd_rel_err') / SIGMA_CD:.2f} sigma_Cd"
    )
    print(
        f"    rho span        {span(inr, 'rho'):.5f} "
        f"= {span(inr, 'rho') / SIGMA_RHO:.2f} sigma_rho  (underpowered)"
    )
    print(f"    validity        {min(per_n[n]['validity'] for n in inr):.0%} worst")

    print("\n  OUT-OF-RANGE behaviour:")
    for n in out:
        p = per_n[n]
        print(
            f"    N={n:>3} ({p['zone']:>5})  validity {p['validity']:>4.0%}   "
            f"Cd err {p['Cd_rel_err']:>+8.4f}   roughness {p['roughness']:.5f}"
        )

    fp = [
        per_n[n]["peak_index_freq"]
        for n in ladder
        if np.isfinite(per_n[n]["peak_index_freq"])
    ]
    print("\n  Mechanism -- peak frequency in NODE-INDEX units across the ladder:")
    print(
        f"    range {min(fp):.3f} - {max(fp):.3f}  (constant => an index-space prior)"
    )

    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "grading.json").write_text(
        json.dumps(
            {"per_n": per_n, "runs": rows, "train_range": [TRAIN_MIN, TRAIN_MAX]},
            indent=2,
            default=float,
        )
    )
    print(f"\n  -> {(_OUT / 'grading.json').relative_to(_ROOT)}\n")


if __name__ == "__main__":
    main()
