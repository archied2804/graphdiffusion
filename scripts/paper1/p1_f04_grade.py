"""
P1-F04 -- grade the noise-floor sweep.

Twenty-seven solver runs: three shapes spanning the envelope, each run nine
times with the geometry held fixed and one nuisance factor varied at a time.
Anything that moves is measurement noise, because nothing physical changed.

  RESOLUTION          rule, rule +/- one 25% step
  node phase          3 rolls of the point list -- identical body, offset nodes
  AVG_WINDOW_CHORDS   15, 25, 35
  MAX_PHYS_T          40, 50, 60

Each run is graded against its own dataset reference curve through the same
``cp_metrics`` path as every other result, so the sigmas are in the paper's
units. Two summaries come out:

  pooled sigma    sqrt of the mean within-shape variance across all 9 runs.
                  Pooling within shape is what makes it a *measurement*
                  uncertainty -- the between-shape spread is real physics and
                  must not enter.

  per-factor      the same, restricted to the runs varying one factor. Tells
                  you which knob to tighten if the floor is too high.

Also compares the between-run sigma_Cd against the within-run ``Cd_std`` that
the solver reports for free: if the within-run unsteadiness dominates, the
averaging window is too short and MAX_PHYS_T must rise.

Usage:
    python scripts/paper1/p1_f04_grade.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from graph_diffusion.postprocessing.cp_metrics import (  # noqa: E402
    cp_agreement,
    dead_probe_mask,
    read_cp_csv,
    read_force_csv,
    read_reference_cp,
)

_ROOT = Path(__file__).resolve().parents[2]
_SHAPES = _ROOT / "cfd" / "ellipse2d" / "shapes" / "eval_grid"
_WORK = _ROOT / "outputs" / "paper1" / "P1-F04_noise-floor"

# label -> (dataset index, Re). Must match p1_f04_make_variants.py.
SHAPES = {
    "lowre_thin": (31, 502.0),
    "mid": (2824, 750.0),
    "highre_thick": (4018, 880.0),
}

METRICS = ("rho", "a", "Cd", "Cl")

# Pre-registered thresholds (spec "Hypothesis" and "Falsifier").
HYP_SIGMA_RHO = 0.005
HYP_SIGMA_A = 0.05
FALSIFIER_2SIGMA_RHO = 0.03


def factor_of(shape: str, tag: str) -> str:
    """Classify a run directory name into the nuisance factor it varies.

    The baseline run (``<shape>_phase0``) belongs to every factor group: each
    group is a one-factor-at-a-time sweep through the common baseline.

    Args:
        shape (str): Shape label, e.g. ``"mid"``.
        tag (str): Run directory name, e.g. ``"mid_res188"``.

    Returns:
        str: One of ``"phase"``, ``"resolution"``, ``"avg_window"``,
            ``"phys_time"``, or ``"?"`` if unrecognised.
    """
    suffix = tag[len(shape) + 1 :]
    if suffix.startswith("phase"):
        return "phase"
    if suffix.startswith("res"):
        return "resolution"
    if suffix.startswith("avg"):
        return "avg_window"
    if suffix.startswith("t"):
        return "phys_time"
    return "?"


def grade_run(d: Path, idx: int) -> dict | None:
    """Grade one run against its dataset reference curve.

    Args:
        d (Path): Run directory.
        idx (int): Dataset sample index locating the reference curve.

    Returns:
        dict | None: Metrics, or ``None`` if the run did not complete.
    """
    meas = d / "tmp" / "ellipse2d_cp.csv"
    ref = _SHAPES / f"sim_{idx}_ref_cp.csv"
    if not meas.exists() or not ref.exists():
        return None

    # Count the solid-cell probes this run lost before they are filtered, so
    # the instrument defect stays visible in the record rather than being
    # silently repaired.
    x_raw, cp_raw, rms_raw = read_cp_csv(meas, drop_dead=False)
    n_dead = int(dead_probe_mask(cp_raw, rms_raw).sum())

    x_m, cp_m, _ = read_cp_csv(meas)
    x_r, cp_r = read_reference_cp(ref)  # dataset p* -> Cp
    agr = cp_agreement(x_r, cp_r, x_m, cp_m, mode="collapsed")
    out: dict[str, float] = {
        "rho": agr.rho,
        "a": agr.a,
        "b_offset": agr.b,
        "affine_rms": agr.affine_rms,
        "n_dead": n_dead,
        "n_nodes": int(x_raw.size),
    }
    force = d / "tmp" / "ellipse2d_force.csv"
    if force.exists():
        try:
            f = read_force_csv(force)
            out |= {"Cd": f.cd, "Cl": f.cl, "Cd_std": f.cd_std, "Cl_std": f.cl_std}
        except (ValueError, KeyError, IndexError):
            pass
    return out


def pooled_sd(groups: list[np.ndarray]) -> float:
    """Pool standard deviations across shapes.

    Pools *variances* weighted by degrees of freedom, which is the standard
    within-group estimator: it answers "how much does one measurement of one
    shape wander", never mixing in the between-shape spread, which is physics.

    Args:
        groups (list[np.ndarray]): One array of values per shape.

    Returns:
        float: Pooled standard deviation, ``nan`` if no group has >= 2 values.
    """
    num = 0.0
    dof = 0
    for g in groups:
        g = g[np.isfinite(g)]
        if g.size < 2:
            continue
        num += float(np.var(g, ddof=1)) * (g.size - 1)
        dof += g.size - 1
    return float(np.sqrt(num / dof)) if dof else float("nan")


def main() -> None:
    rows: list[dict] = []
    print("\nP1-F04 -- grading noise floor\n")
    print(
        f"  {'run':<26}{'factor':<12}{'rho':>9}{'a':>8}{'Cd':>9}"
        f"{'Cd_std':>9}{'Cl':>10}{'dead':>7}"
    )
    for shape, (idx, _re) in SHAPES.items():
        for d in sorted(_WORK.glob(f"{shape}_*")):
            if not d.is_dir():
                continue
            g = grade_run(d, idx)
            if g is None:
                print(f"  {d.name:<26}{'':<12}  MISSING / incomplete")
                continue
            factor = factor_of(shape, d.name)
            rows.append({"shape": shape, "tag": d.name, "factor": factor} | g)
            dead = f"{g['n_dead']}/{g['n_nodes']}"
            print(
                f"  {d.name:<26}{factor:<12}{g['rho']:>9.5f}"
                f"{g['a']:>8.4f}{g.get('Cd', np.nan):>9.4f}"
                f"{g.get('Cd_std', np.nan):>9.4f}{g.get('Cl', np.nan):>10.5f}"
                f"{dead:>7}"
            )

    if not rows:
        sys.exit("no graded runs found -- run scripts/paper1/run_p1_f04.sh first")

    def vals(metric: str, subset: list[dict]) -> np.ndarray:
        return np.array([float(r.get(metric, np.nan)) for r in subset])

    # ---- pooled sigma, all factors together -------------------------------
    print("\n  Pooled noise floor (within shape, all 9 runs):\n")
    print(
        f"  {'metric':<10}{'mean':>10}{'sigma':>11}{'CV':>9}"
        f"{'2sigma':>11}{'3sigma':>11}"
    )
    summary: dict[str, dict] = {}
    for m in METRICS:
        per_shape = [vals(m, [r for r in rows if r["shape"] == s]) for s in SHAPES]
        sd = pooled_sd(per_shape)
        mean = float(np.nanmean(vals(m, rows)))
        cv = abs(sd / mean) if mean else float("nan")
        summary[m] = {
            "mean": mean,
            "sigma": sd,
            "cv": cv,
            "two_sigma": 2 * sd,
            "three_sigma": 3 * sd,
        }
        print(
            f"  {m:<10}{mean:>10.4f}{sd:>11.5f}{cv:>9.2%}"
            f"{2 * sd:>11.5f}{3 * sd:>11.5f}"
        )

    # ---- variance decomposition by factor ---------------------------------
    factors = ["phase", "resolution", "avg_window", "phys_time"]
    print("\n  Per-factor sigma (which knob dominates):\n")
    header = "  " + f"{'metric':<10}" + "".join(f"{f:>13}" for f in factors)
    print(header)
    by_factor: dict[str, dict[str, float]] = {}
    for m in METRICS:
        line = f"  {m:<10}"
        by_factor[m] = {}
        for f in factors:
            per_shape = []
            for s in SHAPES:
                sub = [
                    r
                    for r in rows
                    if r["shape"] == s
                    and (r["factor"] == f or r["tag"] == f"{s}_phase0")
                ]
                per_shape.append(vals(m, sub))
            sd = pooled_sd(per_shape)
            by_factor[m][f] = sd
            line += f"{sd:>13.5f}"
        print(line)
        sds = by_factor[m]
        dom = max(sds, key=sds.__getitem__)
        print(f"  {'':<10}{'dominant: ' + dom:>52}")

    # ---- within-run vs between-run unsteadiness ---------------------------
    print("\n  Within-run unsteadiness vs between-run scatter (Cd):\n")
    print(
        f"  {'shape':<16}{'sigma_Cd (between)':>20}"
        f"{'mean Cd_std (within)':>23}{'ratio':>9}"
    )
    unsteady: dict[str, dict] = {}
    for s in SHAPES:
        sub = [r for r in rows if r["shape"] == s]
        between = float(np.nanstd(vals("Cd", sub), ddof=1))
        within = float(np.nanmean(vals("Cd_std", sub)))
        ratio = within / between if between else float("inf")
        unsteady[s] = {"between": between, "within": within, "ratio": ratio}
        print(f"  {s:<16}{between:>20.5f}{within:>23.5f}{ratio:>9.1f}")
    print(
        "\n  A ratio >> 1 means the run-to-run scatter is far smaller than the\n"
        "  flow's own unsteadiness -- the time average is doing its job."
    )

    # ---- verdict against the pre-registered rule --------------------------
    s_rho = summary["rho"]["sigma"]
    s_a = summary["a"]["sigma"]
    hyp = (s_rho < HYP_SIGMA_RHO) and (s_a < HYP_SIGMA_A)
    falsified = (2 * s_rho) > FALSIFIER_2SIGMA_RHO
    verdict = "FALSIFIED" if falsified else ("SUPPORTED" if hyp else "MARGINAL")

    print("\n  Pre-registered decision rule:")
    print(
        f"    sigma_rho < {HYP_SIGMA_RHO}      : "
        f"{s_rho < HYP_SIGMA_RHO} ({s_rho:.5f})"
    )
    print(f"    sigma_a   < {HYP_SIGMA_A}       : {s_a < HYP_SIGMA_A} ({s_a:.5f})")
    print(
        f"    falsifier 2*sigma_rho > {FALSIFIER_2SIGMA_RHO} : "
        f"{falsified} ({2 * s_rho:.5f})"
    )
    print(f"\n  VERDICT: {verdict}\n")

    (_WORK / "grading.json").write_text(
        json.dumps(
            {
                "runs": rows,
                "pooled": summary,
                "by_factor": by_factor,
                "unsteadiness": unsteady,
                "verdict": verdict,
            },
            indent=2,
            default=float,
        )
    )
    print(f"  -> {(_WORK / 'grading.json').relative_to(_ROOT)}\n")


if __name__ == "__main__":
    main()
