"""
P1-F03 -- grade the solver-qualification runs.

Two parts, both graded with the consolidated ``cp_metrics`` module from P1-F02:

  Part A  envelope gate     15 cases spanning (Re, H) at the stability-rule
                            resolution RES = max(100, 0.2*Re)
  Part B  resolution sweep  3 cases at {0.5, 1.0, 1.5, 2.0} x that rule

The pre-registered questions, in order of what they decide:

  1. Is mean rho >= 0.95 across the envelope?               (pass criterion)
  2. Does the affine slope `a` drift systematically with Re, H or b?
     That is the falsifier: a grading scale that depends on the operating
     point confounds every cross-condition comparison in the paper.
  3. Does rho change by less than 0.01 when resolution is doubled above the
     stability rule?                                        (rule validation)

Regression of `a` on the operating point is reported with a bootstrap CI rather
than a normal-theory one: n = 15 is small and `a` is not obviously Gaussian.

Usage:
    python scripts/paper1/p1_f03_grade.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from graph_diffusion.postprocessing.cp_metrics import (  # noqa: E402
    cp_agreement,
    read_cp_csv,
    read_force_csv,
    read_reference_cp,
)

_ROOT = Path(__file__).resolve().parents[2]
_SHAPES = _ROOT / "cfd" / "ellipse2d" / "shapes" / "eval_grid"
_RUNS = _ROOT / "cfd" / "ellipse2d" / "runs"
_OUT = _ROOT / "outputs" / "paper1" / "P1-F03_solver-qualification"

N_BOOT = 10000
RNG = np.random.default_rng(0)


def load_manifest() -> list[dict]:
    """Read the eval_grid manifest into a list of case dicts."""
    rows: list[dict] = []
    lines = (_SHAPES / "manifest.csv").read_text().strip().splitlines()
    head = lines[0].split(",")
    for line in lines[1:]:
        vals = line.split(",")
        rec = dict(zip(head, vals, strict=True))
        rows.append(
            {
                "name": rec["name"],
                "idx": int(rec["idx"]),
                "Re": float(rec["Re"]),
                "H": float(rec["H"]),
                "b": float(rec["b"]),
                "n_nodes": int(rec["n_nodes"]),
            }
        )
    return rows


def grade_run(name: str, idx: int) -> dict | None:
    """Grade one solver run against its dataset reference curve.

    Args:
        name (str): Run directory name.
        idx (int): Dataset sample index, used to locate the reference curve.

    Returns:
        dict | None: Metrics, or ``None`` if either file is missing.
    """
    meas = _RUNS / name / "tmp" / "ellipse2d_cp.csv"
    ref = _SHAPES / f"sim_{idx}_ref_cp.csv"
    if not meas.exists() or not ref.exists():
        return None

    x_m, cp_m, _ = read_cp_csv(meas)
    # The reference is dataset p*; read_reference_cp converts it to Cp so `a`
    # measures disagreement rather than a units convention.
    x_r, cp_r = read_reference_cp(ref)
    agr = cp_agreement(x_r, cp_r, x_m, cp_m, mode="collapsed")

    out: dict[str, object] = {
        "rho": agr.rho,
        "a": agr.a,
        "b_offset": agr.b,
        "affine_rms": agr.affine_rms,
        "residual_runs_z": agr.residual_runs_z,
        "n_meas": int(x_m.size),
        "n_ref": int(x_r.size),
    }
    force = _RUNS / name / "tmp" / "ellipse2d_force.csv"
    if force.exists():
        try:
            f = read_force_csv(force)
            out |= {"Cd": f.cd, "Cl": f.cl, "Cd_std": f.cd_std, "Cl_std": f.cl_std}
        except (ValueError, KeyError, IndexError):
            pass
    return out


def boot_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least-squares slope with a bootstrap 95% CI.

    Returns:
        tuple[float, float, float]: ``(slope, lo, hi)``.
    """
    n = x.size
    slope = float(np.polyfit(x, y, 1)[0])
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        j = RNG.integers(0, n, n)
        if np.ptp(x[j]) < 1e-12:
            boots[i] = np.nan
            continue
        boots[i] = np.polyfit(x[j], y[j], 1)[0]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return slope, float(lo), float(hi)


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    cases = load_manifest()

    # ---- Part A: envelope gate --------------------------------------------
    print("\nP1-F03 -- solver qualification\n")
    print("Part A: envelope gate")
    print(
        f"  {'case':<24}{'Re':>6}{'H':>6}{'b':>7}{'rho':>8}{'a':>8}"
        f"{'b_off':>8}{'rms':>8}{'|z|':>7}{'Cd':>8}"
    )
    rows: list[dict] = []
    for c in cases:
        g = grade_run(c["name"], c["idx"])
        if g is None:
            print(f"  {c['name']:<24}  MISSING")
            continue
        rows.append(c | g)
        print(
            f"  {c['name']:<24}{c['Re']:>6.0f}{c['H']:>6.2f}{c['b']:>7.3f}"
            f"{g['rho']:>8.4f}{g['a']:>8.3f}{g['b_offset']:>8.3f}"
            f"{g['affine_rms']:>8.4f}{abs(g['residual_runs_z']):>7.1f}"
            f"{g.get('Cd', float('nan')):>8.3f}"
        )

    rho = np.array([r["rho"] for r in rows])
    a = np.array([r["a"] for r in rows])
    re_vals = np.array([r["Re"] for r in rows])  # not `re`: shadows the module
    hh = np.array([r["H"] for r in rows])
    bb = np.array([r["b"] for r in rows])
    z = np.array([abs(r["residual_runs_z"]) for r in rows])

    print(f"\n  n = {len(rows)} of {len(cases)}")
    print(f"  rho : mean {rho.mean():.4f}  min {rho.min():.4f}  max {rho.max():.4f}")
    print(
        f"  a   : mean {a.mean():.4f}  sd {a.std(ddof=1):.4f}  "
        f"CV {a.std(ddof=1) / a.mean():.2%}  range {a.min():.3f}-{a.max():.3f}"
    )
    print(f"  |residual runs z| : median {np.median(z):.2f}  max {z.max():.2f}")

    # ---- The falsifier: does `a` depend on the operating point? -----------
    print("\n  Regression of `a` on the operating point (bootstrap 95% CI):")
    reg: dict[str, dict] = {}
    span = {"Re": re_vals, "H": hh, "b": bb}
    for label, v in span.items():
        s, lo, hi = boot_slope(v, a)
        # Express the drift as a percentage of mean `a` across the observed
        # span of the predictor -- that is the quantity the +-10% rule bounds.
        drift = s * (v.max() - v.min()) / a.mean() * 100.0
        excl = "yes" if (lo > 0) or (hi < 0) else "no"
        reg[label] = {
            "slope": s,
            "ci_lo": lo,
            "ci_hi": hi,
            "drift_pct_over_span": drift,
            "excludes_zero": excl == "yes",
        }
        print(
            f"    a ~ {label:<3} slope {s:>11.3e}  CI [{lo:>10.3e},{hi:>10.3e}]"
            f"  drift over span {drift:>7.2f}%   CI excludes 0: {excl}"
        )

    # ---- Part B: resolution sensitivity -----------------------------------
    print("\nPart B: resolution sensitivity")
    res_rows: list[dict] = []
    for d in sorted(_RUNS.glob("re*_res*")):
        m = re.match(r"(.+)_res(\d+)$", d.name)
        if not m:
            continue
        base, res = m.group(1), int(m.group(2))
        case = next((c for c in cases if c["name"] == base), None)
        if case is None:
            continue
        g = grade_run(d.name, case["idx"])
        if g is None:
            continue
        res_rows.append({"base": base, "res": res, "Re": case["Re"], **g})

    print(f"  {'case':<24}{'RES':>6}{'mult':>7}{'rho':>9}{'a':>8}{'Cd':>9}")
    sweeps: dict[str, list] = {}
    for r in sorted(res_rows, key=lambda r: (r["base"], r["res"])):
        rule = max(100, round(0.2 * r["Re"]))
        mult = r["res"] / rule
        sweeps.setdefault(r["base"], []).append(r)
        print(
            f"  {r['base']:<24}{r['res']:>6}{mult:>7.1f}{r['rho']:>9.4f}"
            f"{r['a']:>8.3f}{r.get('Cd', float('nan')):>9.4f}"
        )

    print("\n  Change from the 1.0x rule to 2.0x (the stability-rule check):")
    conv: dict[str, dict] = {}
    for base, rs in sweeps.items():
        rule = max(100, round(0.2 * rs[0]["Re"]))
        at = {round(r["res"] / rule, 2): r for r in rs}
        if 1.0 in at and 2.0 in at:
            d_rho = at[2.0]["rho"] - at[1.0]["rho"]
            d_a = at[2.0]["a"] - at[1.0]["a"]
            d_cd = at[2.0].get("Cd", np.nan) - at[1.0].get("Cd", np.nan)
            conv[base] = {
                "d_rho": d_rho,
                "d_a": d_a,
                "d_a_pct": d_a / at[1.0]["a"] * 100.0,
                "d_Cd": d_cd,
            }
            print(
                f"    {base:<24} d_rho {d_rho:>+8.5f}   d_a {d_a:>+7.4f} "
                f"({d_a / at[1.0]['a'] * 100:>+6.2f}%)   d_Cd {d_cd:>+7.4f}"
            )

    # ---- Verdict against the pre-registered rule --------------------------
    worst_drift = max(abs(v["drift_pct_over_span"]) for v in reg.values())
    sig = [k for k, v in reg.items() if v["excludes_zero"]]
    rho_pass = rho.mean() >= 0.95
    if not sig and worst_drift < 10.0 and rho_pass:
        verdict = "SUPPORTED"
    elif worst_drift > 10.0:
        verdict = "FALSIFIED"
    else:
        verdict = "MARGINAL"

    print(f"\n  mean rho >= 0.95 : {rho_pass} ({rho.mean():.4f})")
    print(f"  worst |drift| in `a` over a predictor span : {worst_drift:.2f}%")
    print(f"  predictors whose CI excludes zero : {sig or 'none'}")
    print(f"\n  VERDICT: {verdict}\n")

    (_OUT / "grading.json").write_text(
        json.dumps(
            {
                "envelope": rows,
                "resolution": res_rows,
                "summary": {
                    "n": len(rows),
                    "rho_mean": float(rho.mean()),
                    "rho_min": float(rho.min()),
                    "a_mean": float(a.mean()),
                    "a_sd": float(a.std(ddof=1)),
                    "a_cv": float(a.std(ddof=1) / a.mean()),
                    "residual_z_median": float(np.median(z)),
                },
                "regression": reg,
                "resolution_convergence": conv,
                "verdict": verdict,
            },
            indent=2,
            default=float,
        )
    )
    print(f"  -> {(_OUT / 'grading.json').relative_to(_ROOT)}\n")


if __name__ == "__main__":
    main()
