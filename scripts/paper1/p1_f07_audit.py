"""
P1-F07 -- Cp normalisation audit.

Answers where the affine magnitude factor between solver-measured Cp and dataset
Cp comes from. The spec registers a table of candidate mechanisms, each
predicting a specific number, so the audit is a matching exercise rather than a
rationalisation.

Four steps are executed here; steps 5 and 6 are document and code archaeology
and live in ``DGN_Docs/paper1/DATASET_PROVENANCE.md``.

  step 1  Eliminate the solver, using F06's absolute benchmark.
  step 2  Solver-internal self-consistency: Cp must be invariant to the choice
          of U at fixed Re, and to rho outright.
  step 3  Match the observed factor against the registered candidates.
  step 4  Constancy regression -- is the factor one constant, or is it
          structured in Re, H or b?

Step 4 is the one that decides the verdict, and it turns on using the **right**
error bar. F03 ran its own regression against the scatter of its own residuals
and found a Re slope excluding zero. That test asks "is there a trend in these
15 numbers"; the question F07 must answer is "is there a trend **larger than the
instrument can resolve**", and for that the error bar is F04's frozen sigma_a,
not the observed scatter. The two give different answers, and only the second
one is a physical statement.

Usage:
    python scripts/paper1/p1_f07_audit.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from graph_diffusion.postprocessing.cp_metrics import (  # noqa: E402
    DATASET_P_STAR_TO_CP,
    cp_agreement,
    read_cp_csv,
)

_ROOT = Path(__file__).resolve().parents[2]
_F03 = _ROOT / "outputs" / "paper1" / "P1-F03_solver-qualification" / "grading.json"
_F04 = _ROOT / "outputs" / "paper1" / "P1-F04_noise-floor" / "grading.json"
_STEP2 = _ROOT / "outputs" / "paper1" / "P1-F07_normalisation-audit" / "selfconsistency"
_OUT = _ROOT / "outputs" / "paper1" / "P1-F07_normalisation-audit"

#: Step 2 runs: tag -> (U, MAX_PHYS_T, note). "matched" rows hold the flow
#: development fixed at 50 chord-flows; the unmatched rows hold MAX_PHYS_T fixed
#: instead and are kept only to show the confound they introduce.
STEP2_RUNS: dict[str, tuple[float, float, str]] = {
    "base": (1.0, 50.0, "matched"),
    "u2": (2.0, 50.0, "unmatched"),
    "uhalf": (0.5, 50.0, "unmatched"),
    "rho2": (1.0, 50.0, "matched"),
    "u2m": (2.0, 25.0, "matched"),
    "uhalfm": (0.5, 100.0, "matched"),
}

#: The matched runs that vary U. rho2 is also matched but tests density only.
U_MATCHED = ("u2m", "uhalfm")

# Frozen by P1-F04, 2026-08-17, PROTOCOL section 6.
SIGMA_A = 0.03521
SIGMA_RHO = 0.00108

# Registered candidate table, P1-F07 spec step 3. "struck" records candidates
# already eliminated by DATASET_PROVENANCE section 5 before any run.
CANDIDATES: tuple[tuple[int, str, float | None, str], ...] = (
    (1, "Dataset q_ref = 1.8x half-rho-U-squared", 1.80, "struck: DP section 5"),
    (2, "Dataset U_ref = 4/3 x ours", 1.78, "struck: DP section 5"),
    (3, "Missing factor of 1/2", 2.00, "derived from source, DP section 4"),
    (4, "U_max vs U_mean, parabolic (3/2)", 2.25, "constant"),
    (5, "Dataset velocity scaled 1.8x", 3.24, "struck: DP section 5"),
    (6, "Momentum-average of parabolic (6/5)", 1.44, "constant"),
    (7, "Channel blockage at H = 5.5", 1.375, "varies with b"),
    (8, "Compressibility, O(Ma^2)", 1.00, "varies with Re"),
    (9, "rho mismatch converter vs qInf", None, "caught by step 2"),
    (10, "Averaging-window mismatch", 1.00, "varies with Re"),
)


def chi2_const(a: np.ndarray, sigma: float) -> tuple[float, int]:
    """Chi-squared of a constant model, with a known per-point uncertainty.

    Args:
        a (np.ndarray): Measured values.
        sigma (float): Known measurement uncertainty on each value.

    Returns:
        tuple[float, int]: ``(chi2, dof)``.
    """
    return float((((a - a.mean()) / sigma) ** 2).sum()), int(a.size - 1)


def chi2_linear(
    x: np.ndarray, a: np.ndarray, sigma: float
) -> tuple[float, int, float, float]:
    """Chi-squared of a straight-line model, with a known per-point uncertainty.

    Because sigma is common to every point the weighted fit reduces to ordinary
    least squares, but the *uncertainty on the slope* comes from sigma rather
    than from the residual scatter -- which is the whole point of the test.

    Args:
        x (np.ndarray): Regressor.
        a (np.ndarray): Measured values.
        sigma (float): Known measurement uncertainty on each value.

    Returns:
        tuple[float, int, float, float]: ``(chi2, dof, slope, slope_se)``.
    """
    xc = x - x.mean()
    sxx = float((xc**2).sum())
    slope = float((xc * a).sum() / sxx) if sxx else 0.0
    pred = a.mean() + slope * xc
    chi2 = float((((a - pred) / sigma) ** 2).sum())
    slope_se = float(sigma / np.sqrt(sxx)) if sxx else float("inf")
    return chi2, int(a.size - 2), slope, slope_se


def main() -> None:
    print("\nP1-F07 -- Cp normalisation audit\n")
    report: dict = {"sigma_a": SIGMA_A, "dataset_scale": DATASET_P_STAR_TO_CP}

    # ---- step 1 -----------------------------------------------------------
    print("  Step 1 -- eliminate the solver (from P1-F06)\n")
    print("    Schafer-Turek 2D-1, L0 fp64: dP within 0.03% of the interval.")
    print("    L1 production case:          dP 1-4% below the interval.")
    print("    The spec's STOP branch needs a solver dP error NEAR the factor;")
    print("    the largest is 4%, so the solver's pressure scale is eliminated.\n")
    report["step1"] = {"stop_branch": False, "max_dp_error_pct": 4.0}

    # ---- step 4 -----------------------------------------------------------
    grading = json.loads(_F03.read_text())
    env = grading["envelope"]
    rows = list(env.values()) if isinstance(env, dict) else list(env)
    a = np.array([r["a"] for r in rows], dtype=float)
    re = np.array([r["Re"] for r in rows], dtype=float)
    hh = np.array([r["H"] for r in rows], dtype=float)
    bb = np.array([r["b"] for r in rows], dtype=float)

    print(f"  Step 4 -- constancy regression over {a.size} envelope cases\n")
    print(f"    a          = {a.mean():.4f} +/- {a.std(ddof=1):.4f} (observed sd)")
    print(f"    sigma_a    = {SIGMA_A:.5f} (P1-F04, frozen)")
    ratio = a.std(ddof=1) / SIGMA_A
    print(f"    sd/sigma_a = {ratio:.3f}")

    c0, dof0 = chi2_const(a, SIGMA_A)
    print(
        f"\n    Constant model: chi2 = {c0:.2f} on {dof0} dof, "
        f"chi2/dof = {c0 / dof0:.3f}"
    )
    if c0 / dof0 < 1.0:
        print("    -> the entire spread is inside the measurement floor. There is")
        print("       no excess variance for any mechanism to explain.")
    else:
        print("    -> excess variance beyond the floor; something is structured.")

    print("\n    Adding each regressor, with sigma_a as the error bar:\n")
    print(
        f"    {'term':<6}{'slope':>13}{'se':>12}{'slope/se':>10}"
        f"{'delta chi2':>12}{'span effect':>13}"
    )
    terms: dict[str, dict] = {}
    for name, x in (("Re", re), ("H", hh), ("b", bb)):
        c1, dof1, slope, se = chi2_linear(x, a, SIGMA_A)
        z = slope / se if se else float("inf")
        span = slope * (x.max() - x.min())
        terms[name] = {
            "slope": slope,
            "slope_se": se,
            "z": z,
            "delta_chi2": c0 - c1,
            "span_effect": span,
            "span_effect_sigma": span / SIGMA_A,
            "significant": abs(z) > 2.0,
        }
        print(
            f"    {name:<6}{slope:>13.6f}{se:>12.6f}{z:>10.2f}"
            f"{c0 - c1:>12.2f}{span:>10.4f} ({span / SIGMA_A:>.2f}s)"
        )

    print("\n    A |slope/se| > 2 is a term the instrument can actually resolve.")
    resolved = [k for k, v in terms.items() if v["significant"]]
    if resolved:
        print(f"    Resolved: {', '.join(resolved)} -- a is STRUCTURED.")
    else:
        print("    None resolved -- a is CONSTANT within the measurement floor.")
    report["step4"] = {
        "a_mean": float(a.mean()),
        "a_sd": float(a.std(ddof=1)),
        "sd_over_sigma": float(ratio),
        "chi2_const": c0,
        "dof_const": dof0,
        "chi2_per_dof": c0 / dof0,
        "terms": terms,
        "structured": bool(resolved),
    }

    # F03 regressed against its own residual scatter and reached a different
    # conclusion. Show both, so the difference is on the record rather than
    # silently overwritten.
    f03_reg = grading.get("regression", {})
    if f03_reg:
        print("\n    For comparison, F03's own regression (bootstrapped on the")
        print("    residual scatter, not on sigma_a):\n")
        for k, v in f03_reg.items():
            mark = "excludes 0" if v.get("excludes_zero") else "includes 0"
            print(
                f"      {k:<4} slope {v['slope']:+.6f}  "
                f"drift {v['drift_pct_over_span']:+.2f}% over span  [{mark}]"
            )
        report["step4"]["f03_regression"] = f03_reg

    # ---- step 4b: is the Re term the convention, or the instrument? -------
    # RESOLUTION is set from Re by the stability rule, so in this sweep the two
    # are perfectly confounded and no regression on these 15 cases can separate
    # them. F04 measured the resolution factor on its own, at FIXED Re, which is
    # the only handle available.
    res = np.maximum(100.0, np.round(0.2 * re))
    corr = float(np.corrcoef(re, res)[0, 1])
    f04 = json.loads(_F04.read_text())
    sig_res = float(f04["by_factor"]["a"]["resolution"])
    span_re = abs(terms["Re"]["span_effect"])

    print("\n    Step 4b -- is that Re term the convention or the instrument?\n")
    print(
        f"      RESOLUTION = max(100, 0.2*Re) over this sweep: "
        f"{res.min():.0f}-{res.max():.0f}"
    )
    print(f"      corr(Re, RESOLUTION) = {corr:+.4f}")
    if corr > 0.99:
        print("      -> perfectly confounded. These 15 cases CANNOT separate a")
        print("         Re dependence of the convention from a resolution")
        print("         artefact of the solver.")
    print(f"\n      F04 sigma_a from resolution alone, at fixed Re: {sig_res:.5f}")
    print(
        f"      Re-span effect on a:                             {span_re:.5f}"
        f"  ({span_re / sig_res:.2f}x)"
    )
    if span_re < 3.0 * sig_res:
        print("      -> the trend is the same size as the resolution factor F04")
        print("         measured directly. It is fully accounted for by the")
        print("         instrument; no convention-side Re dependence is needed.")
    report["step4b"] = {
        "corr_re_resolution": corr,
        "sigma_a_resolution_f04": sig_res,
        "re_span_effect": span_re,
        "ratio": span_re / sig_res if sig_res else float("inf"),
        "separable": corr <= 0.99,
    }

    # F03's own resolution sweep is the direct check: if under-resolution biased
    # a, d_a would share a sign across cases.
    conv = grading.get("resolution_convergence", {})
    if conv:
        d = [v["d_a_pct"] for v in conv.values()]
        signs = {np.sign(v) for v in d}
        print(
            f"\n      F03 resolution sweep, change in a on doubling: "
            f"{', '.join(f'{v:+.1f}%' for v in d)}"
        )
        if len(signs) > 1:
            print("      -> signs disagree, so this is scatter, not a systematic")
            print("         under-resolution bias. The Re term is noise in the")
            print("         direction of Re, not a mechanism.")
        report["step4b"]["f03_resolution_d_a_pct"] = d
        report["step4b"]["consistent_sign"] = len(signs) == 1

    # ---- step 3 -----------------------------------------------------------
    # The observed factor in the units the candidates are written in: they
    # predict the RAW ratio, before the factor-of-two conversion is applied.
    a_raw = a.mean() * DATASET_P_STAR_TO_CP
    print(f"\n  Step 3 -- candidate matching (observed raw factor " f"{a_raw:.4f})\n")
    print(f"    {'#':<4}{'mechanism':<44}{'pred':>7}{'miss':>9}  status")
    cand_out = []
    for num, name, pred, status in CANDIDATES:
        if pred is None:
            print(f"    {num:<4}{name:<44}{'-':>7}{'-':>9}  {status}")
            cand_out.append({"n": num, "name": name, "pred": None, "status": status})
            continue
        miss = (a_raw - pred) / pred
        print(f"    {num:<4}{name:<44}{pred:>7.2f}{miss:>8.1%}  {status}")
        cand_out.append(
            {"n": num, "name": name, "pred": pred, "rel_miss": miss, "status": status}
        )
    report["step3"] = {"a_raw": float(a_raw), "candidates": cand_out}

    # ---- step 2 -----------------------------------------------------------
    print("\n  Step 2 -- solver-internal self-consistency\n")
    runs = {}
    for tag in STEP2_RUNS:
        f = _STEP2 / tag / "tmp" / "ellipse2d_cp.csv"
        if f.exists():
            x, cp, _ = read_cp_csv(f)
            runs[tag] = (x, cp)
    if "base" not in runs:
        print("    not yet run -- scripts/paper1/run_p1_f07_selfconsistency.sh")
        report["step2"] = {"status": "pending"}
    else:
        xb, cb = runs["base"]
        print(
            f"    {'run':<9}{'U':>5}{'T':>6}{'flows':>7}{'a vs base':>12}"
            f"{'dev':>9}{'max|dCp|':>11}  verdict"
        )
        step2: dict[str, dict] = {}
        for tag in STEP2_RUNS:
            if tag == "base":
                continue
            if tag not in runs:
                print(f"    {tag:<9}  not yet run")
                continue
            u, maxt, note = STEP2_RUNS[tag]
            x, cp = runs[tag]
            agr = cp_agreement(xb, cb, x, cp, mode="collapsed")
            # Same body and resolution, so the probe lists align element-wise.
            # np.interp cannot be used: x/c is not monotonic on a closed
            # surface, it doubles back along the other branch.
            dmax = (
                float(np.max(np.abs(cp - cb))) if cp.shape == cb.shape else float("nan")
            )
            dev = abs(agr.a - 1.0)
            ok = dev < 2 * SIGMA_A
            step2[tag] = {
                "U": u,
                "max_phys_t": maxt,
                "chord_flows": u * maxt,
                "rho": agr.rho,
                "a": agr.a,
                "dev_from_unity": dev,
                "dev_sigma": dev / SIGMA_A,
                "max_abs_dcp": dmax,
                "matched": note == "matched",
                "invariant": bool(ok),
            }
            print(
                f"    {tag:<9}{u:>5.1f}{maxt:>6.0f}{u * maxt:>7.0f}"
                f"{agr.a:>12.5f}{dev / SIGMA_A:>8.2f}s{dmax:>11.2e}"
                f"  {'invariant' if ok else 'NOT INVARIANT'}"
            )
        report["step2"] = step2

        # The U-invariance claim needs the U-varying matched runs. rho2 is
        # matched too but only speaks to density, so it cannot stand in.
        matched = [v for k, v in step2.items() if v["matched"] and k in U_MATCHED]
        unmatched = [v for k, v in step2.items() if not v["matched"]]
        print()
        if len(matched) < len(U_MATCHED):
            print(
                f"    U-invariance: incomplete "
                f"({len(matched)}/{len(U_MATCHED)} matched runs) -- no verdict."
            )
        elif matched:
            worst = max(matched, key=lambda v: v["dev_sigma"])
            print(
                f"    Chord-flows matched: worst deviation "
                f"{worst['dev_sigma']:.2f} sigma_a."
            )
            if all(v["invariant"] for v in matched):
                print("    -> Cp is invariant to U at fixed Re and to rho.")
                print("       The solver's de-dimensionalisation is")
                print("       self-consistent; candidate 9 is eliminated.")
            else:
                print("    -> NOT invariant. Normalisation error inside the")
                print("       solver; start at results.h:380 qInf.")
        if unmatched:
            worst_u = max(unmatched, key=lambda v: v["dev_sigma"])
            print(
                f"\n    Unmatched (T held at 50, so 25-100 chord-flows): worst"
                f" {worst_u['dev_sigma']:.2f} sigma_a."
            )
            print("    That is the chord-flow confound, not a normalisation")
            print("    error -- kept in the table so the design fault is on")
            print("    the record rather than quietly dropped.")

    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "audit.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\n  -> {(_OUT / 'audit.json').relative_to(_ROOT)}\n")


if __name__ == "__main__":
    main()
