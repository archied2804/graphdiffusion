"""
P1-G01 -- grade held-out inverse design. The paper's headline numbers.

Each generated shape is graded against its **target's** dataset Cp curve, and
its drag against the **target geometry's own solver run**, so both sides of the
Cd comparison pass through one instrument and every shared systematic --
including F07's constant 9.35% offset -- cancels.

Three things this reports that a bare mean would hide:

  the distribution   Per-target rho, not just its mean. The mean alone cannot
                     say whether failures are rare and severe or uniform and
                     mild, and those imply very different things.

  the leak, labelled The two Year-1 targets are TRAINING samples. They are
                     graded with the same model and reported beside the clean
                     numbers, which is the clearest statement that the problem
                     was found and corrected.

  the comparator     P1-B01 ran nearest-neighbour retrieval on these exact
                     targets. Retrieval reached the instrument's self-grading
                     ceiling, so the honest question is not "is rho high" but
                     "is the model any better than a lookup".

rho is reported against its chance level throughout (P1-F04 section 4: unrelated
dataset shapes correlate at 0.9954), because on this corpus an unreferenced rho
carries almost no information.

Usage:
    python scripts/paper1/p1_g01_grade.py
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
    dead_probe_mask,
    polygon_validity,
    read_cp_csv,
    read_force_csv,
    read_reference_cp,
)

_SHAPES = _ROOT / "cfd" / "ellipse2d" / "shapes" / "g01_holdout"
_TRUTH_SHAPES = _ROOT / "cfd" / "ellipse2d" / "shapes" / "g01_truth"
_WORK = _ROOT / "outputs" / "paper1" / "P1-G01_holdout-inverse-design" / "solver"
_TRUTH_WORK = _ROOT / "outputs" / "paper1" / "P1-G01_holdout-inverse-design" / "truth"
_B01 = (
    _ROOT
    / "outputs"
    / "paper1"
    / "P1-B01_retrieval-baseline"
    / "retrieval_grading.json"
)
_OUT = _ROOT / "outputs" / "paper1" / "P1-G01_holdout-inverse-design"

# Frozen by P1-F04, PROTOCOL section 6.
SIGMA_RHO = 0.00108
SIGMA_A = 0.03521
SIGMA_CD = 0.04742
RHO_CHANCE = 0.9954

# Pre-registered in the P1-G01 spec, before any of this ran.
HYP_RHO_LO, HYP_RHO_HI = 0.88, 0.94
FALSIFIER_RHO = 0.80


def ci95(x: np.ndarray) -> tuple[float, float]:
    """Mean and half-width of a 95% CI on the mean.

    Args:
        x (np.ndarray): Sample values.

    Returns:
        tuple[float, float]: ``(mean, half_width)``; half-width is ``nan`` for
            fewer than two finite values.
    """
    x = x[np.isfinite(x)]
    if x.size < 2:
        return (float(x.mean()) if x.size else float("nan"), float("nan"))
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(x.size))


def read_body(path: Path) -> np.ndarray:
    """Read a solver body file back as a point array.

    Args:
        path (Path): The ``.dat`` file.

    Returns:
        np.ndarray: Points, shape ``(N, 2)``.
    """
    pts = [
        tuple(float(v) for v in ln.split())
        for ln in path.read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    return np.asarray(pts, dtype=float)


def grade_run(work: Path, tag: str, ref: Path) -> dict | None:
    """Grade one solver run against a reference Cp curve.

    Args:
        work (Path): Root directory holding run subdirectories.
        tag (str): Run tag.
        ref (Path): Reference Cp CSV for the target.

    Returns:
        dict | None: Metrics, or ``None`` if the run did not complete.
    """
    meas = work / tag / "tmp" / "ellipse2d_cp.csv"
    if not meas.exists() or not ref.exists():
        return None
    x_raw, cp_raw, rms_raw = read_cp_csv(meas, drop_dead=False)
    n_dead = int(dead_probe_mask(cp_raw, rms_raw).sum())
    x_m, cp_m, _ = read_cp_csv(meas)
    x_r, cp_r = read_reference_cp(ref)
    agr = cp_agreement(x_r, cp_r, x_m, cp_m, mode="collapsed")
    out: dict = {
        "rho": agr.rho,
        "a": agr.a,
        "affine_rms": agr.affine_rms,
        "band_le": agr.band_rms.get("le", float("nan")),
        "band_peak": agr.band_rms.get("peak", float("nan")),
        "band_aft": agr.band_rms.get("aft", float("nan")),
        "peak_ref": agr.suction_peak_ref,
        "peak_meas": agr.suction_peak_meas,
        "peak_x_ref": agr.suction_peak_x_ref,
        "peak_x_meas": agr.suction_peak_x_meas,
        "n_dead": n_dead,
    }
    force = work / tag / "tmp" / "ellipse2d_force.csv"
    if force.exists():
        try:
            f = read_force_csv(force)
            out |= {"Cd": f.cd, "Cl": f.cl}
        except (ValueError, KeyError, IndexError):
            pass
    return out


def main() -> None:
    man = list(csv.DictReader((_SHAPES / "manifest.csv").open()))
    if not man:
        sys.exit("no manifest -- run scripts/paper1/p1_g01_sample.py first")

    # ---- reference drag, from the target geometries themselves -----------
    truth: dict[str, dict] = {}
    tman = _TRUTH_SHAPES / "manifest.csv"
    if tman.exists():
        for r in csv.DictReader(tman.open()):
            g = grade_run(_TRUTH_WORK, r["tag"], _SHAPES / f"{r['label']}_ref_cp.csv")
            if g is not None:
                truth[r["label"]] = g

    print("\nP1-G01 -- held-out inverse design\n")
    if not truth:
        print("  ⚠  no target runs found -- Cd error cannot be computed.")
        print("     run: SHAPES=.../g01_truth WORK=.../truth run_p1_g01_solver.sh\n")

    rows: list[dict] = []
    n_invalid = 0
    for m in man:
        g = grade_run(_WORK, m["tag"], _SHAPES / f"{m['label']}_ref_cp.csv")
        body = _SHAPES / f"{m['tag']}.dat"
        val = polygon_validity(read_body(body)) if body.exists() else None
        ok = bool(getattr(val, "is_valid", False))
        if not ok:
            n_invalid += 1
        if g is None:
            rows.append({**m, "completed": 0, "valid": int(ok)})
            continue
        t = truth.get(m["label"], {})
        cd_ref = t.get("Cd", float("nan"))
        cd = g.get("Cd", float("nan"))
        rows.append(
            {
                **m,
                **g,
                "completed": 1,
                "valid": int(ok),
                "Cd_target": cd_ref,
                "Cd_rel_err": (
                    (cd - cd_ref) / cd_ref
                    if np.isfinite(cd_ref) and cd_ref
                    else float("nan")
                ),
            }
        )

    done = [r for r in rows if r.get("completed")]
    hold = [r for r in done if not int(r["is_training_sample"])]
    leak = [r for r in done if int(r["is_training_sample"])]
    print(f"  completed {len(done)}/{len(rows)}   invalid geometry {n_invalid}")

    if not hold:
        sys.exit("\n  no completed held-out runs -- run the solver sweep first\n")

    # ---- headline --------------------------------------------------------
    rho = np.array([r["rho"] for r in hold])
    a = np.array([r["a"] for r in hold])
    eaff = np.array([r["affine_rms"] for r in hold])
    cderr = np.array([r.get("Cd_rel_err", np.nan) for r in hold], dtype=float)

    print(
        f"\n  HELD-OUT (n = {len(hold)} = "
        f"{len({r['label'] for r in hold})} targets x seeds)\n"
    )
    print(f"  {'quantity':<22}{'mean':>11}{'95% CI':>11}{'median':>11}")
    for name, arr, fmt in (
        ("rho", rho, "{:.5f}"),
        ("a", a, "{:.4f}"),
        ("eps_affine", eaff, "{:.4f}"),
        ("Cd rel. error", cderr, "{:.4f}"),
    ):
        m, h = ci95(arr)
        med = float(np.nanmedian(arr)) if np.isfinite(arr).any() else float("nan")
        ci = ("+/- " + fmt.format(h)) if np.isfinite(h) else ""
        print(f"  {name:<22}{fmt.format(m):>11}{ci:>11}{fmt.format(med):>11}")

    print(
        f"\n  rho vs chance level ({RHO_CHANCE}): "
        f"{float(np.nanmean(rho)) - RHO_CHANCE:+.5f}"
        f"  ({(float(np.nanmean(rho)) - RHO_CHANCE) / SIGMA_RHO:+.1f} sigma)"
    )

    # ---- the comparison that carries the gate ---------------------------
    if _B01.exists():
        b = json.loads(_B01.read_text())
        ret = [r for r in b["runs"] if r["set"] == "indist"]
        rret = float(np.median([r["rho"] for r in ret]))
        print(f"\n  P1-B01 retrieval on the SAME targets: median rho {rret:.5f}")
        print(
            f"  P1-G01 model:                        median rho "
            f"{float(np.nanmedian(rho)):.5f}"
        )
        d = float(np.nanmedian(rho)) - rret
        print(f"  difference: {d:+.5f}  ({d / SIGMA_RHO:+.1f} sigma)")
        if abs(d) < 2 * SIGMA_RHO:
            print("  -> within 2 sigma: NO DETECTABLE DIFFERENCE")
        elif d > 0:
            print("  -> model better")
        else:
            print("  -> RETRIEVAL BETTER")

    # ---- per-target distribution ----------------------------------------
    print("\n  Per-target (mean over seeds), sorted by rho:\n")
    print(f"  {'target':<16}{'n':>3}{'rho':>10}{'a':>9}{'Cd err':>10}{'peak err':>10}")
    per: dict[str, dict] = {}
    for lab in sorted({r["label"] for r in hold}):
        sub = [r for r in hold if r["label"] == lab]
        per[lab] = {
            "n": len(sub),
            "rho": float(np.mean([r["rho"] for r in sub])),
            "a": float(np.mean([r["a"] for r in sub])),
            "Cd_rel_err": float(np.nanmean([r.get("Cd_rel_err", np.nan) for r in sub])),
            "peak_err": float(
                np.nanmean([r["peak_meas"] - r["peak_ref"] for r in sub])
            ),
        }
    for lab, v in sorted(per.items(), key=lambda kv: kv[1]["rho"]):
        print(
            f"  {lab:<16}{v['n']:>3}{v['rho']:>10.5f}{v['a']:>9.4f}"
            f"{v['Cd_rel_err']:>10.4f}{v['peak_err']:>10.4f}"
        )

    # ---- the leak, labelled ---------------------------------------------
    if leak:
        lrho = np.array([r["rho"] for r in leak])
        print(f"\n  YEAR-1 TARGETS -- these are TRAINING SAMPLES (n = {len(leak)}):")
        print(
            f"    mean rho {float(np.mean(lrho)):.5f}   "
            f"held-out mean {float(np.mean(rho)):.5f}   "
            f"gap {float(np.mean(lrho) - np.mean(rho)):+.5f}"
        )

    # ---- pre-registered decision rule -----------------------------------
    mean_rho = float(np.nanmean(rho))
    if mean_rho < FALSIFIER_RHO:
        verdict = "FALSIFIED (memorisation branch)"
    elif HYP_RHO_LO <= mean_rho <= HYP_RHO_HI:
        verdict = "SUPPORTED"
    elif mean_rho > HYP_RHO_HI:
        verdict = "ABOVE PREDICTION -- check the partition audit before believing it"
    else:
        verdict = "BELOW PREDICTION but above the falsifier"
    print("\n  Pre-registered decision rule:")
    print(f"    hypothesis  rho in [{HYP_RHO_LO}, {HYP_RHO_HI}]")
    print(f"    falsifier   rho < {FALSIFIER_RHO}")
    print(f"    measured    rho = {mean_rho:.5f}")
    print(f"\n  VERDICT: {verdict}\n")

    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "grading.json").write_text(
        json.dumps(
            {
                "sigma": {"rho": SIGMA_RHO, "a": SIGMA_A, "Cd": SIGMA_CD},
                "rho_chance_level": RHO_CHANCE,
                "runs": rows,
                "per_target": per,
                "n_invalid_geometry": n_invalid,
                "verdict": verdict,
            },
            indent=2,
            default=float,
        )
    )
    print(f"  -> {(_OUT / 'grading.json').relative_to(_ROOT)}\n")


if __name__ == "__main__":
    main()
