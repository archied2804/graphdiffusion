"""
P1-F02 — do the four historical (rho, a) implementations agree?

Four independent copies of the correlation / affine-fit logic had accumulated
across ``scripts/``.  This reruns every one of them over the same stored OpenLB
runs, alongside the consolidated ``cp_metrics`` module, and tabulates the
deviations.

The two historical variants differ in which grid they resample onto:

  ``target_on_meas``  interpolate the target onto the measured nodes and fit
                      there -- used by make_exp04x_doc_figures, analyze_ood_sweep
                      and compare_dps_sweep
  ``meas_on_ref``     interpolate the measurement onto the reference nodes and
                      fit there -- used by compare_cp_baseline

Both weight the least-squares fit by one side's node distribution, which
clusters near the leading and trailing edges.  ``cp_metrics`` resamples both
onto a common uniform grid instead.

Usage:
    python scripts/paper1/p1_f02_metric_audit.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from graph_diffusion.postprocessing.cp_metrics import cp_agreement, read_cp_csv

_ROOT = Path(__file__).resolve().parents[2]
_SHAPES = _ROOT / "cfd" / "ellipse2d" / "shapes"
_RUNS = _ROOT / "cfd" / "ellipse2d" / "runs"
_OUT = _ROOT / "outputs" / "paper1" / "P1-F02_grading-metric"


def _read(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read (x_over_c, Cp) exactly as the historical scripts did."""
    x, cp, _ = read_cp_csv(path)
    return x, cp


def hist_target_on_meas(
    xt: np.ndarray, cpt: np.ndarray, xo: np.ndarray, cpo: np.ndarray
) -> tuple[float, float]:
    """Historical variant used by three of the four scripts."""
    o = np.argsort(xt)
    cpt_n = np.interp(xo, xt[o], cpt[o])
    corr = float(np.corrcoef(cpt_n, cpo)[0, 1])
    design = np.vstack([cpt_n, np.ones_like(cpt_n)]).T
    (a, _), *_ = np.linalg.lstsq(design, cpo, rcond=None)
    return corr, float(a)


def hist_meas_on_ref(
    xt: np.ndarray, cpt: np.ndarray, xo: np.ndarray, cpo: np.ndarray
) -> tuple[float, float]:
    """Historical variant used by compare_cp_baseline."""
    cpo_on_ref = np.interp(xt, xo, cpo)
    design = np.vstack([cpt, np.ones_like(cpt)]).T
    (a, _), *_ = np.linalg.lstsq(design, cpo_on_ref, rcond=None)
    corr = float(np.corrcoef(cpt, cpo_on_ref)[0, 1])
    return corr, float(a)


def _pairs() -> list[tuple[str, Path, Path]]:
    """Find (label, reference_csv, measured_csv) triples among stored runs."""
    out: list[tuple[str, Path, Path]] = []
    for ref in sorted(_SHAPES.rglob("*_ref_cp.csv")) + sorted(
        _SHAPES.rglob("*_target_cp.csv")
    ):
        name = ref.name.replace("_ref_cp.csv", "").replace("_target_cp.csv", "")
        meas = _RUNS / name / "tmp" / "ellipse2d_cp.csv"
        if meas.exists():
            out.append((name, ref, meas))
    return out


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    pairs = _pairs()
    if not pairs:
        raise SystemExit("no (reference, measured) pairs found")

    rows: list[dict[str, object]] = []
    print(f"\nComparing {len(pairs)} stored runs across 4 methods\n")
    hdr = (
        f"  {'run':<22} {'rho_A':>7} {'rho_B':>7} {'rho_new':>8} "
        f"{'a_A':>6} {'a_B':>6} {'a_new':>6} {'a_split':>8} {'asym':>6}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for name, ref, meas in pairs:
        xt, cpt = _read(ref)
        xo, cpo = _read(meas)
        rho_a, a_a = hist_target_on_meas(xt, cpt, xo, cpo)
        rho_b, a_b = hist_meas_on_ref(xt, cpt, xo, cpo)
        new_c = cp_agreement(xt, cpt, xo, cpo, mode="collapsed")
        new_s = cp_agreement(xt, cpt, xo, cpo, mode="split")
        rows.append(
            {
                "run": name,
                "rho_target_on_meas": rho_a,
                "rho_meas_on_ref": rho_b,
                "rho_new_collapsed": new_c.rho,
                "rho_new_split": new_s.rho,
                "a_target_on_meas": a_a,
                "a_meas_on_ref": a_b,
                "a_new_collapsed": new_c.a,
                "a_new_split": new_s.a,
                "asymmetry_meas": new_s.asymmetry_meas,
                "residual_runs_z": new_c.residual_runs_z,
                "n_ref": int(xt.size),
                "n_meas": int(xo.size),
            }
        )
        print(
            f"  {name:<22} {rho_a:7.4f} {rho_b:7.4f} {new_c.rho:8.4f} "
            f"{a_a:6.3f} {a_b:6.3f} {new_c.a:6.3f} {new_s.a:8.3f} "
            f"{new_s.asymmetry_meas:6.3f}"
        )

    d_rho = [abs(r["rho_target_on_meas"] - r["rho_meas_on_ref"]) for r in rows]
    d_a = [abs(r["a_target_on_meas"] - r["a_meas_on_ref"]) for r in rows]
    dn_a = [abs(r["a_target_on_meas"] - r["a_new_collapsed"]) for r in rows]
    z = [abs(r["residual_runs_z"]) for r in rows]

    print("\n  Disagreement between the two historical variants:")
    print(f"    max |d rho| = {max(d_rho):.2e}   (spec tolerance 1e-6)")
    print(f"    max |d a|   = {max(d_a):.2e}   (spec tolerance 1e-4)")
    print("\n  Historical (target_on_meas) vs consolidated common-grid:")
    print(f"    max |d a|   = {max(dn_a):.2e}")
    print("\n  Affine residual structure (runs-test |z|):")
    print(f"    median {np.median(z):.2f}   max {max(z):.2f}   (|z|>2 => structured)")

    verdict = "AGREE" if max(d_rho) < 1e-6 and max(d_a) < 1e-4 else "DISAGREE"
    print(f"\n  VERDICT: the two historical variants {verdict}\n")

    (_OUT / "audit.json").write_text(
        json.dumps(
            {
                "n_runs": len(rows),
                "max_abs_drho_between_historical": max(d_rho),
                "max_abs_da_between_historical": max(d_a),
                "max_abs_da_historical_vs_new": max(dn_a),
                "median_abs_residual_runs_z": float(np.median(z)),
                "verdict": verdict,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"  -> {(_OUT / 'audit.json').relative_to(_ROOT)}\n")


if __name__ == "__main__":
    main()
