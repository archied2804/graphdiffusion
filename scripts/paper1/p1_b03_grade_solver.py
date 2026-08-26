"""
P1-B03 -- grade the solver half.

Answers the question the analytic half could not: at how many basis modes does
reconstruction error become **invisible to the grading instrument**?

Four held-out shapes, six bodies each -- the original, a no-truncation roundtrip
control, and PCA reconstructions at M = 1, 2, 3, 10 -- all pushed through the
production solver at the PROTOCOL section 6 standard settings.

Two framings, both reported:

  vs dataset    each variant graded against its own dataset reference curve,
                exactly as every other Paper 1 result is graded. ``d_rho`` is
                then the change in that grade caused by reconstruction, and it
                is directly comparable to the frozen F04 sigma_rho, which was
                measured on the same quantity.

  vs original   the variant's solver Cp against the *original's* solver Cp.
                A purer measure of "did the geometry change alter the flow",
                free of any dataset-side offset -- but with no pre-registered
                threshold attached, so it is a diagnostic, not the verdict.

The ``roundtrip`` control carries the load. Reconstruction happens on a uniform
128-point theta grid and is interpolated back to the body's own node angles, so
``orig`` vs ``pcaNN`` confounds basis truncation with resampling. ``roundtrip``
is that resampling with no truncation at all: if ``roundtrip`` already moves the
grade as much as ``pca10`` does, the registered M = 3 and M = 10 levels are
measuring the resampler, not the basis.

Usage:
    python scripts/paper1/p1_b03_grade_solver.py
"""

import csv
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
_SHAPES = _ROOT / "cfd" / "ellipse2d" / "shapes" / "b03_recon"
_EVAL = _ROOT / "cfd" / "ellipse2d" / "shapes" / "eval_grid"
_WORK = _ROOT / "outputs" / "paper1" / "P1-B03_basis-reachability" / "solver"
_OUT = _ROOT / "outputs" / "paper1" / "P1-B03_basis-reachability"

VARIANTS = ("orig", "roundtrip", "pca01", "pca02", "pca03", "pca10")
REGISTERED = ("pca03", "pca10")

# Frozen in PROTOCOL section 6 by P1-F04, 2026-08-17.
SIGMA_RHO = 0.00108
SIGMA_A = 0.03521
SIGMA_CD = 0.04742


_LABELS: dict[str, int] = {}


def reference_curve(idx: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Locate the dataset reference Cp curve for a sample index.

    Args:
        idx (int): Dataset sample index.

    Returns:
        tuple[np.ndarray, np.ndarray] | None: ``(x_over_c, cp)``, or ``None``
            if no exported reference exists for this sample.
    """
    label = next((k for k, v in _LABELS.items() if v == idx), None)
    for p in (
        _SHAPES / f"{label}_ref_cp.csv" if label else None,
        _EVAL / f"sim_{idx}_ref_cp.csv",
    ):
        if p is not None and p.exists():
            return read_reference_cp(p)
    return None


def grade(d: Path) -> dict | None:
    """Grade one solver run, returning its Cp curve alongside the metrics.

    Args:
        d (Path): Run directory.

    Returns:
        dict | None: Metrics and the filtered Cp curve, or ``None`` if the run
            did not complete.
    """
    meas = d / "tmp" / "ellipse2d_cp.csv"
    if not meas.exists():
        return None
    x_raw, cp_raw, rms_raw = read_cp_csv(meas, drop_dead=False)
    n_dead = int(dead_probe_mask(cp_raw, rms_raw).sum())
    x, cp, _ = read_cp_csv(meas)
    out: dict = {"x": x, "cp": cp, "n_dead": n_dead, "n_nodes": int(x_raw.size)}
    force = d / "tmp" / "ellipse2d_force.csv"
    if force.exists():
        try:
            f = read_force_csv(force)
            out |= {"Cd": f.cd, "Cl": f.cl}
        except (ValueError, KeyError, IndexError):
            pass
    return out


def main() -> None:
    man = list(csv.DictReader((_SHAPES / "manifest.csv").open()))
    labels: dict[str, int] = {r["label"]: int(r["idx"]) for r in man}
    _LABELS.update(labels)
    dev = {(r["label"], r["variant"]): float(r["max_dev_norm"]) for r in man}

    print("\nP1-B03 solver half -- reconstruction tolerance\n")
    runs: dict[tuple[str, str], dict] = {}
    for label in labels:
        for v in VARIANTS:
            g = grade(_WORK / f"{label}_{v}")
            if g is not None:
                runs[(label, v)] = g

    missing = [f"{a}_{b}" for a in labels for b in VARIANTS if (a, b) not in runs]
    if missing:
        print(
            f"  incomplete ({len(missing)}): {', '.join(missing[:6])}"
            f"{' ...' if len(missing) > 6 else ''}\n"
        )
    if not runs:
        sys.exit("no graded runs -- run scripts/paper1/run_p1_b03_solver.sh first")

    rows: list[dict] = []
    print(
        f"  {'shape':<14}{'variant':<11}{'max|dr|':>10}{'rho':>10}"
        f"{'d_rho':>10}{'a':>9}{'Cd':>9}{'d_Cd':>9}{'dead':>7}"
    )

    for label, idx in labels.items():
        ref = reference_curve(idx)
        base = runs.get((label, "orig"))
        if base is None:
            print(f"  {label:<14}  original missing -- cannot difference")
            continue
        if ref is None:
            print(f"  {label:<14}  no dataset reference for idx {idx}")
            continue
        x_r, cp_r = ref
        base_rho = cp_agreement(x_r, cp_r, base["x"], base["cp"], mode="collapsed").rho

        for v in VARIANTS:
            g = runs.get((label, v))
            if g is None:
                continue
            agr = cp_agreement(x_r, cp_r, g["x"], g["cp"], mode="collapsed")
            # Direct variant-against-original comparison, on the original's grid.
            direct = cp_agreement(
                base["x"], base["cp"], g["x"], g["cp"], mode="collapsed"
            )
            d_cd = g.get("Cd", np.nan) - base.get("Cd", np.nan)
            row = {
                "label": label,
                "idx": idx,
                "variant": v,
                "max_dev_norm": dev.get((label, v), 0.0),
                "rho": agr.rho,
                "d_rho": agr.rho - base_rho,
                "a": agr.a,
                "Cd": g.get("Cd", float("nan")),
                "d_Cd": float(d_cd),
                "rho_vs_orig": direct.rho,
                "rms_vs_orig": direct.affine_rms,
                "n_dead": g["n_dead"],
                "n_nodes": g["n_nodes"],
            }
            rows.append(row)
            deadf = f"{g['n_dead']}/{g['n_nodes']}"
            print(
                f"  {label:<14}{v:<11}{row['max_dev_norm']:>10.2e}"
                f"{row['rho']:>10.5f}{row['d_rho']:>+10.5f}{row['a']:>9.4f}"
                f"{row['Cd']:>9.4f}{row['d_Cd']:>+9.4f}{deadf:>7}"
            )
        print()

    # ---- verdict: is each level detectable against the frozen floor? -------
    print(
        f"  Detectability against the frozen F04 floor "
        f"(sigma_rho = {SIGMA_RHO}, 2sigma = {2 * SIGMA_RHO:.5f}):\n"
    )
    print(
        f"  {'variant':<12}{'max|d_rho|':>12}{'rho sig':>10}"
        f"{'max|d_Cd|':>12}{'Cd sig':>9}{'worst shape':>14}  verdict"
    )

    summary: dict[str, dict] = {}
    for v in VARIANTS:
        if v == "orig":
            continue
        sub = [r for r in rows if r["variant"] == v]
        dr = np.array([abs(r["d_rho"]) for r in sub])
        dc_all = np.array([abs(r["d_Cd"]) for r in sub])
        ok = np.isfinite(dc_all)
        dc = dc_all[ok]
        if dr.size == 0:
            continue
        # The decision rule is a per-shape test, so the worst shape decides.
        n_sig = float(dr.max() / SIGMA_RHO)
        n_sig_cd = float(dc.max() / SIGMA_CD) if dc.size else float("nan")
        worst = (
            sub[int(np.argmax(np.where(ok, dc_all, -np.inf)))]["label"]
            if dc.size
            else "-"
        )
        detect = (
            "DETECTABLE"
            if (dr.max() > 2 * SIGMA_RHO or (dc.size and dc.max() > 2 * SIGMA_CD))
            else "invisible"
        )
        summary[v] = {
            "mean_abs_d_rho": float(dr.mean()),
            "max_abs_d_rho": float(dr.max()),
            "sigma_multiples_rho": n_sig,
            "mean_abs_d_Cd": float(dc.mean()) if dc.size else float("nan"),
            "max_abs_d_Cd": float(dc.max()) if dc.size else float("nan"),
            "sigma_multiples_Cd": n_sig_cd,
            "worst_shape_Cd": worst,
            "detectable": detect == "DETECTABLE",
            "registered": v in REGISTERED,
        }
        print(
            f"  {v:<12}{dr.max():>12.5f}{n_sig:>9.2f}s"
            f"{(dc.max() if dc.size else np.nan):>12.5f}{n_sig_cd:>8.2f}s"
            f"{worst:>14}  {detect}"
        )

    print(
        "\n  Reported on the WORST shape, not the mean -- the decision rule is a\n"
        "  per-shape test, and averaging four shapes hides the one that moved.\n"
        "  'invisible' means no shape moved past 2 sigma on either metric.\n"
    )

    print("  Metric sensitivity (P1-F04 section 4 -- rho has almost no range):\n")
    span_rho = max(r["rho"] for r in rows) - min(r["rho"] for r in rows)
    cds = [r["Cd"] for r in rows if np.isfinite(r["Cd"])]
    span_cd = max(cds) - min(cds)
    print(
        f"    rho spans {span_rho:.5f} over all 24 runs = "
        f"{span_rho / SIGMA_RHO:.1f} sigma"
    )
    print(
        f"    Cd  spans {span_cd:.4f} ({span_cd / float(np.mean(cds)):.1%} of mean) = "
        f"{span_cd / SIGMA_CD:.1f} sigma"
    )
    print(
        "    Cd is the discriminating metric here; read the rho column as"
        " corroboration.\n"
    )

    # The control decides whether the registered levels mean anything.
    rt = summary.get("roundtrip")
    if rt is not None:
        print("  Control check -- resampling vs basis truncation:\n")
        for v in ("pca10", "pca03", "pca02", "pca01"):
            s = summary.get(v)
            if s is None:
                continue
            floor = rt["mean_abs_d_rho"]
            ratio = s["mean_abs_d_rho"] / floor if floor else float("inf")
            note = (
                "buried in the resampling floor"
                if ratio < 2.0
                else "above the resampling floor"
            )
            print(f"    {v:<8} {ratio:>6.2f}x roundtrip   {note}")
        print()

    (_OUT / "solver_tolerance.json").write_text(
        json.dumps(
            {
                "sigma": {"rho": SIGMA_RHO, "a": SIGMA_A, "Cd": SIGMA_CD},
                "runs": [
                    {k: v for k, v in r.items() if k not in ("x", "cp")} for r in rows
                ],
                "summary": summary,
            },
            indent=2,
            default=float,
        )
    )
    print(f"  -> {(_OUT / 'solver_tolerance.json').relative_to(_ROOT)}\n")


if __name__ == "__main__":
    main()
