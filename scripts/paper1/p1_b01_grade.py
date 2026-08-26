"""
P1-B01 -- grade the retrieval baseline.

For every evaluation target, the nearest **training** shape in K=8 DCT-Cp space
was run through the solver at the **target's** Re and H, and is graded here
against the **target's** dataset reference curve.

What this can and cannot conclude on its own:

  can    how good retrieval is in absolute terms, and how it decays with
         distance to the nearest training neighbour -- the difficulty axis the
         spec calls "the single best summary of how hard each target actually
         was", and the x-axis of the key figure in section 4.3.

  cannot the model-versus-retrieval verdict. That needs P1-G01, which has not
         run. The spec's decision rule is a comparison, so no verdict is
         assigned here; this half fixes the baseline and the axis.

Metrics follow [[PROTOCOL]] section 2.1 as amended by F04: rho is reported but
is corroboration only -- unrelated shapes on this corpus correlate at 0.9954 --
and the discriminating statistics are rank/top-k and Cd.

Usage:
    python scripts/paper1/p1_b01_grade.py
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
_SHAPES = _ROOT / "cfd" / "ellipse2d" / "shapes" / "b01_retrieval"
_WORK = _ROOT / "outputs" / "paper1" / "P1-B01_retrieval-baseline" / "solver"
_OUT = _ROOT / "outputs" / "paper1" / "P1-B01_retrieval-baseline"

# Frozen by P1-F04, PROTOCOL section 6.
SIGMA_RHO = 0.00108
SIGMA_A = 0.03521
# P1-F04 section 4: rho between unrelated dataset shapes.
RHO_CHANCE = 0.9954

SETS = ("indist", "lowre", "highre", "thin", "thick", "aoa10")


def grade_one(tag: str) -> dict | None:
    """Grade one retrieved body against its target's reference curve.

    Args:
        tag (str): Run tag, e.g. ``"indist_t004"``.

    Returns:
        dict | None: Metrics, or ``None`` if the run did not complete.
    """
    meas = _WORK / tag / "tmp" / "ellipse2d_cp.csv"
    ref = _SHAPES / f"{tag}_ref_cp.csv"
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
        "n_dead": n_dead,
        "n_nodes": int(x_raw.size),
        "_x": x_m,
        "_cp": cp_m,
    }
    force = _WORK / tag / "tmp" / "ellipse2d_force.csv"
    if force.exists():
        try:
            f = read_force_csv(force)
            out |= {"Cd": f.cd, "Cl": f.cl}
        except (ValueError, KeyError, IndexError):
            pass
    return out


def cross_grade(rows: list[dict]) -> dict:
    """Rank each retrieved run against every in-distribution target.

    The discriminative test F04 adopted in place of a bare rho: does a retrieved
    shape's Cp match *its own* target better than it matches the others? Chance
    top-1 is ``1/n``.

    Args:
        rows (list[dict]): Graded in-distribution rows, each carrying its
            measured curve and its target's reference path.

    Returns:
        dict: Top-1 and top-3 rates with their chance levels.
    """
    refs = []
    for r in rows:
        x_r, cp_r = read_reference_cp(_SHAPES / f"{r['tag']}_ref_cp.csv")
        refs.append((x_r, cp_r))
    n = len(rows)
    top1 = top3 = 0
    for i, r in enumerate(rows):
        scores = []
        for j, (x_r, cp_r) in enumerate(refs):
            agr = cp_agreement(x_r, cp_r, r["_x"], r["_cp"], mode="collapsed")
            scores.append((agr.rho, j))
        order = [j for _, j in sorted(scores, reverse=True)]
        if order[0] == i:
            top1 += 1
        if i in order[:3]:
            top3 += 1
    return {
        "n": n,
        "top1": top1 / n,
        "top1_chance": 1.0 / n,
        "top3": top3 / n,
        "top3_chance": 3.0 / n,
    }


def main() -> None:
    man = list(csv.DictReader((_SHAPES / "manifest.csv").open()))
    print("\nP1-B01 -- retrieval baseline\n")

    rows: list[dict] = []
    for m in man:
        g = grade_one(m["tag"])
        if g is None:
            print(f"  {m['tag']:<18} MISSING")
            continue
        rows.append(
            {
                **{
                    k: m[k]
                    for k in (
                        "tag",
                        "set",
                        "target_idx",
                        "retrieved_train_idx",
                        "nn_distance",
                        "Re",
                        "H",
                    )
                },
                **g,
            }
        )

    if not rows:
        sys.exit("no graded runs -- run scripts/paper1/run_p1_b01_solver.sh first")

    print(
        f"  {'target':<16}{'nn dist':>10}{'rho':>10}{'vs chance':>11}"
        f"{'a':>9}{'Cd':>9}{'dead':>7}"
    )
    for r in rows:
        nn = float(r["nn_distance"])
        dead = f"{r['n_dead']}/{r['n_nodes']}"
        print(
            f"  {r['tag']:<16}{nn:>10.5f}{r['rho']:>10.5f}"
            f"{r['rho'] - RHO_CHANCE:>+11.5f}{r['a']:>9.4f}"
            f"{r.get('Cd', np.nan):>9.4f}{dead:>7}"
        )

    # ---- per set --------------------------------------------------------
    print("\n  By evaluation set:\n")
    print(
        f"  {'set':<10}{'n':>4}{'med nn':>10}{'med rho':>10}{'vs chance':>11}"
        f"{'med a':>9}{'med Cd':>9}"
    )
    per_set: dict[str, dict] = {}
    for s in SETS:
        sub = [r for r in rows if r["set"] == s]
        if not sub:
            continue
        nn = float(np.median([float(r["nn_distance"]) for r in sub]))
        rho = float(np.median([r["rho"] for r in sub]))
        a = float(np.median([r["a"] for r in sub]))
        cd = float(np.nanmedian([r.get("Cd", np.nan) for r in sub]))
        per_set[s] = {
            "n": len(sub),
            "median_nn": nn,
            "median_rho": rho,
            "rho_over_chance": rho - RHO_CHANCE,
            "median_a": a,
            "median_Cd": cd,
        }
        print(
            f"  {s:<10}{len(sub):>4}{nn:>10.5f}{rho:>10.5f}"
            f"{rho - RHO_CHANCE:>+11.5f}{a:>9.4f}{cd:>9.4f}"
        )

    # ---- difficulty axis -------------------------------------------------
    nn_all = np.array([float(r["nn_distance"]) for r in rows])
    rho_all = np.array([r["rho"] for r in rows])
    ok = np.isfinite(nn_all) & np.isfinite(rho_all)
    corr = (
        float(np.corrcoef(nn_all[ok], rho_all[ok])[0, 1])
        if ok.sum() > 2
        else float("nan")
    )
    print(f"\n  Difficulty axis: corr(NN distance, rho) = {corr:+.4f}")
    print("  Negative and large means retrieval degrades as the nearest training")
    print("  neighbour gets further away -- which is what makes it a difficulty axis.")

    # ---- discrimination --------------------------------------------------
    ind = [r for r in rows if r["set"] == "indist"]
    xg = cross_grade(ind)
    print(
        f"\n  Discrimination on {xg['n']} in-distribution targets "
        "(cross-graded, rho):\n"
    )
    print(f"    top-1  {xg['top1']:>6.1%}   chance {xg['top1_chance']:.1%}")
    print(f"    top-3  {xg['top3']:>6.1%}   chance {xg['top3_chance']:.1%}")

    print("\n  No verdict is assigned. The spec's decision rule compares against")
    print("  the model, and P1-G01 has not run. This half fixes the baseline and")
    print("  the difficulty axis; the comparison figure follows G01.\n")

    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "retrieval_grading.json").write_text(
        json.dumps(
            {
                "sigma": {"rho": SIGMA_RHO, "a": SIGMA_A},
                "rho_chance_level": RHO_CHANCE,
                "runs": [
                    {k: v for k, v in r.items() if not k.startswith("_")} for r in rows
                ],
                "per_set": per_set,
                "corr_nn_rho": corr,
                "discrimination_indist": xg,
                "verdict": "baseline measured; comparison awaits P1-G01",
            },
            indent=2,
            default=float,
        )
    )
    print(f"  -> {(_OUT / 'retrieval_grading.json').relative_to(_ROOT)}\n")


if __name__ == "__main__":
    main()
