"""
Paper 1 status board.

Derives status from the filesystem rather than a hand-maintained table, so it
cannot go stale.  For each of the 29 registered experiments it reports whether a
spec, a config, outputs and a results write-up exist, and summarises any training
runs found.

Usage:
    python scripts/paper1/p1_status.py            # status board
    python scripts/paper1/p1_status.py --runs     # per-seed training detail
"""

import argparse
import json
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SPECS = _ROOT / "DGN_Docs" / "paper1" / "specs"
_RESULTS = _ROOT / "DGN_Docs" / "paper1" / "results"
_CONFIGS = _ROOT / "configs" / "paper1"
_OUTPUTS = _ROOT / "outputs" / "paper1"

_GROUP_NAMES = {
    "F": "Foundation & protocol",
    "A": "Architecture necessity",
    "B": "Basis necessity",
    "D": "Discretisation independence",
    "C": "Conditioning",
    "G": "Generalisation & failure",
    "S": "Scale & data efficiency",
}


def _read_front_matter(path: Path) -> dict[str, str]:
    """Parse the simple ``key: value`` YAML front matter of a spec file."""
    out: dict[str, str] = {}
    text = path.read_text()
    if not text.startswith("---"):
        return out
    body = text.split("---", 2)[1]
    for line in body.splitlines():
        m = re.match(r"^(\w+):\s*(.*)$", line)
        if m:
            out[m.group(1)] = m.group(2).strip().strip('"')
    return out


def _summarise_run(run_dir: Path) -> dict[str, object]:
    """Summarise one training run directory."""
    info: dict[str, object] = {"dir": run_dir.name}
    meta_p = run_dir / "run_meta.json"
    log_p = run_dir / "loss_log.json"
    train_log = run_dir / "train.log"

    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        info["state"] = "done"
        info["seed"] = meta.get("train_seed")
        info["params"] = meta.get("param_count")
        info["wall_s"] = meta.get("wall_clock_s")
        info["dirty"] = meta.get("git_dirty")
    elif train_log.exists():
        info["state"] = "running"
        tail = train_log.read_text(errors="ignore")[-4000:].replace("\r", "\n")
        epochs = re.findall(r"Epoch\s+(\d+)/(\d+)", tail)
        if epochs:
            info["progress"] = f"{epochs[-1][0]}/{epochs[-1][1]}"
    else:
        info["state"] = "empty"

    if log_p.exists():
        try:
            log = json.loads(log_p.read_text())
            vals = [e["val_loss"] for e in log if e.get("val_loss") is not None]
            if vals:
                info["best_val"] = round(min(vals), 6)
                info["epochs_logged"] = len(log)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return info


def main() -> None:
    p = argparse.ArgumentParser(description="Paper 1 status board")
    p.add_argument("--runs", action="store_true", help="Show per-seed run detail")
    args = p.parse_args()

    specs = sorted(_SPECS.glob("P1-*.md"))
    if not specs:
        raise SystemExit(f"no specs found under {_SPECS}")

    print(f"\nPaper 1 — {len(specs)} registered experiments\n")
    header = (
        f"  {'ID':<8} {'tier':<11} {'spec':<5} {'cfg':<4} "
        f"{'out':<4} {'result':<10} title"
    )
    current_group = ""
    n_reported = 0

    for spec in specs:
        fm = _read_front_matter(spec)
        exp_id = fm.get("id", spec.stem)
        group = exp_id.split("-")[1][0] if "-" in exp_id else "?"
        if group != current_group:
            current_group = group
            print(f"\n{group} — {_GROUP_NAMES.get(group, '')}")
            print(header)

        slug = spec.stem
        # Match on the ID prefix: config and output slugs need not equal the
        # spec slug (e.g. P1-F05_reference.yaml vs P1-F05_reference-model.md).
        has_cfg = (
            bool(list(_CONFIGS.glob(f"{exp_id}_*.yaml")))
            if _CONFIGS.exists()
            else False
        )
        out_dirs = list(_OUTPUTS.glob(f"{exp_id}_*")) if _OUTPUTS.exists() else []
        has_out = any(d.is_dir() and any(d.iterdir()) for d in out_dirs)
        res = _RESULTS / f"{slug}.md"
        has_res = res.exists()
        if has_res:
            n_reported += 1
            verdict = _read_front_matter(res).get("verdict", "?")
        else:
            verdict = "-"

        print(
            f"  {exp_id:<8} {fm.get('tier', '?'):<11} "
            f"{'yes':<5} {('yes' if has_cfg else '-'):<4} "
            f"{('yes' if has_out else '-'):<4} {verdict:<7} "
            f"{fm.get('title', '')[:44]}"
        )

    print(f"\n  reported: {n_reported}/{len(specs)}")

    # --- training runs -------------------------------------------------------
    if not _OUTPUTS.exists():
        print("\n  no outputs yet\n")
        return

    print("\nTraining runs\n")
    for exp_dir in sorted(d for d in _OUTPUTS.iterdir() if d.is_dir()):
        seeds = sorted(d for d in exp_dir.iterdir() if d.is_dir())
        if not seeds:
            continue
        runs = [_summarise_run(s) for s in seeds]
        done = sum(1 for r in runs if r["state"] == "done")
        running = sum(1 for r in runs if r["state"] == "running")
        print(f"  {exp_dir.name}: {done} done, {running} running, {len(runs)} total")
        if args.runs or running:
            for r in runs:
                bits = [f"{r['dir']:<8}", f"{str(r['state']):<8}"]
                if "progress" in r:
                    bits.append(f"epoch {r['progress']}")
                if "best_val" in r:
                    bits.append(f"best_val={r['best_val']}")
                if r.get("wall_s"):
                    bits.append(f"{float(r['wall_s']) / 60:.0f} min")
                if r.get("dirty"):
                    bits.append("GIT-DIRTY (unusable per PROTOCOL §7)")
                print("      " + "  ".join(bits))
    print()


if __name__ == "__main__":
    main()
