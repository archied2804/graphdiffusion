"""
scripts/run_experiments.py — Local sequential sweep runner for EXP-002 to EXP-006
===================================================================================

Runs training + postprocessing for each experiment in dependency order.
EXP-005 and EXP-006 are gated: the runner pauses after EXP-002/003/004
and prompts you to review results and confirm configs before continuing.

Usage:
    # Run all ablations (EXP-002/003/004 only — stops before EXP-005)
    python scripts/run_experiments.py --phase ablations

    # Run EXP-005 (after reviewing 002/003 results)
    python scripts/run_experiments.py --phase full

    # Run EXP-006 (after reviewing 004/005 results)
    python scripts/run_experiments.py --phase rich

    # Run a single named experiment
    python scripts/run_experiments.py --only EXP-002a_circle_radial_k1

    # Dry run (print commands without executing)
    python scripts/run_experiments.py --phase ablations --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------
# Each entry: (slug, config_path, epochs, n_samples)
# Dependency phases:
#   ablations — EXP-002 and EXP-003 k/amplitude sweeps + EXP-004 logit
#   full      — EXP-005 full training (run after reviewing ablation results)
#   rich      — EXP-006 richer features (run after reviewing 004/005 results)

ABLATION_EXPERIMENTS: list[tuple[str, str, int, int]] = [
    ("EXP-002a_circle_radial_k1", "configs/EXP-002a_circle_radial_k1.yaml", 50, 16),
    ("EXP-002b_circle_radial_k2", "configs/EXP-002b_circle_radial_k2.yaml", 50, 16),
    ("EXP-002c_circle_radial_k4", "configs/EXP-002c_circle_radial_k4.yaml", 50, 16),
    ("EXP-002d_circle_radial_k6", "configs/EXP-002d_circle_radial_k6.yaml", 50, 16),
    (
        "EXP-003a_circle_radial_amp005",
        "configs/EXP-003a_circle_radial_amp005.yaml",
        50,
        16,
    ),
    (
        "EXP-003b_circle_radial_amp015",
        "configs/EXP-003b_circle_radial_amp015.yaml",
        50,
        16,
    ),
    (
        "EXP-003c_circle_radial_amp030",
        "configs/EXP-003c_circle_radial_amp030.yaml",
        50,
        16,
    ),
    ("EXP-004_circle_radial_logit", "configs/EXP-004_circle_radial_logit.yaml", 50, 16),
]

FULL_EXPERIMENTS: list[tuple[str, str, int, int]] = [
    ("EXP-005_circle_radial_full", "configs/EXP-005_circle_radial_full.yaml", 100, 50),
]

RICH_EXPERIMENTS: list[tuple[str, str, int, int]] = [
    (
        "EXP-006_circle_radial_rich",
        "configs/EXP-006_circle_radial_rich-features.yaml",
        100,
        50,
    ),
]

ALL_EXPERIMENTS = ABLATION_EXPERIMENTS + FULL_EXPERIMENTS + RICH_EXPERIMENTS


def run_experiment(
    slug: str,
    config: str,
    epochs: int,
    n_samples: int,
    dry_run: bool = False,
) -> bool:
    """Run training then postprocessing for one experiment.

    Returns True on success, False if either step fails.
    """
    output_dir = ROOT / "outputs" / slug
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  {slug}")
    print(f"  config: {config}  epochs: {epochs}")
    print(f"  output: outputs/{slug}/")
    print(f"{'=' * 60}")

    # --- Train ---
    train_cmd = [
        PYTHON,
        "train_circle.py",
        "--config",
        config,
        "--epochs",
        str(epochs),
        "--device",
        "cuda" if _cuda_available() else "cpu",
        "--n_samples",
        str(n_samples),
        "--output",
        str(output_dir / "generated_shapes.png"),
    ]
    print(f"[TRAIN] {' '.join(train_cmd)}")
    if not dry_run:
        t0 = time.time()
        result = subprocess.run(
            train_cmd,
            cwd=str(ROOT),
            tee_log=str(output_dir / "train.log"),
        )
        elapsed = time.time() - t0
        print(f"  → training finished in {elapsed:.0f}s  (exit {result.returncode})")
        if result.returncode != 0:
            print(f"  ✗ TRAINING FAILED for {slug}")
            return False

    # --- Postprocess ---
    postprocess_cmd = [
        PYTHON,
        "scripts/postprocess_circle.py",
        "--experiment-dir",
        str(output_dir),
        "--config",
        config,
        "--n-samples",
        str(n_samples),
        "--visualize-diffusion",
        "--save-samples",
    ]
    print(f"[POST]  {' '.join(postprocess_cmd)}")
    if not dry_run:
        t0 = time.time()
        result = subprocess.run(
            postprocess_cmd,
            cwd=str(ROOT),
            tee_log=str(output_dir / "postprocess.log"),
        )
        elapsed = time.time() - t0
        print(f"  → postprocess finished in {elapsed:.0f}s  (exit {result.returncode})")
        if result.returncode != 0:
            print(f"  ✗ POSTPROCESS FAILED for {slug}")
            return False

    return True


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _subprocess_run_with_log(
    cmd: list[str],
    cwd: str,
    log_path: str,
) -> subprocess.CompletedProcess:  # type: ignore[type-arg]
    """Run a command, tee-ing stdout+stderr to both console and log file."""
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_f.write(line)
        proc.wait()
    return subprocess.CompletedProcess(cmd, proc.returncode)


# Monkey-patch so run_experiment can use tee_log kwarg
_orig_run = subprocess.run


def subprocess_run_tee(
    cmd: list[str],
    tee_log: str | None = None,
    **kwargs: object,
) -> subprocess.CompletedProcess:  # type: ignore[type-arg]
    if tee_log is None:
        return _orig_run(cmd, **kwargs)  # type: ignore[call-overload]
    return _subprocess_run_with_log(cmd, str(kwargs.get("cwd", ".")), tee_log)


subprocess.run = subprocess_run_tee  # type: ignore[assignment]


def print_ablation_summary() -> None:
    """Print a summary table of ablation results after phase 1."""
    print("\n" + "=" * 60)
    print("  ABLATION RESULTS SUMMARY")
    print("=" * 60)
    for slug, *_ in ABLATION_EXPERIMENTS:
        report_path = ROOT / "outputs" / slug / "evaluation_report.json"
        if not report_path.exists():
            print(f"  {slug:<45} — no report")
            continue
        with open(report_path) as f:
            report = json.load(f)
        metrics = report.get("aggregate_metrics", {})
        print(
            f"  {slug:<45}  "
            f"smooth={metrics.get('smoothness', float('nan')):.4f}  "
            f"ks={metrics.get('ks_statistic', float('nan')):.4f}  "
            f"bviol={metrics.get('boundary_violation_rate', float('nan')):.4f}"
        )
    print()
    print("Next step: review outputs above and update EXP-005 config if needed:")
    print("  configs/EXP-005_circle_radial_full.yaml  (k_neighbors, amplitude_scale)")
    print("Then re-run:  python scripts/run_experiments.py --phase full")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EXP-002 through EXP-006 locally")
    parser.add_argument(
        "--phase",
        choices=["ablations", "full", "rich", "all"],
        default="ablations",
        help="Which phase to run (default: ablations)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Run only the named experiment slug",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    args = parser.parse_args()

    if args.only:
        experiments = [e for e in ALL_EXPERIMENTS if e[0] == args.only]
        if not experiments:
            print(f"Unknown experiment slug: {args.only}")
            print("Available:", [e[0] for e in ALL_EXPERIMENTS])
            sys.exit(1)
    elif args.phase == "ablations":
        experiments = ABLATION_EXPERIMENTS
    elif args.phase == "full":
        experiments = FULL_EXPERIMENTS
    elif args.phase == "rich":
        experiments = RICH_EXPERIMENTS
    else:
        experiments = ALL_EXPERIMENTS

    failed: list[str] = []
    for slug, config, epochs, n_samples in experiments:
        ok = run_experiment(slug, config, epochs, n_samples, dry_run=args.dry_run)
        if not ok:
            failed.append(slug)

    # Summary
    print(f"\n{'=' * 60}")
    total = len(experiments)
    succeeded = total - len(failed)
    print(f"  Completed {succeeded}/{total} experiments")
    if failed:
        print(f"  Failed: {failed}")
    print(f"{'=' * 60}")

    if args.phase == "ablations" and not failed and not args.dry_run:
        print_ablation_summary()

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
