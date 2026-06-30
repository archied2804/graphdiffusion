"""
make_cp_morph_animation
=======================

Render the scripted Cp-morph generation clip: the Cp spline morphs through
three in-distribution target Cp curves (auto-selected from the EXP-043
``eval_grid`` sweep to span the Reynolds range) and one 3-sigma OOD target
(``pc1_p3sig`` from the ``ood_systematic`` sweep). One shape is sampled per
target from the EXP-021 conditional diffusion model; the morph itself is
cheap (no per-frame sampling).

Example:
    python scripts/make_cp_morph_animation.py \
        --experiment-dir outputs/EXP-021_long_train \
        --config configs/EXP-021_long_train.yaml --device cpu
"""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import numpy as np
import torch

from graph_diffusion.data.pOnEllipseConditional import dct_ii
from graph_diffusion.postprocessing.inference import (
    radial_to_xy,
    sample_shapes_from_cond,
    template_thetas,
)
from graph_diffusion.postprocessing.loaders import load_exp020
from graph_diffusion.visualisation.cp_morph import write_cp_morph_animation

N_CP_GRID = 128
EVAL_GRID_DIR = Path("cfd/ellipse2d/shapes/eval_grid")
OOD_DIR = Path("cfd/ellipse2d/shapes/ood_systematic")


def _read_cp_csv(path: Path, value_col: str, x_grid: np.ndarray) -> np.ndarray:
    """Read a 2-column Cp CSV and resample its curve onto ``x_grid``.

    Args:
        path: CSV with header ``x_over_c,<value_col>``.
        value_col: Name of the Cp column (``cp_ref`` or ``cp_target``).
        x_grid: Dense grid to resample onto, shape ``(N_CP_GRID,)``.

    Returns:
        Dense Cp curve on ``x_grid``, shape ``(N_CP_GRID,)`` ``float32``.
    """
    xs: list[float] = []
    ys: list[float] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            xs.append(float(row["x_over_c"]))
            ys.append(float(row[value_col]))
    order = np.argsort(xs)
    x_sorted = np.asarray(xs)[order]
    y_sorted = np.asarray(ys)[order]
    return np.interp(x_grid, x_sorted, y_sorted).astype(np.float32)


def _select_indist_cases(n_indist: int) -> list[tuple[str, Path]]:
    """Pick ``n_indist`` Re-spanning eval_grid cases as ``(label, csv_path)``.

    Reads ``eval_grid/manifest.csv``, sorts by Reynolds number, and takes
    evenly-spaced cases (first .. last) so the clip traverses the dataset.
    """
    with (EVAL_GRID_DIR / "manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda r: float(r["Re"]))
    picks = np.linspace(0, len(rows) - 1, n_indist).round().astype(int)
    selected: list[tuple[str, Path]] = []
    for i in picks:
        row = rows[int(i)]
        label = f"Re={float(row['Re']):.0f}, H={float(row['H']):.1f}"
        csv_path = EVAL_GRID_DIR / f"sim_{row['idx']}_ref_cp.csv"
        selected.append((label, csv_path))
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir", type=Path, default=Path("outputs/EXP-021_long_train")
    )
    parser.add_argument("--config", type=str, default="configs/EXP-021_long_train.yaml")
    parser.add_argument("--checkpoint-name", type=str, default="checkpoint_best.pt")
    parser.add_argument("--ood-case", type=str, default="pc1_p3sig")
    parser.add_argument("--n-indist", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x_grid = np.linspace(0.0, 1.0, N_CP_GRID, dtype=np.float32)

    model, dataset, config = load_exp020(
        experiment_dir=str(args.experiment_dir),
        config_path=args.config,
        device=args.device,
        checkpoint_name=args.checkpoint_name,
    )
    k_modes = int(config["ellipse_dataset"].get("k_modes", 8))
    guidance_scale = float(config.get("sampling", {}).get("guidance_scale", 1.0))

    template = copy.copy(dataset[0]).to(args.device)
    theta = template_thetas(template)

    # Build the ordered target list: in-distribution cases, then the OOD climax.
    targets_meta: list[tuple[str, np.ndarray, bool]] = []
    for label, csv_path in _select_indist_cases(args.n_indist):
        targets_meta.append((label, _read_cp_csv(csv_path, "cp_ref", x_grid), False))
    ood_csv = OOD_DIR / f"{args.ood_case}_target_cp.csv"
    targets_meta.append(
        (f"{args.ood_case}", _read_cp_csv(ood_csv, "cp_target", x_grid), True)
    )

    targets = np.stack([cp for _, cp, _ in targets_meta], axis=0)
    labels = [lab for lab, _, _ in targets_meta]
    ood_flags = [flag for _, _, flag in targets_meta]

    # Sample one shape per target (single sample = fast, reproducible).
    shapes: list[np.ndarray] = []
    for _, cp_dense, _ in targets_meta:
        cond_vec = torch.from_numpy(dct_ii(cp_dense, k_modes)).float()
        radii, _ = sample_shapes_from_cond(
            model=model,
            template=template,
            cond_vec=cond_vec,
            n_samples=1,
            guidance_scale=guidance_scale,
            device=args.device,
            seed=args.seed,
        )
        shapes.append(radial_to_xy(radii[0], theta))

    out_dir = args.out_dir if args.out_dir is not None else args.experiment_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_cp_morph_animation(
        targets=targets,
        shapes=shapes,
        labels=labels,
        ood_flags=ood_flags,
        handle_x=np.linspace(0.0, 1.0, 7, dtype=np.float32),
        x_grid=x_grid,
        k_modes=k_modes,
        out_path_mp4=out_dir / "cp_morph.mp4",
        out_path_gif=out_dir / "cp_morph.gif",
        fps=args.fps,
    )
    print(f"[cp_morph] wrote {out_dir / 'cp_morph.gif'} (+ mp4 if ffmpeg available)")


if __name__ == "__main__":
    main()
