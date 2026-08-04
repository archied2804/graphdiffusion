"""
make_cp_morph_animation
=======================

Render the scripted Cp-morph generation clip: the Cp spline morphs through
several in-distribution target Cp curves, a 3-sigma OOD target (``pc1_p3sig``
from the ``ood_systematic`` sweep), and finally a random far-OOD Cp that no
longer yields an ellipse. The in-distribution targets are chosen from the
EXP-043 ``eval_grid`` sweep by sampling every case, scoring how smoothly each
generates, picking the smoothest set (plus one flat body, which generates
wigglier), and ordering them so consecutive Cp curves vary as much as
possible. One shape is sampled per target from the EXP-021 conditional
diffusion model; the morph itself is cheap (no per-frame sampling).

Example:
    python scripts/make_cp_morph_animation.py \
        --experiment-dir outputs/EXP-021_long_train \
        --config configs/EXP-021_long_train.yaml --device cpu
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
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
from graph_diffusion.visualisation.cp_editor import cp_spline_and_dct
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


def _eval_grid_candidates(x_grid: np.ndarray) -> list[tuple[str, float, np.ndarray]]:
    """All eval_grid in-distribution cases as ``(label, thickness, cp_dense)``.

    Reads ``eval_grid/manifest.csv`` and loads each case's reference Cp curve
    resampled onto ``x_grid``. ``thickness`` is the ellipse aspect ratio
    ``b / a`` (vertical semi-axis over horizontal), used downstream to span a
    range of body heights.
    """
    with (EVAL_GRID_DIR / "manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    candidates: list[tuple[str, float, np.ndarray]] = []
    for row in rows:
        aspect = float(row["b"]) / float(row["a"])
        label = f"Re={float(row['Re']):.0f}  b/a={aspect:.2f}"
        cp_path = EVAL_GRID_DIR / f"sim_{row['idx']}_ref_cp.csv"
        candidates.append((label, aspect, _read_cp_csv(cp_path, "cp_ref", x_grid)))
    return candidates


def _sample_shape(
    model: object,
    template: object,
    theta: np.ndarray,
    cp_dense: np.ndarray,
    k_modes: int,
    guidance_scale: float,
    device: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one boundary shape for a dense target Cp curve.

    Returns:
        Tuple ``(radii, xy)``: the raw per-node radii ``(N,)`` and the
        theta-ordered Cartesian outline ``(N, 2)``.
    """
    cond_vec = torch.from_numpy(dct_ii(cp_dense, k_modes)).float()
    radii, _ = sample_shapes_from_cond(
        model=model,  # type: ignore[arg-type]
        template=template,  # type: ignore[arg-type]
        cond_vec=cond_vec,
        n_samples=1,
        guidance_scale=guidance_scale,
        device=device,
        seed=seed,
    )
    return radii[0], radial_to_xy(radii[0], theta)


def _radial_roughness(radii: np.ndarray, theta: np.ndarray) -> float:
    """High-frequency wiggle of a generated outline (lower = smoother).

    Orders the per-node radii by angle and returns the RMS of the discrete
    second difference (a curvature proxy), normalised by the mean radius so
    the metric is scale-free across thin and thick bodies.
    """
    order = np.argsort(theta)
    r = radii[order].astype(np.float64)
    second_diff = r - 0.5 * (np.roll(r, 1) + np.roll(r, -1))
    return float(np.sqrt(np.mean(second_diff**2)) / np.mean(r))


def _random_ood_cp(x_grid: np.ndarray, k_modes: int, seed: int) -> np.ndarray:
    """A far-out-of-distribution Cp curve from random handle positions.

    Mimics dragging the editor handles to random heights over a wide range,
    well outside any training Cp. Conditioning on it pushes the model off the
    ellipse manifold, so the generated shape no longer resembles a clean body.

    Returns:
        Dense Cp curve on ``x_grid``, shape ``(N_CP_GRID,)`` ``float32``.
    """
    rng = np.random.default_rng(seed)
    handle_x = np.linspace(0.0, 1.0, 7, dtype=np.float32)
    handle_y = rng.uniform(-1.6, 1.6, size=handle_x.shape[0]).astype(np.float32)
    spline, _ = cp_spline_and_dct(handle_x, handle_y, x_grid, k_modes)
    return spline.astype(np.float32)


def _max_variation_order(curves: list[np.ndarray]) -> list[int]:
    """Visiting order maximising consecutive Cp-curve variation.

    Brute-forces the open path that maximises the sum of consecutive L2
    distances between curves, so the morph makes the biggest possible jumps
    each step. Cheap for the handful of in-distribution targets; falls back to
    the input order when there are too many curves to enumerate.
    """
    n = len(curves)
    if n <= 2 or n > 8:
        return list(range(n))
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(curves[i] - curves[j]))
            dist[i, j] = dist[j, i] = d
    best_order = list(range(n))
    best_score = -1.0
    for perm in itertools.permutations(range(n)):
        score = sum(dist[perm[k], perm[k + 1]] for k in range(n - 1))
        if score > best_score:
            best_score = score
            best_order = list(perm)
    return best_order


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir", type=Path, default=Path("outputs/EXP-021_long_train")
    )
    parser.add_argument("--config", type=str, default="configs/EXP-021_long_train.yaml")
    parser.add_argument("--checkpoint-name", type=str, default="checkpoint_best.pt")
    parser.add_argument("--ood-case", type=str, default="pc1_p3sig")
    parser.add_argument("--n-indist", type=int, default=4)
    parser.add_argument(
        "--random-seed",
        type=int,
        default=3,
        help="Seed for the final random far-OOD Cp target (changes the "
        "non-ellipse shape the clip ends on).",
    )
    parser.add_argument(
        "--flat-aspect",
        type=float,
        default=0.60,
        help="Aspect ratio b/a below which a body counts as 'flat' for the "
        "one flat case mixed into the smoothest set.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument(
        "--n-hold",
        type=int,
        default=50,
        help="Frames each target is held (longer = longer pause per stage).",
    )
    parser.add_argument(
        "--n-tween",
        type=int,
        default=25,
        help="Frames per morph between consecutive targets.",
    )
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

    # Sample every in-distribution candidate once and score how smoothly it
    # generates, so we can switch between clean (un-wiggly) outlines.
    candidates = _eval_grid_candidates(x_grid)
    scored: list[tuple[str, float, np.ndarray, np.ndarray, float]] = []
    for label, aspect, cp_dense in candidates:
        radii, xy = _sample_shape(
            model,
            template,
            theta,
            cp_dense,
            k_modes,
            guidance_scale,
            args.device,
            args.seed,
        )
        scored.append((label, aspect, cp_dense, xy, _radial_roughness(radii, theta)))

    # Report the full roughness ranking (smoothest first) so the selection is
    # transparent; thin bodies tend to generate wigglier with this model.
    print("[cp_morph] eval_grid roughness ranking (smoothest first):")
    for label, _aspect, _cp, _xy, rough in sorted(scored, key=lambda s: s[4]):
        print(f"    {rough:.4f}  {label}")

    # Pick the (n_indist - 1) smoothest cases plus the smoothest genuinely-flat
    # body, so the set is clean but still includes a flat shape (flat bodies
    # generate wigglier, so we take the cleanest available one).
    by_rough = sorted(range(len(scored)), key=lambda i: scored[i][4])
    core_idx = by_rough[: max(args.n_indist - 1, 0)]
    flat_idx = [
        i for i in by_rough if scored[i][1] < args.flat_aspect and i not in core_idx
    ]
    if flat_idx and len(core_idx) < args.n_indist:
        chosen_idx = core_idx + [flat_idx[0]]
    else:
        chosen_idx = by_rough[: args.n_indist]
    chosen = [scored[i] for i in chosen_idx]

    # Switch between them in the order that maximises Cp-curve variation.
    chosen = [chosen[i] for i in _max_variation_order([c[2] for c in chosen])]

    # Ordered targets: smooth in-distribution shapes, then the OOD climax.
    labels = [c[0] for c in chosen]
    targets_list = [c[2] for c in chosen]
    shapes = [c[3] for c in chosen]
    ood_flags = [False] * len(chosen)

    ood_cp = _read_cp_csv(
        OOD_DIR / f"{args.ood_case}_target_cp.csv", "cp_target", x_grid
    )
    _, ood_xy = _sample_shape(
        model, template, theta, ood_cp, k_modes, guidance_scale, args.device, args.seed
    )
    labels.append(args.ood_case)
    targets_list.append(ood_cp)
    shapes.append(ood_xy)
    ood_flags.append(True)

    # Final far-OOD target: a random Cp that no longer yields an ellipse.
    rand_cp = _random_ood_cp(x_grid, k_modes, args.random_seed)
    _, rand_xy = _sample_shape(
        model, template, theta, rand_cp, k_modes, guidance_scale, args.device, args.seed
    )
    labels.append("random Cp")
    targets_list.append(rand_cp)
    shapes.append(rand_xy)
    ood_flags.append(True)

    targets = np.stack(targets_list, axis=0)

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
        n_tween=args.n_tween,
        n_hold=args.n_hold,
        fps=args.fps,
    )
    print(f"[cp_morph] wrote {out_dir / 'cp_morph.gif'} (+ mp4 if ffmpeg available)")


if __name__ == "__main__":
    main()
