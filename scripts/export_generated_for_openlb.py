"""Sample EXP-022 generated shapes for target Cp curves and export to OpenLB.

For each target — a few farthest-first in-distribution training Cp curves plus
one synthetic asymmetric (OOD) curve — sample a shape from the conditional
diffusion model, then export its boundary polygon as an ``x y`` ``.dat`` for the
OpenLB ``ellipse2d`` case, in both **raw** and **Fourier-smoothed** variants.
Also save the target Cp(x/c) the shape was conditioned on, plus a manifest with
the Re/H to run each shape at.

This is the EXP-041 inverse-design test: the OpenLB-measured Cp of each
generated shape is later compared against its target Cp (reconciled through the
EXP-040 normalisation factor a≈1.78) to check whether the generator's shapes
actually produce the requested pressure — independently of the model's own
self-trained pressure head.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from graph_diffusion.data.pOnEllipse import (
    _H5_COL_H,
    _H5_COL_RE,
    _load_h5_raw,
)
from graph_diffusion.data.pOnEllipseConditional import dct_ii_inverse
from graph_diffusion.postprocessing.loaders import (
    build_dataset,
    build_model,
    load_config,
)

# Reuse the EXP-020/022 sampling machinery verbatim.
from postprocess_exp020 import (  # type: ignore[import-not-found]
    N_CP_GRID,
    _resolve_clamp_range,
    make_synthetic_target,
    pick_targets_farthest_first,
    sample_shapes_for_target,
    shape_template_from_dataset,
)

# Mid-range fallback for the synthetic target (no source simulation).
_SYNTH_RE = 754.0
_SYNTH_H = 5.5


def fourier_smooth(xy: np.ndarray, keep_modes: int) -> np.ndarray:
    """Low-pass a closed boundary by truncating the Fourier series of r(θ).

    The boundary is naturally periodic in θ, so retaining only the lowest
    ``keep_modes`` radial Fourier modes removes the high-frequency wobble the
    diffusion sampler leaves on the surface while preserving the ellipse shape.

    Args:
        xy (np.ndarray): Boundary points ``(N, 2)``, ordered by angle.
        keep_modes (int): Number of low Fourier modes of r(θ) to retain.

    Returns:
        np.ndarray: Smoothed boundary points ``(N, 2)``.
    """
    centroid = xy.mean(axis=0)
    rel = xy - centroid
    theta = np.arctan2(rel[:, 1], rel[:, 0])
    radius = np.hypot(rel[:, 0], rel[:, 1])
    order = np.argsort(theta)
    r_sorted = radius[order]
    th_sorted = theta[order]

    coeffs = np.fft.rfft(r_sorted)
    coeffs[keep_modes + 1 :] = 0.0
    r_smooth = np.fft.irfft(coeffs, n=len(r_sorted))

    return np.column_stack(
        [
            centroid[0] + r_smooth * np.cos(th_sorted),
            centroid[1] + r_smooth * np.sin(th_sorted),
        ]
    )


def write_dat(path: Path, xy: np.ndarray, header: str) -> None:
    """Write a boundary polygon as an ``x y`` point file for OpenLB.

    Args:
        path (Path): Output ``.dat`` path.
        xy (np.ndarray): Boundary points ``(N, 2)``, ordered.
        header (str): One-line description (written as a ``#`` comment).
    """
    with path.open("w") as fh:
        fh.write(f"# {header}\n# x y\n")
        for x, y in xy:
            fh.write(f"{x:.10f} {y:.10f}\n")


def target_cp(cond_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode a K-mode DCT conditioning vector into a target Cp(x/c) curve.

    Args:
        cond_vec (np.ndarray): The first ``K`` DCT-II coefficients.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(x_over_c, cp_target)`` on the
            ``N_CP_GRID`` grid.
    """
    x_over_c = np.linspace(0.0, 1.0, N_CP_GRID)
    cp = dct_ii_inverse(cond_vec.astype(np.float32), N_CP_GRID)
    return x_over_c, cp


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/EXP-022_T1000.yaml", help="Model config YAML."
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/EXP-022_T1000/checkpoint_best.pt",
        help="Trained checkpoint.",
    )
    parser.add_argument(
        "--h5",
        type=Path,
        default=Path("data/ellipse/raw/pOnEllipseTrain.h5"),
        help="Dataset HDF5 (for in-distribution targets' Re/H).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("cfd/ellipse2d/shapes/generated"),
        help="Directory for the .dat / target_cp.csv / manifest.csv outputs.",
    )
    parser.add_argument("--n-indist", type=int, default=2, help="In-dist targets.")
    parser.add_argument("--guidance", type=float, default=3.0, help="CFG scale.")
    parser.add_argument(
        "--smooth-modes", type=int, default=6, help="Fourier modes kept when smoothing."
    )
    parser.add_argument("--seed", type=int, default=0, help="Target-selection seed.")
    parser.add_argument("--device", default="cpu", help="Torch device.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    dataset = build_dataset(config)
    model = build_model(config, args.device)
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()

    clamp_range = _resolve_clamp_range(config)
    k_modes = dataset.k_modes
    template = shape_template_from_dataset(dataset, args.device)
    raw = _load_h5_raw(args.h5)

    # Build the target list: in-distribution (with their source Re/H) + synthetic.
    indist = pick_targets_farthest_first(dataset, args.n_indist, args.seed)
    targets: list[dict] = []
    for idx in indist:
        cond = dataset[idx].cond.squeeze(0).cpu().numpy()
        targets.append(
            {
                "label": f"indist_{idx}",
                "cond": cond,
                "re": float(raw[idx, 0, _H5_COL_RE]),
                "h": float(raw[idx, 0, _H5_COL_H]),
            }
        )
    _, synth_cond = make_synthetic_target(dataset, k_modes)
    targets.append(
        {"label": "synth_ood", "cond": synth_cond, "re": _SYNTH_RE, "h": _SYNTH_H}
    )

    rows = []
    for tgt in targets:
        cond_t = torch.tensor(tgt["cond"], dtype=torch.float32)
        shapes, _ = sample_shapes_for_target(
            model,
            template,
            cond_t,
            n_samples=1,
            guidance_scale=args.guidance,
            device=args.device,
            clamp_range=clamp_range,
            k_modes=k_modes,
        )
        xy = shapes[0]
        xy_smooth = fourier_smooth(xy, args.smooth_modes)

        hdr = f"EXP-041 generated shape, target={tgt['label']} Re={tgt['re']:.4g} H={tgt['h']:.4g}"
        raw_path = args.out_dir / f"{tgt['label']}_raw.dat"
        smooth_path = args.out_dir / f"{tgt['label']}_smooth.dat"
        write_dat(raw_path, xy, hdr + " (raw)")
        write_dat(
            smooth_path,
            xy_smooth,
            hdr + f" (Fourier-smoothed, {args.smooth_modes} modes)",
        )

        x_over_c, cp = target_cp(tgt["cond"])
        cp_path = args.out_dir / f"{tgt['label']}_target_cp.csv"
        with cp_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["x_over_c", "cp_target"])
            for xc, c in zip(x_over_c, cp, strict=True):
                writer.writerow([f"{xc:.6f}", f"{c:.6f}"])

        rows.append(
            {"label": tgt["label"], "re": tgt["re"], "h": tgt["h"], "n_nodes": len(xy)}
        )

    manifest = args.out_dir / "manifest.csv"
    with manifest.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["label", "re", "h", "n_nodes"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Exported {len(rows)} generated shapes (raw + smoothed) to {args.out_dir}")
    for row in rows:
        print(
            f"  {row['label']:>12}: Re={row['re']:.4g} H={row['h']:.4g} n={row['n_nodes']}"
        )


if __name__ == "__main__":
    main()
