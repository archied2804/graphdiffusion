"""Generate a spread of out-of-distribution Cp targets for EXP-041/042.

The original OOD test used a single synthetic target (``mean + 0.3·sin(πx)``),
so we could not tell whether the ~0.86 correlation / ~3.5 magnitude creep is a
general OOD property or a quirk of that one curve. This script builds a *family*
of distinct OOD targets by perturbing the dataset-mean ``Cp(x/c)`` in several
controlled ways (amplitude, sign, frequency, chordwise tilt), quantifies how far
each sits from the training set (``ood_dist`` = min L2 distance in DCT-cond
space), samples a shape for each with plain classifier-free guidance (DPS is
known to hurt OOD — see EXP-042), and exports each as an OpenLB ``.dat`` plus its
decoded target ``Cp`` and a manifest. The OpenLB-measured Cp of each is then
compared to its target to map fidelity against OOD-ness.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from postprocess_exp020 import (  # type: ignore[import-not-found]
    N_CP_GRID,
    _resolve_clamp_range,
    shape_template_from_dataset,
)

from graph_diffusion.data.pOnEllipseConditional import dct_ii, dct_ii_inverse
from graph_diffusion.postprocessing.inference import radial_to_xy, template_thetas
from graph_diffusion.postprocessing.loaders import (
    build_dataset,
    build_model,
    load_config,
)

# Fixed run conditions so the OOD-target morphology is the only variable.
_OOD_RE = 754.0
_OOD_H = 5.5


def perturbations(x: np.ndarray) -> dict[str, np.ndarray]:
    """OOD perturbation family added to the dataset-mean Cp curve.

    Args:
        x (np.ndarray): Chordwise coordinate ``x/c`` on the ``N_CP_GRID`` grid.

    Returns:
        dict[str, np.ndarray]: Named additive perturbations spanning amplitude,
            sign, frequency, and chordwise tilt.
    """
    return {
        "sin_p3": 0.3 * np.sin(np.pi * x),  # original synthetic target
        "sin_p5": 0.5 * np.sin(np.pi * x),  # stronger suction bump
        "sin_m3": -0.3 * np.sin(np.pi * x),  # inverted bump
        "sin2_p3": 0.3 * np.sin(2 * np.pi * x),  # higher-frequency double bump
        "tilt_p4": 0.4 * (x - 0.5),  # chordwise pressure-recovery skew
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/EXP-022_T1000.yaml")
    parser.add_argument(
        "--checkpoint", default="outputs/EXP-022_T1000/checkpoint_best.pt"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("cfd/ellipse2d/shapes/ood_sweep")
    )
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
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
    theta = template_thetas(template)

    # Dataset-mean Cp and the training cond cloud (for the OOD distance).
    train_conds = torch.stack([g.cond.squeeze(0) for g in dataset], dim=0).numpy()
    mean_cond = train_conds.mean(axis=0)
    x_over_c = np.linspace(0.0, 1.0, N_CP_GRID)
    cp_mean = dct_ii_inverse(mean_cond.astype(np.float32), N_CP_GRID)

    rows = []
    for tag, delta in perturbations(x_over_c).items():
        cp_target = (cp_mean + delta).astype(np.float32)
        cond = dct_ii(cp_target, k_modes, norm="ortho").astype(np.float32)
        ood_dist = float(np.linalg.norm(train_conds - cond, axis=1).min())

        # Sample a shape (plain CFG; DPS degrades OOD — see EXP-042).
        template.cond = (
            torch.tensor(cond, dtype=torch.float32).unsqueeze(0).to(args.device)
        )
        torch.manual_seed(args.seed)
        out = model.sample(
            template, clamp_range=clamp_range, guidance_scale=args.guidance
        )
        radii = out.x[:, 0].detach().cpu().numpy()
        xy = radial_to_xy(radii, theta)

        dat = args.out_dir / f"ood_{tag}.dat"
        with dat.open("w") as fh:
            fh.write(
                f"# EXP-041 OOD sweep: target={tag} ood_dist={ood_dist:.4g}\n# x y\n"
            )
            for x, y in xy:
                fh.write(f"{x:.10f} {y:.10f}\n")

        with (args.out_dir / f"ood_{tag}_target_cp.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["x_over_c", "cp_target"])
            for xc, c in zip(x_over_c, cp_target, strict=True):
                writer.writerow([f"{xc:.6f}", f"{c:.6f}"])

        rows.append({"tag": tag, "ood_dist": ood_dist, "re": _OOD_RE, "h": _OOD_H})
        print(f"{tag:>10}: ood_dist={ood_dist:.4f} -> {dat.name} ({len(xy)} pts)")

    with (args.out_dir / "manifest.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["tag", "ood_dist", "re", "h"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {len(rows)} OOD targets to {args.out_dir}")


if __name__ == "__main__":
    main()
