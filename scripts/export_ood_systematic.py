"""Generate the systematic PC-mode OOD Cp sweep for EXP-043.

Builds OOD Cp targets along the dataset's principal Cp modes (a controlled
in-distribution -> border -> OOD ladder via cp_manifold.pc_ladder_targets),
samples a shape for each with the inverse-design checkpoint (plain
classifier-free guidance; DPS hurts OOD per EXP-042), and exports each as an
OpenLB ``.dat`` plus its target Cp and a run-driver-compatible manifest
(name,idx,Re,H,ood_dist,mode,sign,amp_sigma). Run conditions are fixed at
Re/H = 754/5.5 so the Cp-target morphology is the only variable.
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

from graph_diffusion.data.cp_manifold import dataset_cp_pca, pc_ladder_targets
from graph_diffusion.data.pOnEllipse import _load_h5_raw
from graph_diffusion.data.pOnEllipseConditional import dct_ii
from graph_diffusion.postprocessing.inference import radial_to_xy, template_thetas
from graph_diffusion.postprocessing.loaders import (
    build_dataset,
    build_model,
    load_config,
)

_OOD_RE = 754.0
_OOD_H = 5.5


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/EXP-022_T1000.yaml")
    parser.add_argument(
        "--checkpoint", default="outputs/EXP-022_T1000/checkpoint_best.pt"
    )
    parser.add_argument(
        "--h5", type=Path, default=Path("data/ellipse/raw/pOnEllipseTrain.h5")
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("cfd/ellipse2d/shapes/ood_systematic")
    )
    parser.add_argument("--n-modes", type=int, default=2)
    parser.add_argument(
        "--amplitudes", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0]
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

    train_conds = torch.stack([g.cond.squeeze(0) for g in dataset], dim=0).numpy()

    raw = _load_h5_raw(args.h5)
    _, basis = dataset_cp_pca(raw, n_grid=N_CP_GRID, n_components=args.n_modes)
    targets = pc_ladder_targets(
        basis, amplitudes=tuple(args.amplitudes), n_modes=args.n_modes
    )
    x_over_c = np.linspace(0.0, 1.0, N_CP_GRID)

    rows = []
    for i, t in enumerate(targets):
        cp_target = t["cp_target"].astype(np.float32)
        cond = dct_ii(cp_target, k_modes, norm="ortho").astype(np.float32)
        ood_dist = float(np.linalg.norm(train_conds - cond, axis=1).min())

        template.cond = (
            torch.tensor(cond, dtype=torch.float32).unsqueeze(0).to(args.device)
        )
        torch.manual_seed(args.seed)
        out = model.sample(
            template, clamp_range=clamp_range, guidance_scale=args.guidance
        )
        radii = out.x[:, 0].detach().cpu().numpy()
        xy = radial_to_xy(radii, theta)

        name = t["name"]
        dat = args.out_dir / f"{name}.dat"
        with dat.open("w") as fh:
            fh.write(
                f"# EXP-043 systematic OOD: {name} mode={t['mode']} "
                f"amp={t['amp_sigma']}sig ood_dist={ood_dist:.4g}\n# x y\n"
            )
            for x, y in xy:
                fh.write(f"{x:.10f} {y:.10f}\n")

        with (args.out_dir / f"{name}_target_cp.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["x_over_c", "cp_target"])
            for xc, c in zip(x_over_c, cp_target, strict=True):
                writer.writerow([f"{xc:.6f}", f"{c:.6f}"])

        rows.append(
            {
                "name": name,
                "idx": i,
                "Re": _OOD_RE,
                "H": _OOD_H,
                "ood_dist": ood_dist,
                "mode": t["mode"],
                "sign": t["sign"],
                "amp_sigma": t["amp_sigma"],
            }
        )
        print(f"{name:>10}: ood_dist={ood_dist:.4f} -> {dat.name} ({len(xy)} pts)")

    fields = ["name", "idx", "Re", "H", "ood_dist", "mode", "sign", "amp_sigma"]
    with (args.out_dir / "manifest.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {len(rows)} systematic OOD targets to {args.out_dir}")


if __name__ == "__main__":
    main()
