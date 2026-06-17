"""DPS-guidance sweep on the synthetic OOD target (EXP-041 follow-up).

Samples the EXP-022 generator for the synthetic asymmetric (out-of-distribution)
Cp target at several ``dps_guidance_weight`` values and exports each generated
shape as an OpenLB ``x y`` ``.dat``. The OpenLB-measured Cp of each is then
compared against the target to test whether DPS (Diffusion Posterior Sampling,
which steers via the model's *self-trained* pressure head) improves the **true**
Cp fidelity out-of-distribution, or merely chases the head's own OOD error.

DPS injects ``∇‖h(x̂₀, pos) − c_target‖²`` into each reverse step; weight 0 is the
plain CFG baseline. Same seed across weights so shapes differ only by DPS.
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
    make_synthetic_target,
    shape_template_from_dataset,
)

from graph_diffusion.data.pOnEllipseConditional import dct_ii_inverse
from graph_diffusion.postprocessing.inference import radial_to_xy, template_thetas
from graph_diffusion.postprocessing.loaders import (
    build_dataset,
    build_model,
    load_config,
)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/EXP-022_T1000.yaml")
    parser.add_argument(
        "--checkpoint", default="outputs/EXP-022_T1000/checkpoint_best.pt"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("cfd/ellipse2d/shapes/dps_sweep")
    )
    parser.add_argument(
        "--dps",
        type=float,
        nargs="+",
        default=[0.0, 0.3, 0.5, 1.0],
        help="DPS weights.",
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

    _, synth_cond = make_synthetic_target(dataset, k_modes)
    template.cond = (
        torch.tensor(synth_cond, dtype=torch.float32).unsqueeze(0).to(args.device)
    )

    # Target Cp (shared across all DPS weights).
    x_over_c = np.linspace(0.0, 1.0, N_CP_GRID)
    cp_target = dct_ii_inverse(synth_cond.astype(np.float32), N_CP_GRID)
    with (args.out_dir / "synth_target_cp.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["x_over_c", "cp_target"])
        for xc, c in zip(x_over_c, cp_target, strict=True):
            writer.writerow([f"{xc:.6f}", f"{c:.6f}"])

    labels = []
    for dps in args.dps:
        torch.manual_seed(args.seed)
        out = model.sample(
            template,
            clamp_range=clamp_range,
            guidance_scale=args.guidance,
            dps_guidance_weight=dps,
        )
        radii = out.x[:, 0].detach().cpu().numpy()
        xy = radial_to_xy(radii, theta)
        tag = f"dps{dps:g}".replace(".", "p")
        path = args.out_dir / f"synth_{tag}.dat"
        with path.open("w") as fh:
            fh.write(f"# EXP-041 DPS sweep: synthetic OOD target, dps={dps:g}\n# x y\n")
            for x, y in xy:
                fh.write(f"{x:.10f} {y:.10f}\n")
        labels.append((dps, tag, len(xy)))
        print(f"dps={dps:g} -> {path.name} ({len(xy)} pts)")

    with (args.out_dir / "manifest.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dps", "tag", "n_nodes"])
        for dps, tag, n in labels:
            writer.writerow([dps, tag, n])
    print(f"Wrote {len(labels)} DPS-swept shapes to {args.out_dir}")


if __name__ == "__main__":
    main()
