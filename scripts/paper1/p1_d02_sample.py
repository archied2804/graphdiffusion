"""
P1-D02 -- generate at node counts the model was never trained on.

The model is trained across 52-96 boundary nodes. This asks whether it performs
uniformly *inside* that range and how it behaves *outside* it, by rebuilding the
sampling template at each node count and sampling again. No retraining, no
architecture change -- message passing on a ring is node-count agnostic, and
P1-D01 made the conditioning descriptor node-count invariant, which is what makes
the question askable at all.

**This is the capability a lookup table does not have.** Nearest-neighbour
retrieval can only return a stored 52-96-node body; a PCA basis fitted at one
node count cannot be evaluated at another. Asked for a 24- or 384-node design,
both have nothing to return. So D02 is not an accuracy race against a baseline --
it is a test of something only a discretisation-native method can do.

The template is built with **uniformly spaced** theta at every rung. Real corpus
samples are not uniformly spaced, so in-range rungs differ from training data in
spacing as well as count; uniform throughout keeps N the only variable, which is
the cleaner design. Recorded as a deviation.

Bodies are exported at chord 1.0 per [[PROTOCOL]] section 2.1b.

Usage:
    python scripts/paper1/p1_d02_sample.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_ellipse_for_openlb import _ordered_polygon, _reference_cp  # noqa: E402
from p1_b01_retrieval_index import descriptor  # noqa: E402
from postprocess_exp020 import (  # noqa: E402
    _resolve_clamp_range,
    sample_shapes_for_target,
)

from graph_diffusion.data.pOnEllipse import (  # noqa: E402
    _H5_COL_H,
    _H5_COL_RE,
    _H5_COL_Y,
    _load_h5_raw,
    _node_count,
)
from graph_diffusion.postprocessing.loaders import (  # noqa: E402
    build_dataset,
    build_model,
    load_config,
)

_RAW = _ROOT / "data" / "ellipse" / "raw"
_TARGETS = _ROOT / "data" / "splits" / "paper1_eval_targets.json"
_CONFIG = _ROOT / "configs" / "paper1" / "P1-F05_reference.yaml"
_REF = _ROOT / "outputs" / "paper1" / "P1-F05_reference"
_OUT = _ROOT / "cfd" / "ellipse2d" / "shapes" / "d02_nodes"

#: The ladder. Training spans 52-96, so four rungs sit below, four inside and
#: four above -- 0.46x to 4.0x the trained range. The spec registered seven
#: rungs {40,52,64,80,96,128,192}; this is a widening, not a change of rule.
LADDER = (24, 32, 40, 48, 52, 64, 80, 96, 128, 192, 256, 384)
TRAIN_MIN, TRAIN_MAX = 52, 96

#: Three, not five. Seed-to-seed sd on `Cd` is 0.58 sigma_Cd, so 4 targets x 3
#: seeds per rung gives SEM 0.17 sigma -- ample. On rho the sd is 5.0 sigma_rho
#: and no affordable seed count resolves the registered 2 sigma rule; five seeds
#: would not fix that and would halve the ladder.
SEEDS = (0, 1, 2)

#: Four held-out targets, picked on a 2x2 (thickness, Reynolds) factorial of the
#: frozen evaluation set -- fixed before any Cp exists.
N_TARGETS = 4
COND_NORM = "forward"
CHORD = 1.0
K_NEIGHBOURS = 6

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def ring_template(theta: np.ndarray, ref: Data, device: str) -> Data:
    """Build a sampling template on a ring at the given node angles.

    Mirrors the dataset's own construction: ``pos`` and ``p_cond`` are
    ``[cos, sin]``, the ring is connected to ``K_NEIGHBOURS`` neighbours each
    side, and ``edge_attr`` is ``[sin(dtheta), cos(dtheta)]``.

    Args:
        theta (np.ndarray): Node angles in radians, shape ``(N,)``.
        ref (Data): A real dataset template, used only for dtype and for the
            global tensors' shapes.
        device (str): Torch device.

    Returns:
        Data: A template with ``N`` nodes, ready for sampling.
    """
    n = theta.size
    pos = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)

    src, dst = [], []
    for k in range(1, K_NEIGHBOURS + 1):
        for i in range(n):
            src += [i, i]
            dst += [(i + k) % n, (i - k) % n]
    src_a, dst_a = np.asarray(src), np.asarray(dst)
    dtheta = theta[dst_a] - theta[src_a]
    edge_attr = np.stack([np.sin(dtheta), np.cos(dtheta)], axis=1).astype(np.float32)

    return Data(
        x=torch.zeros((n, 1), dtype=torch.float32),
        pos=torch.from_numpy(pos),
        p_cond=torch.from_numpy(pos.copy()),
        edge_index=torch.from_numpy(np.stack([src_a, dst_a])).long(),
        edge_attr=torch.from_numpy(edge_attr),
        u=torch.zeros_like(ref.u),
        cond=torch.zeros_like(ref.cond),
        r_scale=torch.ones_like(ref.r_scale),
    ).to(device)


def verify_builder(ref: Data) -> None:
    """Check the builder reproduces a real dataset template's graph exactly.

    Builds a template from the reference's *own* node angles and compares the
    edge set and edge features. A mismatch means the ladder templates would be
    subtly wrong in a way no downstream number would reveal.

    Args:
        ref (Data): A real template from the dataset.

    Raises:
        SystemExit: If the reconstruction does not match.
    """
    pos = ref.pos.cpu().numpy()
    theta = np.arctan2(pos[:, 1], pos[:, 0])
    built = ring_template(theta, ref, "cpu")

    got = {
        (int(a), int(b)) for a, b in zip(*built.edge_index.cpu().numpy(), strict=True)
    }
    want = {
        (int(a), int(b)) for a, b in zip(*ref.edge_index.cpu().numpy(), strict=True)
    }
    if got != want:
        sys.exit(
            f"template builder MISMATCH: {len(want - got)} edges missing, "
            f"{len(got - want)} extra"
        )

    # Compare edge_attr on the shared ordering.
    ref_map = {
        (int(a), int(b)): v
        for (a, b), v in zip(
            zip(*ref.edge_index.cpu().numpy(), strict=True),
            ref.edge_attr.cpu().numpy(),
            strict=True,
        )
    }
    worst = 0.0
    for (a, b), v in zip(
        zip(*built.edge_index.cpu().numpy(), strict=True),
        built.edge_attr.cpu().numpy(),
        strict=True,
    ):
        worst = max(worst, float(np.abs(ref_map[(int(a), int(b))] - v).max()))
    if worst > 1e-5:
        sys.exit(f"template builder MISMATCH: edge_attr differs by {worst:.2e}")
    print(
        f"  template builder verified against dataset[0] "
        f"(edges exact, edge_attr max diff {worst:.1e})"
    )


def select_targets(raw: np.ndarray, picks: list[int]) -> list[int]:
    """Pick four targets on a 2x2 (thickness, Reynolds) factorial.

    Args:
        raw (np.ndarray): The test HDF5 array.
        picks (list[int]): Candidate row indices (the frozen evaluation set).

    Returns:
        list[int]: Four indices, thin/low-Re first.
    """
    bs, res = [], []
    for i in picks:
        s = raw[i][: _node_count(raw[i])]
        bs.append((s[:, _H5_COL_Y].max() - s[:, _H5_COL_Y].min()) / 2.0)
        res.append(float(s[0, _H5_COL_RE]))
    b_u = np.asarray(bs).argsort().argsort() / (len(bs) - 1)
    re_u = np.asarray(res).argsort().argsort() / (len(res) - 1)
    out: list[int] = []
    for tb in (0.15, 0.85):
        for tre in (0.15, 0.85):
            d = np.hypot(b_u - tb, re_u - tre)
            for k in np.argsort(d):
                if picks[int(k)] not in out:
                    out.append(picks[int(k)])
                    break
    return out


def write_dat(path: Path, xy: np.ndarray, header: str) -> None:
    """Write a generated boundary at physical scale, per PROTOCOL 2.1b.

    Args:
        path (Path): Destination ``.dat``.
        xy (np.ndarray): Boundary points, shape ``(N, 2)``.
        header (str): First comment line, without the leading ``#``.
    """
    a = np.asarray(xy, dtype=float)
    a = a - a.mean(axis=0)
    span = float(a[:, 0].max() - a[:, 0].min())
    poly = _ordered_polygon(a * (CHORD / span) if span > 0 else a)
    with path.open("w") as fh:
        fh.write(f"# {header}\n# x y\n")
        for x, y in poly:
            fh.write(f"{x:.10f} {y:.10f}\n")


def main() -> None:
    meta = json.loads(_TARGETS.read_text())
    indist = meta["sets"]["indist"]["target_indices"]
    config = load_config(str(_CONFIG))
    dataset = build_dataset(config)
    ref = dataset[0]
    clamp_range = _resolve_clamp_range(config)
    k_modes = dataset.k_modes
    guidance = float(config.get("sampling", {}).get("guidance_scale", 3.0))

    print(f"\nP1-D02 -- node-count transfer ({DEVICE})\n")
    verify_builder(ref)

    raw = _load_h5_raw(_RAW / "pOnEllipseInDist.h5")
    picks = select_targets(raw, indist)[:N_TARGETS]
    _OUT.mkdir(parents=True, exist_ok=True)

    tmeta: dict[str, dict] = {}
    conds: dict[str, np.ndarray] = {}
    for t in picks:
        sample = raw[t]
        data = sample[: _node_count(sample)]
        lab = f"t{t:03d}"
        conds[lab] = descriptor(sample, COND_NORM)[0]
        tmeta[lab] = {
            "target_idx": t,
            "Re": float(data[0, _H5_COL_RE]),
            "H": float(data[0, _H5_COL_H]),
            "n_nodes_target": int(_node_count(sample)),
        }
        x, cp = _reference_cp(data)
        with (_OUT / f"{lab}_ref_cp.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["x_over_c", "cp_ref"])
            for a, b in zip(x, cp, strict=True):
                w.writerow([f"{a:.6f}", f"{b:.6f}"])
        print(
            f"  target {lab}: Re={tmeta[lab]['Re']:.0f} "
            f"H={tmeta[lab]['H']:.3g} n={tmeta[lab]['n_nodes_target']}"
        )

    templates = {
        n: ring_template(np.linspace(0.0, 2.0 * np.pi, n, endpoint=False), ref, DEVICE)
        for n in LADDER
    }
    print(f"\n  ladder {LADDER}")
    print(f"  trained range {TRAIN_MIN}-{TRAIN_MAX}   seeds {SEEDS}")
    print(f"  {len(LADDER) * len(picks) * len(SEEDS)} bodies\n")

    rows: list[dict] = []
    for seed in SEEDS:
        ckpt = _REF / f"seed{seed}" / "checkpoint_best.pt"
        model = build_model(config, DEVICE)
        state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state.get("model_state_dict", state))
        model.eval()
        print(f"  seed{seed}")
        for n in LADDER:
            for lab in tmeta:
                shapes, _ = sample_shapes_for_target(
                    model,
                    templates[n],
                    torch.tensor(conds[lab], dtype=torch.float32),
                    n_samples=1,
                    guidance_scale=guidance,
                    device=DEVICE,
                    clamp_range=clamp_range,
                    k_modes=k_modes,
                )
                xy = shapes[0]
                tag = f"{lab}_n{n:03d}_s{seed}"
                m = tmeta[lab]
                zone = (
                    "in"
                    if TRAIN_MIN <= n <= TRAIN_MAX
                    else ("below" if n < TRAIN_MIN else "above")
                )
                write_dat(
                    _OUT / f"{tag}.dat",
                    xy,
                    f"P1-D02 {tag}: N={n} ({zone} trained range) "
                    f"Re={m['Re']:.4g} H={m['H']:.4g}",
                )
                rows.append(
                    {
                        "tag": tag,
                        "label": lab,
                        "n_nodes": n,
                        "zone": zone,
                        "seed": seed,
                        "target_idx": m["target_idx"],
                        "Re": m["Re"],
                        "H": m["H"],
                        "n_generated": int(len(xy)),
                    }
                )
            print(f"    N={n:>3}  ok")

    man = _OUT / "manifest.csv"
    with man.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  {len(rows)} bodies -> {_OUT.relative_to(_ROOT)}\n")


if __name__ == "__main__":
    main()
