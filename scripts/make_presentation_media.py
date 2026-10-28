"""
make_presentation_media
=======================

Presentation clips and fallback stills from the P1-REF model (P1-F05, T=1000,
``cond_norm="forward"``):

  * ``reverse_logtime.mp4/.gif`` -- the reverse process on one held-out target,
    played on a logarithmic time axis so the clip slows down as the shape
    resolves. Every frame is a real ``x_t``; nothing is interpolated.
  * ``reverse_frame_*.png`` + ``reverse_frames_strip.png`` -- six stills of the
    same trajectory, for when the video will not play.
  * ``cp_morph.mp4/.gif`` -- the Cp spline morphing through held-out targets,
    a 3-sigma OOD target and a random far-OOD Cp.
  * ``cp_morph_frame_*.png`` + ``cp_morph_frames_strip.png`` -- one still per
    morph target.

The in-distribution targets are drawn from the frozen held-out list
(``paper1_eval_targets.json``) by farthest-first selection on the conditioning
descriptor, i.e. on the *inputs* only -- nothing is chosen by looking at what
the model generated.

Example:
    uv run python scripts/make_presentation_media.py --seed-dir seed2
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as manim  # noqa: E402
import matplotlib.figure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "scripts" / "paper1"))

from _report_style import GREEN, GREY, NAVY, RED  # noqa: E402
from export_ellipse_for_openlb import _reference_cp  # noqa: E402
from p1_b01_retrieval_index import descriptor  # noqa: E402

from graph_diffusion.data.pOnEllipse import (  # noqa: E402
    _H5_COL_RE,
    _H5_COL_X,
    _H5_COL_Y,
    _load_h5_raw,
    _node_count,
)
from graph_diffusion.data.pOnEllipseConditional import (  # noqa: E402
    dct_ii,
    dct_ii_inverse,
)
from graph_diffusion.postprocessing.inference import (  # noqa: E402
    radial_to_xy,
    sample_shapes_from_cond,
    template_thetas,
)
from graph_diffusion.postprocessing.loaders import (  # noqa: E402
    build_dataset,
    build_model,
    load_config,
)
from graph_diffusion.visualisation.cp_editor import cp_spline_and_dct  # noqa: E402
from graph_diffusion.visualisation.cp_morph import (  # noqa: E402
    build_handle_keyframes,
    tween_handles,
)
from graph_diffusion.visualisation.trajectory import collect_reverse  # noqa: E402

_CONFIG = _ROOT / "configs" / "paper1" / "P1-F05_reference.yaml"
_REF = _ROOT / "outputs" / "paper1" / "P1-F05_reference"
_TARGETS = _ROOT / "data" / "splits" / "paper1_eval_targets.json"
_TEST_H5 = _ROOT / "data" / "ellipse" / "raw" / "pOnEllipseInDist.h5"
_OOD_DIR = _ROOT / "cfd" / "ellipse2d" / "shapes" / "ood_systematic"

COND_NORM = "forward"
N_CP_GRID = 128
HANDLE_X = np.linspace(0.0, 1.0, 7, dtype=np.float32)
#: Timesteps shown in the six fallback stills (roughly log-uniform).
STILL_STEPS = (999, 300, 100, 30, 10, 0)

_RC = {
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.25,
    "lines.linewidth": 1.8,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed-dir", default="seed2", help="P1-REF seed directory.")
    p.add_argument("--out-dir", type=Path, default=_ROOT / "outputs" / "presentation")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--reverse-target", type=int, default=None)
    p.add_argument("--n-indist", type=int, default=4)
    p.add_argument("--ood-case", default="pc1_p3sig")
    p.add_argument("--random-seed", type=int, default=3)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--reverse-seconds", type=float, default=12.0)
    p.add_argument("--log-offset", type=float, default=4.0)
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def save_animation(
    fig: matplotlib.figure.Figure,
    update: object,
    n_frames: int,
    stem: Path,
    fps: int,
    dpi: int,
) -> None:
    """Write ``<stem>.mp4`` (if ffmpeg is present) and ``<stem>.gif``."""
    anim = manim.FuncAnimation(
        fig, update, frames=n_frames, interval=1000.0 / fps, blit=False  # type: ignore[arg-type]
    )
    if manim.writers.is_available("ffmpeg"):
        # yuv420p + even dimensions: plays in PowerPoint / Keynote / browsers.
        writer = manim.FFMpegWriter(
            fps=fps,
            codec="libx264",
            extra_args=[
                "-pix_fmt",
                "yuv420p",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white",
                "-crf",
                "18",
            ],
        )
        anim.save(str(stem.with_suffix(".mp4")), writer=writer, dpi=dpi)
    else:
        print(f"[presentation] ffmpeg not available; skipping {stem.name}.mp4")
    anim.save(str(stem.with_suffix(".gif")), writer="pillow", fps=fps, dpi=80)


def closed(xy: np.ndarray) -> np.ndarray:
    """Append the first point so the outline plots as a closed curve."""
    return np.vstack([xy, xy[:1]])


def read_cp_csv(path: Path, value_col: str, x_grid: np.ndarray) -> np.ndarray:
    """Read a 2-column Cp CSV and resample it onto ``x_grid``."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    xs = np.array([float(r["x_over_c"]) for r in rows])
    ys = np.array([float(r[value_col]) for r in rows])
    order = np.argsort(xs)
    return np.interp(x_grid, xs[order], ys[order]).astype(np.float32)


def farthest_first(points: np.ndarray, n: int) -> list[int]:
    """Greedy farthest-first subset, seeded at the point nearest the mean."""
    chosen = [int(np.argmin(np.linalg.norm(points - points.mean(0), axis=1)))]
    while len(chosen) < min(n, len(points)):
        d = np.linalg.norm(points[:, None] - points[chosen][None], axis=-1).min(1)
        chosen.append(int(np.argmax(d)))
    return chosen


def log_time_frames(total_t: int, n_frames: int, offset: float) -> np.ndarray:
    """Timestep shown at each frame, log-spaced from ``T-1`` down to ``0``.

    ``log(t + offset)`` falls linearly with frame index, so equal screen time
    is spent per *decade* of remaining noise and the clip decelerates towards
    ``t = 0``. ``offset`` stops the last handful of steps eating the whole
    clip.
    """
    u = np.linspace(np.log(total_t - 1 + offset), np.log(offset), n_frames)
    return np.clip(np.round(np.exp(u) - offset), 0, total_t - 1).astype(int)


# --------------------------------------------------------------------------
# Reverse process
# --------------------------------------------------------------------------


class ReverseFigure:
    """Shape (left), target Cp (right) and a log-time progress bar (bottom)."""

    def __init__(
        self,
        states: dict[int, np.ndarray],
        total_t: int,
        offset: float,
        x_true: np.ndarray,
        cp_true: np.ndarray,
        cp_seen: np.ndarray,
        x_grid: np.ndarray,
        k_modes: int,
        caption: str,
    ) -> None:
        self.states = states
        self.total_t = total_t
        self.offset = offset
        self.fig = plt.figure(figsize=(11, 5.6))
        gs = self.fig.add_gridspec(
            2, 2, height_ratios=[1, 0.045], width_ratios=[1, 1.15], hspace=0.32
        )
        self.ax_shape = self.fig.add_subplot(gs[0, 0])
        ax_cp = self.fig.add_subplot(gs[0, 1])
        self.ax_bar = self.fig.add_subplot(gs[1, :])

        all_xy = np.concatenate(list(states.values()), axis=0)
        lim = float(np.abs(all_xy).max()) + 0.1
        self.ax_shape.set_aspect("equal")
        self.ax_shape.set_xlim(-lim, lim)
        self.ax_shape.set_ylim(-lim, lim)
        self.ax_shape.set_xticks([])
        self.ax_shape.set_yticks([])
        for spine in self.ax_shape.spines.values():
            spine.set_visible(False)
        (self.line,) = self.ax_shape.plot([], [], color=GREEN, lw=2.0)
        self.title = self.ax_shape.set_title("")

        ax_cp.plot(x_true, cp_true, "o", color=GREY, ms=3.5, alpha=0.55, label="target")
        ax_cp.plot(x_grid, cp_seen, color=NAVY, lw=2.2, label=f"DCT-{k_modes} (input)")
        ax_cp.set_xlabel("$x/c$")
        ax_cp.set_ylabel("$C_p$")
        ax_cp.set_title(caption)
        ax_cp.grid(True)
        ax_cp.legend(loc="best")

        # Log-time bar: noise on the left, clean on the right.
        self.ax_bar.set_xlim(total_t - 1 + offset, offset)
        self.ax_bar.set_xscale("log")
        self.ax_bar.set_ylim(0, 1)
        self.ax_bar.set_yticks([])
        ticks = [t for t in (999, 300, 100, 30, 10, 3, 0) if t < total_t]
        self.ax_bar.set_xticks([t + offset for t in ticks])
        self.ax_bar.set_xticklabels([str(t) for t in ticks])
        self.ax_bar.minorticks_off()
        self.ax_bar.set_xlabel("diffusion step $t$ (log scale)")
        self.ax_bar.spines["left"].set_visible(False)
        self.ax_bar.axvspan(total_t - 1 + offset, offset, color="0.9", lw=0)
        self.fill = self.ax_bar.axvspan(
            total_t - 1 + offset, total_t - 1 + offset, color=GREEN, lw=0
        )

    def draw(self, t: int) -> None:
        """Show the state at timestep ``t``."""
        xy = closed(self.states[t])
        self.line.set_data(xy[:, 0], xy[:, 1])
        self.title.set_text(f"reverse diffusion   $t$ = {t}")
        self.fill.remove()
        self.fill = self.ax_bar.axvspan(
            self.total_t - 1 + self.offset, t + self.offset, color=GREEN, lw=0
        )


def reverse_strip(
    states: dict[int, np.ndarray],
    steps: tuple[int, ...],
    x_true: np.ndarray,
    cp_true: np.ndarray,
    cp_seen: np.ndarray,
    x_grid: np.ndarray,
    k_modes: int,
) -> matplotlib.figure.Figure:
    """One-row filmstrip of the reverse process plus the target Cp panel."""
    n = len(steps)
    fig = plt.figure(figsize=(2.5 * n + 4.2, 3.3))
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1] * n + [1.6], wspace=0.08)
    lim = float(np.abs(np.concatenate([states[t] for t in steps])).max()) + 0.1
    for col, t in enumerate(steps):
        ax = fig.add_subplot(gs[0, col])
        xy = closed(states[t])
        ax.plot(xy[:, 0], xy[:, 1], color=GREEN, lw=1.6)
        ax.set_aspect("equal")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"$t$ = {t}")
    ax_cp = fig.add_subplot(gs[0, n])
    ax_cp.plot(x_true, cp_true, "o", color=GREY, ms=3, alpha=0.55, label="target")
    ax_cp.plot(x_grid, cp_seen, color=NAVY, lw=2.0, label=f"DCT-{k_modes} (input)")
    ax_cp.set_xlabel("$x/c$")
    ax_cp.set_title("target $C_p$")
    ax_cp.yaxis.tick_right()
    ax_cp.grid(True)
    ax_cp.legend(loc="best", fontsize=9)
    return fig


# --------------------------------------------------------------------------
# Cp morph
# --------------------------------------------------------------------------


class MorphFigure:
    """Cp editor view (left) and the shape for the last-reached target."""

    def __init__(
        self,
        handle_keys: np.ndarray,
        shapes: list[np.ndarray],
        labels: list[str],
        ood_flags: list[bool],
        x_grid: np.ndarray,
        k_modes: int,
    ) -> None:
        self.shapes = shapes
        self.labels = labels
        self.ood_flags = ood_flags
        self.x_grid = x_grid
        self.k_modes = k_modes
        self.fig, (ax_cp, ax_shape) = plt.subplots(
            1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1.2, 1]}
        )
        splines = [
            cp_spline_and_dct(HANDLE_X, h, x_grid, k_modes, COND_NORM)[0]
            for h in handle_keys
        ]
        ax_cp.set_xlim(-0.03, 1.03)
        ax_cp.set_ylim(float(np.min(splines)) - 0.2, float(np.max(splines)) + 0.2)
        ax_cp.set_xlabel("$x/c$")
        ax_cp.set_ylabel("$C_p$")
        ax_cp.set_title("target $C_p$ (drag handles)")
        ax_cp.grid(True)
        (self.spline_line,) = ax_cp.plot([], [], color="0.7", lw=1.2, label="spline")
        (self.dct_line,) = ax_cp.plot(
            [], [], color=NAVY, lw=2.2, label=f"DCT-{k_modes} (input)"
        )
        (self.handles_line,) = ax_cp.plot([], [], "o", color=RED, ms=8, label="handles")
        ax_cp.legend(loc="upper center", ncols=3)

        all_xy = np.concatenate(shapes, axis=0)
        lo, hi = all_xy.min(axis=0) - 0.1, all_xy.max(axis=0) + 0.1
        ax_shape.set_aspect("equal")
        ax_shape.set_xlim(lo[0], hi[0])
        ax_shape.set_ylim(lo[1], hi[1])
        ax_shape.set_xticks([])
        ax_shape.set_yticks([])
        for spine in ax_shape.spines.values():
            spine.set_visible(False)
        (self.shape_line,) = ax_shape.plot([], [], color=GREEN, lw=2.0)
        self.shape_title = ax_shape.set_title("")
        self.fig.tight_layout()

    def draw(self, handle_y: np.ndarray, target_idx: int) -> None:
        """Show one frame: handle state plus the shape for ``target_idx``."""
        spline, dct_curve = cp_spline_and_dct(
            HANDLE_X, handle_y, self.x_grid, self.k_modes, COND_NORM
        )
        self.spline_line.set_data(self.x_grid, spline)
        self.dct_line.set_data(self.x_grid, dct_curve)
        self.handles_line.set_data(HANDLE_X, handle_y)
        xy = closed(self.shapes[target_idx])
        self.shape_line.set_data(xy[:, 0], xy[:, 1])
        is_ood = self.ood_flags[target_idx]
        self.shape_title.set_text(
            f"{self.labels[target_idx].replace(chr(10), '   ')}"
            f"{'  (OOD)' if is_ood else ''}"
        )
        self.shape_title.set_color(RED if is_ood else "black")
        self.shape_line.set_color(RED if is_ood else GREEN)


def morph_strip(
    handle_keys: np.ndarray,
    shapes: list[np.ndarray],
    labels: list[str],
    ood_flags: list[bool],
    x_grid: np.ndarray,
    k_modes: int,
) -> matplotlib.figure.Figure:
    """Two-row strip: the Cp input above the shape it generates."""
    n = len(shapes)
    fig, axes = plt.subplots(
        2, n, figsize=(2.7 * n, 5.4), gridspec_kw={"height_ratios": [1, 1.1]}
    )
    curves = [
        cp_spline_and_dct(HANDLE_X, h, x_grid, k_modes, COND_NORM) for h in handle_keys
    ]
    cp_lo = min(float(c[0].min()) for c in curves) - 0.2
    cp_hi = max(float(c[0].max()) for c in curves) + 0.2
    all_xy = np.concatenate(shapes, axis=0)
    lo, hi = all_xy.min(axis=0) - 0.1, all_xy.max(axis=0) + 0.1
    for col in range(n):
        colour = RED if ood_flags[col] else GREEN
        ax_cp: Axes = axes[0, col]
        ax_cp.plot(x_grid, curves[col][0], color="0.7", lw=1.0)
        ax_cp.plot(x_grid, curves[col][1], color=NAVY, lw=2.0)
        ax_cp.plot(HANDLE_X, handle_keys[col], "o", color=RED, ms=5)
        ax_cp.set_ylim(cp_lo, cp_hi)
        ax_cp.grid(True)
        ax_cp.set_xlabel("$x/c$")
        ax_cp.set_title(
            f"{labels[col]}{chr(10) + '(OOD)' if ood_flags[col] else ''}",
            fontsize=11,
            color=RED if ood_flags[col] else "black",
        )
        if col == 0:
            ax_cp.set_ylabel("$C_p$")
        else:
            ax_cp.set_yticklabels([])
        ax_sh: Axes = axes[1, col]
        xy = closed(shapes[col])
        ax_sh.plot(xy[:, 0], xy[:, 1], color=colour, lw=1.8)
        ax_sh.set_aspect("equal")
        ax_sh.set_xlim(lo[0], hi[0])
        ax_sh.set_ylim(lo[1], hi[1])
        ax_sh.set_xticks([])
        ax_sh.set_yticks([])
        for spine in ax_sh.spines.values():
            spine.set_visible(False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    plt.rcParams.update(_RC)
    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    device = args.device
    x_grid = np.linspace(0.0, 1.0, N_CP_GRID, dtype=np.float32)

    config = load_config(str(_CONFIG))
    dataset = build_dataset(config)
    k_modes = int(dataset.k_modes)
    guidance = float(config.get("sampling", {}).get("guidance_scale", 1.0))
    clamp_range = (float(config["clamp_range"][0]), float(config["clamp_range"][1]))
    total_t = int(config["noise_schedule"]["T"])

    ckpt = _REF / args.seed_dir / "checkpoint_best.pt"
    model = build_model(config, device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()
    print(f"[presentation] model: {ckpt.relative_to(_ROOT)}  ({device})")

    template = copy.copy(dataset[0]).to(device)
    theta = template_thetas(template)

    # Held-out targets: descriptor, truth Cp, Re and thickness for each.
    indist = json.loads(_TARGETS.read_text())["sets"]["indist"]["target_indices"]
    test_raw = _load_h5_raw(_TEST_H5)
    conds = np.stack([descriptor(test_raw[i], COND_NORM)[0] for i in indist])

    def target_info(idx: int) -> tuple[np.ndarray, np.ndarray, float, float]:
        data = test_raw[idx][: _node_count(test_raw[idx])]
        x_true, cp_true = _reference_cp(data)
        aspect = float(np.ptp(data[:, _H5_COL_Y]) / np.ptp(data[:, _H5_COL_X]))
        return x_true, cp_true, float(data[0, _H5_COL_RE]), aspect

    picks = farthest_first(conds, args.n_indist)
    print(
        f"[presentation] held-out targets (farthest-first): "
        f"{[indist[i] for i in picks]}"
    )

    # ---- 1 + 2: reverse process, log time --------------------------------
    rev_idx = (
        args.reverse_target if args.reverse_target is not None else indist[picks[0]]
    )
    x_true, cp_true, re_num, aspect = target_info(rev_idx)
    cond = descriptor(test_raw[rev_idx], COND_NORM)[0]
    cp_seen = dct_ii_inverse(cond, positions=x_grid, norm=COND_NORM)

    tpl = copy.copy(template)
    tpl.cond = torch.tensor(cond, dtype=torch.float32, device=device).unsqueeze(0)
    all_steps = list(range(total_t - 1, -1, -1))
    snaps = collect_reverse(
        model,
        tpl,
        cond=tpl.cond,
        snapshot_steps=all_steps,
        guidance_scale=guidance,
        seed=args.sample_seed,
        clamp_range=clamp_range,
    )
    states = {
        t: radial_to_xy(s[:, 0].numpy(), theta)
        for t, s in zip(all_steps, snaps, strict=True)
    }

    caption = f"held-out target #{rev_idx}   $Re$ = {re_num:.0f}"
    n_frames = int(round(args.reverse_seconds * args.fps))
    frame_t = log_time_frames(total_t, n_frames, args.log_offset)
    frame_t = np.concatenate([frame_t, np.zeros(2 * args.fps, dtype=int)])  # hold

    rev = ReverseFigure(
        states,
        total_t,
        args.log_offset,
        x_true,
        cp_true,
        cp_seen,
        x_grid,
        k_modes,
        caption,
    )
    save_animation(
        rev.fig,
        lambda f: rev.draw(int(frame_t[f])),
        len(frame_t),
        out / "reverse_logtime",
        args.fps,
        args.dpi,
    )
    stills = tuple(t for t in STILL_STEPS if t < total_t)
    for i, t in enumerate(stills, start=1):
        rev.draw(t)
        rev.fig.savefig(out / f"reverse_frame_{i}_t{t:03d}.png", dpi=args.dpi)
    plt.close(rev.fig)
    strip = reverse_strip(states, stills, x_true, cp_true, cp_seen, x_grid, k_modes)
    strip.savefig(out / "reverse_frames_strip.png", dpi=200, bbox_inches="tight")
    plt.close(strip)
    print(
        f"[presentation] reverse: target #{rev_idx}, b/a={aspect:.2f}, "
        f"{len(frame_t)} frames"
    )

    # ---- 3: Cp morph -----------------------------------------------------
    dense: list[np.ndarray] = []
    labels: list[str] = []
    ood_flags: list[bool] = []
    for i in picks:
        idx = indist[i]
        xt, ct, re_i, asp_i = target_info(idx)
        dense.append(np.interp(x_grid, xt, ct).astype(np.float32))
        labels.append(f"held-out #{idx}\nRe={re_i:.0f}, b/a={asp_i:.2f}")
        ood_flags.append(False)
    dense.append(
        read_cp_csv(_OOD_DIR / f"{args.ood_case}_target_cp.csv", "cp_target", x_grid)
    )
    labels.append(f"{args.ood_case}")
    ood_flags.append(True)
    rng = np.random.default_rng(args.random_seed)
    rand_y = rng.uniform(-1.6, 1.6, size=HANDLE_X.shape[0]).astype(np.float32)
    dense.append(cp_spline_and_dct(HANDLE_X, rand_y, x_grid, k_modes, COND_NORM)[0])
    labels.append("random Cp")
    ood_flags.append(True)

    handle_keys = build_handle_keyframes(np.stack(dense), HANDLE_X, x_grid)

    # Condition on the DCT of the *displayed* spline, so the blue curve in
    # every frame is exactly the vector the model was given.
    shapes: list[np.ndarray] = []
    for j, handle_y in enumerate(handle_keys):
        spline, _ = cp_spline_and_dct(HANDLE_X, handle_y, x_grid, k_modes, COND_NORM)
        cond_j = dct_ii(spline, k_modes, positions=x_grid, norm=COND_NORM)
        if not ood_flags[j]:
            gap = float(np.linalg.norm(cond_j - conds[picks[j]]))
            print(
                f"    {labels[j].replace(chr(10), ' ')}: "
                f"|spline cond - true cond| = {gap:.4f} "
                f"(|true| = {float(np.linalg.norm(conds[picks[j]])):.4f})"
            )
        radii, _ = sample_shapes_from_cond(
            model=model,
            template=template,
            cond_vec=torch.from_numpy(cond_j).float(),
            n_samples=1,
            guidance_scale=guidance,
            device=device,
            clamp_range=clamp_range,
            seed=args.sample_seed,
        )
        shapes.append(radial_to_xy(radii[0], theta))

    frames = tween_handles(handle_keys, n_tween=25, n_hold=50)
    morph = MorphFigure(handle_keys, shapes, labels, ood_flags, x_grid, k_modes)
    save_animation(
        morph.fig,
        lambda f: morph.draw(frames[f].handle_y, frames[f].target_idx),
        len(frames),
        out / "cp_morph",
        args.fps,
        args.dpi,
    )
    for j, handle_y in enumerate(handle_keys):
        morph.draw(handle_y, j)
        morph.fig.savefig(out / f"cp_morph_frame_{j + 1}.png", dpi=args.dpi)
    plt.close(morph.fig)
    strip = morph_strip(handle_keys, shapes, labels, ood_flags, x_grid, k_modes)
    strip.savefig(out / "cp_morph_frames_strip.png", dpi=200, bbox_inches="tight")
    plt.close(strip)
    print(f"[presentation] wrote {out.relative_to(_ROOT)}/")


if __name__ == "__main__":
    main()
