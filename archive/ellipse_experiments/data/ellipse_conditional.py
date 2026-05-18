"""
archive/ellipse_experiments/data/ellipse_conditional.py
=========================================================

Archived from src/graph_diffusion/data/ellipsedataset.py.
EllipseConditionalDataset: shape + pressure conditioning for inverse design
(EXP-015/016).

NOTE: imports reference graph_diffusion.data.base_dataset — update if reactivating.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from graph_diffusion.data.base_dataset import BaseGraphDataset

_H5_DATA_KEY = "data"
_H5_COL_X = 0
_H5_COL_Y = 1
_H5_COL_RE = 2
_H5_COL_AOA = 3
_H5_COL_P_START = 4
_H5_N_TIMESTEPS = 101
_VALID_FEATURE_MODES = ("radial", "radial_norm", "cartesian", "normalised")
_VALID_COND_TYPES = ("global_summary", "node_concat")


class DatasetUrl(enum.Enum):
    TRAIN_H5 = (
        "https://huggingface.co/datasets/mariolinov/Ellipse"
        "/resolve/main/pOnEllipseTrain.h5"
    )
    TEST_H5 = (
        "https://huggingface.co/datasets/mariolinov/Ellipse"
        "/resolve/main/pOnEllipseTest.h5"
    )
    TIME_TRAIN_NPY = (
        "https://huggingface.co/datasets/mariolinov/Ellipse"
        "/resolve/main/TimeEllipseTrain.npy"
    )


class DatasetDownloader:
    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def local_path(self, url: DatasetUrl) -> Path:
        filename = url.value.rsplit("/", 1)[-1]
        return self.root / filename

    def is_downloaded(self, url: DatasetUrl) -> bool:
        return self.local_path(url).exists()

    def download(self, url: DatasetUrl) -> Path:
        local = self.local_path(url)
        if local.exists():
            return local
        import requests
        from tqdm import tqdm
        resp = requests.get(url.value, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(local, "wb") as fh, tqdm(total=total or None, unit="B", unit_scale=True, desc=local.name) as pbar:
            for chunk in resp.iter_content(chunk_size=65536):
                fh.write(chunk)
                pbar.update(len(chunk))
        return local


def _build_ring_edge_index(n_nodes: int, k_neighbors: int) -> torch.Tensor:
    sources: list[int] = []
    targets: list[int] = []
    for i in range(n_nodes):
        for k in range(1, k_neighbors + 1):
            sources.append(i)
            targets.append((i + k) % n_nodes)
            sources.append(i)
            targets.append((i - k) % n_nodes)
    return torch.tensor([sources, targets], dtype=torch.long)


def _load_h5_raw(h5_path: Path) -> np.ndarray:
    import h5py
    with h5py.File(h5_path, "r") as f:
        return np.array(f[_H5_DATA_KEY], dtype=np.float32)


def _node_count(sample: np.ndarray) -> int:
    return int((sample[:, _H5_COL_X] == sample[:, _H5_COL_X]).sum())


class EllipseConditionalDataset(BaseGraphDataset):
    """Ellipse shape dataset with pressure-field conditioning.

    Combines ellipse boundary geometry with surface pressure conditioning.
    The base shape representation is controlled by ``feature_mode``; the
    conditioning mode is controlled by ``cond_type``.

    **Feature modes** (same semantics as EllipseShapeDataset):

    - ``"radial"`` (EXP-011/012): ``x = [r]`` where ``r = √(x²+y²)``;
      ``pos = [cos θ, sin θ]``.  Use with ``ComputeAngularEdgeFeatures``.
    - ``"normalised"``: ``x = [r_norm]`` normalised by semi-axes ``(a, b)``;
      semi-axes stored in ``u[0:2]``.  Use with ``ComputeAngularEdgeFeatures``.
    - ``"cartesian"``: ``x = [x_norm, y_norm]``; use with
      ``ComputeArcLengthEdgeFeatures``.

    **Conditioning types**:

    - ``"global_summary"`` (EXP-011): pressure summarised as
      ``cond = [p_mean, p_std, p_max, p_min]``, shape ``(1, 4)``.
    - ``"node_concat"`` (EXP-012): per-node pressure stored as
      ``p_cond = [Cp_norm]``, shape ``(N, 1)``, concatenated to ``x`` at
      each reverse diffusion step.

    Args:
        root (str): Root directory for downloading and caching.
        feature_mode (str): Base shape representation. Defaults to ``"normalised"``.
        cond_type (str): One of ``"global_summary"`` or ``"node_concat"``.
            Defaults to ``"global_summary"``.
        split (str): ``"train"`` or ``"test"``. Defaults to ``"train"``.
        n_samples (int | None): Limit the number of samples. Defaults to ``None``.
        k_neighbors (int): Ring-edge neighbours on each side. Defaults to ``2``.
        global_dim (int): Size of global attribute ``u``. Defaults to ``8``.
        time_index (int): Pressure timestep index (``-1`` = last). Defaults to ``-1``.
        coord_scale (float | None): Pre-computed coordinate scale for cartesian mode.
        pressure_norm_mean (float | None): Global mean for Cp normalisation.
        pressure_norm_std (float | None): Global std for Cp normalisation.
        transform (Callable | None): Runtime transform. Defaults to ``None``.
        pre_transform (Callable | None): Processing-time transform.
            Defaults to ``None``.

    Raises:
        ValueError: If ``feature_mode`` is not one of the supported strings.
        ValueError: If ``cond_type`` is not one of the supported strings.
        ValueError: If ``global_dim < 2`` when ``feature_mode="normalised"``.
    """

    def __init__(
        self,
        root: str,
        feature_mode: str = "normalised",
        cond_type: str = "global_summary",
        split: str = "train",
        n_samples: int | None = None,
        k_neighbors: int = 2,
        global_dim: int = 8,
        time_index: int = -1,
        coord_scale: float | None = None,
        pressure_norm_mean: float | None = None,
        pressure_norm_std: float | None = None,
        transform: Callable[[Data], Data] | None = None,
        pre_transform: Callable[[Data], Data] | None = None,
    ) -> None:
        if feature_mode not in _VALID_FEATURE_MODES:
            raise ValueError(
                f"feature_mode must be one of {_VALID_FEATURE_MODES}, "
                f"got '{feature_mode}'"
            )
        if cond_type not in _VALID_COND_TYPES:
            raise ValueError(
                f"cond_type must be one of {_VALID_COND_TYPES}, got '{cond_type}'"
            )
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got '{split}'")
        if k_neighbors < 1:
            raise ValueError(f"k_neighbors must be >= 1, got {k_neighbors}")
        if global_dim < 1:
            raise ValueError(f"global_dim must be >= 1, got {global_dim}")
        if feature_mode == "normalised" and global_dim < 2:
            raise ValueError(
                "global_dim must be >= 2 for feature_mode='normalised' "
                f"(stores a and b), got {global_dim}"
            )

        self.feature_mode = feature_mode
        self.cond_type = cond_type
        self.split = split
        self.n_samples = n_samples
        self.k_neighbors = k_neighbors
        self.global_dim = global_dim
        self.time_index = time_index
        self.coord_scale = coord_scale
        self.pressure_norm_mean = pressure_norm_mean
        self.pressure_norm_std = pressure_norm_std

        super().__init__(root, transform=transform, pre_transform=pre_transform)

    @property
    def raw_file_names(self) -> list[str]:
        url = DatasetUrl.TRAIN_H5 if self.split == "train" else DatasetUrl.TEST_H5
        return [url.value.rsplit("/", 1)[-1]]

    @property
    def processed_file_names(self) -> list[str]:
        tidx = self.time_index % _H5_N_TIMESTEPS
        return [
            f"data_cond_{self.feature_mode}_{self.cond_type}"
            f"_{self.split}_t{tidx}_k{self.k_neighbors}.pt"
        ]

    def download(self) -> None:
        url = DatasetUrl.TRAIN_H5 if self.split == "train" else DatasetUrl.TEST_H5
        DatasetDownloader(root=self.raw_dir).download(url)

    def _build_graphs(self) -> list[Data]:
        """Load shape + pressure conditioning from the HDF5 file.

        Returns:
            list[Data]: One graph per CFD simulation with shape features and
                pressure conditioning attached.
        """
        h5_path = Path(self.raw_dir) / self.raw_file_names[0]
        raw = _load_h5_raw(h5_path)

        limit = (
            min(self.n_samples, raw.shape[0])
            if self.n_samples is not None
            else raw.shape[0]
        )
        p_col = _H5_COL_P_START + (self.time_index % _H5_N_TIMESTEPS)

        if self.pressure_norm_mean is not None and self.pressure_norm_std is not None:
            p_mean = float(self.pressure_norm_mean)
            p_std = float(self.pressure_norm_std)
        else:
            all_p: list[float] = []
            for i in range(limit):
                N = _node_count(raw[i])
                all_p.extend(raw[i, :N, p_col].tolist())
            p_mean = float(np.mean(all_p))
            p_std = float(np.std(all_p) + 1e-8)

        scale: float = 1.0
        if self.feature_mode == "cartesian":
            if self.coord_scale is not None:
                scale = float(self.coord_scale)
            else:
                max_abs = 0.0
                for i in range(limit):
                    N = _node_count(raw[i])
                    xc = raw[i, :N, _H5_COL_X]
                    yc = raw[i, :N, _H5_COL_Y]
                    xc = xc - xc.mean()
                    yc = yc - yc.mean()
                    max_abs = max(
                        max_abs,
                        float(np.abs(xc).max()),
                        float(np.abs(yc).max()),
                    )
                scale = max_abs + 1e-10
            print(f"[EllipseConditionalDataset] Cartesian coord_scale={scale:.6f}")

        edge_cache: dict[int, torch.Tensor] = {}

        graphs: list[Data] = []
        for i in range(limit):
            sample = raw[i]
            N = _node_count(sample)
            data = sample[:N]

            xc = data[:, _H5_COL_X].astype(np.float32)
            yc = data[:, _H5_COL_Y].astype(np.float32)
            xc = xc - xc.mean()
            yc = yc - yc.mean()

            theta = np.arctan2(yc, xc)
            unit_pos = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(
                np.float32
            )

            r_scale_val: float | None = None

            if self.feature_mode == "radial":
                r = np.sqrt(xc**2 + yc**2)
                x_feat = torch.tensor(r[:, None], dtype=torch.float32)
                pos = torch.tensor(unit_pos, dtype=torch.float32)
                u = torch.zeros(1, self.global_dim)

            elif self.feature_mode == "radial_norm":
                r = np.sqrt(xc**2 + yc**2)
                r_scale_val = float(r.mean())
                x_feat = torch.tensor(
                    (r / r_scale_val)[:, None], dtype=torch.float32
                )
                pos = torch.tensor(unit_pos, dtype=torch.float32)
                u = torch.zeros(1, self.global_dim)

            elif self.feature_mode == "cartesian":
                xn = xc / scale
                yn = yc / scale
                x_feat = torch.tensor(
                    np.stack([xn, yn], axis=1), dtype=torch.float32
                )
                pos = x_feat.clone()
                u = torch.zeros(1, self.global_dim)

            else:  # normalised
                a = float((xc.max() - xc.min()) / 2.0 + 1e-10)
                b = float((yc.max() - yc.min()) / 2.0 + 1e-10)
                r_norm = np.sqrt((xc / a) ** 2 + (yc / b) ** 2)
                x_feat = torch.tensor(r_norm[:, None], dtype=torch.float32)
                pos = torch.tensor(unit_pos, dtype=torch.float32)
                u_vals = np.zeros(self.global_dim, dtype=np.float32)
                u_vals[0] = a
                u_vals[1] = b
                u = torch.tensor(u_vals[None, :])

            p_raw = data[:, p_col].astype(np.float32)
            p_norm = (p_raw - p_mean) / p_std

            if N not in edge_cache:
                edge_cache[N] = _build_ring_edge_index(N, self.k_neighbors)

            extra: dict[str, torch.Tensor] = {}
            if r_scale_val is not None:
                extra["r_scale"] = torch.tensor(
                    [r_scale_val], dtype=torch.float32
                )

            if self.cond_type == "global_summary":
                cond = torch.tensor(
                    [[p_norm.mean(), p_norm.std(), p_norm.max(), p_norm.min()]],
                    dtype=torch.float32,
                )
                if self.feature_mode == "radial_norm":
                    extra["p_cond"] = pos.clone()
                graphs.append(
                    Data(
                        x=x_feat,
                        edge_index=edge_cache[N].clone(),
                        pos=pos,
                        u=u,
                        cond=cond,
                        **extra,
                    )
                )
            else:  # node_concat
                p_pressure = torch.tensor(
                    p_norm[:, None], dtype=torch.float32
                )
                if self.feature_mode == "radial_norm":
                    p_cond = torch.cat([pos.clone(), p_pressure], dim=-1)
                else:
                    p_cond = p_pressure
                graphs.append(
                    Data(
                        x=x_feat,
                        edge_index=edge_cache[N].clone(),
                        pos=pos,
                        u=u,
                        p_cond=p_cond,
                        **extra,
                    )
                )

        return graphs
