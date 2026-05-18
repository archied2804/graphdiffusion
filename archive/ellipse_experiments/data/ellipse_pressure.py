"""
archive/ellipse_experiments/data/ellipse_pressure.py
======================================================

Archived from src/graph_diffusion/data/ellipsedataset.py.
EllipseDataset: pressure field on ellipse boundary (legacy pressure baseline,
EXP-010).

NOTE: The import below references the old path. If reactivating, update to:
    from graph_diffusion.data.base_dataset import BaseGraphDataset
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from graph_diffusion.data.base_dataset import BaseGraphDataset

__all__ = ["EllipseDataset"]

_H5_DATA_KEY = "data"
_H5_COL_X = 0
_H5_COL_Y = 1
_H5_COL_RE = 2
_H5_COL_AOA = 3
_H5_COL_P_START = 4
_H5_N_TIMESTEPS = 101


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


class EllipseDataset(BaseGraphDataset):
    """Pressure field on the ellipse boundary (legacy pressure baseline, EXP-010).

    Archived. See archive/ellipse_experiments/README.md.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        n_samples: int | None = None,
        k_neighbors: int = 2,
        global_dim: int = 8,
        time_index: int = -1,
        pressure_norm_mean: float | None = None,
        pressure_norm_std: float | None = None,
        transform: Callable[[Data], Data] | None = None,
        pre_transform: Callable[[Data], Data] | None = None,
    ) -> None:
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got '{split}'")
        if k_neighbors < 1:
            raise ValueError(f"k_neighbors must be >= 1, got {k_neighbors}")
        if global_dim < 2:
            raise ValueError(f"global_dim must be >= 2 (stores Re, AoA), got {global_dim}")

        self.split = split
        self.n_samples = n_samples
        self.k_neighbors = k_neighbors
        self.global_dim = global_dim
        self.time_index = time_index
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
        return [f"data_pressure_{self.split}_t{tidx}_k{self.k_neighbors}.pt"]

    def download(self) -> None:
        url = DatasetUrl.TRAIN_H5 if self.split == "train" else DatasetUrl.TEST_H5
        DatasetDownloader(root=self.raw_dir).download(url)

    def _build_graphs(self) -> list[Data]:
        h5_path = Path(self.raw_dir) / self.raw_file_names[0]
        raw = _load_h5_raw(h5_path)
        limit = min(self.n_samples, raw.shape[0]) if self.n_samples is not None else raw.shape[0]

        if self.pressure_norm_mean is not None and self.pressure_norm_std is not None:
            p_mean = float(self.pressure_norm_mean)
            p_std = float(self.pressure_norm_std)
        else:
            all_p: list[float] = []
            for i in range(limit):
                N = _node_count(raw[i])
                p_col = _H5_COL_P_START + (self.time_index % _H5_N_TIMESTEPS)
                all_p.extend(raw[i, :N, p_col].tolist())
            p_mean = float(np.mean(all_p))
            p_std = float(np.std(all_p) + 1e-8)

        edge_cache: dict[int, torch.Tensor] = {}
        p_col = _H5_COL_P_START + (self.time_index % _H5_N_TIMESTEPS)

        graphs: list[Data] = []
        for i in range(limit):
            sample = raw[i]
            N = _node_count(sample)
            data = sample[:N]
            re = float(data[0, _H5_COL_RE])
            aoa = float(data[0, _H5_COL_AOA])
            xc = data[:, _H5_COL_X].astype(np.float32)
            yc = data[:, _H5_COL_Y].astype(np.float32)
            xc -= xc.mean()
            yc -= yc.mean()
            order = np.argsort(np.arctan2(yc, xc))
            xc = xc[order]
            yc = yc[order]
            p_raw = data[:, p_col][order]
            p_norm = ((p_raw - p_mean) / p_std).astype(np.float32)
            x_feat = torch.tensor(p_norm[:, None], dtype=torch.float32)
            pos = torch.tensor(np.stack([xc, yc], axis=1), dtype=torch.float32)
            u_vals = np.zeros(self.global_dim, dtype=np.float32)
            u_vals[0] = re
            u_vals[1] = aoa
            u = torch.tensor(u_vals[None, :])
            if N not in edge_cache:
                edge_cache[N] = _build_ring_edge_index(N, self.k_neighbors)
            graphs.append(Data(x=x_feat, edge_index=edge_cache[N].clone(), pos=pos, u=u))

        return graphs
