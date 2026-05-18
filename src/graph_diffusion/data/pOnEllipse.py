"""
graph_diffusion.data.pOnEllipse
=================================

Dataset for the pOnEllipse aerodynamic boundary mesh dataset
(HuggingFace ``mariolinov/Ellipse``).

Provides:

- ``DatasetUrl``         — enum of HuggingFace file URLs
- ``DatasetDownloader``  — downloads files with a streaming progress bar
- ``pOnEllipseDataset``  — ellipse boundary geometry with configurable
  feature representation for shape generation experiments

**HDF5 structure** (confirmed by EXP-010 data inspection):

The file has a single key ``"data"`` of shape ``(5701, 96, 105)``, dtype float32:

- Axis 0 (5701): distinct (ellipse, Re, AoA) simulations.
- Axis 1 (96): surface boundary nodes per simulation; excess rows are NaN-padded.
  Valid node counts: {52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96}.
- Axis 2 (105 columns per node-row):

  ======  =======  =====================================================
  Col     Name     Description
  ======  =======  =====================================================
  0       x        Node x-coordinate on the ellipse surface
  1       y        Node y-coordinate on the ellipse surface
  2       Re       Reynolds number, 500–1000
  3       AoA      Angle of attack in degrees, 5.0–6.0
  4–104   p_0..100 Pressure coefficient at 101 timesteps
  ======  =======  =====================================================
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from graph_diffusion.data.base_dataset import BaseGraphDataset

__all__ = [
    "DatasetUrl",
    "DatasetDownloader",
    "pOnEllipseDataset",
]

# ---------------------------------------------------------------------------
# HDF5 layout constants
# ---------------------------------------------------------------------------
_H5_DATA_KEY = "data"
_H5_COL_X = 0
_H5_COL_Y = 1
_H5_COL_RE = 2
_H5_COL_AOA = 3
_H5_COL_P_START = 4
_H5_N_TIMESTEPS = 101


class DatasetUrl(enum.Enum):
    """HuggingFace URLs for the pOnEllipse dataset files."""

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
    """Download HuggingFace dataset files with a streaming progress bar.

    Files are downloaded once and cached under ``root``.  Subsequent calls
    to :meth:`download` return the cached path immediately.

    Args:
        root (str): Directory where downloaded files are saved.
            Created automatically if it does not exist.

    Raises:
        requests.HTTPError: If the remote server returns a non-200 status.
    """

    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def local_path(self, url: DatasetUrl) -> Path:
        """Return the local cache path for a given URL.

        Args:
            url (DatasetUrl): The dataset file to locate.

        Returns:
            Path: The local file path under ``self.root``.
        """
        filename = url.value.rsplit("/", 1)[-1]
        return self.root / filename

    def is_downloaded(self, url: DatasetUrl) -> bool:
        """Check whether a file has already been downloaded.

        Args:
            url (DatasetUrl): The dataset file to check.

        Returns:
            bool: ``True`` if the local cache file exists.
        """
        return self.local_path(url).exists()

    def download(self, url: DatasetUrl) -> Path:
        """Download a file if not already cached, then return its local path.

        Args:
            url (DatasetUrl): The dataset file to download.

        Returns:
            Path: The local path to the downloaded file.
        """
        local = self.local_path(url)
        if local.exists():
            return local

        import requests
        from tqdm import tqdm

        print(f"Downloading {url.value} → {local}")
        resp = requests.get(url.value, stream=True, timeout=120)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        with open(local, "wb") as fh, tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            desc=local.name,
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=65536):
                fh.write(chunk)
                pbar.update(len(chunk))

        return local


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_ring_edge_index(n_nodes: int, k_neighbors: int) -> torch.Tensor:
    """Build a bidirectional ring edge index connecting each node to its
    ``k_neighbors`` nearest neighbours on each side.

    Args:
        n_nodes (int): Number of nodes in the ring.
        k_neighbors (int): Number of neighbours on each side.

    Returns:
        torch.Tensor: Shape ``(2, 2 * n_nodes * k_neighbors)``.
    """
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
    """Load the full ``(N_sims, MAX_NODES, 105)`` array from the HDF5 file.

    Args:
        h5_path (Path): Path to the HDF5 file.

    Returns:
        np.ndarray: Float32 array of shape ``(5701, 96, 105)``.
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        return np.array(f[_H5_DATA_KEY], dtype=np.float32)


def _node_count(sample: np.ndarray) -> int:
    """Return the number of valid (non-NaN) node rows in a sample slice.

    Args:
        sample (np.ndarray): Shape ``(MAX_NODES, N_features)``.

    Returns:
        int: Number of finite rows.
    """
    return int((sample[:, _H5_COL_X] == sample[:, _H5_COL_X]).sum())


# ---------------------------------------------------------------------------
# pOnEllipseDataset
# ---------------------------------------------------------------------------

_VALID_FEATURE_MODES = ("radial", "radial_norm", "cartesian", "normalised")


class pOnEllipseDataset(BaseGraphDataset):
    """Ellipse boundary geometry for aerodynamic shape generation.

    Each graph is one CFD simulation: N surface boundary nodes
    (52–96, varies per simulation) connected by a bidirectional ring.
    The node feature ``x`` is the quantity the diffusion model denoises;
    ``pos`` drives the edge-feature transforms.

    Supports four coordinate representations via ``feature_mode``:

    - ``"radial"`` — node feature ``x = [r]`` where ``r = √(x²+y²)``.
      ``pos = [cos θ, sin θ]`` (unit-circle reference).
      Use with ``ComputeAngularEdgeFeatures``.
    - ``"radial_norm"`` — node feature ``x = [r / r̄]`` (mean-normalised).
      ``Data.r_scale`` stores the per-graph mean radius for reconstruction.
      ``Data.p_cond = [cos θ, sin θ]`` supplies positional context.
      Use with ``ComputeAngularEdgeFeatures`` and ``clamp_range=[0.5, 2.0]``.
    - ``"cartesian"`` — node feature ``x = [x_norm, y_norm]``.
      Use with ``ComputeArcLengthEdgeFeatures``.
    - ``"normalised"`` — centred positions divided by semi-axes ``(a, b)``;
      ``x = [r_norm]``. ``u[0:2] = [a, b]``.
      Use with ``ComputeAngularEdgeFeatures``.

    Args:
        root (str): Root directory for downloading and caching.
        feature_mode (str): One of ``"radial"``, ``"radial_norm"``,
            ``"cartesian"``, or ``"normalised"``. Defaults to ``"radial"``.
        split (str): ``"train"`` or ``"test"``. Defaults to ``"train"``.
        n_samples (int | None): Limit the number of simulations loaded.
            ``None`` uses all 5 701.
        k_neighbors (int): Ring-edge neighbours on each side. Defaults to ``2``.
        global_dim (int): Size of the global attribute ``u``. Defaults to ``8``.
        coord_scale (float | None): Global coordinate scale for ``"cartesian"``
            normalisation. ``None`` computes from the loaded split.
        transform (Callable | None): Runtime transform. Defaults to ``None``.
        pre_transform (Callable | None): Processing-time transform.
            Defaults to ``None``.

    Raises:
        ValueError: If ``feature_mode`` is not one of the supported strings.
        ValueError: If ``split`` is not ``"train"`` or ``"test"``.
        ValueError: If ``k_neighbors < 1``.
        ValueError: If ``global_dim < 1``.
        ValueError: If ``global_dim < 2`` when ``feature_mode="normalised"``.
    """

    def __init__(
        self,
        root: str,
        feature_mode: str = "radial",
        split: str = "train",
        n_samples: int | None = None,
        k_neighbors: int = 2,
        global_dim: int = 8,
        coord_scale: float | None = None,
        transform: Callable[[Data], Data] | None = None,
        pre_transform: Callable[[Data], Data] | None = None,
    ) -> None:
        if feature_mode not in _VALID_FEATURE_MODES:
            raise ValueError(
                f"feature_mode must be one of {_VALID_FEATURE_MODES}, "
                f"got '{feature_mode}'"
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
        self.split = split
        self.n_samples = n_samples
        self.k_neighbors = k_neighbors
        self.global_dim = global_dim
        self.coord_scale = coord_scale

        super().__init__(root, transform=transform, pre_transform=pre_transform)

    @property
    def raw_file_names(self) -> list[str]:
        url = DatasetUrl.TRAIN_H5 if self.split == "train" else DatasetUrl.TEST_H5
        return [url.value.rsplit("/", 1)[-1]]

    @property
    def processed_file_names(self) -> list[str]:
        return [f"data_shape_{self.feature_mode}_{self.split}_k{self.k_neighbors}.pt"]

    def download(self) -> None:
        """Download the pOnEllipse HDF5 file from HuggingFace if not cached."""
        url = DatasetUrl.TRAIN_H5 if self.split == "train" else DatasetUrl.TEST_H5
        DatasetDownloader(root=self.raw_dir).download(url)

    def _build_graphs(self) -> list[Data]:
        """Load and convert ellipse boundary coordinates to graph Data objects.

        Returns:
            list[Data]: One graph per CFD simulation.
        """
        h5_path = Path(self.raw_dir) / self.raw_file_names[0]
        raw = _load_h5_raw(h5_path)

        limit = (
            min(self.n_samples, raw.shape[0])
            if self.n_samples is not None
            else raw.shape[0]
        )

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
                    xc -= xc.mean()
                    yc -= yc.mean()
                    max_abs = max(max_abs, float(np.abs(xc).max()), float(np.abs(yc).max()))
                scale = max_abs + 1e-10
            print(f"[pOnEllipseDataset] Cartesian coord_scale={scale:.6f}")

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

            if N not in edge_cache:
                edge_cache[N] = _build_ring_edge_index(N, self.k_neighbors)

            extra: dict[str, torch.Tensor] = {}
            if r_scale_val is not None:
                extra["r_scale"] = torch.tensor([r_scale_val], dtype=torch.float32)
                extra["p_cond"] = pos.clone()

            graphs.append(
                Data(
                    x=x_feat,
                    edge_index=edge_cache[N].clone(),
                    pos=pos,
                    u=u,
                    **extra,
                )
            )

        return graphs
