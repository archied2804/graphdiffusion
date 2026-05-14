# Aerodynamic Mesh Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the repository from an accumulated circle-ablation/experimental codebase into a clean, aerodynamically-framed baseline where `src/graph_diffusion/` contains only what the pOnEllipse shape-generation pipeline needs.

**Architecture:** Archive the complete EXP-00x circle series and ellipse conditional/pressure-baseline code under `archive/`. Extract `BaseGraphDataset` into its own file, rename `EllipseShapeDataset` → `pOnEllipseDataset` in `data/pOnEllipse.py`, and consolidate training to a single `train.py` (from `train_ellipse.py`). Model code is unchanged apart from docstring notes on conditional features.

**Tech Stack:** Python 3.11, PyTorch, PyTorch Geometric, uv, pytest, ruff, mypy, black.

---

## File Map

| Action | Path |
|--------|------|
| Create | `archive/circle/` (reference copy of EXP-00x code) |
| Create | `archive/ellipse_experiments/` (conditional + pressure baseline) |
| Create | `src/graph_diffusion/data/base_dataset.py` |
| Create | `src/graph_diffusion/data/pOnEllipse.py` |
| Create | `tests/test_base_dataset.py` |
| Create | `tests/test_pOnEllipse.py` |
| Create | `train.py` (replaces current stub + train_ellipse.py) |
| Modify | `src/graph_diffusion/data/__init__.py` |
| Modify | `src/graph_diffusion/model/__init__.py` |
| Modify | `src/graph_diffusion/model/score_network.py` (docstrings only) |
| Modify | `src/graph_diffusion/model/graph_diffusion_model.py` (docstrings only) |
| Modify | `CLAUDE.md` |
| Delete | `src/graph_diffusion/data/dataset.py` |
| Delete | `src/graph_diffusion/data/ellipsedataset.py` |
| Delete | `tests/test_dataset.py` |
| Delete | `tests/test_ellipsedataset.py` |
| Delete | `train_ellipse.py`, old `train.py`, `train_circle.py`, `train_ddp.py`, `main.py` |

---

## Task 1: Verify baseline

**Files:** none modified

- [ ] **Step 1: Run the full test suite**

```bash
uv run pytest tests/ -q
```

Expected: all tests pass (or note any pre-existing failures before starting).

- [ ] **Step 2: Run the quality gate**

```bash
uv run ruff check src/ tests/ && uv run mypy src/
```

Expected: no errors. If there are pre-existing errors, note them — do not fix them here.

---

## Task 2: Create archive directory structure

**Files:** new directories only

- [ ] **Step 1: Create all archive subdirectories**

```bash
mkdir -p archive/circle/data \
          archive/circle/model \
          archive/circle/configs \
          archive/circle/tests \
          archive/ellipse_experiments/data \
          archive/ellipse_experiments/configs \
          archive/ellipse_experiments/tests
```

- [ ] **Step 2: Add README files so the archive is self-documenting**

Create `archive/circle/README.md`:

```markdown
# Archive: Circle Series (EXP-001 – EXP-006)

Proof-of-concept DDPM experiments on synthetic unit-circle shapes.
The EXP-00x series is complete. This code is kept for reference only
and is not imported anywhere in the active codebase.

Key results: EXP-005 recipe (k=6 neighbours, amplitude=0.15, single
radial feature, 100 epochs) is the recommended starting point.
See `docs/experiments/EXP-00x_series_summary.md` for full conclusions.
```

Create `archive/ellipse_experiments/README.md`:

```markdown
# Archive: Ellipse Conditional / Pressure Experiments

Contains `EllipseDataset` (pressure baseline, EXP-010) and
`EllipseConditionalDataset` (conditional inverse design, EXP-015/016).
These are not part of the unconditional shape-generation baseline.
They will be re-integrated when EXP-015 is implemented.

Import paths in these files reference
`graph_diffusion.data.base_dataset` — update if reactivating.
```

- [ ] **Step 3: Commit the empty structure**

```bash
git add archive/
git commit -m "chore: create archive directory structure for circle and ellipse experiments"
```

---

## Task 3: Archive circle source files

**Files:** moves from `src/` and root to `archive/circle/`

- [ ] **Step 1: Move circle and synthetic dataset source files**

```bash
git mv src/graph_diffusion/data/circledataset.py archive/circle/data/circledataset.py
git mv src/graph_diffusion/data/dataset.py archive/circle/data/dataset.py
git mv src/graph_diffusion/model/fourier_score_network.py archive/circle/model/fourier_score_network.py
```

- [ ] **Step 2: Move old generic train.py and circle training scripts**

```bash
git mv train.py archive/circle/train.py
git mv train_circle.py archive/circle/train_circle.py
git mv train_ddp.py archive/circle/train_ddp.py
git mv main.py archive/circle/main.py
```

- [ ] **Step 3: Move circle scripts**

```bash
git mv scripts/postprocess_circle.py archive/circle/
git mv scripts/run_circle.sh archive/circle/
git mv scripts/run_experiments.py archive/circle/
git mv scripts/run.sh archive/circle/
```

- [ ] **Step 4: Move circle experiment configs**

```bash
git mv configs/circle_radial.yaml archive/circle/configs/
git mv configs/default.yaml archive/circle/configs/
git mv configs/EXP-002a_circle_radial_k1.yaml archive/circle/configs/
git mv configs/EXP-002b_circle_radial_k2.yaml archive/circle/configs/
git mv configs/EXP-002c_circle_radial_k4.yaml archive/circle/configs/
git mv configs/EXP-002_circle_radial_k-neighbors.yaml archive/circle/configs/
git mv configs/EXP-002d_circle_radial_k6.yaml archive/circle/configs/
git mv configs/EXP-003a_circle_radial_amp005.yaml archive/circle/configs/
git mv configs/EXP-003b_circle_radial_amp015.yaml archive/circle/configs/
git mv configs/EXP-003c_circle_radial_amp030.yaml archive/circle/configs/
git mv configs/EXP-003_circle_radial_amplitude.yaml archive/circle/configs/
git mv configs/EXP-004_circle_radial_logit.yaml archive/circle/configs/
git mv configs/EXP-005_circle_radial_full.yaml archive/circle/configs/
git mv configs/EXP-006_circle_radial_rich-features.yaml archive/circle/configs/
```

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "chore(archive): move circle series source files, scripts, and configs to archive/circle/"
```

---

## Task 4: Archive circle tests

**Files:** moves from `tests/` to `archive/circle/tests/`

- [ ] **Step 1: Move circle test files**

```bash
git mv tests/test_circledataset.py archive/circle/tests/test_circledataset.py
git mv tests/test_circledataset_rich.py archive/circle/tests/test_circledataset_rich.py
git mv tests/test_circle_integration.py archive/circle/tests/test_circle_integration.py
git mv tests/test_integration.py archive/circle/tests/test_integration.py
```

- [ ] **Step 2: Commit**

```bash
git add -u
git commit -m "chore(archive): move circle test files to archive/circle/tests/"
```

---

## Task 5: Archive ellipse experimental tests

**Files:** moves from `tests/` to `archive/ellipse_experiments/tests/`

- [ ] **Step 1: Move ellipse conditional and integration tests**

```bash
git mv tests/test_ellipse_conditional.py archive/ellipse_experiments/tests/test_ellipse_conditional.py
git mv tests/test_ellipse_integration.py archive/ellipse_experiments/tests/test_ellipse_integration.py
```

- [ ] **Step 2: Move EXP-013 configs**

```bash
git mv configs/EXP-013_ellipse_smoothness_reg.yaml archive/ellipse_experiments/configs/
git mv configs/EXP-013b_ellipse_smoothness_reg_strong.yaml archive/ellipse_experiments/configs/
```

- [ ] **Step 3: Commit**

```bash
git add -u
git commit -m "chore(archive): move ellipse conditional tests and EXP-013 configs to archive/"
```

---

## Task 6: Extract archive ellipse experimental source code

These classes come out of `ellipsedataset.py` into separate archive files before that file is replaced in Task 8.

**Files:**
- Create: `archive/ellipse_experiments/data/ellipse_pressure.py`
- Create: `archive/ellipse_experiments/data/ellipse_conditional.py`

- [ ] **Step 1: Create `archive/ellipse_experiments/data/ellipse_pressure.py`**

This is `EllipseDataset` extracted verbatim from `src/graph_diffusion/data/ellipsedataset.py` (lines 224–391). Copy those lines into the new file, prepend this header, and fix the import path note:

```python
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

# NOTE: update this import if reactivating
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
```

- [ ] **Step 2: Create `archive/ellipse_experiments/data/ellipse_conditional.py`**

Copy `EllipseConditionalDataset` verbatim from `src/graph_diffusion/data/ellipsedataset.py` (lines 619–902). Prepend this header:

```python
"""
archive/ellipse_experiments/data/ellipse_conditional.py
=========================================================

Archived from src/graph_diffusion/data/ellipsedataset.py.
EllipseConditionalDataset: shape + pressure conditioning for inverse design
(EXP-015/016).

NOTE: imports reference graph_diffusion.data.base_dataset — update if reactivating.
"""
```

Then include the complete `EllipseConditionalDataset` class as-is from `ellipsedataset.py` (no changes to the class body). Add the necessary imports at the top of the file:

```python
from __future__ import annotations

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
```

Then copy the helper functions `_build_ring_edge_index`, `_load_h5_raw`, `_node_count`, `DatasetUrl`, `DatasetDownloader` from Task 6 Step 1 (same implementations). Then copy `EllipseConditionalDataset` verbatim.

- [ ] **Step 3: Commit**

```bash
git add archive/ellipse_experiments/data/
git commit -m "chore(archive): extract EllipseDataset and EllipseConditionalDataset to archive/"
```

---

## Task 7: Create `src/graph_diffusion/data/base_dataset.py`

Extract `BaseGraphDataset` from `dataset.py` into its own file.

**Files:**
- Create: `src/graph_diffusion/data/base_dataset.py`

- [ ] **Step 1: Write the failing test** (in a temporary location to verify the file works)

In a fresh Python REPL or throwaway script, verify the import will work once the file exists. Skip this step — the test is written in Task 11.

- [ ] **Step 2: Create `src/graph_diffusion/data/base_dataset.py`**

```python
"""
graph_diffusion.data.base_dataset
===================================

Abstract base dataset for all boundary mesh datasets in this library.
Subclass this and implement ``_build_graphs`` to define a new dataset.
"""

import abc
from collections.abc import Callable

from torch_geometric.data import Data, InMemoryDataset

__all__ = [
    "BaseGraphDataset",
]


class BaseGraphDataset(InMemoryDataset, abc.ABC):  # type: ignore[misc]
    """Abstract base class for all graph datasets in this library.

    Subclasses must implement ``_build_graphs`` to define the concrete
    list of ``Data`` objects that constitute the dataset.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (Callable | None): A transform applied to each ``Data``
            object at access time. Defaults to ``None``.
        pre_transform (Callable | None): A transform applied once during
            ``process()`` before saving. Defaults to ``None``.
    """

    def __init__(
        self,
        root: str,
        transform: Callable[[Data], Data] | None = None,
        pre_transform: Callable[[Data], Data] | None = None,
    ) -> None:
        super().__init__(root, transform=transform, pre_transform=pre_transform)
        self.load(self.processed_paths[0])

    @abc.abstractmethod
    def _build_graphs(self) -> list[Data]:
        """Construct and return the list of graph Data objects.

        Returns:
            list[Data]: The raw graph data objects.
        """

    @property
    def raw_file_names(self) -> list[str]:
        """No raw files required by default."""
        return []

    @property
    def processed_file_names(self) -> list[str]:
        """Single processed file containing the collated dataset."""
        return ["data.pt"]

    def download(self) -> None:
        """No-op — override in subclasses that download from remote sources."""

    def process(self) -> None:
        """Build graphs, apply pre_transform, collate and save."""
        graph_list = self._build_graphs()
        if self.pre_transform is not None:
            graph_list = [self.pre_transform(g) for g in graph_list]
        self.save(graph_list, self.processed_paths[0])
```

- [ ] **Step 3: Commit**

```bash
git add src/graph_diffusion/data/base_dataset.py
git commit -m "feat(data): extract BaseGraphDataset into base_dataset.py"
```

---

## Task 8: Create `src/graph_diffusion/data/pOnEllipse.py`

Rename `EllipseShapeDataset` → `pOnEllipseDataset` and update the import of `BaseGraphDataset`.

**Files:**
- Create: `src/graph_diffusion/data/pOnEllipse.py`

- [ ] **Step 1: Create `src/graph_diffusion/data/pOnEllipse.py`**

This is `ellipsedataset.py` with three changes:
1. New module docstring (below)
2. Import updated: `from graph_diffusion.data.base_dataset import BaseGraphDataset`
3. `EllipseShapeDataset` → `pOnEllipseDataset` everywhere in the file
4. `__all__` updated to only export the three symbols that belong here
5. `EllipseDataset`, `EllipseConditionalDataset`, and `_VALID_COND_TYPES` removed

Full file:

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add src/graph_diffusion/data/pOnEllipse.py
git commit -m "feat(data): add pOnEllipseDataset (renamed from EllipseShapeDataset)"
```

---

## Task 9: Update `data/__init__.py` and `model/__init__.py`

**Files:**
- Modify: `src/graph_diffusion/data/__init__.py`
- Modify: `src/graph_diffusion/model/__init__.py`

- [ ] **Step 1: Replace `src/graph_diffusion/data/__init__.py`**

```python
"""
graph_diffusion.data
====================

Data loading, dataset definitions, and graph transforms for aerodynamic
boundary mesh generation.
"""

from graph_diffusion.data.base_dataset import BaseGraphDataset
from graph_diffusion.data.dataloader import GraphDataLoader
from graph_diffusion.data.pOnEllipse import (
    DatasetDownloader,
    DatasetUrl,
    pOnEllipseDataset,
)
from graph_diffusion.data.transforms import (
    AddSelfLoops,
    BaseTransform,
    Compose,
    ComputeAngularEdgeFeatures,
    ComputeArcLengthEdgeFeatures,
    KNNGraph,
    NormalizeNodeFeatures,
)

__all__ = [
    "BaseGraphDataset",
    "GraphDataLoader",
    "DatasetUrl",
    "DatasetDownloader",
    "pOnEllipseDataset",
    "BaseTransform",
    "NormalizeNodeFeatures",
    "AddSelfLoops",
    "KNNGraph",
    "Compose",
    "ComputeAngularEdgeFeatures",
    "ComputeArcLengthEdgeFeatures",
]
```

- [ ] **Step 2: Replace `src/graph_diffusion/model/__init__.py`**

```python
"""
graph_diffusion.model
======================

Score network and diffusion model for aerodynamic boundary mesh generation.
"""

from graph_diffusion.model.graph_diffusion_model import GraphDiffusionModel
from graph_diffusion.model.score_network import ScoreNetwork

__all__ = [
    "ScoreNetwork",
    "GraphDiffusionModel",
]
```

- [ ] **Step 3: Commit**

```bash
git add src/graph_diffusion/data/__init__.py src/graph_diffusion/model/__init__.py
git commit -m "refactor(data,model): update __init__.py exports for aerodynamic rename"
```

---

## Task 10: Add docstring notes to model classes

Mark the conditional/experimental parameters so they are clearly labelled future work.

**Files:**
- Modify: `src/graph_diffusion/model/score_network.py`
- Modify: `src/graph_diffusion/model/graph_diffusion_model.py`

- [ ] **Step 1: Update `ScoreNetwork` docstring in `score_network.py`**

Locate the `Args:` block in the `ScoreNetwork` class docstring (around line 36). After the `output_dim` argument description, add:

```
    Note:
        ``cond_dim`` and ``output_dim`` are future-work parameters for
        conditional inverse design (EXP-015: global pressure conditioning,
        EXP-016: node-level pressure conditioning). Leave as ``None`` for
        the unconditional shape-generation baseline.
```

- [ ] **Step 2: Update `GraphDiffusionModel` docstring in `graph_diffusion_model.py`**

Locate the `Args:` block in the `GraphDiffusionModel` class docstring (around line 37). After the `smoothness_weight` argument description, add:

```
    Note:
        ``feature_transform``, ``n_noise_channels``, and
        ``smoothness_weight`` are future-work parameters for bounded
        diffusion and conditional/regularised training (EXP-013+,
        EXP-015+). Leave at their defaults (``None``, ``None``, ``0.0``)
        for the unconditional shape-generation baseline.
```

- [ ] **Step 3: Commit**

```bash
git add src/graph_diffusion/model/score_network.py \
        src/graph_diffusion/model/graph_diffusion_model.py
git commit -m "docs(model): mark conditional/experimental args as future work (EXP-015+)"
```

---

## Task 11: Create `tests/test_base_dataset.py`

Replace `test_dataset.py` with a focused test for `BaseGraphDataset` only. `SyntheticGraphDataset` tests move to archive.

**Files:**
- Create: `tests/test_base_dataset.py`

- [ ] **Step 1: Write `tests/test_base_dataset.py`**

```python
"""
Tests for graph_diffusion.data.base_dataset
=============================================

Unit tests for BaseGraphDataset. Uses a minimal in-process concrete
subclass to verify the abstract contract without real data.
"""

import pytest
import torch
from context import graph_diffusion  # noqa: F401
from torch_geometric.data import Data

from graph_diffusion.data.base_dataset import BaseGraphDataset


class _MinimalDataset(BaseGraphDataset):
    """Minimal concrete subclass: returns a single fixed graph."""

    def _build_graphs(self) -> list[Data]:
        return [
            Data(
                x=torch.zeros(4, 1),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
            )
        ]


class TestBaseGraphDataset:
    def test_cannot_instantiate_abstract(self) -> None:
        """BaseGraphDataset is abstract — direct instantiation raises TypeError."""
        with pytest.raises(TypeError):
            BaseGraphDataset(root="/tmp/test_abstract")  # type: ignore[abstract]

    def test_concrete_subclass_length(self, tmp_path: pytest.TempPathFactory) -> None:
        """A concrete subclass with one graph has length 1."""
        ds = _MinimalDataset(root=str(tmp_path / "minimal"))
        assert len(ds) == 1

    def test_concrete_subclass_graph_attributes(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """The graph returned by _build_graphs is accessible via indexing."""
        ds = _MinimalDataset(root=str(tmp_path / "minimal"))
        graph = ds[0]
        assert graph.x.shape == (4, 1)

    def test_pre_transform_applied(self, tmp_path: pytest.TempPathFactory) -> None:
        """pre_transform is applied to each graph during processing."""
        from graph_diffusion.data.transforms import NormalizeNodeFeatures

        class _NonZeroDataset(BaseGraphDataset):
            def _build_graphs(self) -> list[Data]:
                return [
                    Data(
                        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
                        edge_index=torch.zeros(2, 0, dtype=torch.long),
                    )
                ]

        ds = _NonZeroDataset(
            root=str(tmp_path / "nonzero"),
            pre_transform=NormalizeNodeFeatures(),
        )
        graph = ds[0]
        mean = graph.x.mean(dim=0)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=0.15)
```

- [ ] **Step 2: Run the new test to confirm it passes**

```bash
uv run pytest tests/test_base_dataset.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_base_dataset.py
git commit -m "test(data): add test_base_dataset.py for BaseGraphDataset contract"
```

---

## Task 12: Create `tests/test_pOnEllipse.py`

Replace `test_ellipsedataset.py` with a version that references `pOnEllipseDataset` and removes the archived dataset tests.

**Files:**
- Create: `tests/test_pOnEllipse.py`

- [ ] **Step 1: Write `tests/test_pOnEllipse.py`**

This keeps: `DatasetUrl`, `DatasetDownloader`, `ComputeArcLengthEdgeFeatures`, and `pOnEllipseDataset` feature-mode tests. It removes: `EllipseDataset` tests and `EllipseConditionalDataset` tests.

```python
"""
Tests for graph_diffusion.data.pOnEllipse
==========================================

Unit tests for DatasetUrl, DatasetDownloader, pOnEllipseDataset, and
ComputeArcLengthEdgeFeatures.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from context import graph_diffusion  # noqa: F401

from graph_diffusion.data.pOnEllipse import (
    _H5_N_TIMESTEPS,
    DatasetDownloader,
    DatasetUrl,
    pOnEllipseDataset,
)
from graph_diffusion.data.transforms import (
    ComputeAngularEdgeFeatures,
    ComputeArcLengthEdgeFeatures,
)

# ---------------------------------------------------------------------------
# Helpers — fake HDF5 matching the real pOnEllipse layout
# ---------------------------------------------------------------------------

_N_SIMS = 6
_MAX_NODES = 8
_N_NODES_A = 6
_N_NODES_B = 8


def _fake_ellipse_nodes(
    n_nodes: int, re: float, aoa: float, rng: np.random.Generator
) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, n_nodes, endpoint=False)
    a = rng.uniform(1.0, 2.0)
    b = rng.uniform(0.5, 1.5)
    x = (a * np.cos(theta)).astype(np.float32)
    y = (b * np.sin(theta)).astype(np.float32)
    re_col = np.full(n_nodes, re, dtype=np.float32)
    aoa_col = np.full(n_nodes, aoa, dtype=np.float32)
    p = rng.standard_normal((n_nodes, _H5_N_TIMESTEPS)).astype(np.float32)
    return np.concatenate(
        [x[:, None], y[:, None], re_col[:, None], aoa_col[:, None], p], axis=1
    )


def _write_fake_h5(path: Path) -> None:
    import h5py

    rng = np.random.default_rng(42)
    n_feat = 4 + _H5_N_TIMESTEPS
    data_arr = np.full((_N_SIMS, _MAX_NODES, n_feat), np.nan, dtype=np.float32)
    for i in range(_N_SIMS):
        n_nodes = _N_NODES_A if i % 2 == 0 else _N_NODES_B
        rows = _fake_ellipse_nodes(n_nodes, re=500.0 + i, aoa=5.0 + 0.1 * i, rng=rng)
        data_arr[i, :n_nodes, :] = rows
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=data_arr)


def _make_dataset_root(tmpdir: str) -> tuple[Path, Path]:
    root = Path(tmpdir) / "dataset"
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True)
    _write_fake_h5(raw_dir / "pOnEllipseTrain.h5")
    return root, raw_dir  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# DatasetUrl
# ---------------------------------------------------------------------------


def test_dataset_url_contains_repo() -> None:
    for url in DatasetUrl:
        assert "mariolinov/Ellipse" in url.value
        assert url.value.startswith("https://")


def test_dataset_url_filenames() -> None:
    assert DatasetUrl.TRAIN_H5.value.endswith("pOnEllipseTrain.h5")
    assert DatasetUrl.TEST_H5.value.endswith("pOnEllipseTest.h5")
    assert DatasetUrl.TIME_TRAIN_NPY.value.endswith("TimeEllipseTrain.npy")


# ---------------------------------------------------------------------------
# DatasetDownloader
# ---------------------------------------------------------------------------


def test_downloader_local_path() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        dl = DatasetDownloader(root=tmpdir)
        p = dl.local_path(DatasetUrl.TRAIN_H5)
        assert p.name == "pOnEllipseTrain.h5"
        assert p.parent == Path(tmpdir)


def test_downloader_is_downloaded_false() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        dl = DatasetDownloader(root=tmpdir)
        assert not dl.is_downloaded(DatasetUrl.TRAIN_H5)


def test_downloader_is_downloaded_after_touch() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        dl = DatasetDownloader(root=tmpdir)
        dl.local_path(DatasetUrl.TRAIN_H5).touch()
        assert dl.is_downloaded(DatasetUrl.TRAIN_H5)


def test_downloader_creates_root_dir() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        nested = Path(tmpdir) / "a" / "b" / "c"
        DatasetDownloader(root=str(nested))
        assert nested.exists()


# ---------------------------------------------------------------------------
# ComputeArcLengthEdgeFeatures
# ---------------------------------------------------------------------------


def test_arc_length_features_shape() -> None:
    torch.manual_seed(0)
    from torch_geometric.data import Data

    N, E = 16, 32
    theta = torch.linspace(0, 2 * torch.pi, N + 1)[:-1]
    pos = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    edge_index = torch.stack([torch.randint(0, N, (E,)), torch.randint(0, N, (E,))])
    data = Data(pos=pos, edge_index=edge_index)
    out = ComputeArcLengthEdgeFeatures()(data)
    assert out.edge_attr.shape == (E, 2)


def test_arc_length_features_unit_values() -> None:
    torch.manual_seed(0)
    from torch_geometric.data import Data

    N = 32
    theta = torch.linspace(0, 2 * torch.pi, N + 1)[:-1]
    pos = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    src = torch.arange(N)
    dst = (src + 1) % N
    edge_index = torch.stack([src, dst])
    data = Data(pos=pos, edge_index=edge_index)
    out = ComputeArcLengthEdgeFeatures()(data)
    assert out.edge_attr[:, 0].abs().max() <= 1.0 + 1e-5
    assert out.edge_attr[:, 1].abs().max() <= 1.0 + 1e-5


def test_arc_length_raises_no_pos() -> None:
    from torch_geometric.data import Data

    data = Data(edge_index=torch.zeros(2, 4, dtype=torch.long))
    with pytest.raises(ValueError, match="pos"):
        ComputeArcLengthEdgeFeatures()(data)


def test_arc_length_raises_no_edge_index() -> None:
    from torch_geometric.data import Data

    data = Data(pos=torch.randn(8, 2))
    with pytest.raises(ValueError, match="edge_index"):
        ComputeArcLengthEdgeFeatures()(data)


# ---------------------------------------------------------------------------
# pOnEllipseDataset — validation
# ---------------------------------------------------------------------------


def test_invalid_feature_mode() -> None:
    with pytest.raises(ValueError, match="feature_mode"):
        pOnEllipseDataset(root="/tmp/dummy", feature_mode="polar")


def test_invalid_split() -> None:
    with pytest.raises(ValueError, match="split"):
        pOnEllipseDataset(root="/tmp/dummy", split="val")


def test_invalid_k_neighbors() -> None:
    with pytest.raises(ValueError, match="k_neighbors"):
        pOnEllipseDataset(root="/tmp/dummy", k_neighbors=0)


def test_normalised_requires_global_dim_2() -> None:
    with pytest.raises(ValueError, match="global_dim"):
        pOnEllipseDataset(root="/tmp/dummy", feature_mode="normalised", global_dim=1)


# ---------------------------------------------------------------------------
# pOnEllipseDataset — feature modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode,expected_x_dim",
    [
        ("radial", 1),
        ("radial_norm", 1),
        ("cartesian", 2),
        ("normalised", 1),
    ],
)
def test_feature_modes(mode: str, expected_x_dim: int) -> None:
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        root, _ = _make_dataset_root(tmpdir)
        pre_transform = (
            ComputeArcLengthEdgeFeatures()
            if mode == "cartesian"
            else ComputeAngularEdgeFeatures()
        )
        ds = pOnEllipseDataset(
            root=str(root),
            feature_mode=mode,
            n_samples=5,
            k_neighbors=2,
            global_dim=8,
            pre_transform=pre_transform,
        )
        graph = ds[0]
        assert graph.x.shape == (
            _N_NODES_A,
            expected_x_dim,
        ), f"mode={mode}: expected x dim {expected_x_dim}, got {graph.x.shape}"
        assert graph.edge_attr.shape[1] == 2
        assert graph.u.shape == (1, 8)


def test_normalised_stores_ab() -> None:
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        root, _ = _make_dataset_root(tmpdir)
        ds = pOnEllipseDataset(
            root=str(root),
            feature_mode="normalised",
            n_samples=5,
            k_neighbors=2,
            global_dim=8,
            pre_transform=ComputeAngularEdgeFeatures(),
        )
        graph = ds[0]
        assert graph.u[0, 0] > 0, "semi-axis a should be positive"
        assert graph.u[0, 1] > 0, "semi-axis b should be positive"
        assert graph.u[0, 2:].abs().sum() == 0, "remaining u dims should be zero"


def test_radial_norm_stores_r_scale_and_p_cond() -> None:
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        root, _ = _make_dataset_root(tmpdir)
        ds = pOnEllipseDataset(
            root=str(root),
            feature_mode="radial_norm",
            n_samples=5,
            k_neighbors=2,
            global_dim=8,
            pre_transform=ComputeAngularEdgeFeatures(),
        )
        graph = ds[0]
        assert graph.x.shape == (_N_NODES_A, 1)
        assert hasattr(graph, "r_scale") and graph.r_scale.item() > 0
        mean_r_norm = graph.x[:, 0].mean().item()
        assert abs(mean_r_norm - 1.0) < 1e-5
        assert hasattr(graph, "p_cond")
        assert graph.p_cond.shape == (_N_NODES_A, 2)
        norms = (graph.p_cond**2).sum(dim=-1).sqrt()
        assert torch.allclose(norms, torch.ones(_N_NODES_A), atol=1e-5)
```

- [ ] **Step 2: Run the new test to confirm it passes**

```bash
uv run pytest tests/test_pOnEllipse.py -v
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_pOnEllipse.py
git commit -m "test(data): add test_pOnEllipse.py for pOnEllipseDataset"
```

---

## Task 13: Delete obsolete source and test files

Now that the new files exist and their tests pass, remove the originals.

**Files:** deletions

- [ ] **Step 1: Delete obsolete source files**

`dataset.py` was already moved to archive in Task 3. Only `ellipsedataset.py` remains to remove:

```bash
git rm src/graph_diffusion/data/ellipsedataset.py
```

- [ ] **Step 2: Delete obsolete test files**

```bash
git rm tests/test_dataset.py
git rm tests/test_ellipsedataset.py
```

- [ ] **Step 3: Delete obsolete training script**

```bash
git rm train_ellipse.py
```

- [ ] **Step 4: Run the full test suite to confirm nothing is broken**

```bash
uv run pytest tests/ -v
```

Expected: all active tests pass. If any test imports from `dataset` or `ellipsedataset`, fix the import to use `base_dataset` or `pOnEllipse` respectively.

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "refactor(data): delete dataset.py, ellipsedataset.py, train_ellipse.py and obsolete tests"
```

---

## Task 14: Create `train.py`

Based on `train_ellipse.py` (now deleted), but with updated framing and simplified to `pOnEllipseDataset` only.

**Files:**
- Create: `train.py`

- [ ] **Step 1: Create `train.py`**

```python
"""
train.py — Aerodynamic boundary mesh shape generation
=======================================================

Train a DDPM over pOnEllipse boundary node positions so that sampling
produces novel, physically plausible surface mesh configurations.

The script loads a YAML config, instantiates pOnEllipseDataset, builds the
ScoreNetwork + GraphDiffusionModel pipeline, runs the training loop, and
saves a checkpoint plus generated-shape plots.

Supported experiments:
  EXP-010  radial shape generation
           (feature_mode=radial, dataset_type=shape)
  EXP-011  radial + cartesian shape ablation
           (feature_mode=radial or cartesian)
  EXP-012  radial_norm shape generation
           (feature_mode=radial_norm)

Usage:
    python train.py --config configs/EXP-010_ellipse_data_pipeline.yaml --epochs 200
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from graph_diffusion.building_blocks.feature_transforms import (
    FeatureTransform,
    LogitNormTransform,
)
from graph_diffusion.building_blocks.noise_schedule import NoiseSchedule
from graph_diffusion.data.dataloader import GraphDataLoader
from graph_diffusion.data.pOnEllipse import pOnEllipseDataset
from graph_diffusion.data.transforms import (
    ComputeAngularEdgeFeatures,
    ComputeArcLengthEdgeFeatures,
)
from graph_diffusion.model.graph_diffusion_model import GraphDiffusionModel
from graph_diffusion.model.score_network import ScoreNetwork


def load_config(path: str) -> dict:  # type: ignore[type-arg]
    """Load YAML configuration file."""
    with open(path) as f:  # noqa: PTH123
        return yaml.safe_load(f)


def _build_dataset(config: dict) -> pOnEllipseDataset:  # type: ignore[type-arg]
    """Instantiate pOnEllipseDataset from config."""
    ds_cfg = config.get("ellipse_dataset", {})
    feature_mode = ds_cfg.get("feature_mode", "radial")
    pre_transform = (
        ComputeArcLengthEdgeFeatures()
        if feature_mode == "cartesian"
        else ComputeAngularEdgeFeatures()
    )
    return pOnEllipseDataset(
        root=ds_cfg.get("root", "data/ellipse"),
        feature_mode=feature_mode,
        split=ds_cfg.get("split", "train"),
        n_samples=ds_cfg.get("n_samples", None),
        k_neighbors=ds_cfg.get("k_neighbors", 2),
        global_dim=ds_cfg.get("global_dim", 8),
        pre_transform=pre_transform,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train aerodynamic boundary mesh shape generator"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/EXP-010_ellipse_data_pipeline.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="Number of shapes to generate after training",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="generated_shapes.png",
        help="Path to save generated shapes plot",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device)

    # --- Feature transform (optional bounded diffusion) ---
    feature_transform: FeatureTransform | None = None
    ft_cfg = config.get("feature_transform")
    if ft_cfg and ft_cfg.get("type") == "logit_norm":
        feature_transform = LogitNormTransform(
            r_min=float(ft_cfg.get("r_min", 0.5)),
            r_max=float(ft_cfg.get("r_max", 1.5)),
        )
        print(
            f"Using LogitNormTransform(r_min={ft_cfg.get('r_min')}, "
            f"r_max={ft_cfg.get('r_max')})"
        )

    # --- Dataset ---
    dataset = _build_dataset(config)

    # --- Data Loader ---
    data_cfg = config.get("data", {})
    loader = GraphDataLoader(
        dataset,
        batch_size=data_cfg.get("batch_size", 32),
        val_split=data_cfg.get("val_split", 0.1),
        num_workers=data_cfg.get("num_workers", 0),
        shuffle=data_cfg.get("shuffle", True),
        seed=42,
    )

    # --- Noise schedule ---
    ns_cfg = config.get("noise_schedule", {})
    noise_schedule = NoiseSchedule(
        T=ns_cfg.get("T", 200),
        schedule_type=ns_cfg.get("schedule_type", "cosine"),
        beta_start=ns_cfg.get("beta_start", 1e-4),
        beta_end=ns_cfg.get("beta_end", 0.02),
    )

    # --- Score network ---
    sn_cfg = config.get("score_network", {})
    mlp_cfg = config.get("mlp", {})
    score_network = ScoreNetwork(
        node_dim=sn_cfg.get("node_dim", 32),
        edge_dim=sn_cfg.get("edge_dim", 2),
        global_dim=sn_cfg.get("global_dim", 8),
        time_embed_dim=sn_cfg.get("time_embed_dim", 64),
        n_layers=sn_cfg.get("n_layers", 4),
        hidden_dims=sn_cfg.get("hidden_dims", [64, 64]),
        activation=mlp_cfg.get("activation", "silu"),
        layer_norm=mlp_cfg.get("layer_norm", True),
        residual=mlp_cfg.get("residual", True),
        input_dim=sn_cfg.get("input_dim", None),
        cond_dim=sn_cfg.get("cond_dim", None),
        output_dim=sn_cfg.get("output_dim", None),
    )

    # --- Diffusion model ---
    model_cfg = config.get("model", {})
    model = GraphDiffusionModel(
        score_network=score_network,
        noise_schedule=noise_schedule,
        feature_transform=feature_transform,
        n_noise_channels=model_cfg.get("n_noise_channels", None),
        smoothness_weight=float(model_cfg.get("smoothness_weight", 0.0)),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Training on: {device}")
    print(f"Dataset: {len(dataset)} graphs, {args.epochs} epochs")

    clamp_cfg = config.get("clamp_range")
    clamp_range: tuple[float, float] | None = None
    if clamp_cfg is not None:
        clamp_range = (float(clamp_cfg[0]), float(clamp_cfg[1]))

    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    training_cfg = config.get("training", {})

    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    if training_cfg.get("scheduler") == "cosine_annealing":
        eta_min = float(training_cfg.get("eta_min", 1e-5))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=eta_min
        )

    early_stopping_patience: int | None = training_cfg.get("early_stopping_patience")
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    loss_log: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(
            loader.train_loader(),
            desc=f"Epoch {epoch:3d}/{args.epochs}",
            leave=False,
        )
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = model.compute_loss(batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(n_batches, 1)
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in loader.val_loader():
                batch = batch.to(device)
                loss = model.compute_loss(batch)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        loss_log.append(
            {"epoch": epoch, "train_loss": avg_loss, "val_loss": avg_val_loss}
        )
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={avg_loss:.4f}  val_loss={avg_val_loss:.4f}"
        )

        if scheduler is not None:
            scheduler.step()

        if early_stopping_patience is not None:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                patience_counter = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config,
                        "epoch": epoch,
                        "lr": args.lr,
                    },
                    output_dir / "checkpoint_best.pt",
                )
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(
                        f"Early stopping at epoch {epoch} "
                        f"(best val_loss={best_val_loss:.4f} at epoch {best_epoch})"
                    )
                    break

    writer.close()
    print("Training complete.")

    checkpoint_path = output_dir / "checkpoint.pt"
    epochs_run = loss_log[-1]["epoch"] if loss_log else 0
    if (
        early_stopping_patience is not None
        and (output_dir / "checkpoint_best.pt").exists()
    ):
        import shutil
        shutil.copy(output_dir / "checkpoint_best.pt", checkpoint_path)
        print(f"Saved best checkpoint (epoch {best_epoch}) to {checkpoint_path}")
    else:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "epoch": epochs_run,
                "lr": args.lr,
            },
            checkpoint_path,
        )

    with open(output_dir / "loss_log.json", "w") as f:
        json.dump(loss_log, f, indent=2)
    print(f"Saved loss log to {output_dir / 'loss_log.json'}")

    # --- Generate and plot sample shapes ---
    model.eval()
    ds_cfg = config.get("ellipse_dataset", {})
    feature_mode = ds_cfg.get("feature_mode", "radial")
    template = dataset[0].to(device)

    fig, axes = plt.subplots(1, args.n_samples, figsize=(4 * args.n_samples, 4))
    if args.n_samples == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        torch.manual_seed(i)
        result = model.sample(template, clamp_range=clamp_range)

        if feature_mode in ("radial", "radial_norm", "normalised"):
            r_raw = result.x[:, 0].cpu().numpy()
            theta = np.arctan2(
                template.pos[:, 1].cpu().numpy(),
                template.pos[:, 0].cpu().numpy(),
            )
            order = np.argsort(theta)
            theta_s = theta[order]
            r_scale = (
                template.r_scale.item() if hasattr(template, "r_scale") else 1.0
            )
            r_s = r_raw[order] * r_scale
            xc = np.append(r_s * np.cos(theta_s), r_s[0] * np.cos(theta_s[0]))
            yc = np.append(r_s * np.sin(theta_s), r_s[0] * np.sin(theta_s[0]))
            r_ref = template.x[:, 0].cpu().numpy()[order] * r_scale
            xr = np.append(r_ref * np.cos(theta_s), r_ref[0] * np.cos(theta_s[0]))
            yr = np.append(r_ref * np.sin(theta_s), r_ref[0] * np.sin(theta_s[0]))
            ax.plot(xr, yr, color="0.7", linewidth=1.0, label="template")
            ax.plot(xc, yc, "b-", linewidth=1.5, label="generated")
            ax.set_aspect("equal")
            ax.legend(fontsize=7, loc="upper right")
        else:  # cartesian
            xc_pred = result.x[:, 0].cpu().numpy()
            yc_pred = result.x[:, 1].cpu().numpy()
            xc_pred = np.append(xc_pred, xc_pred[0])
            yc_pred = np.append(yc_pred, yc_pred[0])
            xt = template.x[:, 0].cpu().numpy()
            yt = template.x[:, 1].cpu().numpy()
            xt = np.append(xt, xt[0])
            yt = np.append(yt, yt[0])
            ax.plot(xt, yt, color="0.7", linewidth=1.0, label="template")
            ax.plot(xc_pred, yc_pred, "b-", linewidth=1.5, label="generated")
            ax.set_aspect("equal")
            ax.legend(fontsize=7, loc="upper right")

        ax.set_title(f"Sample {i + 1}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved generated shapes to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add train.py
git commit -m "feat(train): add train.py — canonical aerodynamic shape generation entrypoint"
```

---

## Task 15: Update `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the Commands section**

Replace the train command line:
```
# Train circle experiment
python train_circle.py --config configs/circle_radial.yaml --epochs 100 --device cuda --output outputs/run/generated_shapes.png
```
with:
```
# Train aerodynamic shape generation
python train.py --config configs/EXP-010_ellipse_data_pipeline.yaml --epochs 200 --device cuda --output outputs/EXP-010/generated_shapes.png
```

Replace the postprocess line:
```
python scripts/postprocess_circle.py --checkpoint outputs/<run>/checkpoint.pt --config configs/circle_radial.yaml
```
with:
```
python scripts/postprocess_ellipse.py --checkpoint outputs/<run>/checkpoint.pt --config configs/EXP-010_ellipse_data_pipeline.yaml
```

- [ ] **Step 2: Update the Architecture / `data/` section**

Replace the entire `data/` bullet list under `### 'data/'` with:

```markdown
- **`BaseGraphDataset`** (`InMemoryDataset`, `base_dataset.py`) — abstract base; subclasses implement `_build_graphs() -> list[Data]`.
- **`pOnEllipseDataset`** (`pOnEllipse.py`) — the canonical aerodynamic dataset: loads the pOnEllipse HDF5 (HuggingFace `mariolinov/Ellipse`), builds surface boundary node graphs with bidirectional ring edges, and supports four coordinate representations via `feature_mode` (`"radial"` | `"radial_norm"` | `"cartesian"` | `"normalised"`). Node feature `x` is the quantity the diffusion model denoises; `pos` drives the edge-feature transforms.
- **`DatasetDownloader`** — streams HuggingFace files with a progress bar, caches locally.
- **`GraphDataLoader`** — wraps PyG `DataLoader`; provides `train_loader()` / `val_loader()` with reproducible `random_split`.
- **Transforms** — `ComputeAngularEdgeFeatures`, `ComputeArcLengthEdgeFeatures`, `NormalizeNodeFeatures`, `AddSelfLoops`, `KNNGraph`, `Compose`. Applied as `pre_transform` in dataset construction.
```

- [ ] **Step 3: Update the Data flow section**

Replace:
```
UnitCircleDataset → ComputeAngularEdgeFeatures (pre_transform)
```
with:
```
pOnEllipseDataset → ComputeAngularEdgeFeatures | ComputeArcLengthEdgeFeatures (pre_transform)
```

- [ ] **Step 4: Update the Experiments & outputs section**

Replace the roadmap line:
```
Current roadmap: EXP-00x circle ablation series complete (EXP-001 → EXP-006) → EXP-01x aerodynamic series (pOnEllipse dataset, HuggingFace `mariolinov/Ellipse`).
```
with:
```
Current roadmap: EXP-01x aerodynamic shape generation series active. EXP-00x circle ablation series complete and archived under `archive/circle/`.
```

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(CLAUDE.md): update architecture and commands for aerodynamic refactor"
```

---

## Task 16: Final quality gate

- [ ] **Step 1: Run the full test suite**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass with no failures. If any test references an archived class, fix the import.

- [ ] **Step 2: Run lint and type check**

```bash
uv run ruff check src/ tests/ && uv run black --check src/ tests/ && uv run mypy src/
```

Expected: no errors. Fix any that arise (typically stale imports or missing type annotations).

- [ ] **Step 3: Confirm no references to archived classes remain in `src/`**

```bash
grep -r "EllipseShapeDataset\|EllipseDataset\|EllipseConditionalDataset\|UnitCircleDataset\|SyntheticGraphDataset\|FourierScoreNetwork" src/ tests/ train.py
```

Expected: no output. If any are found, update the import to the new name or remove.

- [ ] **Step 4: Smoke-test train.py dry run (no real data needed)**

```bash
python train.py --help
```

Expected: argument help printed without errors.

- [ ] **Step 5: Final commit**

```bash
git add -u
git commit -m "chore: final cleanup after aerodynamic refactor — all tests pass, quality gate clean"
```
