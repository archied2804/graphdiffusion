"""
graph_diffusion.data.dataloader
================================

Graph data loader with train/validation splitting, wrapping
``torch_geometric.loader.DataLoader``.  Supports optional
``DistributedSampler`` for multi-GPU (DDP) training.
"""

import torch
import torch.distributed as dist
import torch_geometric.loader
from torch.utils.data import Subset, random_split
from torch.utils.data.distributed import DistributedSampler

from graph_diffusion.data.base_dataset import BaseGraphDataset

__all__ = [
    "GraphDataLoader",
]


class GraphDataLoader:
    """Wraps ``torch_geometric.loader.DataLoader`` with split and sensible defaults.

    Splits the dataset into training and validation subsets using a
    reproducible random split, then exposes ``DataLoader`` instances for each.

    When ``distributed=True``, a ``DistributedSampler`` is attached to
    each split so that every DDP rank receives a disjoint shard of the
    data.  Call :meth:`set_epoch` at the start of every epoch to ensure
    proper shuffling across ranks.

    Args:
        dataset (BaseGraphDataset): The graph dataset to load.
        batch_size (int): Number of graphs per mini-batch *per rank*.
            Defaults to ``32``.
        val_split (float): Fraction of the dataset to reserve for validation.
            Must be in ``(0, 1)``. Defaults to ``0.1``.
        shuffle (bool): Whether to shuffle the training set each epoch.
            Defaults to ``True``.
        num_workers (int): Number of data-loading worker processes.
            Defaults to ``0``.
        seed (int): Random seed for the train/val split.
            Defaults to ``42``.
        distributed (bool): If ``True``, wrap each split with a
            ``DistributedSampler``.  Requires that
            ``torch.distributed`` has already been initialised.
            Defaults to ``False``.

    Raises:
        ValueError: If ``batch_size < 1``.
        ValueError: If ``val_split`` is not in ``(0, 1)``.
        ValueError: If ``num_workers < 0``.
        RuntimeError: If ``distributed=True`` but the default process
            group has not been initialised.
    """

    def __init__(
        self,
        dataset: BaseGraphDataset,
        batch_size: int = 32,
        val_split: float = 0.1,
        shuffle: bool = True,
        num_workers: int = 0,
        seed: int = 42,
        distributed: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if not (0.0 < val_split < 1.0):
            raise ValueError(f"val_split must be in (0, 1), got {val_split}")
        if num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {num_workers}")
        if distributed and not dist.is_initialized():
            raise RuntimeError(
                "distributed=True requires torch.distributed to be initialised "
                "before constructing GraphDataLoader"
            )

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.distributed = distributed

        n_total = len(dataset)
        n_val = max(1, int(n_total * val_split))
        n_train = n_total - n_val

        generator = torch.Generator().manual_seed(seed)
        splits: list[Subset[BaseGraphDataset]] = random_split(
            dataset, [n_train, n_val], generator=generator
        )
        self._train_dataset = splits[0]
        self._val_dataset = splits[1]

        # Build distributed samplers when requested
        self._train_sampler: DistributedSampler[Subset[BaseGraphDataset]] | None = None
        self._val_sampler: DistributedSampler[Subset[BaseGraphDataset]] | None = None
        if distributed:
            self._train_sampler = DistributedSampler(
                self._train_dataset, shuffle=shuffle, seed=seed
            )
            self._val_sampler = DistributedSampler(self._val_dataset, shuffle=False)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch on distributed samplers for proper shuffling.

        Must be called at the start of each training epoch when
        ``distributed=True``.  No-op otherwise.

        Args:
            epoch (int): Current epoch number.
        """
        if self._train_sampler is not None:
            self._train_sampler.set_epoch(epoch)
        if self._val_sampler is not None:
            self._val_sampler.set_epoch(epoch)

    def train_loader(self) -> torch_geometric.loader.DataLoader:
        """Returns a DataLoader over the training split.

        Returns:
            torch_geometric.loader.DataLoader: Training data loader.
        """
        if self._train_sampler is not None:
            return torch_geometric.loader.DataLoader(
                self._train_dataset,
                batch_size=self.batch_size,
                sampler=self._train_sampler,
                num_workers=self.num_workers,
            )
        return torch_geometric.loader.DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def val_loader(self) -> torch_geometric.loader.DataLoader:
        """Returns a DataLoader over the validation split.

        Returns:
            torch_geometric.loader.DataLoader: Validation data loader.
        """
        if self._val_sampler is not None:
            return torch_geometric.loader.DataLoader(
                self._val_dataset,
                batch_size=self.batch_size,
                sampler=self._val_sampler,
                num_workers=self.num_workers,
            )
        return torch_geometric.loader.DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
