"""
graph_diffusion.data
====================

Data loading, dataset definitions, and graph transforms.
"""

from graph_diffusion.data.circledataset import UnitCircleDataset
from graph_diffusion.data.dataloader import GraphDataLoader
from graph_diffusion.data.dataset import BaseGraphDataset, SyntheticGraphDataset
from graph_diffusion.data.ellipsedataset import (
    DatasetDownloader,
    DatasetUrl,
    EllipseConditionalDataset,
    EllipseDataset,
    EllipseShapeDataset,
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
    "SyntheticGraphDataset",
    "UnitCircleDataset",
    "GraphDataLoader",
    "BaseTransform",
    "NormalizeNodeFeatures",
    "AddSelfLoops",
    "KNNGraph",
    "Compose",
    "ComputeAngularEdgeFeatures",
    "ComputeArcLengthEdgeFeatures",
    "DatasetUrl",
    "DatasetDownloader",
    "EllipseDataset",
    "EllipseShapeDataset",
    "EllipseConditionalDataset",
]
