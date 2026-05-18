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
