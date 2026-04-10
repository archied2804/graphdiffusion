# GNN Tutorial – AIRTUK

> **Graph Neural Networks for Flow Field Reconstruction**
> A hands-on tutorial developed for the [UK Turbulence Consortium (AIRTUK)](https://www.ukturbulence.co.uk/).

---

### Authors

- Francesca Cooke (CEO)

## Overview

This tutorial introduces **Graph Neural Networks (GNNs)** as fast surrogate models for Computational Fluid Dynamics (CFD). Rather than solving the governing PDEs numerically at every time step, a GNN is trained to approximate the solution update directly from the mesh — dramatically reducing inference cost while preserving spatial fidelity on unstructured grids.

The tutorial is structured as a Jupyter notebook (`GNNTutorialAIRTUK.ipynb`) and relies on the accompanying **GNN4CFD** library.

---

## Learning Objectives

1. Motivate GNNs for CFD by contrasting them with CNNs and MLPs, and understand why the graph is a natural representation of an unstructured mesh.
2. Load and inspect a CFD dataset (*NsCircle*), understand its physical meaning, and construct a graph from the raw mesh using *k*-nearest-neighbour connectivity.
3. Build and train a **single-scale GNN** surrogate model and evaluate its rollout accuracy using the coefficient of determination R².
4. Extend the architecture to a **multi-scale MuS-GNN**, understand the DownMP/UpMP mechanism, and compare performance against the single-scale baseline.

---

## Background

### Why GNNs for CFD?

Standard deep learning surrogates (CNNs) require data on a regular Cartesian grid — a fundamental mismatch for CFD, where real geometries require unstructured meshes with variable local resolution. GNNs resolve this by treating the mesh *as-is*:

| Mesh concept      | Graph concept      |
|-------------------|--------------------|
| Mesh vertex       | Node vᵢ            |
| Mesh connectivity | Directed edge eᵢⱼ  |
| Physical fields (u, v, p) | Node features fᵢ |
| Relative node positions | Edge features rᵢⱼ |

Key inductive biases exploited: **relational locality** and **permutation invariance**.

### The GNN as a Learned Time-Stepper

The flow fields are governed by the incompressible Navier–Stokes equations. The GNN learns to approximate the time integral at each mesh node and advances the solution via a residual update:

```
u(t₀+Δt, xᵢ) ← u(t₀, xᵢ) + Iᵢ(t₀ → t₀+Δt, F(u, Re))
```

This update is applied **autoregressively** to produce a temporal rollout of arbitrary length.

---

## Dataset – NsCircle

The dataset is generated with [Nektar++](https://www.nektar.info/) for **2D incompressible Navier–Stokes flow past circular cylinders**.

| Parameter               | Value / Range                          |
|-------------------------|----------------------------------------|
| Cylinder diameter D     | 1 (normalised)                         |
| Centre-to-centre spacing H | Randomly sampled ∈ [4D, 6D]        |
| Reynolds number Re      | 500–1000 (laminar shedding)            |
| Training simulations    | 1 000 independent cases                |
| Nodes per mesh          | ~7 000                                 |
| Time step Δt            | 0.1                                    |
| Snapshots per simulation| 100 (after periodic steady-state)      |

**State variables per node:** u (streamwise velocity), v (cross-stream velocity), p (pressure). Re is broadcast as an additional node-level scalar.

**Dataset links:**
- Primary: https://zenodo.org/records/7870707
- Secondary (fallback): https://huggingface.co/datasets/mariolino/Ellipse/tree/main

---

## Repository Structure

```
.
├── GNN4CFD/
│   ├── cfddataset.py          # Dataset loaders (NsEllipse, etc.)
│   ├── dataloader.py          # Custom DataLoader
│   ├── graph.py               # Graph data structure utilities
│   ├── metrics.py             # Evaluation metrics (R²)
│   ├── model.py               # High-level model interface
│   ├── nn/
│   │   ├── blocks.py          # GNBlock: edge MLP → aggregate → node MLP
│   │   ├── GNNArchitect.py    # GNNArchBuilder & MultiScaleGNNBuilder
│   │   ├── losses.py          # GraphLoss (L2 + λ·L1 with Dirichlet weighting)
│   │   ├── model.py           # Training loop internals
│   │   └── UniversalGNN.py    # Wraps architecture dict into trainable module
│   └── transforms/
│       ├── connect.py         # ConnectKNN – builds k-NN graph edges
│       ├── geometric.py       # Geometric augmentations
│       ├── interpolate.py     # Inter-scale interpolation
│       ├── noise.py           # AddUniformNoise
│       ├── scale.py           # ScaleNs, ScaleEdgeAttr
│       └── subset.py          # GridClustering (multi-scale hierarchy)
├── GNNTutorialAIRTUK.ipynb    # Main tutorial notebook
├── main.py                    # Script-based training entry point
├── pyproject.toml             # Project metadata and dependencies
├── uv.lock                    # Locked dependency versions
└── README.md
```

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.10+ | Tested with 3.13.9 |
| PyTorch | https://pytorch.org |
| PyTorch Geometric (PyG) | https://pytorch-geometric.readthedocs.io/en/latest/ |
| CUDA-capable GPU | Strongly recommended; CPU fallback supported |
| `uv` (optional) | Fast dependency manager; `uv.lock` provided |

---

## Installation

### Using `uv` (recommended)

```bash
git clone <repo-url>
cd <repo>
uv sync
```

### Using pip

```bash
pip install torch torchvision
pip install torch-geometric
pip install -e .
```

---

## Running the Tutorial

1. **Download the dataset** from one of the links above and note the path.
2. **Start Jupyter** and open the notebook:
   ```bash
   jupyter notebook GNNTutorialAIRTUK.ipynb
   ```
3. Update the `pathtodata` variable in the notebook to point to your local dataset.
4. Follow the cells in order — each section builds on the previous one.

---

## Tutorial Sections

| Section | Description |
|---------|-------------|
| **1. Theory** | Why GNNs for CFD? Inductive biases, comparison with CNNs/MLPs, GNN as a learned time-stepper |
| **2. Data Representation** | NsCircle dataset, graph construction via k-NN (k=6), feature normalisation |
| **3. Visualisation** | Loading and plotting u, v, p fields and mesh node positions |
| **4. Network Architecture** | Encode-Process-Decode workflow, GN Block, MLP structure, multi-scale (MuS-GNN) architecture |
| **5. Single-Scale GNN** | Architecture definition, training configuration, data loading, training loop, post-processing |
| **6. Multi-Scale GNN** | Architecture construction (3 scales), DownMP/UpMP mechanism, GridClustering transforms, training and evaluation |
| **7. Extensions** | Suggested experiments: optimal number of scales, model complexity scaling, different loss functions, other datasets |

---

## Network Architecture

### Single-Scale GNN

A standard **Encode-Process-Decode** GNN:

```
Input node features  →  Node MLP Encoder  →  latent v (size: hidden_dim)
Input edge features  →  Edge MLP Encoder  →  latent e (size: hidden_dim)
                          ↓
                     M × GN Block
                     (Edge update → Aggregate → Node update)
                          ↓
                     Node MLP Decoder  →  predicted Δu
```

Default hyperparameters:

| Parameter       | Value |
|-----------------|-------|
| `num_mp_blocks` | 8     |
| `hidden_dim`    | 128   |
| `edge_encoder_in` | 2  |
| `node_encoder_in` | 5  |
| `decoder_out`   | 3     |
| `mlp_depth`     | 3     |

### Multi-Scale GNN (MuS-GNN)

Extends the single-scale architecture with **DownMP** (fine → coarse) and **UpMP** (coarse → fine) message-passing across 3 graph scales:

```
G_fine   (~7k nodes):   MP → DownMP → UpMP → MP → Decode
G_coarse (~1.7k nodes):        MP → DownMP → UpMP
G_coarsest (~400 nodes):              MP
```

Grid resolutions: `[0.15, 0.30, 0.60]`

---

## Training Configuration

```python
gfd.nn.TrainConfig(
    epochs        = ...,
    num_steps     = [i for i in range(1, 11)],   # curriculum rollout
    add_steps     = {"tolerance": 0.005, "loss": "training"},
    batch_size    = 1,
    lr            = 1e-5,
    grad_clip     = {"epoch": 0, "limit": 1},
    scheduler     = {"factor": 0.5, "patience": 5, "loss": "training"},
    stopping      = 1e-8,
    mixed_precision = True,
)
```

### Loss Function

```
L = Σᵢ ‖uᵢ - ûᵢ‖² + λ_d · Σᵢ ‖uᵢ - ûᵢ‖₁
```

where λ_d = 0.25 applies an additional L1 penalty on Dirichlet boundary nodes (cylinder walls) to improve near-wall accuracy.

### Multi-Step Rollout Curriculum

Training begins on single-step predictions; once the loss drops below `tolerance=0.005`, the rollout is extended by one step (up to 10 steps). This prevents error accumulation during autoregressive inference.

---

## Evaluation

Model accuracy is reported using the **coefficient of determination R²**, computed node-wise over the full rollout:

```
R² = 1 - [ Σᵢ,ₜ (uᵢₜ - ûᵢₜ)² ] / [ Σᵢ,ₜ (uᵢₜ - ū)² ]
```

R² is computed separately for each field (u, v, p). Pressure typically shows a lower R² than velocity due to global divergence-free constraints.

---

## References

- Battaglia et al. (2018). *Relational inductive biases, deep learning, and graph networks.* https://arxiv.org/abs/1806.01261
- Lino et al. (2022). *Multi-scale rotation-equivariant graph neural networks for unsteady Eulerian fluid dynamics.* https://pubs.aip.org/aip/pof/article/34/8/087850

**Code adapted from:**
- https://github.com/archied2804/graphs4cfd/cardiac (fork)
- https://github.com/mario-linov/graphs4cfd (original, Mario Lino)

---

## Acknowledgements

Developed for the **UK Turbulence Consortium (AIRTUK)**. Dataset generated using [Nektar++](https://www.nektar.info/).
