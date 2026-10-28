---
tags: [architecture, classes, mermaid]
---

# Architecture Overview

This document describes the full class hierarchy of the `graph_diffusion`
package.  All diagrams use Mermaid and render natively in Obsidian.

> **Vault tip:** Enable the *Mermaid* core plugin in Obsidian settings
> (Settings → Core plugins → toggle on) to render diagrams inline.

---

## Package dependency graph

The sub-packages form a strict DAG — no circular dependencies allowed.

```mermaid
graph TD
    subgraph graph_diffusion
        data["data/"]
        bb["building_blocks/"]
        model["model/"]
        viz["visualisation/"]
    end

    model -->|imports| bb
    model -.->|"uses at runtime<br/>(Data objects)"| data
    viz -->|"reads raw HDF5<br/>(pOnEllipse)"| data
    data -.x bb
    data -.x model
    bb -.x data
    viz -.x model
    viz -.x bb

    style data fill:#4a9eff,stroke:#333,color:#fff
    style bb fill:#ff9f43,stroke:#333,color:#fff
    style model fill:#ee5a24,stroke:#333,color:#fff
    style viz fill:#10ac84,stroke:#333,color:#fff
```

**Rule:** `data` and `building_blocks` have **no** upstream dependencies.
`model` depends on `building_blocks` only. `visualisation` is read-only and depends on
`data` (the pOnEllipse raw arrays) only — never on `model/` or `building_blocks/`.
Data objects (`torch_geometric.data.Data`) flow through the system without creating
import dependencies.

---

## Full class diagram

```mermaid
classDiagram
    direction TB

    %% ─── data/ ───
    class BaseGraphDataset {
        <<abstract>>
        +root: str
        +transform: Callable | None
        +pre_transform: Callable | None
        +_build_graphs()* list~Data~
        +download() None
        +process() None
    }

    class SyntheticGraphDataset {
        +n_graphs: int
        +n_nodes_range: tuple
        +node_feature_dim: int
        +k: int
        +global_dim: int
        +seed: int
        +_build_graphs() list~Data~
    }

    class UnitCircleDataset {
        +n_graphs: int
        +n_nodes: int
        +n_fourier_modes: int
        +amplitude_scale: float
        +r_min: float
        +r_max: float
        +k_neighbors: int
        +global_dim: int
        +seed: int
        +_build_graphs() list~Data~
        -_build_ring_edge_index() Tensor
    }

    class GraphDataLoader {
        +batch_size: int
        +shuffle: bool
        +num_workers: int
        +train_loader() DataLoader
        +val_loader() DataLoader
    }

    class BaseTransform {
        <<abstract>>
        +forward(data: Data)* Data
    }

    class NormalizeNodeFeatures {
        +eps: float
        +forward(data: Data) Data
    }

    class AddSelfLoops {
        +fill_value: float
        +forward(data: Data) Data
    }

    class KNNGraph {
        +k: int
        +loop: bool
        +forward(data: Data) Data
    }

    class Compose {
        +transforms: list~BaseTransform~
        +forward(data: Data) Data
    }

    class ComputeAngularEdgeFeatures {
        +forward(data: Data) Data
    }

    BaseGraphDataset <|-- SyntheticGraphDataset
    BaseGraphDataset <|-- UnitCircleDataset
    BaseTransform <|-- NormalizeNodeFeatures
    BaseTransform <|-- AddSelfLoops
    BaseTransform <|-- KNNGraph
    BaseTransform <|-- Compose
    BaseTransform <|-- ComputeAngularEdgeFeatures
    GraphDataLoader --> BaseGraphDataset : wraps

    %% ─── building_blocks/ ───
    class MLP {
        +in_dim: int
        +hidden_dims: list~int~
        +out_dim: int
        +activation: str
        +layer_norm: bool
        +residual: bool
        +forward(x: Tensor) Tensor
    }

    class SinusoidalTimeEmbedding {
        +embed_dim: int
        +max_period: int
        +mlp: MLP
        +forward(t: Tensor) Tensor
    }

    class EdgeModel {
        +mlp: MLP
        +forward(src, dest, edge_attr, u, batch) Tensor
    }

    class NodeModel {
        +mlp: MLP
        +forward(x, edge_index, edge_attr, u, batch) Tensor
    }

    class GlobalModel {
        +mlp: MLP
        +forward(x, edge_index, edge_attr, u, batch) Tensor
    }

    class GraphNetworkBlock {
        +edge_model: EdgeModel
        +node_model: NodeModel
        +global_model: GlobalModel
        +forward(x, edge_index, edge_attr, u, batch) tuple
    }

    class NoiseSchedule {
        +T: int
        +betas: Tensor
        +alphas: Tensor
        +alphas_cumprod: Tensor
        +sqrt_alphas_cumprod: Tensor
        +sqrt_one_minus_alphas_cumprod: Tensor
        +get_t(t: Tensor, buffer_name: str) Tensor
    }

    SinusoidalTimeEmbedding --> MLP : contains
    EdgeModel --> MLP : contains
    NodeModel --> MLP : contains
    GlobalModel --> MLP : contains
    GraphNetworkBlock --> EdgeModel : contains
    GraphNetworkBlock --> NodeModel : contains
    GraphNetworkBlock --> GlobalModel : contains

    %% ─── model/ ───
    class ScoreNetwork {
        +node_dim: int
        +edge_dim: int
        +global_dim: int
        +input_dim: int | None
        +input_proj: Linear | None
        +output_decode: Linear | None
        +time_embedding: SinusoidalTimeEmbedding
        +time_proj: Linear
        +gn_layers: ModuleList
        +output_proj: MLP
        +forward(data: Data, t: Tensor) Tensor
    }

    class GraphDiffusionModel {
        +score_network: ScoreNetwork
        +noise_schedule: NoiseSchedule
        +forward_diffusion(x_0, t, batch) tuple
        +compute_loss(batch: Data) Tensor
        +sample(graph_template, n_steps, sampler, clamp_range) Data
    }

    ScoreNetwork --> SinusoidalTimeEmbedding : contains
    ScoreNetwork --> GraphNetworkBlock : contains N×
    ScoreNetwork --> MLP : output_proj
    GraphDiffusionModel --> ScoreNetwork : contains
    GraphDiffusionModel --> NoiseSchedule : contains
```

---

## Sub-package details

### `data/` — datasets & transforms

```mermaid
classDiagram
    direction LR

    class BaseGraphDataset {
        <<abstract, InMemoryDataset>>
        +_build_graphs()* list~Data~
    }

    class SyntheticGraphDataset {
        n_graphs, n_nodes_range
        node_feature_dim, k
        global_dim, seed
    }

    class UnitCircleDataset {
        n_graphs, n_nodes
        n_fourier_modes
        amplitude_scale
        r_min, r_max
        k_neighbors
    }

    BaseGraphDataset <|-- SyntheticGraphDataset
    BaseGraphDataset <|-- UnitCircleDataset

    note for BaseGraphDataset "Inherits torch_geometric\nInMemoryDataset + abc.ABC"
```

### `building_blocks/` — reusable components

```mermaid
flowchart TD
    subgraph GraphNetworkBlock
        direction LR
        E["EdgeModel<br/>φ_e: [e‖v_s‖v_r‖u] → e'"]
        N["NodeModel<br/>φ_v: [v‖ē‖u] → v'"]
        G["GlobalModel<br/>φ_u: [ē‖v̄‖u] → u'"]
        E --> N --> G
    end

    subgraph Internal
        MLP1["MLP"]
        MLP2["MLP"]
        MLP3["MLP"]
    end

    E --> MLP1
    N --> MLP2
    G --> MLP3

    TE["SinusoidalTimeEmbedding<br/>t → sin/cos → MLP"]
    NS["NoiseSchedule<br/>β, α, ᾱ buffers"]
```

### `model/` — diffusion pipeline

```mermaid
flowchart LR
    subgraph GraphDiffusionModel
        direction TB
        FD["forward_diffusion()<br/>x₀, t → x_t, ε"]
        CL["compute_loss()<br/>batch → MSE(ε̂, ε)"]
        SA["sample()<br/>x_T → x_{T-1} → … → x₀"]
    end

    subgraph ScoreNetwork
        direction TB
        IP["input_proj<br/>(optional)"]
        TE["time_embed<br/>sinusoidal"]
        GN["N × GraphNetworkBlock"]
        OP["output_proj<br/>MLP"]
        OD["output_decode<br/>(optional)"]
        IP --> GN --> OP --> OD
        TE --> GN
    end

    GraphDiffusionModel --> ScoreNetwork
    GraphDiffusionModel --> NS["NoiseSchedule"]
```

### `visualisation/` — dataset overview suite (read-only)

`dataset_overview.py` is a read-only analysis module operating on the raw pOnEllipse
HDF5 arrays (`[x, y, Re, H, p0..p100]`). It imports nothing from `model/` or
`building_blocks/`, preserving the one-way DAG.

- **`pressure_drag_coefficient(polygon_xy, cp)`** — discrete line integral
  `Cd_p = ∮ Cp (n·x̂) ds`. Pressure drag only (the dataset is pressure-only; no skin
  friction).
- **`sim_summary(raw, idx)`** — per-sim accessor returning `Re`, `H`, semi-axes `a`/`b`,
  the ordered boundary polygon, `cp_mean`, `x/c`, the `Cp(x/c)` curve, and `Cd_p`.
- **Figure builders** — `plot_design_space_drag` (Re×b hexbin coloured by `Cd_p`),
  `plot_cp_envelope` (all `Cp(x/c)` curves + mean±std, coloured by Re or b),
  `plot_design_space_coverage` (Re×b scatter, eval samples highlighted),
  `plot_shape_gallery`, `plot_drag_vs_re` (binned by thickness), and
  `plot_cp_pca` (2-D PCA of the `Cp` curves). All re-exported from
  `graph_diffusion.visualisation`.

The CLI `scripts/plot_dataset_overview.py` renders all 8 figure variants to PNG+PDF
under `outputs/dataset_overview/` (gitignored).

### `data/design_space.py` — systematic (Re, H) sampling + case naming

A small helper module (depends only on the raw pOnEllipse arrays):

- **`make_case_name(re, h, idx)`** → a sortable case name
  `re{RRRR}_h{HH}_idx{NNNN}` (e.g. `re0500_h53_idx0000`), shared by evaluation and CFD
  cases so they live in one naming namespace.
- **`sample_design_space(raw, n_re, n_h)`** — lays an (Re, H) grid, snaps each node to
  the nearest real dataset sim, and guarantees the corners. Sampling axes are **Re and H
  only**; thickness `b` is emergent (recorded but not sampled).

`scripts/export_ellipse_for_openlb.py` exposes this via a `--grid` mode
(`--n-re`, `--n-h`) and writes a `name`-first manifest column
(`name,idx,Re,H,a,b,n_nodes`). It also calls `verify_manifest_row` to assert each
manifest row's Re/H equal the HDF5 columns.

### `cfd/ellipse2d/` — OpenLB domain correctness & per-sample runs

- **`ellipse2d.cpp`** now **requires** `--CHANNEL_HEIGHT`: the old silent default `8.0`
  is replaced by a `-1.0` sentinel plus a post-CLI guard that aborts FATAL (dataset
  domain height is 5–6), preventing silently simulating the wrong domain.
- **`runs/run_from_manifest.sh`** launches one OpenLB run per manifest row using that
  row's own Re and H (per-sample domain); `--SEMI_MINOR_B` is deliberately omitted since
  the body shape is fully determined by the `.dat` polygon.

> [!NOTE] Planned: physical conditioning (deferred)
> Augmenting the Cp-conditioned model with physical Re (+ optional H) conditioning and
> retraining is future work (Part D), independent of validation. See
> [[superpowers/specs/2026-06-23-dataset-viz-physical-conditioning-design|the design spec]].

---

## Data flow

```mermaid
sequenceDiagram
    participant D as Dataset
    participant DL as DataLoader
    participant GDM as GraphDiffusionModel
    participant SN as ScoreNetwork
    participant NS as NoiseSchedule

    D->>DL: Data objects (x, edge_index, edge_attr, u, batch)
    DL->>GDM: mini-batch (Data)
    GDM->>NS: sample t, get √ᾱ_t, √(1-ᾱ_t)
    GDM->>GDM: forward_diffusion(x₀, t) → x_t, ε
    GDM->>SN: forward(data_noisy, t) → ε̂
    SN-->>GDM: predicted noise ε̂
    GDM->>GDM: MSE(ε̂, ε) → loss

    Note over GDM: Sampling (reverse)
    loop t = T down to 1
        GDM->>SN: forward(data_t, t) → ε̂
        GDM->>NS: get_t(t, buffers)
        GDM->>GDM: x_{t-1} = denoise_step(x_t, ε̂)
    end
```

---

## See also

- [[README|Experiment index]] — list of all experiments
- [[changelog|Feature changelog]] — code changes per experiment
- [[EXPERIMENT_PLAN|Experimentation plan]] — planned experiment sequence
