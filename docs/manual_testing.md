---
tags: [testing, workflow, csf, cluster, manual]
---

# Manual Testing Guide — CSF Cluster Experiments

This document describes the manual workflow for running `graph_diffusion`
experiments on the University of Manchester CSF3 cluster entirely through
the CLI.  Every experiment follows the same clean-slate pattern:

**SSH → clone → edit config → submit → monitor → scp results → delete.**

No persistent state is left on the cluster between experiments.

---

## 1. One-Time Pre-Work

Before running any experiment for the first time, complete these steps once.

### 1.1 Know your CSF account code

Every `sbatch` submission requires `--account=<code>`.  Your account code
is visible on the CSF user portal or in your project allocation email.
Keep it handy — it appears on every submission command below.

### 1.2 Ensure `uv` is available on CSF

Check whether `uv` is already on your PATH after logging in:

```bash
which uv
```

If not found, install it to your home directory:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"   # add to ~/.bashrc to persist
```

### 1.3 Confirm the CUDA module name

The script uses `module load libs/cuda/12.8.1`.  If your CSF allocation is
on a different node type, verify the correct module name once you are logged in:

```bash
module avail cuda
```

If it differs, update the `module load` line in `scripts/run_circle.sh`
locally, then commit and push before cloning on CSF.

---

## 2. Standard Per-Experiment Workflow

Follow these steps for every experiment.  Experiment-specific commands are
in Section 4.

### Step 1 — Check the feasibility checklist

See Section 4 for the checklist for your experiment.
**If any item is marked ✘, do not proceed — complete it first.**

### Step 2 — SSH into CSF

```bash
ssh <username>@csf3.itservices.manchester.ac.uk
```

### Step 3 — Clone the repository and install dependencies

```bash
cd ~
git clone https://github.com/<your-username>/DGN_Simple.git
cd DGN_Simple
uv sync
```

`uv sync` reads `pyproject.toml` and `uv.lock`, creates `.venv/`, and
installs all dependencies exactly.  This only needs to run once per
clone — the SLURM job reuses the same `.venv`.

> Always clone fresh.  Never reuse a previous clone from an earlier experiment.

### Step 4 — Edit the config (if the experiment requires it)

Some experiment configs contain a placeholder value with a comment
`# ← CHANGE THIS`.  Edit that value now:

```bash
nano configs/<experiment-config>.yaml
# make the change, then Ctrl+O → Enter → Ctrl+X to save
```

### Step 5 — Submit the SLURM job

Make sure you are in the repo root (`~/DGN_Simple`) before running `sbatch`.
SLURM passes `$SLURM_SUBMIT_DIR` to the job so it knows where the repo is.

Use the exact command from the experiment checklist in Section 4.
Note the job ID printed:

```
Submitted batch job 1234567
```

### Step 6 — Monitor the job

```bash
squeue -u $USER                          # check status (PD=pending, R=running)
tail -f dgn_circle_1234567.out           # live training log
```

The job is complete when `squeue -u $USER` shows no entry for it, or the
log prints `=== Experiment complete ===`.

### Step 7 — Copy results to your local machine

Run the following **from your local terminal** (not the cluster):

```bash
scp -r <username>@csf3.itservices.manchester.ac.uk:~/dgn_outputs/<EXP_SLUG>/ \
       ./outputs/<EXP_SLUG>/
```

Replace `<EXP_SLUG>` with the exact slug used in the `sbatch` command.

### Step 8 — Verify the transfer

```bash
ls outputs/<EXP_SLUG>/
```

Expected files (all generated automatically by `run_circle.sh`):

| File | Source |
|------|--------|
| `config_used.yaml` | Exact config copy |
| `run_metadata.json` | Job ID, git commit, node, timestamps |
| `checkpoint.pt` | Final model weights + optimizer state |
| `loss_log.json` | Per-epoch train/val loss |
| `tensorboard/` | TensorBoard event files |
| `generated_shapes.png` | 4-sample shape plot from end of training |
| `evaluation_report.json` | Quantitative metrics (50 samples) |
| `loss_curves.png` | Train/val loss plot |
| `radii_histogram.png` | Reference vs generated radii distributions |
| `sample_gallery.png` | Gallery of 16 generated shapes |
| `quality_distributions.png` | Smoothness & circularity histograms |
| `generated_samples.pt` | Raw generated radii + angles tensor |

### Step 9 — Clean up CSF

Once the `scp` has completed and you have verified the local files,
delete everything from CSF:

```bash
# Run on CSF
rm -rf ~/DGN_Simple
rm -rf ~/dgn_outputs/<EXP_SLUG>/
```

Confirm nothing remains:

```bash
ls ~/dgn_outputs/     # should show nothing or only unrelated directories
ls ~/DGN_Simple 2>&1  # should say "No such file or directory"
```

---

## 3. Local Post-Processing

All quantitative evaluation runs automatically on the cluster as part of
`run_circle.sh` (via `scripts/postprocess_circle.py`).  To re-run or
extend post-processing locally after `scp`:

```bash
cd /path/to/DGN_Simple
python scripts/postprocess_circle.py \
    --experiment-dir outputs/<EXP_SLUG> \
    --config         configs/<experiment-config>.yaml \
    --n-samples      50 \
    --save-samples \
    --visualize-diffusion
```

---

## 4. Experiment Checklists

---

### EXP-001 — Radial diffusion baseline

**Status:** ✅ Runnable (already completed; re-run to reproduce)

**Pre-flight checklist:**

- [x] `configs/circle_radial.yaml` exists and committed
- [x] `scripts/run_circle.sh` exists and committed
- [x] No placeholder values to edit

**Submission command:**

```bash
sbatch --account=<your-account> \
       --export=EXP_SLUG=EXP-001_circle_radial_baseline,\
CONFIG=configs/circle_radial.yaml,\
EPOCHS=100 \
       scripts/run_circle.sh
```

**Local scp:**

```bash
scp -r <user>@csf3.itservices.manchester.ac.uk:~/dgn_outputs/EXP-001_circle_radial_baseline/ \
       ./outputs/EXP-001_circle_radial_baseline/
```

**CSF cleanup:**

```bash
rm -rf ~/DGN_Simple && rm -rf ~/dgn_outputs/EXP-001_circle_radial_baseline/
```

---

### EXP-002 — k-neighbors ablation (4 variants)

**Status:** ✅ Runnable — requires editing one line in the config per variant

**Pre-flight checklist:**

- [x] `configs/EXP-002_circle_radial_k-neighbors.yaml` exists and committed
- [x] `scripts/run_circle.sh` exists and committed
- [ ] **Config edited:** `k_neighbors` set to the value for this variant after cloning

Run one variant at a time — clone, `uv sync`, edit config, submit, copy, delete, then repeat
for the next variant.

#### Variant a — k_neighbors = 1

Edit `configs/EXP-002_circle_radial_k-neighbors.yaml`:

```yaml
k_neighbors: 1       # ← set this
```

Submit:

```bash
sbatch --account=<your-account> \
       --export=EXP_SLUG=EXP-002a_circle_radial_k1,\
CONFIG=configs/EXP-002_circle_radial_k-neighbors.yaml,\
EPOCHS=100 \
       scripts/run_circle.sh
```

scp + cleanup:

```bash
# local
scp -r <user>@csf3.itservices.manchester.ac.uk:~/dgn_outputs/EXP-002a_circle_radial_k1/ \
       ./outputs/EXP-002a_circle_radial_k1/
# cluster
rm -rf ~/DGN_Simple && rm -rf ~/dgn_outputs/EXP-002a_circle_radial_k1/
```

#### Variant b — k_neighbors = 2

Edit `configs/EXP-002_circle_radial_k-neighbors.yaml`:

```yaml
k_neighbors: 2
```

```bash
sbatch --account=<your-account> \
       --export=EXP_SLUG=EXP-002b_circle_radial_k2,\
CONFIG=configs/EXP-002_circle_radial_k-neighbors.yaml,\
EPOCHS=100 \
       scripts/run_circle.sh
```

scp + cleanup:

```bash
scp -r <user>@csf3.itservices.manchester.ac.uk:~/dgn_outputs/EXP-002b_circle_radial_k2/ \
       ./outputs/EXP-002b_circle_radial_k2/
rm -rf ~/DGN_Simple && rm -rf ~/dgn_outputs/EXP-002b_circle_radial_k2/
```

#### Variant c — k_neighbors = 4

```yaml
k_neighbors: 4
```

```bash
sbatch --account=<your-account> \
       --export=EXP_SLUG=EXP-002c_circle_radial_k4,\
CONFIG=configs/EXP-002_circle_radial_k-neighbors.yaml,\
EPOCHS=100 \
       scripts/run_circle.sh
```

```bash
scp -r <user>@csf3.itservices.manchester.ac.uk:~/dgn_outputs/EXP-002c_circle_radial_k4/ \
       ./outputs/EXP-002c_circle_radial_k4/
rm -rf ~/DGN_Simple && rm -rf ~/dgn_outputs/EXP-002c_circle_radial_k4/
```

#### Variant d — k_neighbors = 6

```yaml
k_neighbors: 6
```

```bash
sbatch --account=<your-account> \
       --export=EXP_SLUG=EXP-002d_circle_radial_k6,\
CONFIG=configs/EXP-002_circle_radial_k-neighbors.yaml,\
EPOCHS=100 \
       scripts/run_circle.sh
```

```bash
scp -r <user>@csf3.itservices.manchester.ac.uk:~/dgn_outputs/EXP-002d_circle_radial_k6/ \
       ./outputs/EXP-002d_circle_radial_k6/
rm -rf ~/DGN_Simple && rm -rf ~/dgn_outputs/EXP-002d_circle_radial_k6/
```

---

### EXP-003 — Amplitude-scale ablation (3 variants)

**Status:** ✅ Runnable — requires editing one line in the config per variant

**Pre-flight checklist:**

- [x] `configs/EXP-003_circle_radial_amplitude.yaml` exists and committed
- [x] `scripts/run_circle.sh` exists and committed
- [ ] **Config edited:** `amplitude_scale` set to the value for this variant after cloning

#### Variant a — amplitude_scale = 0.05

```yaml
amplitude_scale: 0.05
```

```bash
sbatch --account=<your-account> \
       --export=EXP_SLUG=EXP-003a_circle_radial_amp005,\
CONFIG=configs/EXP-003_circle_radial_amplitude.yaml,\
EPOCHS=100 \
       scripts/run_circle.sh
```

```bash
scp -r <user>@csf3.itservices.manchester.ac.uk:~/dgn_outputs/EXP-003a_circle_radial_amp005/ \
       ./outputs/EXP-003a_circle_radial_amp005/
rm -rf ~/DGN_Simple && rm -rf ~/dgn_outputs/EXP-003a_circle_radial_amp005/
```

#### Variant b — amplitude_scale = 0.15

```yaml
amplitude_scale: 0.15
```

```bash
sbatch --account=<your-account> \
       --export=EXP_SLUG=EXP-003b_circle_radial_amp015,\
CONFIG=configs/EXP-003_circle_radial_amplitude.yaml,\
EPOCHS=100 \
       scripts/run_circle.sh
```

```bash
scp -r <user>@csf3.itservices.manchester.ac.uk:~/dgn_outputs/EXP-003b_circle_radial_amp015/ \
       ./outputs/EXP-003b_circle_radial_amp015/
rm -rf ~/DGN_Simple && rm -rf ~/dgn_outputs/EXP-003b_circle_radial_amp015/
```

#### Variant c — amplitude_scale = 0.30

```yaml
amplitude_scale: 0.30
```

```bash
sbatch --account=<your-account> \
       --export=EXP_SLUG=EXP-003c_circle_radial_amp030,\
CONFIG=configs/EXP-003_circle_radial_amplitude.yaml,\
EPOCHS=100 \
       scripts/run_circle.sh
```

```bash
scp -r <user>@csf3.itservices.manchester.ac.uk:~/dgn_outputs/EXP-003c_circle_radial_amp030/ \
       ./outputs/EXP-003c_circle_radial_amp030/
rm -rf ~/DGN_Simple && rm -rf ~/dgn_outputs/EXP-003c_circle_radial_amp030/
```

---

### EXP-004 — Logit-transform bounded diffusion

**Status:** ❌ Blocked — code changes required before this experiment can run

**Pre-flight checklist:**

- [ ] `GraphDiffusionModel.forward_diffusion()` applies `logit(r)` before
      the forward process (in `src/graph_diffusion/model/graph_diffusion_model.py`)
- [ ] `GraphDiffusionModel.sample()` applies `sigmoid()` during the reverse
      process to map back to the bounded range
- [ ] Both transforms are opt-in via a `transform: logit` key in config
      (so existing experiments are unaffected)
- [ ] `configs/EXP-004_circle_radial_logit.yaml` created and committed
- [ ] New code has passing tests

**Do not proceed until all items above are checked.**

---

### EXP-005 — Full 100-epoch training with scheduling

**Status:** ⚠️ Partially blocked — can run 100 epochs now but without scheduler

The training script already accepts `--epochs 100`.  It does not yet support
cosine LR scheduling or early stopping.

**Pre-flight checklist:**

- [ ] `torch.optim.lr_scheduler.CosineAnnealingLR` added to `train_circle.py`
      behind a `--lr_scheduler cosine` flag (off by default)
- [ ] Validation-patience early stopping added behind a `--patience N` flag
- [ ] `configs/EXP-005_circle_radial_full.yaml` created and committed
      (base config from best of EXP-002/EXP-003, 100 epochs, scheduler on)

**Submission command (after above tasks are done):**

```bash
sbatch --account=<your-account> \
       --export=EXP_SLUG=EXP-005_circle_radial_full,\
CONFIG=configs/EXP-005_circle_radial_full.yaml,\
EPOCHS=100 \
       scripts/run_circle.sh
```

---

### EXP-006 — Richer node features [r, κ, s/L]

**Status:** ❌ Blocked — dataset code changes required

**Pre-flight checklist:**

- [ ] `UnitCircleDataset._build_graphs()` computes curvature `κ(θ)` and
      normalised arc-length `s/L` per node and stacks them as `x` of shape
      `(n_nodes, 3)` when `rich_features: true` is set in config
      (in `src/graph_diffusion/data/circledataset.py`)
- [ ] `configs/EXP-006_circle_radial_rich-features.yaml` created with
      `input_dim: 3` in `score_network` and `rich_features: true` in
      `circle_dataset`
- [ ] New feature computation has passing tests

**Do not proceed until all items above are checked.**

---

### EXP-007 — NACA aerofoil geometry

**Status:** ❌ Blocked — new dataset class required

**Pre-flight checklist:**

- [ ] `src/graph_diffusion/data/nacadataset.py` implemented:
      `NACADataset(BaseGraphDataset)` generating NACA 4-digit profiles
- [ ] `NACADataset` exported from `src/graph_diffusion/data/__init__.py`
- [ ] A training entry point (`train_naca.py` or extended `train_circle.py`)
      exists that loads `NACADataset` with its appropriate transforms
- [ ] `configs/EXP-007_naca_radial_baseline.yaml` created and committed
- [ ] New dataset has passing tests

**Do not proceed until all items above are checked.**

---

## 5. Troubleshooting

### Job stays in pending (PD) for a long time

Check the reason:

```bash
squeue -u $USER -o "%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R"
```

Common reasons: `Resources` (waiting for a GPU to free), `Priority`
(fair-share queue), `QOSMaxJobsPerUser` (too many jobs submitted at once).

### Job exits immediately with non-zero code

Check the `.err` file:

```bash
cat dgn_circle_<JOBID>.err
```

Common causes: `REPO_URL_NOT_SET` (forgot to set the URL), incorrect CUDA
module, missing Python package, config key not recognised.

### Postprocessing fails but training succeeded

`run_circle.sh` prints a warning and exits with `0` if postprocessing
fails — training outputs (`checkpoint.pt`, `loss_log.json`) are preserved.
Re-run postprocessing locally after `scp`:

```bash
python scripts/postprocess_circle.py \
    --experiment-dir outputs/<EXP_SLUG> \
    --config         configs/<experiment-config>.yaml \
    --n-samples 50 --save-samples
```

### `scp` is slow / times out

Use `rsync` instead:

```bash
rsync -avz --progress \
    <user>@csf3.itservices.manchester.ac.uk:~/dgn_outputs/<EXP_SLUG>/ \
    ./outputs/<EXP_SLUG>/
```
