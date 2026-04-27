#!/bin/bash --login
#
# run_circle.sh — Single-GPU training for circle diffusion experiments
# =====================================================================
#
# Submit this from the ROOT of the cloned DGN_Simple repository after
# running `uv sync` once to create the virtual environment.
#
# Prerequisites (done once after cloning, before first sbatch):
#   cd ~/DGN_Simple
#   uv sync
#
# Usage:
#   sbatch --account=<your-account-code> \
#          --export=EXP_SLUG=EXP-001_circle_radial_baseline,\
#                   CONFIG=configs/circle_radial.yaml,\
#                   EPOCHS=100 \
#          scripts/run_circle.sh
#
# Required export variable:
#   EXP_SLUG   Experiment slug used as the output directory name.
#              Examples: EXP-001_circle_radial_baseline
#                        EXP-002a_circle_radial_k1
#                        EXP-003c_circle_radial_amp030
#
# Optional export variables (all have sensible defaults):
#   CONFIG       Path to YAML config, relative to repo root.
#                Default: configs/circle_radial.yaml
#   EPOCHS       Number of training epochs.             Default: 100
#   LR           Learning rate.                         Default: 1e-3
#   N_SAMPLES    Shapes generated for evaluation.       Default: 50
#   OUTPUT_BASE  Parent directory for outputs.          Default: $HOME/dgn_outputs
# -------------------------------------------------------------------------

# ── Configurable variables ──
EXP_SLUG="${EXP_SLUG:-EXP-UNKNOWN}"
CONFIG="${CONFIG:-configs/circle_radial.yaml}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-1e-3}"
N_SAMPLES="${N_SAMPLES:-50}"
OUTPUT_BASE="${OUTPUT_BASE:-$HOME/dgn_outputs}"

# ── SLURM directives ──
#SBATCH --partition=gpuH_short
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:00
#SBATCH --job-name=dgn_circle
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ── Derived paths ──
REPO_DIR="${SLURM_SUBMIT_DIR}"
OUTPUT_DIR="${OUTPUT_BASE}/${EXP_SLUG}"

# ── Validate required inputs ──
if [[ "${EXP_SLUG}" == "EXP-UNKNOWN" ]]; then
    echo "ERROR: EXP_SLUG must be set via --export=EXP_SLUG=EXP-NNN_..."
    exit 1
fi

# ── 1. Print job info ──
echo "=== Job Info ==="
echo "Job ID     : ${SLURM_JOB_ID}"
echo "EXP_SLUG   : ${EXP_SLUG}"
echo "Config     : ${CONFIG}"
echo "Epochs     : ${EPOCHS}"
echo "LR         : ${LR}"
echo "N_samples  : ${N_SAMPLES}"
echo "Node       : $(hostname)"
echo "Date       : $(date)"
echo "Repo dir   : ${REPO_DIR}"
echo "Output dir : ${OUTPUT_DIR}"
echo "================"
echo ""

START_TIME=$(date +%s)

# ── 2. Load modules ──
module purge
module load libs/cuda/12.8.1

# ── 3. Move to repo and activate environment ──
cd "${REPO_DIR}" || { echo "ERROR: cannot cd to ${REPO_DIR}"; exit 1; }

GIT_COMMIT=$(git rev-parse HEAD)
echo "Repo   : ${REPO_DIR}"
echo "Commit : ${GIT_COMMIT}"

echo ""
echo "=== Activating environment ==="
uv sync --quiet
source .venv/bin/activate

echo "Python : $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA   : $(python -c 'import torch; print(torch.version.cuda)')"
nvidia-smi -L

# ── 5. Create output directory ──
mkdir -p "${OUTPUT_DIR}"

# Write a metadata file so results are traceable without opening the checkpoint
cat > "${OUTPUT_DIR}/run_metadata.json" <<EOF
{
  "exp_slug":   "${EXP_SLUG}",
  "job_id":     "${SLURM_JOB_ID}",
  "node":       "$(hostname)",
  "git_commit": "${GIT_COMMIT}",
  "repo_dir":   "${REPO_DIR}",
  "config":     "${CONFIG}",
  "epochs":     ${EPOCHS},
  "lr":         ${LR},
  "submitted":  "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

# Copy the exact config used so the output dir is self-contained
cp "${CONFIG}" "${OUTPUT_DIR}/config_used.yaml"

# ── 6. Train ──
echo ""
echo "=== Training ==="
python train_circle.py \
    --config    "${CONFIG}" \
    --epochs    "${EPOCHS}" \
    --lr        "${LR}" \
    --device    cuda \
    --n_samples 4 \
    --output    "${OUTPUT_DIR}/generated_shapes.png"
TRAIN_EXIT=$?

if [[ "${TRAIN_EXIT}" -ne 0 ]]; then
    echo "ERROR: training exited with code ${TRAIN_EXIT}"
    exit "${TRAIN_EXIT}"
fi

# ── 7. Postprocess ──
echo ""
echo "=== Postprocessing ==="
python scripts/postprocess_circle.py \
    --experiment-dir "${OUTPUT_DIR}" \
    --config         "${CONFIG}" \
    --n-samples      "${N_SAMPLES}" \
    --device         cuda \
    --save-samples
POST_EXIT=$?

if [[ "${POST_EXIT}" -ne 0 ]]; then
    echo "WARNING: postprocessing exited with code ${POST_EXIT} — training outputs are still intact"
fi

# ── 8. Summary ──
END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))
HOURS=$((RUNTIME / 3600))
MINUTES=$(( (RUNTIME % 3600) / 60 ))
SECONDS_REM=$((RUNTIME % 60))

echo ""
echo "=== Experiment complete ==="
echo "EXP_SLUG  : ${EXP_SLUG}"
echo "Runtime   : ${HOURS}h ${MINUTES}m ${SECONDS_REM}s"
echo "Outputs   : ${OUTPUT_DIR}"
echo ""
echo "Files produced:"
ls -lh "${OUTPUT_DIR}/"
echo ""
echo "To copy results to your local machine, run FROM your local terminal:"
echo "  scp -r \$(whoami)@csf3.itservices.manchester.ac.uk:${OUTPUT_DIR}/ ./outputs/${EXP_SLUG}/"
echo ""
echo "To clean up after copying:"
echo "  rm -rf ~/DGN_Simple && rm -rf ${OUTPUT_DIR}"
