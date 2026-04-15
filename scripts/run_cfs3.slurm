#!/bin/bash --login
#
# run_cfs3.slurm — Ephemeral multi-GPU training on CFS3 (UoM H200 cluster)
# =========================================================================
#
# Clones the repo into $TMPDIR, installs deps, runs training, copies
# outputs to a persistent location, and cleans up everything.
#
# Usage:
#   sbatch scripts/run_cfs3.slurm                          # defaults (1 GPU)
#   sbatch --export=NGPUS=4 scripts/run_cfs3.slurm         # 4 GPUs
#   sbatch --export=NGPUS=8,EPOCHS=200,CONFIG=configs/circle_radial.yaml \
#          scripts/run_cfs3.slurm
#
# Required: set ACCOUNT below or pass --account=<code> on the sbatch line.
# -------------------------------------------------------------------------

# ── Configurable variables (override via --export or environment) ──
NGPUS="${NGPUS:-1}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-1e-3}"
CONFIG="${CONFIG:-configs/default.yaml}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_ddp.py}"
REPO_URL="${REPO_URL:-https://github.com/YOUR_USER/DGN_Simple.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
OUTPUT_BASE="${OUTPUT_BASE:-$HOME/dgn_outputs}"
SCALE_LR="${SCALE_LR:-0}"          # set to 1 to enable linear LR scaling
USE_AMP="${USE_AMP:-1}"            # set to 0 to disable mixed precision
SAVE_EVERY="${SAVE_EVERY:-0}"      # checkpoint interval (0 = final only)
ACCOUNT="${ACCOUNT:-TO_ADD}"       # MANDATORY: replace with your account code

# ── SLURM directives ──
#SBATCH --partition=gpuH_short
#SBATCH --account=${ACCOUNT}
#SBATCH --gres=gpu:${NGPUS}
#SBATCH --ntasks-per-node=${NGPUS}
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:00
#SBATCH --job-name=dgn_ddp
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ── Derived paths ──
WORK_DIR="${TMPDIR:-/tmp}/dgn_${SLURM_JOB_ID}"
OUTPUT_DIR="${OUTPUT_BASE}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
CHECKPOINT_DIR="${WORK_DIR}/checkpoints"

# ── Cleanup function (runs on EXIT, including timeout/failure) ──
cleanup() {
    echo ""
    echo "=== Cleanup ==="

    # Copy outputs to persistent storage before deleting workspace
    if [[ -d "${CHECKPOINT_DIR}" ]]; then
        mkdir -p "${OUTPUT_DIR}/checkpoints"
        cp -r "${CHECKPOINT_DIR}"/* "${OUTPUT_DIR}/checkpoints/" 2>/dev/null
        echo "Checkpoints copied to ${OUTPUT_DIR}/checkpoints/"
    fi

    # Copy any generated plots/figures
    if compgen -G "${WORK_DIR}/*.png" > /dev/null 2>&1; then
        mkdir -p "${OUTPUT_DIR}"
        cp "${WORK_DIR}"/*.png "${OUTPUT_DIR}/" 2>/dev/null
        echo "Plots copied to ${OUTPUT_DIR}/"
    fi

    # Copy SLURM log files if they exist in the workspace
    if compgen -G "${WORK_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}."* > /dev/null 2>&1; then
        cp "${WORK_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}."* "${OUTPUT_DIR}/" 2>/dev/null
    fi

    # Remove ephemeral workspace
    if [[ -d "${WORK_DIR}" ]]; then
        rm -rf "${WORK_DIR}"
        echo "Cleaned up workspace: ${WORK_DIR}"
    fi

    echo "Outputs preserved in: ${OUTPUT_DIR}"
    echo "=== Cleanup complete ==="
}
trap cleanup EXIT

# ── 1. Environment setup ──
echo "=== Job Setup ==="
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Job Name   : ${SLURM_JOB_NAME}"
echo "Partition  : ${SLURM_JOB_PARTITION}"
echo "Node       : $(hostname)"
echo "GPUs       : ${NGPUS}"
echo "Config     : ${CONFIG}"
echo "Epochs     : ${EPOCHS}"
echo "Date       : $(date)"
echo "Work Dir   : ${WORK_DIR}"
echo "Output Dir : ${OUTPUT_DIR}"
echo "==================="
echo ""

START_TIME=$(date +%s)

module purge
module load libs/cuda/12.8.1

# ── 2. Clone repository ──
echo "=== Cloning repository ==="
git clone --depth 1 --branch "${REPO_BRANCH}" "${REPO_URL}" "${WORK_DIR}"
if [[ $? -ne 0 ]]; then
    echo "ERROR: git clone failed"
    exit 1
fi
cd "${WORK_DIR}" || exit 1

# ── 3. Create virtual environment and install dependencies ──
echo "=== Setting up Python environment ==="
python3 -m venv .venv
source .venv/bin/activate

pip install --quiet --upgrade pip
pip install --quiet uv
uv pip install -e . --quiet

# ── 4. Environment diagnostics ──
echo ""
echo "=== Environment ==="
echo "Python : $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA   : $(python -c 'import torch; print(torch.version.cuda)')"
echo "Host   : $(hostname)"
echo "Date   : $(date)"
echo "==================="
echo ""

nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# ── 5. Create output directories ──
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${CHECKPOINT_DIR}"

# ── 6. Build training command ──
EXTRA_FLAGS=""
if [[ "${SCALE_LR}" == "1" ]]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --scale_lr"
fi
if [[ "${USE_AMP}" == "1" ]]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --amp"
fi
if [[ "${SAVE_EVERY}" -gt 0 ]]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --save_every ${SAVE_EVERY}"
fi

echo "=== Starting training ==="

if [[ "${NGPUS}" -gt 1 ]]; then
    # Multi-GPU: use torchrun for DDP
    echo "Mode: DDP with ${NGPUS} GPUs via torchrun"
    torchrun \
        --standalone \
        --nproc_per_node="${NGPUS}" \
        "${TRAIN_SCRIPT}" \
        --config "${CONFIG}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --checkpoint_dir "${CHECKPOINT_DIR}" \
        ${EXTRA_FLAGS}
    EXIT_CODE=$?
else
    # Single GPU: run directly (can use either train.py or train_ddp.py)
    echo "Mode: Single GPU"
    # For single-GPU, prefer the standard training script if specified
    if [[ "${TRAIN_SCRIPT}" == "train_ddp.py" ]]; then
        torchrun \
            --standalone \
            --nproc_per_node=1 \
            "${TRAIN_SCRIPT}" \
            --config "${CONFIG}" \
            --epochs "${EPOCHS}" \
            --lr "${LR}" \
            --checkpoint_dir "${CHECKPOINT_DIR}" \
            ${EXTRA_FLAGS}
        EXIT_CODE=$?
    else
        python "${TRAIN_SCRIPT}" \
            --config "${CONFIG}" \
            --epochs "${EPOCHS}" \
            --lr "${LR}" \
            ${EXTRA_FLAGS}
        EXIT_CODE=$?
    fi
fi

# ── 7. Copy training log to output dir ──
# The SLURM stdout/stderr are written to the submission directory,
# but we can save a copy of any local logs
if [[ -f "${WORK_DIR}/train.log" ]]; then
    cp "${WORK_DIR}/train.log" "${OUTPUT_DIR}/" 2>/dev/null
fi

# ── 8. Print exit summary ──
END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))
HOURS=$((RUNTIME / 3600))
MINUTES=$(( (RUNTIME % 3600) / 60 ))
SECONDS=$((RUNTIME % 60))

echo ""
echo "==================="
echo "=== Exit Summary ==="
echo "Exit Code  : ${EXIT_CODE}"
echo "Exit Time  : $(date)"
echo "Runtime    : ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Outputs    : ${OUTPUT_DIR}"
echo "==================="

exit "${EXIT_CODE}"
