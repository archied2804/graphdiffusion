#!/bin/bash
#SBATCH --job-name=graphdiff
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# ── Load modules (adjust to your cluster) ──
module load python/3.11
module load cuda/12.4

# ── Activate environment ──
cd ~/Projects/DGN_Simple
source .venv/bin/activate

# ── Run experiment ──
CONFIG="${CONFIG:-configs/circle_radial.yaml}"
EPOCHS="${EPOCHS:-100}"
DEVICE="${DEVICE:-cuda}"
OUTPUT="${OUTPUT:-outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}/generated_shapes.png}"

mkdir -p "$(dirname "$OUTPUT")"

python train_circle.py \
    --config "$CONFIG" \
    --epochs "$EPOCHS" \
    --device "$DEVICE" \
    --output "$OUTPUT" \
    2>&1 | tee "outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}/train.log"

echo "Job $SLURM_JOB_ID completed at $(date)"
