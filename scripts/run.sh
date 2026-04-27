#!/bin/bash --login

#SBATCH --partition=gpuH_short    # Use 'gpuH' (max 4 days) or 'gpuH_short' (max 1 day)
#SBATCH --account gpu-ar-nwtf     # MANDATORY: Replace with your approved H200 account code
#SBATCH --gres=gpu:1              # Request 1 H200 GPU (change if you need more, up to 8)
#SBATCH --ntasks=1                # Number of tasks
#SBATCH --cpus-per-task=8         # Number of CPU cores per GPU (default is 1, max 8 per GPU)
#SBATCH --time=23:59:00         # Wallclock time (Days(m)-Hours:Minutes:Seconds)
#SBATCH --job-name=h200_dgn_job   # Name of the job
#SBATCH --output=%x_%j.out        # Standard output log (%x = job name, %j = job ID)
#SBATCH --error=%x_%j.err         # Standard error log

# 1. Navigate to project
cd /mnt/iusers01/mace01/m22729ad/h200-scratch/tmp_DGN/graphdiffusion


# 2. Load the required CUDA module (H200 supports 11.8+ or 12.0+) and purge environment
module purge
module load libs/cuda/12.8.1

# (Optional) Load an underlying Python module if project dependant
# module load apps/python3/3.x.x

# 3. Activate your Python Virtual Environment
source ./.venv/bin/activate

# After activating venv, add:
START_TIME=$(date +%s)
echo "=== Environment ==="
echo "Python : $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA   : $(python -c 'import torch; print(torch.version.cuda)')"
echo "Host   : $(hostname)"
echo "Date   : $(date)"
echo "==================="

# 4. Print stats of assigned GPU(s) to verify allocation
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# 5. Run script
python train_circle.py \
    --config "${CONFIG:-configs/circle_radial.yaml}" \
    --epochs "${EPOCHS:-500}" \
    --device "${DEVICE:-cuda}" \
    --output "${OUTPUT:-outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}/generated_shapes.png}" \
    2>&1 | tee "outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}/train.log"
EXIT_CODE=$?

# 6. Print exit summary
END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))
MINUTES=$((RUNTIME / 60))
SECONDS=$((RUNTIME % 60))

echo ""
echo "==================="
echo "=== Exit Summary ==="
echo "Exit Code  : $EXIT_CODE"
echo "Exit Time  : $(date)"
echo "Runtime    : ${MINUTES}m ${SECONDS}s"
echo "==================="