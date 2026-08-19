#!/bin/bash
#SBATCH --job-name=drishti_s2
#SBATCH --partition=interruptible_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:30:00
#SBATCH --output=/users/k23123868/edward/logs/drishti_s2_%j.out
#SBATCH --error=/users/k23123868/edward/logs/drishti_s2_%j.err
#SBATCH --exclude=erc-hpc-comp[048,054,170-175,177,178,196]
#
# Step 2 in-domain baseline: plain U-Net on refuge_zeiss, full image, 512px.
# Submit: sbatch /users/k23123868/edward/spfilm/submit_refuge_s2.sh
# Smoke: sbatch --time=0-00:20:00 /users/k23123868/edward/spfilm/submit_refuge_s2.sh --smoke
# NOT resumable — no --requeue, no SIGUSR1. If preempted, the job dies
# visibly and must be resubmitted from scratch.

set -euo pipefail

CODE_ROOT="/users/k23123868/edward/spfilm"
CONFIG="$CODE_ROOT/configs/stage2_drishti_create.json"
OUT_DIR="$CODE_ROOT/artifacts/runs/drishti_s2_${SLURM_JOB_ID}"

mkdir -p /users/k23123868/edward/logs "$OUT_DIR"

module load cuda
module load anaconda3/2022.10-gcc-13.2.0
eval "$(conda shell.bash hook)"
conda activate spfilm

python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "FATAL: no usable CUDA on $(hostname)"; exit 1; }

cd "$CODE_ROOT"
echo "[$(date -u +%FT%TZ)] starting refuge_s2 on $(hostname) (job $SLURM_JOB_ID)"
echo "git commit: $(git rev-parse HEAD)"
git diff --quiet || echo "WARNING: working tree is dirty"
nvidia-smi -L

python -u run_refuge_s2.py --config "$CONFIG" --out-dir "$OUT_DIR" --seed 42 "$@"

echo "[$(date -u +%FT%TZ)] refuge_s2 finished"
