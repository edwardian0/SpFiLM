#!/bin/bash
#SBATCH --job-name=single_s3
#SBATCH --partition=interruptible_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=0-06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/users/k23123868/edward/logs/single_s3_%j.out
#SBATCH --error=/users/k23123868/edward/logs/single_s3_%j.err
#SBATCH --constraint="a100|a40|a30|l40s|h100"
#SBATCH --exclude=erc-hpc-comp[048,054,170-175,177,178,196,235-239,242,252,253]
#
# Stage 3 train-on-one, test-on-three: plain U-Net, full image, 512px.
# Submit one run:
#   sbatch /users/k23123868/edward/spfilm/submit_lodo_stage3_single_source.sh \
#     refuge_zeiss 42
# Smoke one run:
#   sbatch --time=0-00:20:00 \
#     /users/k23123868/edward/spfilm/submit_lodo_stage3_single_source.sh \
#     refuge_zeiss 42 --smoke
# The full 4-source-domain x 5-seed protocol is 20 independent submissions;
# each model scores the other three domains separately.

set -euo pipefail

CODE_ROOT="/users/k23123868/edward/spfilm"
CONFIG="$CODE_ROOT/configs/stage3_lodo_single_create.json"

if (( $# < 2 )); then
  echo "usage: sbatch $0 <source-domain> <seed> [--smoke]" >&2
  echo "domains: refuge_zeiss refuge_canon_val drishti_gs rim_one_dl" >&2
  exit 64
fi

SOURCE_DOMAIN="$1"
RUN_SEED="$2"
shift 2
# A requeued job keeps the same SLURM_JOB_ID and so reuses this directory on
# purpose: run_experiment finds resume_state.pt there and continues from the last
# completed epoch instead of restarting.
ATTEMPT="${SLURM_RESTART_COUNT:-0}"
OUT_DIR="$CODE_ROOT/artifacts/runs/single_s3_${SOURCE_DOMAIN}_seed_${RUN_SEED}_${SLURM_JOB_ID}"

mkdir -p /users/k23123868/edward/logs "$OUT_DIR"

module load cuda
module load anaconda3/2022.10-gcc-13.2.0
eval "$(conda shell.bash hook)"
conda activate spfilm

python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "FATAL: no usable CUDA on $(hostname)"; exit 1; }

cd "$CODE_ROOT"
echo "[$(date -u +%FT%TZ)] starting single_s3 on $(hostname) (job $SLURM_JOB_ID)"
echo "source domain: $SOURCE_DOMAIN"
echo "attempt: $ATTEMPT (0 = first run; >0 = requeued, resuming from checkpoint)"
echo "run seed: $RUN_SEED"
echo "git commit: $(git rev-parse HEAD)"
git diff --quiet || echo "WARNING: working tree is dirty"
nvidia-smi -L

python -u run_stage3_lodo_1_3.py --config "$CONFIG" run \
  --source-domain "$SOURCE_DOMAIN" \
  --seed "$RUN_SEED" \
  --out-dir "$OUT_DIR" \
  "$@"

echo "[$(date -u +%FT%TZ)] single_s3 finished"
