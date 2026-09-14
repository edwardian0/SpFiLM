#!/bin/bash
#SBATCH --job-name=ssfilm_s4
#SBATCH --partition=interruptible_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=0-06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/users/k23123868/edward/logs/ssfilm_s4_%j.out
#SBATCH --error=/users/k23123868/edward/logs/ssfilm_s4_%j.err
#SBATCH --constraint="a100|a40|a30|l40s|h100"
#SBATCH --exclude=erc-hpc-comp[048,050,054,170-175,177,178,196,235-239,242,252,253]
#
# Step 4 Global FiLM under the train-on-one protocol over three domains
# (RIM-ONE-DL dropped for now): train on one source, score the other two,
# 40 train / 10 val / 2 x 50 test. Same runner, seeds and schedule as the Stage 3
# plain single-source arm; only the config differs (arm=global_film). It is
# paired with the EXISTING single_s3_* plain runs on identical target images:
# train-on-one training does not depend on the other domains, so plain is not
# re-run. One source domain is one code, so this arm is a sanity control that
# must match plain within seed noise, not a conditioning test; the runner prints
# that warning on start. Wall time matches the plain arm (same 40-image source).
# The full 3-source x 5-seed protocol is 15 independent submissions.
# Submit one run:
#   sbatch /users/k23123868/edward/spfilm/submit_stage4_single_source_film.sh \
#     refuge_zeiss 42
# Smoke one run:
#   sbatch --time=0-00:20:00 \
#     /users/k23123868/edward/spfilm/submit_stage4_single_source_film.sh \
#     refuge_zeiss 42 --smoke

set -euo pipefail

CODE_ROOT="/users/k23123868/edward/spfilm"
CONFIG="$CODE_ROOT/configs/stage4_single_source_global_film_3dom_create.json"

if (( $# < 2 )); then
  echo "usage: sbatch $0 <source-domain> <seed> [--smoke]" >&2
  echo "domains: refuge_zeiss refuge_canon_val drishti_gs (rim_one_dl is inactive in Step 4)" >&2
  exit 64
fi

SOURCE_DOMAIN="$1"
RUN_SEED="$2"
shift 2
# A requeued job keeps the same SLURM_JOB_ID and so reuses this directory on
# purpose: run_experiment finds resume_state.pt there and continues from the last
# completed epoch instead of restarting.
ATTEMPT="${SLURM_RESTART_COUNT:-0}"
OUT_DIR="$CODE_ROOT/artifacts/runs/ssfilm_s4_${SOURCE_DOMAIN}_seed_${RUN_SEED}_${SLURM_JOB_ID}"

mkdir -p /users/k23123868/edward/logs "$OUT_DIR"

module load cuda
module load anaconda3/2022.10-gcc-13.2.0
eval "$(conda shell.bash hook)"
conda activate spfilm

python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "FATAL: no usable CUDA on $(hostname)"; exit 1; }

cd "$CODE_ROOT"
echo "[$(date -u +%FT%TZ)] starting ssfilm_s4 on $(hostname) (job $SLURM_JOB_ID)"
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

echo "[$(date -u +%FT%TZ)] ssfilm_s4 finished"
