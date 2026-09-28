#!/bin/bash -l
#SBATCH --job-name=allsf_s4
#SBATCH --partition=interruptible_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --export=NONE
#SBATCH --output=/users/k23123868/edward/logs/allsf_s4_%j.out
#SBATCH --error=/users/k23123868/edward/logs/allsf_s4_%j.err
#SBATCH --constraint="a100|a40|a30|l40s|h100"
#SBATCH --exclude=erc-hpc-comp[048,050,054,170-175,177,178,196,223,235-239,242,252,253]
#
# Train-on-all protocol, SpFiLM arm (Step 5 of the brief; rank K=8, FOV gating
# off): one model per seed trained on the pooled budgeted train partitions of the
# three active domains (120 train / 30 val, RIM-ONE-DL inactive) with each image's
# true conditioning signal, scored on each domain's own 50 test images with its
# own signal, plus the sweep under every other training domain's signal.
# Paired with submit_stage4_all_domains_film.sh (Global FiLM, which it differs
# from only in the spatial term) and submit_stage4_all_domains_plain.sh on
# identical test sets. SpFiLM costs more per step than Global FiLM (two 3x3 basis
# convolutions on the resampled image and a bmm per level, and 16-channel fields
# at 512 px at level 0): read the seed-42 job's Elapsed before submitting 43-46.
# Submit one seed:
#   sbatch /users/k23123868/edward/spfilm/submit_stage4_all_domains_spatial_film.sh 42
# Smoke one seed:
#   sbatch --time=0-00:20:00 /users/k23123868/edward/spfilm/submit_stage4_all_domains_spatial_film.sh 42 --smoke
# The SpFiLM arm is 5 independent submissions (seeds 42-46).

set -euo pipefail

CODE_ROOT="/users/k23123868/edward/spfilm"
CONFIG="$CODE_ROOT/configs/stage4_all_domains_spatial_film_k8_3dom_create.json"

if (( $# < 1 )); then
  echo "usage: sbatch $0 <seed> [--smoke]" >&2
  exit 64
fi

RUN_SEED="$1"
shift 1
# A requeued job keeps the same SLURM_JOB_ID and so reuses this directory on
# purpose: run_experiment finds resume_state.pt there and continues from the last
# completed epoch instead of restarting.
ATTEMPT="${SLURM_RESTART_COUNT:-0}"
OUT_DIR="$CODE_ROOT/artifacts/runs/allsf_s4_seed_${RUN_SEED}_${SLURM_JOB_ID}"

mkdir -p /users/k23123868/edward/logs "$OUT_DIR"

# The job builds its own environment (login shell, --export=NONE) instead of
# inheriting the submitting shell's. On 2026-09-14 five copies of this script
# submitted from arc-hpc-login4 died at `module load` with Lmod 8.5.6's usage
# text in .err and an empty .out, while the same script submitted earlier from
# erc-hpc-login2 ran: the exported `module` function / Lmod state of the newer
# login node does not work on the compute nodes, so it must not be inherited.
echo "[$(date -u +%FT%TZ)] job $SLURM_JOB_ID on $(hostname), submitted from ${SLURM_SUBMIT_HOST:-unknown}"
type module >/dev/null 2>&1 \
  || { echo "FATAL: no module command on $(hostname); login profile not sourced" >&2; exit 1; }
module load cuda
module load anaconda3/2022.10-gcc-13.2.0
eval "$(conda shell.bash hook)"
conda activate spfilm

python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "FATAL: no usable CUDA on $(hostname)"; exit 1; }

cd "$CODE_ROOT"
echo "[$(date -u +%FT%TZ)] starting allsf_s4 on $(hostname) (job $SLURM_JOB_ID)"
echo "train on all active domains, test on each"
echo "attempt: $ATTEMPT (0 = first run; >0 = requeued, resuming from checkpoint)"
echo "run seed: $RUN_SEED"
echo "git commit: $(git rev-parse HEAD)"
git diff --quiet || echo "WARNING: working tree is dirty"
nvidia-smi -L

python -u run_stage4_all_domains.py --config "$CONFIG" run \
  --seed "$RUN_SEED" \
  --out-dir "$OUT_DIR" \
  "$@"

echo "[$(date -u +%FT%TZ)] allsf_s4 finished"
