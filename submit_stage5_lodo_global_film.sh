#!/bin/bash -l
#SBATCH --job-name=gfilm_s5
#SBATCH --partition=interruptible_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=0-03:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --export=NONE
#SBATCH --output=/users/k23123868/edward/logs/gfilm_s5_%j.out
#SBATCH --error=/users/k23123868/edward/logs/gfilm_s5_%j.err
#SBATCH --constraint="a100|a40|a30|l40s|h100"
#SBATCH --exclude=erc-hpc-comp[048,050,054,170-175,177,178,196,223,235-239,242,252,253]
#
# Step 5 leave-one-domain-out over three domains (RIM-ONE-DL inactive): train
# on two, test on the held-out third, 80 train / 20 val / 50 test. Global FiLM
# arm, run by run_stage5_lodo.py; the held-out domain gets one code, the source
# domain nearest to its unlabelled reference sample. Paired with
# submit_stage5_lodo_plain.sh on identical test images: same folds, seeds and
# schedule, only the config's arm differs. The Step 4 LODO scripts
# (submit_stage4_plain.sh / submit_stage4_global_film.sh) define the same
# comparison; launching both sets would duplicate the 30 jobs. FiLM adds five
# small MLPs; train-on-all FiLM runs (120 train) finished in ~20 min, so the 3 h
# limit is ample; confirm from the smoke job's epoch_seconds.
# The full 3-domain x 5-seed arm is 15 independent submissions.
# Submit one run:
#   sbatch /users/k23123868/edward/spfilm/submit_stage5_lodo_global_film.sh \
#     refuge_zeiss 42
# Smoke one run:
#   sbatch --time=0-00:20:00 \
#     /users/k23123868/edward/spfilm/submit_stage5_lodo_global_film.sh \
#     refuge_zeiss 42 --smoke

set -euo pipefail

CODE_ROOT="/users/k23123868/edward/spfilm"
CONFIG="$CODE_ROOT/configs/stage5_lodo_global_film_3dom_create.json"

if (( $# < 2 )); then
  echo "usage: sbatch $0 <held-out-domain> <seed> [--smoke]" >&2
  echo "domains: refuge_zeiss refuge_canon_val drishti_gs (rim_one_dl is inactive in Step 5)" >&2
  exit 64
fi

HELD_OUT_DOMAIN="$1"
RUN_SEED="$2"
shift 2
# A requeued job keeps the same SLURM_JOB_ID and so reuses this directory on
# purpose: run_experiment finds resume_state.pt there and continues from the last
# completed epoch instead of restarting.
ATTEMPT="${SLURM_RESTART_COUNT:-0}"
OUT_DIR="$CODE_ROOT/artifacts/runs/gfilm_s5_${HELD_OUT_DOMAIN}_seed_${RUN_SEED}_${SLURM_JOB_ID}"

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
echo "[$(date -u +%FT%TZ)] starting gfilm_s5 on $(hostname) (job $SLURM_JOB_ID)"
echo "held-out domain: $HELD_OUT_DOMAIN"
echo "attempt: $ATTEMPT (0 = first run; >0 = requeued, resuming from checkpoint)"
echo "run seed: $RUN_SEED"
echo "git commit: $(git rev-parse HEAD)"
git diff --quiet || echo "WARNING: working tree is dirty"
nvidia-smi -L

python -u run_stage5_lodo.py --config "$CONFIG" run \
  --held-out-domain "$HELD_OUT_DOMAIN" \
  --seed "$RUN_SEED" \
  --out-dir "$OUT_DIR" \
  "$@"

echo "[$(date -u +%FT%TZ)] gfilm_s5 finished"
