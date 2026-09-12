#!/bin/bash
# Render the prediction-overlay figures for the cells named in
# HANDOFF_prediction_overlays.md section 5.
#
# Step 1 needs CREATE access. Runs synced back from CREATE did not bring their
# weights, so the checkpoints have to be pulled first. CREATE's SSH requires an
# MFA session that expires: if scp exits with
#
#     You may need to authenticate your SSH access by visiting the e-Research Portal
#
# then refresh it at https://portal.er.kcl.ac.uk/mfa/ and run this again.
#
# Three checkpoints cover all five figures. Each seed was picked because its own
# cross-domain Dice is the closest of the five seeds to that cell's five-seed mean,
# so the figure is representative of the number in the brief rather than of a
# lucky or unlucky run.
#
#   single_s3_refuge_zeiss_seed_46      zeiss->drishti 0.726 (cell mean 0.727)
#                                       zeiss->canon   0.752 (cell mean 0.733)
#   single_s3_refuge_canon_val_seed_43  canon->zeiss   0.900 (cell mean 0.891)
#                                       canon->drishti 0.879 (cell mean 0.879)
#   fixed_s3_rim_one_dl_seed_45         LODO held-out  0.072 (arm mean 0.074)
#
# Usage:  ./render_overlay_figures.sh [--skip-pull]

set -euo pipefail
cd "$(dirname "$0")"

# This script runs in two places and they do not share an interpreter: the laptop
# has the .spfilm venv, CREATE has a conda env that oncompute.sh activates. Pick
# whichever is present, and say which, so a wrong choice is visible rather than
# showing up later as a missing torch.
if [ -n "${PYTHON:-}" ]; then
  read -r -a PY <<< "$PYTHON"
elif [ -x .spfilm/bin/python ]; then
  PY=(.spfilm/bin/python)
elif [ -x ./oncompute.sh ]; then
  PY=(./oncompute.sh python)
else
  echo "No interpreter found: expected .spfilm/bin/python or ./oncompute.sh." >&2
  echo "Set PYTHON=... to choose one explicitly." >&2
  exit 1
fi
echo "interpreter: ${PY[*]}"

REMOTE=create
REMOTE_ROOT=/cephfs/volumes/hpc_home/k23123868/627d4c89-cd29-4f24-b3ae-4f85744f010c/edward/spfilm/artifacts/runs
OUT=artifacts/prediction_overlays

ZEISS=single_s3_refuge_zeiss_seed_46_36984715
CANON=single_s3_refuge_canon_val_seed_43_36984707
RIMONE=fixed_s3_rim_one_dl_seed_45_37071727

# On CREATE the weights are already where they were written, and there is no
# route to scp from `create` to itself, so only fetch when running elsewhere.
ON_CREATE=0
[ -d "$REMOTE_ROOT" ] && ON_CREATE=1

for run in "$ZEISS" "$CANON" "$RIMONE"; do
  if [ -f "artifacts/runs/$run/best_model.pt" ]; then
    echo "have  artifacts/runs/$run/best_model.pt"
  elif [ "$ON_CREATE" = 1 ] || [ "${1:-}" = "--skip-pull" ]; then
    echo "MISSING artifacts/runs/$run/best_model.pt" >&2
    exit 1
  else
    echo "pull  $run/best_model.pt"
    scp "$REMOTE:$REMOTE_ROOT/$run/best_model.pt" "artifacts/runs/$run/best_model.pt"
  fi
done

# The worst cell in the matrix, and the pair that is photometrically almost
# identical: if these failures look geometric, that is the image-level support
# for the brief's conclusion.
"${PY[@]}" visualize_predictions.py --run-dir "artifacts/runs/$ZEISS" \
  --target-domain drishti_gs --worst 4 --median 2 --best 2 \
  --output-dir "$OUT/zeiss_to_drishti"

# One pair, both directions, 0.891 vs 0.733. A symmetric photometric story cannot
# explain the gap, so the two directions belong side by side.
"${PY[@]}" visualize_predictions.py --run-dir "artifacts/runs/$ZEISS" \
  --target-domain refuge_canon_val --worst 3 --best 1 \
  --output-dir "$OUT/zeiss_to_canon"
"${PY[@]}" visualize_predictions.py --run-dir "artifacts/runs/$CANON" \
  --target-domain refuge_zeiss --worst 3 --best 1 \
  --output-dir "$OUT/canon_to_zeiss"

# The LODO collapse. Expect predictions at entirely the wrong scale.
"${PY[@]}" visualize_predictions.py --run-dir "artifacts/runs/$RIMONE" \
  --worst 4 --median 2 \
  --output-dir "$OUT/lodo_rim_one_dl"

# A control. Failure figures on their own are not evidence.
"${PY[@]}" visualize_predictions.py --run-dir "artifacts/runs/$CANON" \
  --target-domain drishti_gs --worst 2 --median 2 --best 2 \
  --output-dir "$OUT/canon_to_drishti_control"

echo
echo "figures under $OUT/"
