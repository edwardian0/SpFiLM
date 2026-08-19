#!/bin/bash

set -euo pipefail

DATA_ROOT="/scratch/prj/bc_ca_segmentation_in_tb_anatomy/glaucoma_datasets/DRISHTI-GS"

# discover_drishti keeps these literals inside the function rather than exposing
# module-level constants, so this CREATE-only verifier mirrors them exactly.
TRAIN_IMAGES_REL="Training-20211018T055246Z-001/Training/Images"
TRAIN_GT_REL="Training-20211018T055246Z-001/Training/GT"
TEST_IMAGES_REL="Test-20211018T060000Z-001/Test/Images"
TEST_GT_REL="Test-20211018T060000Z-001/Test/Test_GT"

fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

if [[ ! -d "$DATA_ROOT" ]]; then
  fail "data_root is not a directory: $DATA_ROOT"
fi
if [[ ! -r "$DATA_ROOT" || ! -x "$DATA_ROOT" ]]; then
  fail "data_root is not readable/searchable: $DATA_ROOT"
fi

TRAIN_IMAGES="$DATA_ROOT/$TRAIN_IMAGES_REL"
TRAIN_GT="$DATA_ROOT/$TRAIN_GT_REL"
TEST_IMAGES="$DATA_ROOT/$TEST_IMAGES_REL"
TEST_GT="$DATA_ROOT/$TEST_GT_REL"

for expected_dir in "$TRAIN_IMAGES" "$TRAIN_GT" "$TEST_IMAGES" "$TEST_GT"; do
  if [[ ! -d "$expected_dir" ]]; then
    fail "expected directory is missing (exact spelling required): $expected_dir"
  fi
done

TRAIN_IMAGE_PATHS=()
while IFS= read -r -d '' image_path; do
  TRAIN_IMAGE_PATHS+=("$image_path")
done < <(
  find "$TRAIN_IMAGES" -type f \
    \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) \
    ! -name '._*' ! -path '*/__MACOSX/*' -print0 | sort -z
)

TEST_IMAGE_PATHS=()
while IFS= read -r -d '' image_path; do
  TEST_IMAGE_PATHS+=("$image_path")
done < <(
  find "$TEST_IMAGES" -type f \
    \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) \
    ! -name '._*' ! -path '*/__MACOSX/*' -print0 | sort -z
)

if (( ${#TRAIN_IMAGE_PATHS[@]} != 50 )); then
  fail "provider_train image count mismatch: expected 50, actual ${#TRAIN_IMAGE_PATHS[@]} ($TRAIN_IMAGES)"
fi
if (( ${#TEST_IMAGE_PATHS[@]} != 51 )); then
  fail "provider_test image count mismatch: expected 51, actual ${#TEST_IMAGE_PATHS[@]} ($TEST_IMAGES)"
fi

TRAIN_GT_COUNT=$(find "$TRAIN_GT" -mindepth 1 -maxdepth 1 -type d \
  ! -name '__MACOSX' -print | wc -l | tr -d '[:space:]')
if (( TRAIN_GT_COUNT != ${#TRAIN_IMAGE_PATHS[@]} )); then
  fail "provider_train GT directory count mismatch: expected ${#TRAIN_IMAGE_PATHS[@]} to match images, actual $TRAIN_GT_COUNT ($TRAIN_GT)"
fi

TEST_GT_COUNT=$(find "$TEST_GT" -mindepth 1 -maxdepth 1 -type d \
  ! -name '__MACOSX' -print | wc -l | tr -d '[:space:]')
if (( TEST_GT_COUNT != ${#TEST_IMAGE_PATHS[@]} )); then
  fail "provider_test GT directory count mismatch: expected ${#TEST_IMAGE_PATHS[@]} to match images, actual $TEST_GT_COUNT ($TEST_GT)"
fi

check_image_gt_pairs() {
  local split_name=$1
  local gt_root=$2
  shift 2
  local image_path base sample_id
  local orphan_count=0
  local first_five=()
  for image_path in "$@"; do
    base=${image_path##*/}
    sample_id=${base%.*}
    if [[ ! -d "$gt_root/$sample_id" ]]; then
      ((orphan_count += 1))
      if (( ${#first_five[@]} < 5 )); then
        first_five+=("$sample_id")
      fi
    fi
  done
  if (( orphan_count > 0 )); then
    fail "$split_name has $orphan_count image stem(s) without matching GT directories; first 5: ${first_five[*]}"
  fi
}

check_image_gt_pairs "provider_train" "$TRAIN_GT" "${TRAIN_IMAGE_PATHS[@]}"
check_image_gt_pairs "provider_test" "$TEST_GT" "${TEST_IMAGE_PATHS[@]}"

sample_id_from_path() {
  local base=${1##*/}
  printf '%s\n' "${base%.*}"
}

SAMPLE_IDS=(
  "$(sample_id_from_path "${TRAIN_IMAGE_PATHS[0]}")"
  "$(sample_id_from_path "${TRAIN_IMAGE_PATHS[${#TRAIN_IMAGE_PATHS[@]} / 2]}")"
  "$(sample_id_from_path "${TEST_IMAGE_PATHS[${#TEST_IMAGE_PATHS[@]} - 1]}")"
)
SAMPLE_GT_ROOTS=("$TRAIN_GT" "$TRAIN_GT" "$TEST_GT")

for index in 0 1 2; do
  sample_id=${SAMPLE_IDS[$index]}
  softmap_root="${SAMPLE_GT_ROOTS[$index]}/$sample_id/SoftMap"
  disc_path="$softmap_root/${sample_id}_ODsegSoftmap.png"
  cup_path="$softmap_root/${sample_id}_cupsegSoftmap.png"
  if [[ ! -f "$disc_path" ]]; then
    fail "sampled disc soft map is missing for $sample_id: $disc_path"
  fi
  if [[ ! -f "$cup_path" ]]; then
    fail "sampled cup soft map is missing for $sample_id: $cup_path"
  fi
done

printf 'PASS: provider_train images/GT = 50/50; provider_test images/GT = 51/51\n'
printf 'PASS: sampled soft-map pairs exist for: %s\n' "${SAMPLE_IDS[*]}"
printf 'HOME_USAGE (du -sh ~): '
if ! du -sh ~; then
  fail "du -sh ~ failed"
fi
printf 'PASS: Drishti-GS path verification completed\n'
