#!/bin/bash
set -euo pipefail

# Runs four COFW 50-epoch ablation jobs sequentially:
#   CoordConv on/off x SmoothL1/L2, with AAM disabled.

RUN_ROOT="${RUN_ROOT:-./ablation_runs_coordconv_noaam_loss}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-16}"
LEARN_RATE="${LEARN_RATE:-0.001}"
MAX_EPOCH="${MAX_EPOCH:-50}"
IMAGE_DIR="${IMAGE_DIR:-./image_dir}"
ANNOT_DIR="${ANNOT_DIR:-./annot_dir}"

mkdir -p "${RUN_ROOT}/logs"

run_one() {
  local loss_func="$1"
  local add_coord="$2"
  local tag="$3"
  local out_dir="${RUN_ROOT}/${tag}"
  local log="${RUN_ROOT}/logs/${tag}.log"

  mkdir -p "${out_dir}"
  echo "==== $(date '+%F %T') START ${tag} ====" | tee -a "${RUN_ROOT}/run_all.log"
  env PYTHON_BIN="${PYTHON_BIN}" \
      BATCH_SIZE="${BATCH_SIZE}" \
      NUM_WORKERS="${NUM_WORKERS}" \
      LEARN_RATE="${LEARN_RATE}" \
      MAX_EPOCH="${MAX_EPOCH}" \
      LOSS_FUNC="${loss_func}" \
      ADD_COORD="${add_coord}" \
      USE_AAM=0 \
      IMAGE_DIR="${IMAGE_DIR}" \
      ANNOT_DIR="${ANNOT_DIR}" \
      OUT_DIR="${out_dir}" \
      bash train-COFW-coordconv-ablation.sh > "${log}" 2>&1
  echo "==== $(date '+%F %T') FINISH ${tag} ====" | tee -a "${RUN_ROOT}/run_all.log"
}

run_one smoothl1 1 smoothl1_with_coord
run_one smoothl1 0 smoothl1_no_coord
run_one l2 1 l2_with_coord
run_one l2 0 l2_no_coord

echo "All COFW CoordConv/no-AAM/loss ablations finished." | tee -a "${RUN_ROOT}/run_all.log"
