#!/bin/bash
set -euo pipefail

export PATH=/root/miniconda3/bin:$PATH
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DEVICE_IDS="${DEVICE_IDS:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-16}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
IMAGE_DIR="${IMAGE_DIR:-./image_dir}"
ANNOT_DIR="${ANNOT_DIR:-./annot_dir}"
PRETRAINED_WEIGHT="${PRETRAINED_WEIGHT:-model/300W_STARLoss_NME_2_87.pkl}"
RUN_ROOT="${RUN_ROOT:-./ablation_runs_300W}"
LOG_ROOT="${LOG_ROOT:-./ablation_logs_300W}"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${LOG_ROOT}/test"

run_test() {
  local name="$1"
  local weight="$2"
  local log_file="${LOG_ROOT}/test/${name}_test.log"
  echo "========== TEST ${name} ==========" | tee "${log_file}"
  echo "weight=${weight}" | tee -a "${log_file}"
  "${PYTHON_BIN}" main.py --mode=test \
    --device_ids="${DEVICE_IDS}" \
    --image_dir="${IMAGE_DIR}" \
    --annot_dir="${ANNOT_DIR}" \
    --data_definition=300W \
    --pretrained_weight="${weight}" \
    --ckpt_dir=out_dir 2>&1 | tee -a "${log_file}"
}

run_variant() {
  local name="$1"
  local strategy="$2"
  local lr="$3"
  local epochs="$4"
  local out_dir="${RUN_ROOT}/${name}"
  local log_file="${LOG_ROOT}/${name}.log"

  echo "========== TRAIN ${name} ==========" | tee "${log_file}"
  echo "strategy=${strategy}, lr=${lr}, epochs=${epochs}, batch=${BATCH_SIZE}" | tee -a "${log_file}"

  "${PYTHON_BIN}" main.py --mode=train \
    --device_ids="${DEVICE_IDS}" \
    --batch_size="${BATCH_SIZE}" \
    --val_batch_size="${VAL_BATCH_SIZE}" \
    --train_num_workers="${NUM_WORKERS}" \
    --val_num_workers="${NUM_WORKERS}" \
    --learn_rate="${lr}" \
    --max_epoch="${epochs}" \
    --fine_tune_strategy="${strategy}" \
    --image_dir="${IMAGE_DIR}" \
    --annot_dir="${ANNOT_DIR}" \
    --data_definition=300W \
    --pretrained_weight="${PRETRAINED_WEIGHT}" \
    --ckpt_dir="${out_dir}" 2>&1 | tee -a "${log_file}"
}

best_model_for() {
  local name="$1"
  find "${RUN_ROOT}/${name}" -path '*/model/best_model.pkl' | sort | tail -1
}

run_test author_baseline "${PRETRAINED_WEIGHT}"
run_variant heads_lr5e5_ep3 heads_only 0.00005 3
run_variant full_lr5e5_ep3 full 0.00005 3
run_variant heads_lr1e4_ep3 heads_only 0.0001 3

for name in heads_lr5e5_ep3 full_lr5e5_ep3 heads_lr1e4_ep3; do
  model_path="$(best_model_for "${name}")"
  if [ -z "${model_path}" ]; then
    echo "ERROR: best model not found for ${name}" >&2
    exit 1
  fi
  run_test "${name}" "${model_path}"
done

echo "All 300W ablation runs finished. Logs: ${LOG_ROOT}"
