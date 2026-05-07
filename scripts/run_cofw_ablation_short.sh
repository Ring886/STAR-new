#!/bin/bash
set -euo pipefail

export PATH=/root/miniconda3/bin:$PATH
export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
export DEVICE_IDS="${DEVICE_IDS:-0}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
export IMAGE_DIR="${IMAGE_DIR:-./image_dir}"
export ANNOT_DIR="${ANNOT_DIR:-./annot_dir}"
export PRETRAINED_WEIGHT="${PRETRAINED_WEIGHT:-model/COFW_STARLoss_NME_4_62.pkl}"
export RUN_ROOT="${RUN_ROOT:-./ablation_runs}"
export LOG_ROOT="${LOG_ROOT:-./ablation_logs}"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

run_variant() {
  local name="$1"
  local strategy="$2"
  local lr="$3"
  local epochs="$4"
  local out_dir="$RUN_ROOT/$name"
  local log_file="$LOG_ROOT/${name}.log"
  echo "========== ${name} ==========" | tee "$log_file"
  echo "strategy=${strategy}, lr=${lr}, epochs=${epochs}, batch=${BATCH_SIZE}" | tee -a "$log_file"
  "$PYTHON_BIN" main.py --mode=train \
    --device_ids="$DEVICE_IDS" \
    --batch_size="$BATCH_SIZE" \
    --val_batch_size="$VAL_BATCH_SIZE" \
    --train_num_workers="$NUM_WORKERS" \
    --val_num_workers="$NUM_WORKERS" \
    --learn_rate="$lr" \
    --max_epoch="$epochs" \
    --image_dir="$IMAGE_DIR" \
    --annot_dir="$ANNOT_DIR" \
    --data_definition=COFW \
    --pretrained_weight="$PRETRAINED_WEIGHT" \
    --fine_tune_strategy="$strategy" \
    --ckpt_dir="$out_dir" 2>&1 | tee -a "$log_file"
}

run_variant full_lr3e4_ep3 full 0.0003 3
run_variant heads_lr3e4_ep3 heads_only 0.0003 3
run_variant full_lr1e4_ep3 full 0.0001 3
