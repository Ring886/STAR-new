#!/bin/bash
set -euo pipefail

# opt-3 conservative COFW fine-tuning recipe.
# Goal: start from the author's strong checkpoint (NME ~= 0.046249) and fine-tune
# with a smaller learning rate, without restoring the old optimizer/scheduler.

DEVICE_IDS="${DEVICE_IDS:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LEARN_RATE="${LEARN_RATE:-0.0003}"
MAX_EPOCH="${MAX_EPOCH:-120}"
IMAGE_DIR="${IMAGE_DIR:-./image_dir}"
ANNOT_DIR="${ANNOT_DIR:-./annot_dir}"
OUT_DIR="${OUT_DIR:-./out_dir}"
PRETRAINED_WEIGHT="${PRETRAINED_WEIGHT:-model/COFW_STARLoss_NME_4_62.pkl}"

if [ ! -f "${PRETRAINED_WEIGHT}" ]; then
  echo "ERROR: pretrained checkpoint not found: ${PRETRAINED_WEIGHT}" >&2
  exit 1
fi

echo "Starting opt-3 COFW fine-tuning"
echo "  DEVICE_IDS=${DEVICE_IDS}"
echo "  BATCH_SIZE=${BATCH_SIZE}"
echo "  NUM_WORKERS=${NUM_WORKERS}"
echo "  LEARN_RATE=${LEARN_RATE}"
echo "  MAX_EPOCH=${MAX_EPOCH}"
echo "  PRETRAINED_WEIGHT=${PRETRAINED_WEIGHT}"

python3 main.py --mode=train \
  --device_ids="${DEVICE_IDS}" \
  --batch_size="${BATCH_SIZE}" \
  --val_batch_size=32 \
  --train_num_workers="${NUM_WORKERS}" \
  --val_num_workers="${NUM_WORKERS}" \
  --learn_rate="${LEARN_RATE}" \
  --max_epoch="${MAX_EPOCH}" \
  --image_dir="${IMAGE_DIR}" \
  --annot_dir="${ANNOT_DIR}" \
  --data_definition=COFW \
  --pretrained_weight="${PRETRAINED_WEIGHT}" \
  --ckpt_dir="${OUT_DIR}"
