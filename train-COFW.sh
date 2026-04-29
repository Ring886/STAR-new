#!/bin/bash
set -euo pipefail

# opt-3-gpu-running: COFW fine-tuning script for GPU servers.
# It starts from the author's strong checkpoint and fine-tunes with a smaller LR.
# Safe defaults are tuned for a single-GPU AutoDL/SeetaCloud instance, but the
# script auto-detects visible CUDA GPUs and also allows env overrides.

if [ -z "${DEVICE_IDS:-}" ]; then
  GPU_COUNT=$(python3 - <<'PY_DEVICE'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY_DEVICE
)
  if [ "${GPU_COUNT}" -gt 0 ]; then
    DEVICE_IDS=$(python3 - <<'PY_DEVICE'
import torch
print(",".join(str(i) for i in range(torch.cuda.device_count())))
PY_DEVICE
)
  else
    DEVICE_IDS="-1"
  fi
fi

BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-16}"
LEARN_RATE="${LEARN_RATE:-0.0001}"
MAX_EPOCH="${MAX_EPOCH:-30}"
IMAGE_DIR="${IMAGE_DIR:-./image_dir}"
ANNOT_DIR="${ANNOT_DIR:-./annot_dir}"
OUT_DIR="${OUT_DIR:-./out_dir}"
PRETRAINED_WEIGHT="${PRETRAINED_WEIGHT:-model/COFW_STARLoss_NME_4_62.pkl}"

if [ ! -f "${PRETRAINED_WEIGHT}" ]; then
  echo "ERROR: pretrained checkpoint not found: ${PRETRAINED_WEIGHT}" >&2
  exit 1
fi

echo "Starting opt-3-gpu-running COFW fine-tuning"
echo "  GPU_COUNT=${GPU_COUNT:-manual}"
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
