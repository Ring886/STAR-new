#!/bin/bash
set -euo pipefail

# opt-3-gpu-running: 300W fine-tuning script for GPU servers.
# It mirrors train-COFW.sh, but switches the dataset definition and the author's
# starting checkpoint to 300W. Paths keep pointing to the common project roots
# because STAR-new expects IMAGE_DIR/ANNOT_DIR to be the parent directories that
# contain 300W/, COFW/, etc.; do not set them directly to image_dir/300W.
#
# Usage:
#   bash train-300W.sh
#   DRY_RUN=1 bash train-300W.sh        # only print the command, do not train
#   MAX_EPOCH=5 BATCH_SIZE=16 bash train-300W.sh

if [ -z "${PYTHON_BIN:-}" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif [ -x /root/miniconda3/bin/python ]; then
    PYTHON_BIN="/root/miniconda3/bin/python"
  elif [ -x /root/anaconda3/bin/python ]; then
    PYTHON_BIN="/root/anaconda3/bin/python"
  else
    echo "ERROR: Python interpreter not found. Set PYTHON_BIN=/path/to/python" >&2
    exit 1
  fi
fi

if [ -z "${DEVICE_IDS:-}" ]; then
  GPU_COUNT=$("${PYTHON_BIN}" - <<'PY_DEVICE'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY_DEVICE
)
  if [ "${GPU_COUNT}" -gt 0 ]; then
    DEVICE_IDS=$("${PYTHON_BIN}" - <<'PY_DEVICE'
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
LEARN_RATE="${LEARN_RATE:-0.00005}"
MAX_EPOCH="${MAX_EPOCH:-20}"
FINE_TUNE_STRATEGY="${FINE_TUNE_STRATEGY:-heads_only}"
IMAGE_DIR="${IMAGE_DIR:-./image_dir}"
ANNOT_DIR="${ANNOT_DIR:-./annot_dir}"
OUT_DIR="${OUT_DIR:-./out_dir}"
PRETRAINED_WEIGHT="${PRETRAINED_WEIGHT:-model/300W_STARLoss_NME_2_87.pkl}"
DRY_RUN="${DRY_RUN:-0}"

if [ ! -d "${IMAGE_DIR}/300W" ]; then
  echo "ERROR: 300W image directory not found: ${IMAGE_DIR}/300W" >&2
  exit 1
fi

if [ ! -d "${ANNOT_DIR}/300W" ]; then
  echo "ERROR: 300W annotation directory not found: ${ANNOT_DIR}/300W" >&2
  exit 1
fi

if [ ! -f "${PRETRAINED_WEIGHT}" ]; then
  echo "ERROR: pretrained checkpoint not found: ${PRETRAINED_WEIGHT}" >&2
  exit 1
fi

CMD=("${PYTHON_BIN}" main.py --mode=train
  --device_ids="${DEVICE_IDS}"
  --batch_size="${BATCH_SIZE}"
  --val_batch_size=32
  --train_num_workers="${NUM_WORKERS}"
  --val_num_workers="${NUM_WORKERS}"
  --learn_rate="${LEARN_RATE}"
  --max_epoch="${MAX_EPOCH}"
  --fine_tune_strategy="${FINE_TUNE_STRATEGY}"
  --image_dir="${IMAGE_DIR}"
  --annot_dir="${ANNOT_DIR}"
  --data_definition=300W
  --pretrained_weight="${PRETRAINED_WEIGHT}"
  --ckpt_dir="${OUT_DIR}")

echo "Starting opt-3-gpu-running 300W fine-tuning"
echo "  GPU_COUNT=${GPU_COUNT:-manual}"
echo "  PYTHON_BIN=${PYTHON_BIN}"
echo "  DEVICE_IDS=${DEVICE_IDS}"
echo "  BATCH_SIZE=${BATCH_SIZE}"
echo "  NUM_WORKERS=${NUM_WORKERS}"
echo "  LEARN_RATE=${LEARN_RATE}"
echo "  MAX_EPOCH=${MAX_EPOCH}"
echo "  FINE_TUNE_STRATEGY=${FINE_TUNE_STRATEGY}"
echo "  IMAGE_DIR=${IMAGE_DIR}"
echo "  ANNOT_DIR=${ANNOT_DIR}"
echo "  PRETRAINED_WEIGHT=${PRETRAINED_WEIGHT}"
printf '  COMMAND='
printf ' %q' "${CMD[@]}"
printf '\n'

if [ "${DRY_RUN}" = "1" ]; then
  echo "DRY_RUN=1, skip training."
  exit 0
fi

"${CMD[@]}"
