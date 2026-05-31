#!/bin/bash
set -euo pipefail

# COFW CoordConv ablation. By default this trains the no-CoordConv variant from
# scratch so it can be compared with the existing 50-epoch STARLoss experiment.

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
LEARN_RATE="${LEARN_RATE:-0.001}"
MAX_EPOCH="${MAX_EPOCH:-50}"
LOSS_FUNC="${LOSS_FUNC:-STARLoss_v2}"
IMAGE_DIR="${IMAGE_DIR:-./image_dir}"
ANNOT_DIR="${ANNOT_DIR:-./annot_dir}"
OUT_DIR="${OUT_DIR:-./out_dir}"
ADD_COORD="${ADD_COORD:-0}"
USE_AAM="${USE_AAM:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [ ! -d "${IMAGE_DIR}/COFW" ]; then
  echo "ERROR: COFW image directory not found: ${IMAGE_DIR}/COFW" >&2
  exit 1
fi

if [ ! -d "${ANNOT_DIR}/COFW" ]; then
  echo "ERROR: COFW annotation directory not found: ${ANNOT_DIR}/COFW" >&2
  exit 1
fi

COORD_FLAG="--no_add_coord"
if [ "${ADD_COORD}" = "1" ]; then
  COORD_FLAG="--add_coord"
fi

AAM_FLAG="--use_AAM"
if [ "${USE_AAM}" = "0" ]; then
  AAM_FLAG="--no_use_AAM"
fi

CMD=("${PYTHON_BIN}" main.py --mode=train
  --device_ids="${DEVICE_IDS}"
  --batch_size="${BATCH_SIZE}"
  --val_batch_size=32
  --train_num_workers="${NUM_WORKERS}"
  --val_num_workers="${NUM_WORKERS}"
  --learn_rate="${LEARN_RATE}"
  --max_epoch="${MAX_EPOCH}"
  --loss_func="${LOSS_FUNC}"
  --image_dir="${IMAGE_DIR}"
  --annot_dir="${ANNOT_DIR}"
  --data_definition=COFW
  --ckpt_dir="${OUT_DIR}"
  "${COORD_FLAG}"
  "${AAM_FLAG}")

echo "Starting COFW CoordConv ablation"
echo "  GPU_COUNT=${GPU_COUNT:-manual}"
echo "  PYTHON_BIN=${PYTHON_BIN}"
echo "  DEVICE_IDS=${DEVICE_IDS}"
echo "  BATCH_SIZE=${BATCH_SIZE}"
echo "  NUM_WORKERS=${NUM_WORKERS}"
echo "  LEARN_RATE=${LEARN_RATE}"
echo "  MAX_EPOCH=${MAX_EPOCH}"
echo "  LOSS_FUNC=${LOSS_FUNC}"
echo "  ADD_COORD=${ADD_COORD}"
echo "  USE_AAM=${USE_AAM}"
echo "  IMAGE_DIR=${IMAGE_DIR}"
echo "  ANNOT_DIR=${ANNOT_DIR}"
printf '  COMMAND='
printf ' %q' "${CMD[@]}"
printf '\n'

if [ "${DRY_RUN}" = "1" ]; then
  echo "DRY_RUN=1, skip training."
  exit 0
fi

"${CMD[@]}"
