#!/bin/bash
set -euo pipefail

# opt-2-gpu-running: COFW SEBlock short fine-tuning script.
# This branch contains SEBlock changes. Prior SEBlock runs often saved best_model
# at epoch 1, so default MAX_EPOCH is intentionally short and all knobs are
# environment-variable overridable.

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
MAX_EPOCH="${MAX_EPOCH:-5}"
IMAGE_DIR="${IMAGE_DIR:-./image_dir}"
ANNOT_DIR="${ANNOT_DIR:-./annot_dir}"
OUT_DIR="${OUT_DIR:-./out_dir}"
PRETRAINED_WEIGHT="${PRETRAINED_WEIGHT:-}"

# If a SEBlock-compatible best checkpoint exists, use it as the default starting point.
# Do not default to the author's no-SE checkpoint because it is architecture-incompatible
# with this SEBlock branch's added *.se.fc.* parameters.
if [ -z "${PRETRAINED_WEIGHT}" ] && [ -f model/best_model_COFW.pkl ]; then
  PRETRAINED_WEIGHT="model/best_model_COFW.pkl"
fi

ARGS=(
  main.py --mode=train
  --device_ids="${DEVICE_IDS}"
  --batch_size="${BATCH_SIZE}"
  --val_batch_size=32
  --train_num_workers="${NUM_WORKERS}"
  --val_num_workers="${NUM_WORKERS}"
  --learn_rate="${LEARN_RATE}"
  --max_epoch="${MAX_EPOCH}"
  --image_dir="${IMAGE_DIR}"
  --annot_dir="${ANNOT_DIR}"
  --data_definition=COFW
  --ckpt_dir="${OUT_DIR}"
)

if [ -n "${PRETRAINED_WEIGHT}" ]; then
  if [ ! -f "${PRETRAINED_WEIGHT}" ]; then
    echo "ERROR: pretrained checkpoint not found: ${PRETRAINED_WEIGHT}" >&2
    exit 1
  fi
  ARGS+=(--pretrained_weight="${PRETRAINED_WEIGHT}")
fi

if [ "${RESUME_TRAINING_STATE:-0}" = "1" ]; then
  ARGS+=(--resume_training_state)
fi

echo "Starting opt-2-gpu-running COFW SEBlock short fine-tuning"
echo "  GPU_COUNT=${GPU_COUNT:-manual}"
echo "  PYTHON_BIN=${PYTHON_BIN}"
echo "  DEVICE_IDS=${DEVICE_IDS}"
echo "  BATCH_SIZE=${BATCH_SIZE}"
echo "  NUM_WORKERS=${NUM_WORKERS}"
echo "  LEARN_RATE=${LEARN_RATE}"
echo "  MAX_EPOCH=${MAX_EPOCH}"
echo "  PRETRAINED_WEIGHT=${PRETRAINED_WEIGHT:-<none; training from scratch>}"
echo "  RESUME_TRAINING_STATE=${RESUME_TRAINING_STATE:-0}"

"${PYTHON_BIN}" "${ARGS[@]}"
