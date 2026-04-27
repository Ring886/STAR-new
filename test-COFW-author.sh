#!/bin/bash
set -euo pipefail

DEVICE_IDS="${DEVICE_IDS:-0}"
IMAGE_DIR="${IMAGE_DIR:-./image_dir}"
ANNOT_DIR="${ANNOT_DIR:-./annot_dir}"
OUT_DIR="${OUT_DIR:-./out_dir}"
PRETRAINED_WEIGHT="${PRETRAINED_WEIGHT:-model/COFW_STARLoss_NME_4_62.pkl}"

python3 main.py --mode=test \
  --device_ids="${DEVICE_IDS}" \
  --image_dir="${IMAGE_DIR}" \
  --annot_dir="${ANNOT_DIR}" \
  --data_definition=COFW \
  --pretrained_weight="${PRETRAINED_WEIGHT}" \
  --ckpt_dir="${OUT_DIR}"
