#!/bin/bash

# High-quality COFW optimization run starting from the original best checkpoint.
# Strategy:
# 1) Keep the original dev_demo_0.0.1 model architecture and training recipe.
# 2) Fine-tune from model/COFW_STARLoss_NME_4_62.pkl instead of training from scratch.
# 3) Use a smaller total batch size (16), which matched the best historical COFW runs in local logs.
# 4) Reset optimizer/scheduler state so the lower fine-tuning learning rate actually takes effect.

DEVICE_IDS="0,1,2,3"
BATCH_SIZE=64
NUM_WORKERS=16
LEARN_RATE=0.0001
IMAGE_DIR="./image_dir"
ANNOT_DIR="./annot_dir"
OUT_DIR="./out_dir"
PRETRAINED_WEIGHT="model/COFW_STARLoss_NME_4_62.pkl"

echo "Starting COFW optimization fine-tune from ${PRETRAINED_WEIGHT}"
echo "DEVICE_IDS=${DEVICE_IDS} BATCH_SIZE=${BATCH_SIZE} NUM_WORKERS=${NUM_WORKERS} LEARN_RATE=${LEARN_RATE}"

python3 main.py --mode=train \
               --device_ids=${DEVICE_IDS} \
               --batch_size=${BATCH_SIZE} \
               --train_num_workers=${NUM_WORKERS} \
               --learn_rate=${LEARN_RATE} \
               --image_dir=${IMAGE_DIR} \
               --annot_dir=${ANNOT_DIR} \
               --data_definition=COFW \
               --pretrained_weight=${PRETRAINED_WEIGHT} \
               --ckpt_dir=${OUT_DIR}
