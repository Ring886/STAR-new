#!/bin/bash

# ==============================================================================
# Training Script for COFW Dataset
# You can modify the variables below depending on your machine (Local Mac vs GPU Server)
# ==============================================================================

# Device configuration: 
# - For Mac (CPU/MPS): set to "0" or "-1"
# - For GPU Server (Multi-GPU): set to "0,1,2,3"
DEVICE_IDS="0,1,2,3"

# Batch size:
# - For Mac (Limited Memory): 8 or 16
# - For GPU Server: 32 or 64
BATCH_SIZE=64

# Dataloader workers:
# - For Mac: 0 or 2
# - For GPU Server: 4 or 8
NUM_WORKERS=16

# Directories
IMAGE_DIR="./image_dir"
ANNOT_DIR="./annot_dir"
OUT_DIR="./out_dir"

echo "Starting training on COFW with BATCH_SIZE=${BATCH_SIZE}, DEVICE_IDS=${DEVICE_IDS}, NUM_WORKERS=${NUM_WORKERS}"

python3 main.py --mode=train \
               --device_ids=${DEVICE_IDS} \
               --batch_size=${BATCH_SIZE} \
               --train_num_workers=${NUM_WORKERS} \
               --image_dir=${IMAGE_DIR} \
               --annot_dir=${ANNOT_DIR} \
               --data_definition=COFW \
               --ckpt_dir=${OUT_DIR}
