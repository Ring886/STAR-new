#!/bin/bash
python main.py --mode=test --device_ids=0 \
               --image_dir=image_dir --annot_dir=annot_dir \
               --data_definition=COFW \
               --pretrained_weight=model/best_model_COFW.pkl \
               --ckpt_dir=out_dir
