#!/bin/bash
python evaluate.py --device_ids=0 \
                   --model_path=model/best_model_SEBlock.pkl \
                   --metadata_path=annot_dir/COFW/test.tsv \
                   --image_dir=image_dir \
                   --data_definition=COFW
