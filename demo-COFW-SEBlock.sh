#!/bin/bash
python demo.py \
  --model_path=model/best_model_SEBlock.pkl \
  --data_definition=COFW \
  --device_ids=0 \
  --predictor_path=model/shape_predictor_68_face_landmarks.dat \
  --my_image_path=my_images/1.png \
  --out_image_path=out_image-COFW-SEBlock.png
