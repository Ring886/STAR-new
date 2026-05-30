# COFW 数据集 demo 启动脚本，指定模型、输入图片和输出路径，运行 demo.py 生成遮挡场景关键点可视化。
python demo.py \
  --model_path=model/best_model_COFW.pkl \
  --data_definition=COFW \
  --device_ids=0 \
  --predictor_path=model/shape_predictor_68_face_landmarks.dat \
  --my_image_path=my_images/3.png \
  --out_image_path=out_image-COFW.png \
  --draw_radius=1
