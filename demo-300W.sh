# 300W 数据集 demo 启动脚本，指定模型、输入图片和输出路径，运行 demo.py 生成绿色关键点可视化。
python demo.py \
  --model_path=model/best_model_300W.pkl \
  --data_definition=300W \
  --device_ids=0 \
  --predictor_path=model/shape_predictor_68_face_landmarks.dat \
  --my_image_path=my_images/2.png \
  --out_image_path=out_image-300W.png \
  --draw_radius=1
