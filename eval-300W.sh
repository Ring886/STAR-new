# 300W 数据集评估启动脚本，指定权重、test.tsv 和图片目录，运行 evaluate.py 计算 NME。
python evaluate.py --device_ids=0 \
                   --model_path=model/best_model_300W.pkl \
                   --metadata_path=annot_dir/300W/test.tsv \
                   --image_dir=image_dir \
                   --data_definition=300W
