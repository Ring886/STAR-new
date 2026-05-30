# COFW 数据集评估启动脚本，指定权重、test.tsv 和图片目录，运行 evaluate.py 计算遮挡场景 NME。
python evaluate.py --device_ids=0 \
                   --model_path=model/best_model_COFW.pkl \
                   --metadata_path=annot_dir/COFW/test.tsv \
                   --image_dir=image_dir \
                   --data_definition=COFW
