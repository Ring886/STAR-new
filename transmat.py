import h5py
import numpy as np
import os
import json
from PIL import Image

# --- 配置路径 ---
mat_file_path = '/Users/mac/Downloads/COFW_color/COFW_train_color.mat'
output_dir = '/Users/mac/Downloads/COFW_color_dataset/train'
output_img_dir = os.path.join(output_dir, 'images')

# 创建输出文件夹
os.makedirs(output_img_dir, exist_ok=True)

annotations = []

print("开始解析和导出数据...")

with h5py.File(mat_file_path, 'r') as f:
    keys = set(f.keys())
    if {'IsTr', 'bboxesTr', 'phisTr'}.issubset(keys):
        image_key, bbox_key, phi_key, split_name = 'IsTr', 'bboxesTr', 'phisTr', 'train'
    elif {'IsT', 'bboxesT', 'phisT'}.issubset(keys):
        image_key, bbox_key, phi_key, split_name = 'IsT', 'bboxesT', 'phisT', 'test'
    else:
        raise KeyError(f"MAT 文件缺少预期字段，实际字段: {sorted(keys)}")

    is_tr = f[image_key]
    bboxes = f[bbox_key]
    phis = f[phi_key]
    
    num_images = is_tr.shape[1] # 1345张图
    
    for i in range(num_images):
        # ==========================================
        # 1. 提取并保存图片
        # ==========================================
        # 获取第 i 张图片的“指针”引用
        img_ref = is_tr[0, i]
        # 顺着指针拿到真实的矩阵数据
        img_data = np.array(f[img_ref])
        
        # 【关键点】MATLAB 与 Python 的维度转换
        # MATLAB存的是 (通道, 宽, 高) 比如 (3, W, H)
        # Python/PIL需要的是 (高, 宽, 通道) 也就是 (H, W, 3)
        if img_data.ndim == 3:
            img_data = img_data.transpose(2, 1, 0)
        elif img_data.ndim == 2:
            # 如果是灰度图，MATLAB存的是 (W, H)，转为 (H, W)
            img_data = img_data.transpose(1, 0)
            
        # 转成图片格式并保存
        img = Image.fromarray(img_data.astype('uint8'))
        img_name = f"{i}.png"
        img.save(os.path.join(output_img_dir, img_name))
        
        # ==========================================
        # 2. 提取标签 (Bounding Box 和 关键点)
        # ==========================================
        # bbox 格式通常是 [x, y, w, h]
        bbox = bboxes[:, i].tolist()
        
        # COFW 有 29 个关键点，phisTr的87维 = 29(x) + 29(y) + 29(遮挡状态)
        phi = phis[:, i].tolist()
        
        # 组装这大概图片的标注信息
        anno = {
            "image_id": img_name,
            "bbox": bbox,
            "keypoints": phi
        }
        annotations.append(anno)
        
        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1} / {num_images} 张图片...")

# 把所有标签写入一个 JSON 文件
json_path = os.path.join(output_dir, 'annotations.json')
with open(json_path, 'w') as jf:
    json.dump(annotations, jf, indent=4)

print(f"\n转换完成！数据集已保存至: {output_dir}")
print(f"包含 {num_images} 张图片 和 1 个 annotations.json 标签文件。")
