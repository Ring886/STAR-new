# 预测阶段的热力图解码器，把模型输出的关键点热力图通过加权平均转换成连续坐标。
import torch


class decoder_default:
    def __init__(self, weight=1, use_weight_map=False):
        self.weight = weight
        self.use_weight_map = use_weight_map

    def _make_grid(self, h, w):
        # 生成 [-1,1] 范围内的坐标网格；后面会用它和热力图做加权平均。
        yy, xx = torch.meshgrid(
            torch.arange(h).float() / (h - 1) * 2 - 1,
            torch.arange(w).float() / (w - 1) * 2 - 1)
        return yy, xx

    def get_coords_from_heatmap(self, heatmap):
        """
            inputs:
            - heatmap: batch x npoints x h x w

            outputs:
            - coords: batch x npoints x 2 (x,y), [-1, +1]
            - radius_sq: batch x npoints
        """
        batch, npoints, h, w = heatmap.shape
        if self.use_weight_map:
            heatmap = heatmap * self.weight

        # xx/yy 表示每个热力图像素位置自己的坐标，范围是 [-1,1]。
        yy, xx = self._make_grid(h, w)
        yy = yy.view(1, 1, h, w).to(heatmap)
        xx = xx.view(1, 1, h, w).to(heatmap)

        # 对每个关键点的整张二维热力图求和；[2,3] 分别对应高和宽两个空间维度。
        heatmap_sum = torch.clamp(heatmap.sum([2, 3]), min=1e-6)

        # 用热力图响应作为权重，对坐标网格做加权平均，得到连续的关键点坐标。
        yy_coord = (yy * heatmap).sum([2, 3]) / heatmap_sum  # batch x npoints
        xx_coord = (xx * heatmap).sum([2, 3]) / heatmap_sum  # batch x npoints
        # 最终每个关键点得到一个 (x,y) 坐标；后处理会把它从 [-1,1] 还原到图片坐标。
        coords = torch.stack([xx_coord, yy_coord], dim=-1)

        return coords
