# 300W 短轮消融实验记录

## 实验目的

本实验基于 `opt-3-gpu-running` 分支新建 `ablation-300w-opt3-short`，在 300W 数据集上补充与 COFW 并列的短轮消融实验。实验不改变网络结构，固定数据集、测试流程和初始公开检查点，只比较微调范围与学习率设置对统一测试指标的影响。

## 实验环境

- 分支：`ablation-300w-opt3-short`
- 基础分支：`origin/opt-3-gpu-running`
- GPU：NVIDIA GeForce RTX 5090 32GB
- 数据集：300W
- 初始权重：`model/300W_STARLoss_NME_2_87.pkl`
- 训练轮数：`max_epoch=3`，项目训练循环实际记录 epoch 0 至 epoch 3
- batch size：16
- 测试集规模：688
- 测试脚本：`python main.py --mode=test --data_definition=300W ...`
- 关键指标：第 9 号输出的 NME / FR / AUC，NME 与 FR 越低越好，AUC 越高越好。

## 消融设置

| 实验编号 | 设置 | 说明 |
| --- | --- | --- |
| B0 | 公开 300W checkpoint 直接测试 | 不做微调，作为统一测试对照 |
| B1 | heads_only, lr=0.00005, epoch=3 | 冻结主干，仅微调输出头相关层 |
| B2 | full, lr=0.00005, epoch=3 | 全网络参与短轮训练 |
| B3 | heads_only, lr=0.0001, epoch=3 | 仅微调输出头，同时提高学习率 |

## 统一测试结果

| 实验编号 | NME ↓ | FR ↓ | AUC ↑ | 说明 |
| --- | ---: | ---: | ---: | --- |
| B0 公开 300W checkpoint | 0.028704 | 0.058100 | 0.437725 | 统一测试对照 |
| B1 heads_only lr=0.00005 | 0.028700 | 0.058100 | 0.437716 | 与对照基本一致 |
| B2 full lr=0.00005 | 0.028731 | 0.059600 | 0.437118 | 全参数短轮训练波动略大 |
| B3 heads_only lr=0.0001 | 0.028698 | 0.058100 | 0.437783 | 本次短轮记录中数值最低 |

## 训练中最佳验证指标

| 实验编号 | 最佳验证 NME |
| --- | ---: |
| B1 heads_only lr=0.00005 | 0.028700 |
| B2 full lr=0.00005 | 0.028731 |
| B3 heads_only lr=0.0001 | 0.028698 |

## 分析结论

1. 300W 公开检查点本身已经很稳定，短轮训练后各组差异很小。
2. `heads_only` 设置在两个学习率下均保持了接近对照的结果，说明只更新输出头相关层不会明显破坏 300W 上已有表现。
3. `full + lr=0.00005` 在本次短轮记录中 NME 和 FR 略高，说明全参数短轮训练对稳定性更敏感。
4. `heads_only + lr=0.0001` 在本次短轮记录中得到最低 NME 0.028698 和最高 AUC 0.437783，但差异极小，论文中应仅作为训练流程和策略对照记录，不表述为性能提升结论。
5. 该实验与 COFW 短轮消融形成并列结构：COFW 说明遮挡场景下的短轮策略对照，300W 说明标准人脸对齐数据集上的短轮策略对照。

## 关键日志位置

- 统一测试日志：`ablation_logs_300W/test/*.log`
- 训练日志：`ablation_logs_300W/*.log`
- 训练输出目录：`ablation_runs_300W/`
- 实验脚本：`scripts/run_300w_ablation_short.sh`

## 复现实验命令

```bash
cd /root/autodl-tmp/STAR-new
bash scripts/run_300w_ablation_short.sh
```
