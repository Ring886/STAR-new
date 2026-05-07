# COFW 保守微调消融实验记录

## 实验目的

本实验基于 `opt-3-gpu-running` 分支，在不引入额外结构模块的前提下，对作者公开 COFW checkpoint 进行短轮保守微调消融。实验重点是观察“是否微调、微调整体网络还是仅微调输出头、学习率大小”对 COFW 测试指标的影响。

## 实验环境

- 分支：`ablation-opt3-short`
- 基础分支：`origin/opt-3-gpu-running`
- GPU：NVIDIA GeForce RTX 5090 32GB
- 数据集：COFW
- 初始权重：`model/COFW_STARLoss_NME_4_62.pkl`
- 训练轮数：`max_epoch=3`，项目训练循环实际记录 epoch 0 至 epoch 3
- batch size：16
- 验证/测试集规模：506
- 测试脚本：`python main.py --mode=test --data_definition=COFW ...`
- 关键指标：第 9 号输出的 NME / FR / AUC，NME 与 FR 越低越好，AUC 越高越好。

## 消融设置

| 实验编号 | 设置 | 说明 |
| --- | --- | --- |
| A0 | 作者 checkpoint 直接测试 | 不做微调，作为基准 |
| A1 | full fine-tune, lr=0.0003, epoch=3 | 全网络参与短轮保守微调 |
| A2 | heads_only fine-tune, lr=0.0003, epoch=3 | 冻结主干，仅微调输出头相关层 |
| A3 | full fine-tune, lr=0.0001, epoch=3 | 全网络参与微调，但学习率更小 |

## 统一测试结果

| 实验编号 | NME ↓ | FR ↓ | AUC ↑ | 相对基线 NME 变化 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| A0 作者 checkpoint | 0.046250 | 0.007900 | 0.539523 | 0.000000 | 强基线 |
| A1 full lr=0.0003 | 0.046225 | 0.007900 | 0.539775 | -0.000025 | 极小幅改善，FR 持平 |
| A2 heads_only lr=0.0003 | 0.046208 | 0.007900 | 0.539957 | -0.000042 | 本次短轮实验中最好，但提升幅度很小 |
| A3 full lr=0.0001 | 0.046245 | 0.007900 | 0.539568 | -0.000005 | 与基线基本持平 |

## 训练中最佳验证指标

| 实验编号 | 最佳验证 NME |
| --- | ---: |
| A1 full lr=0.0003 | 0.046225 |
| A2 heads_only lr=0.0003 | 0.046208 |
| A3 full lr=0.0001 | 0.046245 |

## 分析结论

1. 作者公开 checkpoint 本身已经是强基线，短轮微调的可提升空间很有限。
2. 三组保守微调均没有造成指标崩坏，说明从作者模型出发进行小学习率微调是稳定的。
3. `heads_only + lr=0.0003` 在本次短轮设置下取得最低 NME 0.046208，相比作者 checkpoint 的 0.046250 下降 0.000042，AUC 从 0.539523 上升到 0.539957，FR 保持 0.007900。
4. 由于实验 epoch 较少，且提升幅度非常小，论文中应保守表述为“短轮消融显示输出头微调在 COFW 上略优于直接测试和全量微调，但整体差异有限”，不能夸大为显著提升。
5. 该消融实验适合服务论文最终定位：以复现和工程实现为主，在作者模型基础上进行保守微调验证。

## 关键日志位置

- 作者基线测试：`ablation_logs/test/author_baseline_test.log`
- A1 测试：`ablation_logs/test/full_lr3e4_ep3_test.log`
- A2 测试：`ablation_logs/test/heads_lr3e4_ep3_test.log`
- A3 测试：`ablation_logs/test/full_lr1e4_ep3_test.log`
- 训练日志：`ablation_logs/*.log`
- 训练输出目录：`ablation_runs/`

## 复现实验命令

```bash
cd /root/autodl-tmp/STAR-new
bash scripts/run_cofw_ablation_short.sh
```

测试最佳模型时，使用各实验目录下的 `model/best_model.pkl` 作为 `--pretrained_weight`。
