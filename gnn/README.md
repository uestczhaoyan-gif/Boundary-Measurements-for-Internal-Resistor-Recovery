# GNN 主线

`gnn/` 是本项目的主要建模目录。它把 8x8 电阻网络保留为图结构，用边界电压响应完成变化数量分类、112 条电阻边变化量回归，并把两者组合成联合推理。这里的目录按研究问题组织，不把每个实验版本都当成新的“项目”。

## 主线结构

```text
GNN_CLS/modelo3        变化数量 K ∈ {0,1,2,3}
          │
GNN_REG/o4a2            112 维电阻变化量 ΔR
          │
GNN_CMEI_INFERENCE      用 K 选择 |ΔR| 候选边并计算联合指标
```

外围验证线：

- `GNN_NOISE/`：边界测量噪声下的 fine-tune 和鲁棒性曲线。
- `GNN_EXPAND/`：10x10、矩形、蜂窝和切圆拓扑上的迁移验证。
- `GNN_FULL/`：早期多任务网络，作为历史对照，不是当前默认入口。
- `GNN_CLS/`、`GNN_REG/` 的其他 `modelo*`/`o*`：架构消融和失败路线。

## 输入与输出

主数据的每个样本包含 32 组激励下的 28 个外部节点电压，以及 112 条电阻边的真实变化。GNN 将 64 个节点和 112 条边作为固定图；节点特征包含边界掩码、源/地掩码和电压信息，具体实现以各版本 `model/model.py` 为准。

- CLS 输出 4 个数量类别。`modelo3` 使用 CORAL 风格的有序阈值，并保留对 `2/3` 边界的专门处理。
- REG 输出 112 维 `ΔR`，同时学习变化位置的门控概率；`o4a2` 是当前推荐的回归入口。
- CMEI 读取 CLS 的 `K` 和 REG 的 `ΔR`，按绝对变化量排序后保留前 `K` 条边，再计算数量、位置和数值综合指标。

## 推荐入口

| 任务 | 推荐目录 | 训练 | 推理 |
| --- | --- | --- | --- |
| 数量分类 | `GNN_CLS/modelo3` | `train.py` | `inference.py` |
| 变化量回归 | `GNN_REG/o4a2` | `train.py` | `inference.py` |
| 联合评估 | `GNN_CMEI_INFERENCE` | 不训练 | `inference_gnn_cmei.py` |
| 噪声鲁棒性 | `GNN_NOISE` | 各 `CLS_*`/`REG_*` 分支 | `run_noise_eval_suite.py` |
| 拓扑迁移 | `GNN_EXPAND` | 各 stage 的 `cls/reg/train.py` | 各 stage 的 `joint_inference/inference.py` |

训练前先查看参数：

```powershell
python gnn\GNN_CLS\modelo3\train.py --help
python gnn\GNN_REG\o4a2\train.py --help
python gnn\GNN_CMEI_INFERENCE\inference_gnn_cmei.py --help
```

实际可用参数随版本演进，命令以 `--help` 输出为准。默认输出和缓存被 `.gitignore` 排除，避免把模型权重和重复结果提交进仓库。

## 结果口径

主线日志中记录过的参考点包括：CLS `test_macro_f1` 约 0.90，REG `o4a2` 的 `mae_all=0.4679`、`mae_changed=23.5724`，联合推理 `CMEI=93.53`。这些数字必须和对应输出目录的 `metrics.json`/`cmei_metrics.json` 一起解释，不能脱离数据集标签或随机种子单独引用。

带噪 v2 的阶段曲线为 clean `93.49`、40 dB `92.81`、30 dB `90.44`、20 dB `80.42`；`rand_boundary` 在 20 dB 端点记录过 `82.56`。这说明噪声训练策略存在工作区间，不应被压缩成一个“全面优于”的结论。

## 版本治理

- `modelo*` 和 `o*` 表示同一任务的架构/损失迭代，不代表独立产品。
- 只有经过同口径训练、推理和记录的版本，才能提升为推荐入口。
- 失败版本保留代码和简短结论，详细过程写入 `Log.md`，避免 README 失去导航功能。
- 新实验必须使用独立 `dataset-tag` 和输出目录，防止 cache 复用造成假复现。

## 相关文档

- 总体项目入口：[`../README.md`](../README.md)
- 当前最佳锚点：[`../CURRENT_BEST.md`](../CURRENT_BEST.md)
- 联合推理：[`GNN_CMEI_INFERENCE/README.md`](GNN_CMEI_INFERENCE/README.md)
- 噪声验证：[`GNN_NOISE/README.md`](GNN_NOISE/README.md)
- 拓扑迁移：[`GNN_EXPAND/README.md`](GNN_EXPAND/README.md)
- 实验时间线：[`Log.md`](Log.md)
