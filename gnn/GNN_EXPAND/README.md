# GNN topology expansion

`GNN_EXPAND` 检验主线模型是否能从 8x8 之外迁移到不同规模和不同边界形状。它不是简单扩大训练集，而是重新生成每个目标拓扑的基尔霍夫正演数据，并分别训练 CLS、REG 和 joint inference。

## 四个阶段

| 阶段 | 拓扑 | 边界节点 | 激励组 |
| --- | --- | ---: | ---: |
| stage1 | square 10x10 | 36 | 40 |
| stage2 | rectangle 6x10 | 28 | 32 |
| stage3 | honeycomb 63 | 28 | 32 |
| stage4 | circlecut 69 | 24 | 28 |

真实边界规模随拓扑变化，不能强行套用 8x8 的 28/32 口径。元数据在 `data/*_meta.json`。

## 运行

```powershell
python gnn\GNN_EXPAND\generate_expand_datasets.py --help
python gnn\GNN_EXPAND\stage2_rect_6x10\cls\train.py --help
python gnn\GNN_EXPAND\stage2_rect_6x10\reg\train.py --help
python gnn\GNN_EXPAND\stage2_rect_6x10\joint_inference\inference.py --help
```

每个 stage 还有自己的 README 和 Log。当前阶段汇总指标见 `expand_summary_metrics.json`；CSV、cache、outputs 和本地依赖默认不提交。

## 解释边界

迁移实验回答的是“在新拓扑上重新训练或 warm-start 后是否有效”，不等于零样本泛化。报告结果时必须标明 topology、是否 warm-start、边界节点数、激励组数和训练方式。
