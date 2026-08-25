# Project map

## 读者路径

```text
想了解项目       -> ../README.md
想复现主模型     -> ../gnn/README.md
想看数据口径     -> ../data/README.md
想理解难例原因   -> ../inverse_identifiability/README.md
想研究规模限制   -> ../square_scale_study/README.md
想看基线和失败线 -> ../mlp/README.md / ../history/README.md
想核对指标       -> ../CURRENT_BEST.md + 各 outputs/metrics.json
```

## 目录边界

- **公开研究资产**：Python 源码、README、元数据、可公开图表和必要的配置。
- **本地生成资产**：CSV、cache、outputs、模型权重和 vendor 依赖。它们可由脚本重建，默认不提交。
- **私有材料**：论文、答辩 PPT、诚信声明、外文资料和论文编辑工作台，统一位于 `private_materials/`。

## 结果流

`data` 生成样本，`gnn/mlp` 训练模型，`inference` 写出机器可读指标，`Log.md` 解释决策，`Figure` 保存公开展示图。论文引用的每个数字都应该能够沿这条链回溯到具体数据标签和输出目录。

## 当前主线与归档

当前主线是 `GNN_CLS/modelo3 + GNN_REG/o4a2 + GNN_CMEI_INFERENCE`。`GNN_NOISE` 和 `GNN_EXPAND` 是验证扩展，`MLP` 是固定拓扑 baseline，`inverse_identifiability` 与 `square_scale_study` 是独立的机制/规模研究，`history` 是停止推进的路线。
