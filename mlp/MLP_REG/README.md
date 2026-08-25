# MLP regression family

`MLP_REG/` 预测 112 维电阻变化量，是固定拓扑下的非图回归 baseline。`modelo5` 是推荐起点；训练目标和数据路径以该版本的 `train.py --help` 为准。

```powershell
python mlp\MLP_REG\modelo5\train.py --help
python mlp\MLP_REG\modelo5\inference.py --help
```

至少记录 `mae_all`、`mae_changed` 和阈值化假阳性数量。不要把 MLP 输出直接接入当前 GNN CMEI 入口，除非另行记录联合模型口径。
