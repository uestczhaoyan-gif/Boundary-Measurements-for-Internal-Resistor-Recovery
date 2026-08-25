# GNN regression family

本目录的模型输出 112 条电阻边的变化量 `ΔR`，并用门控/稀疏目标抑制未变化边上的大幅假阳性。`modelo1` 到 `modelo3` 记录基础图回归演进，`o4a2` 是当前推荐的回归入口，`model_tp1` 等物理传播路线保留为对照。

## 推荐入口

```powershell
python gnn\GNN_REG\o4a2\train.py --help
python gnn\GNN_REG\o4a2\inference.py --help
```

`o4a2` 的训练目标由回归误差、变化位置 mask BCE 和稀疏项组成；验证选模同时观察 `mae_changed`、`mae_all` 和稀疏阈值统计。具体默认值以 `train.py --help` 为准。

## 评价

不要只看 `mae_all`。论文和实验记录至少同时报告 `mae_changed`、阈值化假阳性数量，并注明变化数量分布。`modelo3b` 是基于 `modelo3` 的候选集推理版本，不是新的训练主干。

## 与联合推理的关系

REG 负责给出边排序和幅值，CLS 提供截断数量。最终结果由 [`../GNN_CMEI_INFERENCE`](../GNN_CMEI_INFERENCE) 统一计算。
