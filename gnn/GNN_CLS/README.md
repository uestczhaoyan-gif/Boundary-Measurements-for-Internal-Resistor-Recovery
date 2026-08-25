# GNN classification family

本目录的模型只预测内部电阻变化数量 `K ∈ {0,1,2,3}`。模型版本按 `modelo1`、`modelo2`、`modelo3` 递进；当前推荐 `modelo3`，其他版本用于消融和历史对照。

## 推荐入口

```powershell
python gnn\GNN_CLS\modelo3\train.py --help
python gnn\GNN_CLS\modelo3\inference.py --help
```

`modelo3` 以图结构编码 32 组激励的边界响应，并使用有序分类目标处理四个数量类别。训练和推理都支持 `--data-path`、`--dataset-tag`、独立 cache/output 子目录和固定随机种子。

## 评价

主指标是 `test_macro_f1` 和分类混淆矩阵，尤其关注 `2/3` 边界。输出目录中的 `metrics.json` 是结果源；详细架构差异留在各版本代码和 [`../Log.md`](../Log.md)。

## 与联合推理的关系

CLS 不负责定位具体边。它的预测 `K` 会被 [`../GNN_CMEI_INFERENCE`](../GNN_CMEI_INFERENCE) 用来截断 REG 的候选边，因此数量错误会直接影响最终定位结果。
