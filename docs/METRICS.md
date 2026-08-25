# Metrics glossary

## Classification

- `accuracy`：样本数量类别完全正确的比例。
- `macro_f1`：各类别 F1 的无权平均，能看到少数类别是否被忽略。
- `test_macro_f1`：测试集上的 macro F1；只有在相同测试划分下才能比较。

## Regression

- `mae_all`：112 条边全部计入的平均绝对误差。
- `mae_changed`：只在真实发生变化的边上计算的平均绝对误差，更接近幅值恢复能力。
- `avg(|dR|>threshold)`：预测变化量超过阈值的平均边数，用来观察假阳性和过度扩散。

## Support and joint inference

- `support`：真实变化边的集合。
- `candidate_cover`：真实变化边落入前 `K`/前 `M` 候选集合的比例；它不等于最终精确恢复率。
- `id_recall`：最终预测变化边对真实变化边的召回率。
- `CMEI`：项目自定义的联合指标，综合数量、位置和数值误差。具体权重和实现以 `gnn/GNN_CMEI_INFERENCE/inference_gnn_cmei.py` 为准。

## 使用规则

不要在不同数据标签、不同噪声强度、不同拓扑或不同候选截断规则之间直接比较同名指标。结果表必须同时写出数据集、split seed、模型版本和输出 JSON 路径。
