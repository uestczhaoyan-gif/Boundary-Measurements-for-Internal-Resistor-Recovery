# GNN joint inference

本目录把数量分类和变化量回归组合成最终的边界测量推理结果。默认入口是 `inference_gnn_cmei.py`；`inference_gnn_cmei_v2.py` 是实验版，不是当前默认方案。

## 推理链路

1. 加载 `GNN_CLS/modelo3`，得到 `K`。
2. 加载 `GNN_REG/o4a2`，得到 112 维 `ΔR`。
3. 按 `|ΔR|` 排序，保留前 `K` 条候选边。
4. 对数量、位置和数值误差计算 CMEI 及分项指标。

```powershell
python gnn\GNN_CMEI_INFERENCE\inference_gnn_cmei.py --help
python gnn\GNN_CMEI_INFERENCE\inference_gnn_cmei.py `
  --data-path data\training_data64Nodes_2.csv `
  --dataset-tag training_data64Nodes_2
```

脚本会按数据集标签解析 cache、模型和标准化文件。若使用自定义模型，显式传入对应路径，避免从旧输出目录静默回退。

## 结果

阶段性 clean 主线记录为 `CMEI=93.53`。结果文件位于本目录 `outputs/` 下的具体数据集子目录，机器可读入口是 `cmei_metrics.json`。噪声联合评估也统一写入本目录，而不是根目录的临时输出。
