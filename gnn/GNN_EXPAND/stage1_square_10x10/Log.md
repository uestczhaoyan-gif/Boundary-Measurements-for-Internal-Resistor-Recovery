# Stage1 Square 10x10 Log

## 2026-04-02 初始化
- 已建立：
  - `cls/train.py`
  - `cls/inference.py`
  - `reg/train.py`
  - `reg/inference.py`
  - `joint_inference/inference.py`
- 当前状态：
  - 目录与脚本已落地，等待正式训练结果追加

## 2026-04-02 数据生成
- 已生成原生 stage1 数据：
  - `gnn/GNN_EXPAND/data/square_10x10.csv`
  - `gnn/GNN_EXPAND/data/square_10x10_meta.json`
- 已确认：
  - 激励只使用外部节点
  - 测量只输出外部节点电压

## 2026-04-02 数据口径更正
- `stage1` 原生数据不是 `28` 个外部节点、`32` 组激励。
- 已按 `square_10x10_meta.json` 核对：
  - 外部节点数为 `36`
  - 激励组数为 `40`
- 约束保持不变：
  - 激励只使用外部节点
  - 测量只输出外部节点电压
