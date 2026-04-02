# GNN_CMEI_INFERENCE 说明（64Nodes）
说明：
- 本目录只负责 `CLS + REG` 的联合推理与 `CMEI` 评估。
- 从 2026-04-02 起，`inference_gnn_cmei` 及其改版与所有 joint outputs 统一收拢到这里。

## 一、目录结构
- `GNN_CMEI_INFERENCE/inference_gnn_cmei.py`
  - 当前稳定版联合推理入口
- `GNN_CMEI_INFERENCE/inference_gnn_cmei_v2.py`
  - 基于联合优化记录整理出的增强推理版
- `GNN_CMEI_INFERENCE/outputs/`
  - 所有联合推理输出目录

## 二、当前运行约定
- clean / noise 训练仍分别保留在：
  - `GNN_CLS`
  - `GNN_REG`
  - `GNN_NOISE`
- 但 joint inference 一律从本目录执行，并写入本目录 `outputs/`
- 因此后续同步云端结果时，只需要重点下载：
  - `gnn/GNN_NOISE/**/outputs/`
  - `gnn/GNN_CMEI_INFERENCE/outputs/`

## 三、当前正式主线
- `CLS`：`GNN_NOISE/CLS_modelo3_ft`
- `REG`：`GNN_NOISE/REG_o4a2_ft`
- `joint inference`：默认先用 `inference_gnn_cmei.py`
- `v2` 只作为推理层增强实验，不覆盖稳定版

## 四、v2 可借鉴内容
- 已从根目录 `GNN_联合优化.txt` 中筛出真正适合推理层直接落地的两类改动：
  - `near-miss` 高置信保护
  - `REG` 证据驱动的动态 `K` / 数量仲裁
- 训练侧建议如：
  - `Absolute Edge Embedding`
  - `Relaxed Sparsity Loss`
  - `Focal-CORAL`
  - `Pseudo-Edge Pooling`
  已经分别落在或应落在 `REG/CLS` 训练分支，不属于本目录职责

## 五、v2 当前状态
- `inference_gnn_cmei_v2.py` 已实跑验证。
- 当前 `rand_boundary` 主线结果：
  - `v1 clean CMEI=91.01`
  - `v2(guard_only) clean CMEI=90.85`
  - `v2(full arbitration) clean CMEI=90.08`
  - `v1 20dB CMEI=82.56`
  - `v2(guard_only) 20dB CMEI=82.40`
  - `v2(full arbitration) 20dB CMEI=79.40`
- 结论：
  - `v2` 目前没有超过 `v1`
  - 因此继续保留为实验版，不替代稳定版
