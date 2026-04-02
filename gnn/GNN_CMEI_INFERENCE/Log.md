# GNN_CMEI_INFERENCE Log（64Nodes）

## 2026-04-02 - 联合推理目录独立
- 新建：
  - `gnn/GNN_CMEI_INFERENCE`
- 已迁移：
  - `gnn/inference_gnn_cmei.py -> gnn/GNN_CMEI_INFERENCE/inference_gnn_cmei.py`
  - `gnn/outputs -> gnn/GNN_CMEI_INFERENCE/outputs`
- 保留兼容入口：
  - `gnn/inference_gnn_cmei.py`
  - 作用：转发到新目录，避免旧命令立即失效

## 2026-04-02 - `inference_gnn_cmei_v2.py`
- 已根据根目录 `GNN_联合优化.txt` 整理出适合推理层直接落地的部分：
  - `near-miss` 高置信保护
  - `REG` 数量证据仲裁
- 未直接照搬的内容：
  - `Absolute Edge Embedding`
  - `Relaxed Sparsity Loss`
  - `Focal-CORAL`
  - `Pseudo-Edge Pooling`
- 原因：
  - 它们属于训练/结构层优化，而不是纯推理层改动

## 2026-04-02 - `v2` 本地验证
- 已对当前正式主线 `rand_boundary` 进行实跑：
  - `v2(full arbitration)` clean `CMEI=90.08`
  - `v2(full arbitration)` 20dB `CMEI=79.40`
  - `v2(guard_only)` clean `CMEI=90.85`
  - `v2(guard_only)` 20dB `CMEI=82.40`
- 对照基线：
  - `v1 clean CMEI=91.01`
  - `v1 20dB CMEI=82.56`
- 结论：
  - `REG` 数量仲裁当前会过度降低预测数量
  - 高置信 near-miss 保护单独启用时也未超过 `v1`
  - `v2` 暂保留为实验脚本，不转正
