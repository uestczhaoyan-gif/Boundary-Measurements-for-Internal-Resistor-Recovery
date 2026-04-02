# Attention 运行日志（64Nodes）

说明：
- 记录 `64Nodes/attention` 各版本关键指标、问题分析与改进计划。
- 更新规则：仅追加，不覆盖历史。

记录模板：
- 版本与训练命令
- 关键指标（分类混淆矩阵；回归 mae_all/mae_changed/avg(|dR|>50)）
- 现象分析
- 下一轮修改动作

## 2026-03-20 - v1 工程骨架初始化
- 新建 `ATTN_CLS/modelo1`
- 新建 `ATTN_REG/modelo1`
- 新建 `ATTN_FULL/modelo1_h_multitask`
- 架构：Transformer Encoder 基准（`d_model=128, nhead=8, depth=4`）
- 输入：97 通道网格输入映射为 64 节点 token

