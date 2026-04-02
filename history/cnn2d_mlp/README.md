# CNN2D_MLP 子项目说明（64Nodes）

说明：
- 本文件只记录 CNN2D_MLP 方案的架构设计与版本定义。
- 训练日志、问题分析、改进建议请看：`64Nodes/cnn2d_mlp/Log.md`。

## 为什么 CNN2D_MLP 可能适合本课题
- 将外部电压投影到网格边界后，卷积可学习局部空间模式与多激励之间的共享结构。
- `src/gnd` 显式通道把激励条件编码进输入，有助于区分“同样电压分布但激励不同”的情况。
- MLP 任务头保留了对全局组合关系的建模能力，适合作为“空间特征提取 + 全局决策”的折中方案。
- 结构仍相对轻量，便于快速实验不同损失与稀疏约束。

## 一、统一数据与输入规范
- 数据源：`64Nodes/data/training_data64.csv`
- 样本粒度：按 `combo_id` 聚合，每个样本含 32 组激励。
- 先构造 `dV = V_meas - V_base`（`V_base` 来自 0-change 组合均值）。
- 外部节点电压映射到 `8x8` 网格边界位置，内部节点置 0。
- 输入张量（v2）：`[N, 97, 8, 8]`
- 32 通道：电压图（每个激励1通道）
- 32 通道：`src_map`（每个激励源节点位置 one-hot）
- 32 通道：`gnd_map`（每个激励地节点位置 one-hot）
- 1 通道：边界掩码 `boundary_mask`
- 数据划分：`train:val:test = 8:1:1`
- 标准化：按训练集均值/方差（逐通道逐像素）
- 说明：v2 默认缓存文件为 `*_v2.npz`，避免误用旧版缓存。

## 二、项目结构
- `CNN2D_CLS/modelo1`
- `CNN2D_REG/modelo1`
- `CNN2D_FULL/modelo1_h_multitask`

每个版本包含：
- `model/model.py`
- `train.py`
- `inference.py`
- `outputs/`（训练后生成）

## 三、模型定义

### 1) CNN2D_CLS / modelo1
- 任务：变化数量分类（0/1/2/3）
- 架构（v2）：`Residual CNN Backbone -> MLP Head -> CORAL logits(3)`
- 损失：CORAL BCE + class_weight + 2<->3 轻惩罚
- 阈值：验证集“带约束搜索”（`t1 <= t2 <= t3`）
- 训练策略（v2）：
- Early Stopping（监控 `val_macro_f1`）
- 可选阈值稳定：`fixed_t2`（固定 t2，仅搜索 t1/t3）
- 指标：测试集混淆矩阵

### 2) CNN2D_REG / modelo1
- 任务：112 维 `dR` 回归
- 架构（v2）：`Residual CNN Backbone -> MLP Head(112)`，输出 `tanh * max_abs`
- 损失：变化位 SmoothL1 + 未变化位 L1 + hinge + sparse
- 调度：分段稀疏调度（前 20 epoch 弱稀疏，后续增强）
- 阈值：验证集搜索 `40~70`（步长 1）
- 训练策略（v2）：Early Stopping（监控 `val_loss`）
- 指标：`mae_all`、`mae_changed`、`avg(|dR|>thr)`、Derived Count 混淆矩阵

### 3) CNN2D_FULL / modelo1_h_multitask
- 任务：完整问题（数量 + 幅值）
- 架构（v2）：共享 Residual CNN 主干 + 双头（CORAL 分类头 + 回归头）
- 损失：`L_cls + L_change + L_unchange + L_hinge + L_sparse + L_count`
- 训练策略（v2）：
- Stage1：先训分类相关（稳定数量边界）
- Stage2：联合训练回归与数量一致性（更低学习率）
- 各阶段均带早停
- 评估：
- 回归指标：`mae_all`、`mae_changed`、`avg(|dR|>50)`
- 数量指标：分类头混淆矩阵 + 回归阈值派生混淆矩阵

## 四、推荐运行顺序
1. 先跑 `CNN2D_CLS/modelo1` 验证数量可分性
2. 再跑 `CNN2D_REG/modelo1` 验证幅值回归与稀疏平衡
3. 最后跑 `CNN2D_FULL/modelo1_h_multitask` 联合任务

