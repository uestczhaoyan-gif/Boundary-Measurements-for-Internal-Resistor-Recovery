# Attention 子项目说明（64Nodes）

说明：
- 本文件记录 Attention 系列模型的架构与版本定义。
- 训练结果与问题分析请看：`64Nodes/attention/Log.md`。

## 为什么 Attention 可能适合本课题
- 多激励条件下的电压响应存在“长距离依赖”，注意力机制擅长建模全局关联。
- 将 64 节点视为 token 后，模型可自适应关注关键节点组合，而不依赖固定局部卷积核。
- 在变化电阻数量少、信号稀疏时，注意力权重可提升对弱特征的捕捉能力。
- 便于后续扩展到“跨激励注意力”或“节点-边联合注意力”。

## 一、目录结构
- `ATTN_CLS/modelo1`
- `ATTN_REG/modelo1`
- `ATTN_FULL/modelo1_h_multitask`

每个版本包含：
- `model/model.py`
- `train.py`
- `inference.py`
- `outputs/`（训练后生成）

## 二、输入与数据规范
- 数据：`64Nodes/data/training_data64.csv`
- 输入采用与 CNN2D_MLP v2 一致的 97 通道网格：
- `32电压 + 32src_map + 32gnd_map + 1boundary_mask`
- 标准化与划分：`train:val:test = 8:1:1`

## 三、基准架构（modelo1）
- Transformer Encoder（batch_first）
- `d_model=128, nhead=8, depth=4, ff=256, dropout=0.1`
- token 定义：`8x8` 网格每个节点为一个 token（共 64 tokens，token_dim=97）
- 任务头：
- CLS：全局池化后 CORAL 3-logits
- REG：全局池化后回归 112 维 `dR`（`tanh * max_abs`）
- FULL：共享编码器 + 分类头 + 回归头

## 四、训练建议
1. 先跑 `ATTN_CLS/modelo1`
2. 再跑 `ATTN_REG/modelo1`
3. 最后跑 `ATTN_FULL/modelo1_h_multitask`
