# CNN 子项目说明（64Nodes）

说明：
- 本文件记录纯 CNN 系列模型的架构与版本定义。
- 训练结果与问题分析请看：`64Nodes/cnn/Log.md`。

## 为什么纯 CNN 可能适合本课题
- 纯 CNN 强调局部平移共享，可检验“局部图样是否足以判别变化位置与幅值”。
- 对比 CNN2D_MLP，可隔离 MLP 头的影响，帮助判断性能瓶颈来自“特征提取”还是“任务头建模”。
- 参数规模可控，训练速度快，适合作为空间建模方向的对照基准。
- 对噪声通常更平滑，便于评估鲁棒性。

## 一、目录结构
- `CNN_CLS/modelo1`
- `CNN_REG/modelo1`
- `CNN_FULL/modelo1_h_multitask`

每个版本包含：
- `model/model.py`
- `train.py`
- `inference.py`
- `outputs/`（训练后生成）

## 二、输入与数据规范
- 数据：`64Nodes/data/training_data64.csv`
- 输入与 CNN2D_MLP v2 保持一致：
- `97` 通道：`32电压 + 32src_map + 32gnd_map + 1boundary_mask`
- 划分：`8:1:1`，按训练集统计标准化

## 三、基准架构（modelo1）
- 主干：保分辨率 Residual CNN（8x8 不做重池化）
- CLS：`Conv1x1` 分类头 + GAP -> CORAL logits
- REG：`Conv1x1` 回归头 + GAP -> `Linear(112)` -> `tanh*max_abs`
- FULL：共享主干 + 分类头 + 回归头

## 四、训练建议
1. 先跑 `CNN_CLS/modelo1`
2. 再跑 `CNN_REG/modelo1`
3. 最后跑 `CNN_FULL/modelo1_h_multitask`
