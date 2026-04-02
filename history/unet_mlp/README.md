# U-Net_MLP 子项目说明（64Nodes）

说明：
- 本文件记录 U-Net_MLP 系列模型的架构与版本定义。
- 训练结果与问题分析请看：`64Nodes/unet_mlp/Log.md`。

## 为什么 U-Net_MLP 可能适合本课题
- U-Net 的下采样/上采样 + 跳连接可同时保留全局上下文和边界细节，适合“少量局部异常”检测。
- 相比普通 CNN，更容易恢复空间分辨率信息，对变化位置识别潜在更友好。
- 结合 MLP 头后，可在空间增强特征基础上进行全局数量/幅值决策。
- 对后续扩展到“先重建内部场，再反演电阻”的两阶段思路兼容性强。

## 一、目录结构
- `UNET_MLP_CLS/modelo1`
- `UNET_MLP_REG/modelo1`
- `UNET_MLP_FULL/modelo1_h_multitask`

每个版本包含：
- `model/model.py`
- `train.py`
- `inference.py`
- `outputs/`（训练后生成）

## 二、输入与数据规范
- 数据：`64Nodes/data/training_data64.csv`
- 输入：与 CNN2D_MLP v2 一致的 97 通道网格
- `32电压 + 32src_map + 32gnd_map + 1boundary_mask`
- 划分：`8:1:1`，按训练集统计标准化

## 三、基准架构（modelo1）
- 主干：U-Net（下采样 + 上采样 + 跳连接）
- CLS：U-Net 输出特征 -> MLP 分类头（CORAL）
- REG：U-Net 输出特征 -> MLP 回归头（112维，`tanh*max_abs`）
- FULL：共享 U-Net 主干 + 双任务 MLP 头

## 四、训练建议
1. 先跑 `UNET_MLP_CLS/modelo1`
2. 再跑 `UNET_MLP_REG/modelo1`
3. 最后跑 `UNET_MLP_FULL/modelo1_h_multitask`
