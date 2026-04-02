# modelv2_new

- 任务：固定 3 变化场景下的纯重建（输出 112 维 `dR`）
- 架构：延续 `modelv1_new`，仍为 `896 -> 1024 -> 896 -> 512 -> 256 -> 112`
- 每层：`BN + ReLU + Dropout`
- 残差：输入 `896` 到隐藏层 `896` 做跳连
- 输出约束：`tanh(y) * max_abs`，默认 `max_abs=310`

## 相对 modelv1_new 的核心改动
- 回归主损失从“统一 MSE”改为“变化位更重、未变化位较轻”的加权回归：
  - 默认 `w_change=7`
  - 默认 `w_unchange=1`
- 稀疏约束不再压全部 `112` 维，而是主要约束未变化位：
  - `L_sparse_unchange`
  - `L_hinge_unchange`，默认阈值 `50`
- 新增“第 4 个假阳性抑制”分离损失：
  - 约束 `min_true_abs - max_nontrue_abs` 保持正间隔
  - 默认 `sep_margin=12`
- 物理约束改为延后启用：
  - 默认从 `epoch 25` 开始
  - 在 `20 epoch` 内逐步升到目标权重
- 评估补回：
  - `avg(|dR|>50)` 只作为诊断指标，不作为主目标

## 指标输出
- `mae_all`
- `mae_changed`
- `avg(|dR|>50)`（诊断）
- 位置预测准确率（对 0 个 / 对 1 个 / 对 2 个 / 全对）

## 损失构成（主次顺序）
- `L_reg`：变化位加权回归主损失（最高）
- `L_id`：坐标法位置约束
- `L_phys`：延后启用的基尔霍夫约束
- `L_sparse_unchange`：未变化位 L1
- `L_hinge_unchange`：未变化位大幅假阳性抑制
- `L_sep`：第 4 假阳性分离约束

## 默认关键超参
- `epochs=160`
- `patience=30`
- `lambda_reg=1.0`
- `w_change=7.0`
- `w_unchange=1.0`
- `lambda_id=0.35`
- `lambda_phys=0.15`
- `lambda_sparse=0.05`
- `lambda_hinge=0.10`
- `lambda_sep=0.20`
- `hinge_threshold=50`
- `sep_margin=12`
- `phys_start_epoch=25`
- `phys_ramp_epochs=20`

## 运行
```bash
python 64Nodes/mlp/fixed_change_recon/modelv2_new/train.py
python 64Nodes/mlp/fixed_change_recon/modelv2_new/inference.py
```

