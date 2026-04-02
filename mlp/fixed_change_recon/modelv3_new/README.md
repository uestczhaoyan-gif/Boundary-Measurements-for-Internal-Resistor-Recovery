# modelv3_new

- 任务：fixed-change 场景下的纯重建（输出 112 维 `dR`）
- 当前主用：`fixed_3`
- 当前已支持：`fixed_2 / fixed_3`
- 当前已冻结为：只做 `fixed_2 / fixed_3` 两个纯回归任务，不再当作通用 fixed-k 框架。
- 架构：保持 `896 -> 1024 -> 896 -> 512 -> 256 -> 112`
- 每层：`BN + ReLU + Dropout`
- 残差：输入 `896` 到隐藏层 `896` 做跳连
- 输出约束：`tanh(y) * max_abs`，默认 `max_abs=310`

## 设计目标
- 不改骨干架构，只做针对 fixed-change 重构的稳定化。
- 保留 `modelv2_new` 已验证有效的变化位加权回归方向。
- 把旧的“第4大抑制”改成对 fixed-k 通用的“第 k+1 大抑制 + 第 k/k+1 间隔”。
- 但当前运行入口只接受 `fixed_k in {2,3}`。

## 当前损失
- `L_reg`：变化位加权回归
- `L_id`：坐标矩约束
- `L_phys`：延后启用的基尔霍夫约束
- `L_fp_next`：直接压制第 `k+1` 大 `|dR|`
- `L_rank_gap`：拉开第 `k` 和第 `k+1` 大 `|dR|` 的间隔

## 相对上一版的本轮改动
- 新增 `--fixed-k`，训练/推理都可直接切到 `fixed_2 / fixed_3`
- 训练脚本会优先从对应 meta 自动同步：
  - `current_source_a`
  - `change_count_fixed`
- 评分中的“全对位置命中”已从固定 `pos3` 改为通用的 `pos_k`
- cache 与 outputs 现已按 `fixed_2 / fixed_3` 和 `dataset_tag` 双层分开
- 训练/推理现在都会检查：
  - cache 内 `fixed_k` 是否与当前数据一致
  - cache 对应 `source_csv` 是否与当前数据文件一致

## 默认关键超参
- `epochs=160`
- `patience=30`
- `lambda_reg=1.0`
- `w_change=7.0`
- `w_unchange=1.0`
- `lambda_id=0.35`
- `lambda_phys=0.15`
- `lambda_fp_next=0.30`
- `fp_next_threshold=45`
- `lambda_rank_gap=0.18`
- `rank_gap_margin=14`
- `phys_start_epoch=25`
- `phys_ramp_epochs=20`

## best checkpoint 评分
- `score = mae_changed + 2.5 * max(0, avg(|dR|>50) - target_active_count) + score_pos_hit * (1 - pos_k) + 0.05 * mae_all`
- 其中：
  - `target_active_count` 默认自动等于 `fixed_k`
  - `pos_k` 表示“固定 k 个位置全对”的比例

## 运行
```bash
python 64Nodes/mlp/fixed_change_recon/modelv3_new/train.py
python 64Nodes/mlp/fixed_change_recon/modelv3_new/inference.py
python 64Nodes/mlp/fixed_change_recon/modelv3_new/train.py --data-path 64Nodes/mlp/fixed_change_recon/data_fixed/training_data64_fixed_2.csv --fixed-k 2 --dataset-tag 5mA
python 64Nodes/mlp/fixed_change_recon/modelv3_new/inference.py --data-path 64Nodes/mlp/fixed_change_recon/data_fixed/training_data64_fixed_2.csv --fixed-k 2 --dataset-tag 5mA
```

## 输出与缓存
- 默认缓存：
  - `cache/fixed_3/<dataset_tag>/cache_fixed_v3_new.npz`
  - `cache/fixed_2/<dataset_tag>/cache_fixed_v3_new.npz`
- 默认输出：
  - `outputs/fixed_3/<dataset_tag>/...`
  - `outputs/fixed_2/<dataset_tag>/...`
