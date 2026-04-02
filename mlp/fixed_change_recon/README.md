# fixed_change_recon 微项目

## 目的
- 验证一个更“理想化”的子问题：每个样本有固定数量的电阻变化。
- 仅做重构（位置 + 数值），不再单独训练变化数量头。
- 当前已冻结为两个固定子任务：
  - `fixed_3`：固定 3 个变化的纯回归重构
  - `fixed_2`：固定 2 个变化的纯回归重构
- 它们当前都只承担“重构诊断”作用，不再承担数量判断/分类功能。

## 数据
- 默认 `fixed_3` 5mA：`64Nodes/mlp/fixed_change_recon/data_fixed/training_data64_fixed_3.csv`
- 默认 `fixed_2` 5mA：`64Nodes/mlp/fixed_change_recon/data_fixed/training_data64_fixed_2.csv`
- 当前每套默认规模：5000 个不重复电阻组合 × 32 激励 = 160000 行。
- 拓扑/激励：与主项目一致（8x8, 64 节点, 112 电阻, 外部 28 节点, 32 组激励）。
- 变化范围：每个变化电阻在 `±5% ~ ±30%`。
- 坐标文件：`64Nodes/mlp/fixed_change_recon/data_fixed/resistor_coords_bl_origin.json`
- 当前只建议通过 `generate_data_fixed.py --fixed-k {2,3} --current-a ... --dataset-tag ...` 生成专用数据。
- 对于非 5mA 数据，默认文件名会写成 `training_data64_fixed_<k>_<dataset_tag>.csv` 与对应 meta。

## 当前状态（2026-03-25）
- 目录已由 `change3_recon` 更名为 `fixed_change_recon`。
- 原专用数据已重命名为 `fixed_3`，并新增 `fixed_2` 专用数据。
- 当前主线 `modelv3_new` 已固定为只支持 `fixed_2 / fixed_3`，并优先从 meta 自动同步：
  - `current_source_a`
  - `change_count_fixed`
- 当前训练/推理入口已补上：
  - `cache/fixed_2/<dataset_tag>/...` 与 `cache/fixed_3/<dataset_tag>/...` 分离
  - `outputs/fixed_2/<dataset_tag>/...` 与 `outputs/fixed_3/<dataset_tag>/...` 分离
  - cache 内 `fixed_k` 与 `source_csv` 一致性检查
  - 启动时打印最终解析到的 `data_path / cache_path / out_dir / fixed_k`
- `modelv3_new` 的已知 `fixed_3` 首轮结果：
  - `mae_all=14.7789`
  - `mae_changed=64.0375`
  - `avg(|dR|>50)=4.3280`
  - 位置准确率（对0/1/2/3个）=`0.0040/0.1860/0.5720/0.2380`
- 当前额外说明：
  - 2026-03-25 发现 `fixed_2` 首轮训练存在旧 cache 复用/目录未创建问题；
  - 现已在代码层补强隔离与校验，并完成 `fixed_2` 有效重跑。
- 当前 `fixed_2` 有效结果：
  - `mae_all=10.6534`
  - `mae_changed=54.1879`
  - `avg(|dR|>50)=2.3960`
  - 位置准确率（对0/1/2个）=`0.0080/0.4760/0.5160`
- 结论：`fixed_change_recon` 当前作为冻结的诊断子项目保留，主线精力先回到 `MLP/GNN` 的 `REG` 与 `CLS`。

## 版本
- `modelv1`：重构基线（稀疏 + 排序 + 后期 Kirchhoff 一致性）。
- `modelv1_coord`：在 `modelv1` 上增加坐标矩约束。
- `modelv2`：增加固定计数软约束 + hardest 分离约束。
- `modelv2_coord`：`modelv2` 的坐标约束版。
- `modelv1_new`：新重建模型（残差结构 + MSE/ID/Physics/Sparse）。
- `modelv2_new`：变化位加权回归 + 未变化位稀疏/hinge + 延后物理约束 + 第4假阳性抑制。
- `modelv3_new`：当前冻结主线，仅用于 `fixed_2 / fixed_3`，采用“加权回归 + 坐标 + 延后 physics + 第 k+1 大抑制 + 第 k/k+1 间隔”。

## 当前建议
1. 当前这条线先冻结，不作为主线频繁改模对象。
2. 需要诊断时优先重跑 `fixed_3`，再用 `fixed_2` 看 pair 场景本身是否同样不稳定。
3. 只维护 `modelv3_new` 的可用性与数据兼容性，旧版本继续作为历史参照。

## 命令示例
```bash
python 64Nodes/mlp/fixed_change_recon/scripts/generate_data_fixed.py --fixed-k 3 --total-combos 5000
python 64Nodes/mlp/fixed_change_recon/scripts/generate_data_fixed.py --fixed-k 2 --total-combos 5000
python 64Nodes/mlp/fixed_change_recon/scripts/generate_data_fixed.py --fixed-k 3 --current-a 0.01 --dataset-tag 10mA --total-combos 5000
python 64Nodes/mlp/fixed_change_recon/scripts/generate_data_fixed.py --fixed-k 2 --current-a 0.01 --dataset-tag 10mA --total-combos 5000
python 64Nodes/mlp/fixed_change_recon/modelv3_new/train.py
python 64Nodes/mlp/fixed_change_recon/modelv3_new/inference.py
python 64Nodes/mlp/fixed_change_recon/modelv3_new/train.py --data-path 64Nodes/mlp/fixed_change_recon/data_fixed/training_data64_fixed_2.csv --fixed-k 2 --dataset-tag 5mA
python 64Nodes/mlp/fixed_change_recon/modelv3_new/inference.py --data-path 64Nodes/mlp/fixed_change_recon/data_fixed/training_data64_fixed_2.csv --fixed-k 2 --dataset-tag 5mA
```

## 说明
- `modelv3_new` 当前只允许 `fixed_2 / fixed_3`，不再把这条线当成通用 fixed-k 实验框架。
- 即使 `fixed_2` 与 `fixed_3` 都使用同一个 `dataset-tag`（例如都用 `5mA`），cache 和 outputs 也会按 `fixed_2 / fixed_3` 自动分开。
- 当前环境若缺 `numpy`，可以优先复用 `inverse_identifiability/.vendor`。
