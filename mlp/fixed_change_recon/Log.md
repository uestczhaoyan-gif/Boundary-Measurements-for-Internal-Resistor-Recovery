# fixed_change_recon 日志

> 说明：本文件保留了早期以 change3_recon 命名时的历史记录；自 2026-03-24 起当前目录名已调整为 ixed_change_recon。


## 2026-03-21 v0 (初始化)
- 新建验证性微项目：`64Nodes/mlp/change3_recon`
- 新建数据脚本：`scripts/generate_data_change3.py`
- 新建模型：
  - `modelv1`（重构 + 稀疏 + top3排序 + 后期物理约束）
  - `modelv2_coord`（在v1上增加坐标矩约束）
- 新建推理脚本：
  - 输出预测top3 id/变化值/电阻值
  - 输出真实top3对比
  - 输出全112电阻预测与真实数组（json）

## 待跑实验
- 数据正式生成：5000 combos（固定3变化）
- 跑 `modelv1` 与 `modelv2_coord` 对比：
  - `mae_all`
  - `mae_changed`
  - `avg(|dR|>50)`
  - `top3_id_precision`
- 数据已生成：`training_data64_change3.csv`（160000行）
- 元数据与坐标映射已生成：
  - `training_data64_change3_meta.json`
  - `resistor_coords_bl_origin.json`
- 数据校验通过：
  - rows=160000
  - combos=5000
  - 所有样本 change_count=3
  - 每个组合均有32条激励记录
  - 组合唯一性（id+value）=5000

## 2026-03-21 v1 (modelv3_sepcount 代码落地，待训练)
- 新增版本：`modelv3_sepcount`
- 核心改动（基于 `modelv2_coord`）：
  - 新增 `L_count3`（软计数约束，目标固定为3）
  - 新增 `L_sep`（hardest 正负分离约束，抑制第4个假阳性）
  - 验证 score 新增 top3 项：`+ score_top3 * (1 - top3_id_precision)`
  - 新增推理诊断输出：`pred_soft_count`、`sep_gap_min_true_minus_max_non_true`
- 关键默认超参：
  - `lambda_count3=0.25`
  - `lambda_sep=0.30`
  - `count_threshold=52`
  - `count_temp=12`
  - `sep_margin=12`
  - `score_top3=10.0`
  - `count_start_epoch=10`
  - `sep_start_epoch=15`
- 训练命令：
  - `python 64Nodes/mlp/change3_recon/modelv3_sepcount/train.py`
- 推理命令：
  - `python 64Nodes/mlp/change3_recon/modelv3_sepcount/inference.py`
- 本轮目标判据（与主任务对齐）：
  - 首看 `mae_changed` 是否下降
  - 同时看 `top3_id_precision` 是否显著高于 `modelv2_coord`
  - 监控 `avg(|dR|>50)` 是否更接近 3.0

## 2026-03-21 v2 (modelv1 / modelv2_coord 实测结果，用户回传)

### modelv1
- 训练停止：`早停触发于 epoch 116，最佳 epoch=96`
- 测试指标：
  - `mae_all=18.1460`
  - `mae_changed=64.9168`
  - `avg(|dR|>50.0)=6.25`
  - `top3_id_precision=0.6120`

### modelv2_coord
- 训练停止：`早停触发于 epoch 112，最佳 epoch=92`
- 测试指标：
  - `mae_all=17.6922`
  - `mae_changed=65.4877`
  - `avg(|dR|>50.0)=5.94`
  - `top3_id_precision=0.6160`

### 本轮对比记录
- `modelv2_coord` 相比 `modelv1`：
  - `mae_all` 更低（`18.1460 -> 17.6922`）
  - `top3_id_precision` 略升（`0.6120 -> 0.6160`）
  - `mae_changed` 略高（`64.9168 -> 65.4877`）
  - `avg(|dR|>50)` 下降（`6.25 -> 5.94`）

### change3_recon 最需要改进点（用户指定）
- 不再输出和关注分类相关内容。
- 日志与指标口径需改为“重构优先”。
- 损失函数配置需按重构目标重整。
- inference 输出需改为重构导向字段。

## 2026-03-22 v3 (版本重排与目录整理)
- 版本重排（便于区分“坐标版/非坐标版”）：
  - 原 `modelv2_coord` -> `modelv1_coord`
  - 原 `modelv3_sepcount` -> `modelv2_coord`
  - 新增 `modelv2`（由 `modelv2_coord` 复制，默认 `lambda_coord=0.0`）
- 脚本默认参数同步：
  - `modelv1_coord` 缓存默认：`cache_change3_v1_coord.npz`
  - `modelv2` 缓存默认：`cache_change3_v2.npz`
  - `modelv2_coord` 缓存默认：`cache_change3_v2_coord.npz`
- 相关 README 与运行命令已同步更新：
  - `64Nodes/mlp/change3_recon/README.md`
  - `64Nodes/mlp/change3_recon/modelv1_coord/README.md`
  - `64Nodes/mlp/change3_recon/modelv2/README.md`
  - `64Nodes/mlp/change3_recon/modelv2_coord/README.md`

## 2026-03-22 v4（modelv2 实测结果，用户回传）

### modelv2
- 训练过程（每5轮）整体趋势：
  - `val_top3_precision` 从 `0.3060` 提升到约 `0.63`
  - `val_mae_changed` 从 `134.8962` 下降到约 `67.4574`
  - `val_avg(|dR|>50)` 长期在 `5.0~6.3` 区间，明显高于目标 3
- 测试指标：
  - `mae_all=17.6353`
  - `mae_changed=67.9094`
  - `avg(|dR|>50.0)=5.67`
  - `top3_id_precision=0.6127`

### 与已有结果对比（同一微项目）
- 相比 `modelv1`（`mae_changed=64.9168`，`top3_id_precision=0.6120`）：
  - `mae_changed` 变差
  - `top3_id_precision` 仅小幅变化
- 相比 `modelv1_coord`（原记录中的 `modelv2_coord`，`mae_changed=65.4877`，`top3_id_precision=0.6160`）：
  - `mae_changed` 更差
  - `top3_id_precision` 略低

### 文件覆盖检查（本地）
- 检查结论：`modelv2` 与 `modelv2_coord` 在本地**不是同一份文件**。
- 关键差异：
  - `modelv2`：`lambda_coord=0.0`，缓存 `cache_change3_v2.npz`
  - `modelv2_coord`：`lambda_coord=0.12`，缓存 `cache_change3_v2_coord.npz`
- 相同部分：网络结构文件 `model/model.py` 一致（这是预期行为）。

### 本轮现象分析（重构口径）
- 固定3变化并不必然更容易：该子集去掉了大量 0/1 变化的“简单样本”，整体样本难度更集中在定位+幅值精确恢复。
- 当前主要短板仍在“假阳性压不下去”（`avg(|dR|>50)` 偏高），导致 `mae_changed` 与 top3 命中率无法同步提升。
- `modelv2` 默认关闭坐标约束，当前位置信息利用较弱；从已有对比看，坐标信息对定位仍有边际帮助。

## 2026-03-22 v5（modelv2_coord 实测结果，用户回传）

### modelv2_coord
- 训练过程（每5轮）整体趋势：
  - `val_top3_precision` 从 `0.2993` 提升到约 `0.63`
  - `val_mae_changed` 从 `134.4633` 下降到约 `68.7071`
  - `val_avg(|dR|>50)` 长期在 `5.0~6.4` 区间，仍高于目标 3
- 测试指标：
  - `mae_all=17.3997`
  - `mae_changed=66.5398`
  - `avg(|dR|>50.0)=5.59`
  - `top3_id_precision=0.6340`

### 与 modelv2 对比（同轮）
- `mae_all`: `17.6353 -> 17.3997`（改善）
- `mae_changed`: `67.9094 -> 66.5398`（改善）
- `avg(|dR|>50)`: `5.67 -> 5.59`（改善但仍偏高）
- `top3_id_precision`: `0.6127 -> 0.6340`（明显改善）

### 本轮现象分析（重构口径）
- 坐标约束对“位置命中”有帮助（top3 提升明显）。
- 但核心矛盾仍是“假阳性偏多”，导致 `avg(|dR|>50)` 未能压向 3，`mae_changed` 仍未达理想值。

## 2026-03-22 v6（modelv1_new 新重建模型已实现）
- 新增版本目录：`64Nodes/mlp/change3_recon/modelv1_new`
- 架构更新：
  - `896 -> 1024 -> 896 -> 512 -> 256 -> 112`
  - 每层 `BN + ReLU + Dropout`
  - 在输入 `896` 与隐藏层 `896` 间加入残差连接
  - 输出约束：`tanh(y) * max_abs`，默认 `max_abs=310`
- 指标输出更新：
  - `mae_all`
  - `mae_changed`
  - 位置预测准确率（对0个/对1个/对2个/全对）
- inference JSON 字段精简为：
  - `pred_id`
  - `true_id`
  - `pred_id_delta`
  - `true_id_delta`
  - `pred_delta_all`
- 损失函数重构（优先级从高到低）：
  - `L_mse`（MSE）
  - `L_id`（坐标法 ID 误差）
  - `L_phys`（基尔霍夫约束，每批随机4个激励）
  - `L_sparse`（L1 稀疏）
- 训练命令：
  - `python 64Nodes/mlp/change3_recon/modelv1_new/train.py`
- 推理命令：
  - `python 64Nodes/mlp/change3_recon/modelv1_new/inference.py`

## 2026-03-22 v7（modelv1_new 首轮实测结果，用户回传）

### modelv1_new
- 训练过程：
  - 本轮跑满 `120 epoch`，未出现提前停止。
  - 验证集 `val_loss` 基本持续下降：`853.0442 -> 301.8278`
  - 验证集 `val_mae_changed` 明显下降：`146.8975 -> 69.9896`
  - 验证集“全对3个位置”比例从 `0.008` 提升到约 `0.276~0.278`
- 测试指标：
  - `mae_all=5.8611`
  - `mae_changed=70.4115`
  - `位置准确率(对0/1/2/3个)=0.0040/0.1920/0.5140/0.2900`
  - 若按“平均命中数 / 3”换算，派生 `top3_id_precision ≈ 0.6967`

### 与已有版本对比（重构口径）
- 相比当前最优旧版 `modelv2_coord`：
  - `mae_all`: `17.3997 -> 5.8611`（大幅改善）
  - `mae_changed`: `66.5398 -> 70.4115`（变差）
  - 位置命中：从旧版 `top3_id_precision=0.6340` 提升到派生约 `0.6967`
  - “3个位置全对”达到 `0.2900`，说明定位能力明显增强

### 本轮现象分析
- `modelv1_new` 呈现出明显的“位置更准、幅值更弱”特征：
  - 坐标损失与物理约束帮助模型更好地把高响应集中到正确区域，因此位置命中改善明显；
  - 但真实变化电阻上的幅值拟合仍偏弱，导致 `mae_changed` 反而高于 `modelv2_coord`。
- `mae_all` 的大幅改善不能直接等价为“重构全面变好”：
  - 当前损失主项是全量 `112` 维 `MSE`；
  - 在固定3变化任务中，未变化电阻占绝大多数，因此模型更容易通过“把大多数位置压得更接近0”来显著降低 `mae_all`。
- 从训练曲线看，模型还没有完全训满：
  - `val_loss` 到末期仍在下降；
  - 但当前主要矛盾更像是“损失重心偏向全局误差”，而不只是“训练轮数不够”。

### 下一轮优化重点
- 不把数量判断重新拉回主目标，继续围绕“固定3变化重构”优化。
- 优先加强真实变化位的幅值学习，同时尽量保住本轮已经获得的位置命中提升。
- 建议下一轮重点看：
  - `mae_changed` 能否重新压回 `66` 以下；
  - “3个位置全对”是否保持在 `0.29` 左右或继续提高；
  - 增加 `avg(|dR|>50)` 作为诊断指标，重新监控假阳性是否回升。

## 2026-03-22 v8（modelv2_new 代码落地）
- 新增版本目录：`64Nodes/mlp/change3_recon/modelv2_new`
- 架构：
  - 延续 `modelv1_new` 的 `896 -> 1024 -> 896 -> 512 -> 256 -> 112`
  - 保留 `896` 残差连接与 `tanh * 310` 输出约束
- 核心改动（基于 `modelv1_new`）：
  - 统一 `MSE` 改为“变化位更重、未变化位较轻”的加权回归：
    - 默认 `w_change=7.0`
    - 默认 `w_unchange=1.0`
  - 稀疏约束改为主要作用在未变化位：
    - `L_sparse_unchange`
    - `L_hinge_unchange`
  - 新增“第4假阳性抑制”分离损失：
    - 约束 `min_true_abs - max_nontrue_abs >= sep_margin`
    - 默认 `sep_margin=12`
  - 物理约束改为延后启用：
    - 默认 `phys_start_epoch=25`
    - 默认 `phys_ramp_epochs=20`
  - 评估补回：
    - `avg(|dR|>50)`，仅作为诊断指标，不参与主目标排序
- 默认训练超参：
  - `epochs=160`
  - `patience=30`
  - `lambda_reg=1.0`
  - `lambda_id=0.35`
  - `lambda_phys=0.15`
  - `lambda_sparse=0.05`
  - `lambda_hinge=0.10`
  - `lambda_sep=0.20`
- 训练命令：
  - `python 64Nodes/mlp/change3_recon/modelv2_new/train.py`
- 推理命令：
  - `python 64Nodes/mlp/change3_recon/modelv2_new/inference.py`
- 本轮目标：
  - 尽量保住 `modelv1_new` 已获得的位置命中提升；
  - 同时把 `mae_changed` 拉回并压低第4个假阳性。

## 2026-03-22 v9（modelv2_new 首轮实测结果，用户回传）

### modelv2_new
- 训练过程：
  - `phys_scale=0.00` 阶段（epoch 1~24）中，`val_mae_changed` 已从 `105.1456` 快速降到 `68.5000`
  - 物理约束开始逐步启用后，`val_mae_changed` 继续下降，到后段最低约 `55.6696`
  - `val_avg(|dR|>50)` 虽有下降（约 `12.75 -> 6.57`），但始终显著高于目标 `3`
  - 训练于 `epoch 89` 提前停止，记录的最佳 epoch 为 `59`
- 测试指标：
  - `mae_all=18.2218`
  - `mae_changed=56.6768`
  - `avg(|dR|>50)=8.2640`
  - `位置准确率(对0/1/2/3个)=0.0040/0.2740/0.5380/0.1840`
  - 若按“平均命中数 / 3”换算，派生 `top3_id_precision = 0.6340`

### 与已有版本对比（重构口径）
- 相比 `modelv1_new`：
  - `mae_changed`: `70.4115 -> 56.6768`（大幅改善）
  - `mae_all`: `5.8611 -> 18.2218`（明显变差）
  - “3个位置全对”: `0.2900 -> 0.1840`（明显下降）
- 相比当前旧版最好结果 `modelv2_coord`：
  - `mae_changed`: `66.5398 -> 56.6768`（显著改善，为当前最好）
  - `avg(|dR|>50)`: `5.59 -> 8.2640`（显著恶化）
  - 派生 `top3_id_precision`: `0.6340 -> 0.6340`（基本持平）

### 本轮现象分析
- 本轮改动成功把训练重心拉回到了“真实变化位幅值”：
  - `w_change=7` 的加权回归非常有效，`mae_changed` 一次性大幅下降。
- 但副作用也很明显：
  - 未变化位约束仍然不够强，导致假阳性数量明显上升，`avg(|dR|>50)` 恶化到 `8.2640`；
  - 位置命中从 `modelv1_new` 的“更容易全对3个”退化成“更容易对2个”，说明模型更敢报大幅值，但不够克制。
- 当前 `modelv2_new` 可以概括为：
  - “幅值更敢学了，真实变化位 MAE 最好”
  - “但额外报出的高幅值边也更多，位置优势没有保住”

### inference 样例观察
- 样例中经常出现“命中 2 个真实变化 + 1 个高幅值假阳性”的模式。
- 部分误报带有明显的局部/邻边特征：
  - 例如样例里 `true_id=83` 被预测成 `pred_id=84`
  - 说明模型已能大致锁定异常区域，但第 3 个位置仍可能漂到邻近边
- 也存在“正确区域被抬高，但缺失的真实边被另一条高幅值边顶掉”的现象：
  - 这与当前 `avg(|dR|>50)` 过高是一致的
  - 说明 `L_sep` 和未变化位抑制项仍不足以把第4个及之后的伪峰压下去

### 下一轮优化重点
- 下一步不建议回退当前“变化位加权回归”的主思路，因为它已经显著改善了 `mae_changed`。
- 更合理的方向是：
  - 保留加权回归主框架；
  - 进一步增强“未变化位抑制”和“第4假阳性压制”；
  - 同时调整验证 score，使 best checkpoint 选择更贴近我们真正关心的重构指标。

## 2026-03-22 v10（modelv3_new 简化 loss 版本已实现）
- 新增版本目录：`64Nodes/mlp/change3_recon/modelv3_new`
- 架构：
- 保持 `896 -> 1024 -> 896 -> 512 -> 256 -> 112`
  - 保留 `896` 残差连接
  - 保留 `tanh * 310` 输出约束
- 核心思路：
  - 不改骨干架构，只做“简化 loss”对照实验
  - 保留 `modelv2_new` 中已验证有效的变化位加权回归
  - 删除重叠较多的假阳性约束项，改为单一的“第4大幅值抑制”
- 当前损失组成：
  - `L_reg`：变化位加权回归（默认 `w_change=7.0`，`w_unchange=1.0`）
  - `L_id`：坐标矩约束
  - `L_phys`：延后启用的基尔霍夫约束
  - `L_fp4`：`mean(ReLU(top4_abs - fp4_threshold)^2)`，默认 `fp4_threshold=45`
- 删除的旧项：
  - `L_sparse_unchange`
  - `L_hinge_unchange`
  - `L_sep`
- 选模评分更新为更贴近主任务：
  - `score = mae_changed + 2.5 * max(0, avg(|dR|>50) - 3) + 8.0 * (1 - pos3) + 0.05 * mae_all`
- 默认关键超参：
  - `epochs=160`
  - `patience=30`
  - `lambda_reg=1.0`
  - `lambda_id=0.35`
  - `lambda_phys=0.15`
  - `lambda_fp4=0.30`
  - `fp4_threshold=45`
  - `phys_start_epoch=25`
  - `phys_ramp_epochs=20`
- 训练命令：
  - `python 64Nodes/mlp/change3_recon/modelv3_new/train.py`
- 推理命令：
  - `python 64Nodes/mlp/change3_recon/modelv3_new/inference.py`
- 本轮目标：
- 尽量保留 `modelv2_new` 的低 `mae_changed`
- 同时压低 `avg(|dR|>50)`，并把“3个位置全对”从 `0.184` 拉回去

## 2026-03-24 v11（modelv3_new 首轮实测结果，用户回传）

### modelv3_new
- 测试指标：
  - `mae_all=14.7789`
  - `mae_changed=64.0375`
  - `avg(|dR|>50)=4.3280`
  - `位置准确率(对0/1/2/3个)=0.0040/0.1860/0.5720/0.2380`

### 与已有版本对比（重构口径）
- 相比 `modelv2_coord`：
  - `mae_changed`: `66.5398 -> 64.0375`（改善）
  - `avg(|dR|>50)`: `5.59 -> 4.3280`（明显改善）
- 相比 `modelv1_new`：
  - `mae_changed`: `70.4115 -> 64.0375`（改善）
  - `位置全对3个`: `0.2900 -> 0.2380`（回落）

### 阶段结论
- `modelv3_new` 当前不是单项最优，但已经把“变化位幅值 / 假阳性 / 位置命中”拉到了相对更均衡的状态。
- 下一步比起继续叠加更多损失，更适合把它作为 fixed-3 的当前 balanced baseline，转入不同激励电流对比。

## 2026-03-24 v12（change3 多激励数据支持）
- 数据生成脚本 `scripts/generate_data_change3.py` 已支持：
  - `--current-a`
  - `--dataset-tag`
- 对于非 5mA 数据，默认文件名会写成：
  - `training_data64_change3_<dataset_tag>.csv`
  - `training_data64_change3_<dataset_tag>_meta.json`
- `modelv2_coord` 与 `modelv3_new` 当前已支持：
  - `--dataset-tag`
  - 自动写入 `cache/<dataset_tag>/...`
  - 自动写入 `outputs/<dataset_tag>/...`
  - 优先从对应 meta 读取 `current_source_a`
- 目的：
  - 在 fixed-3 场景下安全比较 `5mA / 10mA / 20mA`，避免 cache/outputs 互相覆盖，也避免物理损失仍误用旧电流值。

## 2026-03-24 v13（fixed-change 子项目重命名与 fixed_2 数据落地）
- 子项目目录由 `64Nodes/mlp/change3_recon` 更名为 `64Nodes/mlp/fixed_change_recon`
- 原专用数据重命名为：
  - `training_data64_fixed_3.csv`
  - `training_data64_fixed_3_meta.json`
- 新增专用数据：
  - `training_data64_fixed_2.csv`
  - `training_data64_fixed_2_meta.json`
- 数据生成脚本改为：
  - `scripts/generate_data_fixed.py`
  - 支持 `--fixed-k`
- 本轮已实际生成：
  - `fixed_3` 5mA（重写 meta 与坐标路径）
  - `fixed_2` 5mA（5000 组合）

## 2026-03-24 v14（modelv3_new 扩展到 fixed_2 / fixed_3）
- 当前主线 `modelv3_new` 已不再写死“固定3变化”：
  - 新增 `--fixed-k`
  - 会优先从对应 meta 自动读取 `change_count_fixed`
  - 位置命中统计已从 `top3` 泛化为 `top-k`
- 损失从“第4大抑制”扩展为通用 fixed-k 版本：
  - `L_fp_next`：抑制第 `k+1` 大 `|dR|`
  - `L_rank_gap`：拉开第 `k` 和第 `k+1` 大 `|dR|` 的间隔
- 默认新增超参：
  - `lambda_fp_next=0.30`
  - `fp_next_threshold=45`
  - `lambda_rank_gap=0.18`
  - `rank_gap_margin=14`
- 说明：
  - 这样在 `fixed_3` 时等价于“压第4大”
  - 在 `fixed_2` 时会自动变成“压第3大”

## 2026-03-25 v15（fixed_3 首轮回传；fixed_2 运行结果判定无效）

### fixed_3 / modelv3_new（用户回传）
- 测试指标：
  - `mae_all=15.5451`
  - `mae_changed=62.5130`
  - `avg(|dR|>50)=5.0460`
  - 位置准确率(对0/1/2/3个)=`0.0040/0.1960/0.5680/0.2320`

### 与上一轮 fixed_3 基线对比
- 相比 0324 的 `modelv3_new`：
  - `mae_changed: 64.0375 -> 62.5130`（略改善）
  - `avg(|dR|>50): 4.3280 -> 5.0460`（恶化）
  - `位置全对3个: 0.2380 -> 0.2320`（略回退）
- 阶段判断：
  - 当前更像“真实变化位幅值更好一点，但假阳性和最终定位没有同步受益”；
  - 因此这轮不能作为稳定进步版定型。

### fixed_2 本轮运行结果判定为无效
- 训练记录先出现：
  - `FileNotFoundError: .../cache/change2_5mA/cache_change3_v3_new.npz`
- 随后的 `fixed_2` 训练日志与 `fixed_3` 逐 epoch 完全一致；
- 且日志仍打印 `位置准确率(0..3)`，没有体现 `fixed_2` 应有的 `0..2` 口径。
- 当前结论：
  - 本轮 `fixed_2` 实验结果不可信，不能用于比较 `fixed_2 vs fixed_3`；
  - 当前代码虽已加入 `fixed_k` 参数，但运行时仍存在 cache/路径/一致性校验方面的缺口。

### 下一轮修改重点
- 先修数据与缓存隔离，再谈 fixed_2 指标：
- cache 文件命名去掉旧 `change3` 痕迹；
- 首次运行新 `dataset_tag` 时自动创建 `cache/<dataset_tag>/`；
- 训练和推理都要显式检查“cache 内 fixed_k 是否与当前数据和参数一致”，不一致则拒绝复用旧 cache。
- 修完后，先单独重跑 `fixed_2`，确认日志口径已经变成 `位置准确率(0..2)`，再做和 `fixed_3` 的正式对比。

## 2026-03-25 v16（运行口径修正完成，并冻结为 fixed_2/fixed_3 纯回归）

### 代码修正
- `modelv3_new/train.py` 与 `inference.py` 已完成：
  - 当前仅允许 `fixed_k in {2,3}`
  - 默认 cache 文件名由 `cache_change3_v3_new.npz` 改为 `cache_fixed_v3_new.npz`
  - cache 自动拆分到：
    - `cache/fixed_3/<dataset_tag>/...`
    - `cache/fixed_2/<dataset_tag>/...`
  - outputs 自动拆分到：
    - `outputs/fixed_3/<dataset_tag>/...`
    - `outputs/fixed_2/<dataset_tag>/...`
  - 启动时会打印最终解析到的 `fixed_k / data_path / cache_path / out_dir`
  - cache 中新增 `source_csv` 记录，用于和当前数据路径做一致性检查
- `scripts/generate_data_fixed.py` 已同步改为只接受 `fixed_k=2/3`

### 当前项目定位
- `fixed_change_recon` 当前冻结为两个纯回归诊断子项目：
  - `fixed_3`
  - `fixed_2`
- 它们当前不再承担数量判断/分类功能，也不再作为当前主线持续改模对象。

### 当前建议
- 先重跑 `fixed_2`，确认日志口径已经变成 `位置准确率(0..2)`，且 cache/outputs 路径明确落在 `fixed_2/<dataset_tag>` 下；
- 之后再视需要保留这条线做诊断，但主精力回到 `MLP/GNN` 的 `REG` 与 `CLS`。

## 2026-03-25 v17（fixed_2 有效重跑结果）

### fixed_2 / modelv3_new（用户回传）
- 运行口径确认：
  - `fixed_scope=fixed_2`
  - `cache_path=.../cache/fixed_2/5mA/cache_fixed_v3_new.npz`
  - `out_dir=.../outputs/fixed_2/5mA`
  - 日志已正确显示 `位置准确率(0..2)`
- 测试指标：
  - `mae_all=10.6534`
  - `mae_changed=54.1879`
  - `avg(|dR|>50)=2.3960`
  - 位置准确率(对0/1/2个)=`0.0080/0.4760/0.5160`

### 与 fixed_3 对比
- `fixed_2` 相比 `fixed_3`：
  - `mae_changed: 62.5130 -> 54.1879`（明显改善）
  - `avg(|dR|>50): 5.0460 -> 2.3960`（明显改善）
- 阶段结论：
  - fixed_2 明显更容易；
  - 当前 `2/3` 困难不只是“数量判断模糊”，固定 3 变化场景本身的重构难度就更高。



