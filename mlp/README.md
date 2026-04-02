# MLP 子项目说明（64Nodes）

说明：
- 本文件只记录 MLP 方案的技术设计与版本架构演进。
- 具体训练结果、问题现象、改进结论请看：`64Nodes/mlp/Log.md`。

## 为什么 MLP 可能适合本课题
- 本课题输入本质是固定拓扑下的“全局响应向量”（32组激励叠加），MLP 对这种结构化向量映射非常直接。
- 参数与训练流程简单，可快速形成稳定 baseline，便于做消融和版本迭代。
- 在样本量中等（1万组合）时，MLP 往往先于复杂结构收敛，适合作为“性能下限+调参参照”。
- 结合 `tanh` 输出范围与稀疏/hinge 约束，可把物理先验（±30%、少量变化）直接写进目标函数。

## 一、统一数据与训练规范
- 数据源：`64Nodes/data/training_data64.csv`
- 可切换数据源：`64Nodes/data/training_data64Nodes_2.csv`（10mA 激励版）
- 可切换数据源：`64Nodes/data/training_data64Nodes_3.csv`（20mA 激励版）
- 当前非 fixed-change MLP 主线默认数据：`64Nodes/data/training_data64Nodes_2.csv`（10mA）
- 样本粒度：按 `combo_id` 聚合，每个样本对应 32 组激励。
- 输入特征：`dV = V_meas - V_base`。
- `V_base` 计算：从 `change_count=0` 组合中，按激励序号求外部电压均值。
- 输入维度：`32 x 28 = 896`，训练时展平为向量。
- 数据划分：`train:val:test = 8:1:1`（按 combo 粒度）。
- 标准化：仅用训练集统计量（按特征维度）。
- 日志：每 5 个 epoch 输出一次 train/val loss。
- 当前主线训练/推理入口支持命令行切换：
- `--data-path`：切换主数据 CSV
- `--dataset-tag`：指定数据集标签；默认取 CSV 文件名
- 默认会把 cache 写到 `cache/<dataset_tag>/`，输出写到 `outputs/<dataset_tag>/`
- 当前已支持该机制的主线版本：
- `MLP_CLS/modelo5`
- `MLP_REG/modelo5`
- `MLP_FULL/modelv4_h_multitask`
- 2026-03-24 三套主数据首轮对比结论：
- 在主混合数据集上，增大激励对 MLP 主线的提升有限，`10mA` 只在 `mae_changed` 上略优，`20mA` 没有稳定继续提升。
- 考虑到当前阶段需要收缩战线、减少数据并行，非 fixed-change MLP 主线默认统一改为 `10mA`；`5mA / 20mA` 保留为对照数据，不再作为默认入口。
- `MLP_FULL/modelv4_h_multitask` 的默认 warm start 路径已修正为 `mlp/MLP_REG/modelo4/outputs/<dataset_tag>/model_last.pt`。

## 当前主推进顺序（2026-03-25）
1. `MLP_REG/modelo5`
2. `MLP_CLS/modelo5`
- `mlp/fixed_change_recon` 当前冻结为固定 `2/3` 变化的纯回归诊断子项目，不作为当前主线持续改模对象。
- `MLP_FULL/modelv4_h_multitask` 当前只保留为验证支线，不作为本轮主要改模对象。

## 当前主线小步优化（2026-03-24）
- `MLP_REG/modelo5`：
  - `L_order` 进一步强调真实 `2/3` 样本，新增 `order_margin=12`、`order_weight_k2=1.25`、`order_weight_k3=1.45`
  - 验证选模新增过预测惩罚：`val_overpredict_alpha=0.60`、`val_overpredict_k23_alpha=1.10`
- `MLP_CLS/modelo5`：
  - 阈值搜索评分新增 `bonus_r2=0.05`，不再只偏向 class3 recall
  - Stage2 的 aux 难例集合改为显式包含真实 `2/3` 样本，减少“边界附近采样不全”的问题
- `fixed_change_recon/modelv3_new`：
  - 当前主线已从 fixed-3 单点扩展为 fixed-change 通用入口
  - 新增 `fixed_k` 数据支持、`L_fp_next` 与 `L_rank_gap`
  - 当前专用数据已整理为 `training_data64_fixed_2.csv` 与 `training_data64_fixed_3.csv`

## 当前状态修正（2026-03-25）
- `MLP_REG/modelo5`：
  - 保留 `modelo5` 原本有效的 `L_order + 重建优先选模` 骨架
  - 将 2026-03-24 新增的“更强排序间隔/过预测惩罚”回退到保守默认：
    - `order_margin=0`
    - `order_weight_k2=1.0`
    - `order_weight_k3=1.0`
    - `val_overpredict_alpha=0.0`
    - `val_overpredict_k23_alpha=0.0`
  - 目的：优先恢复 0324 那一版更好的 `mae_changed`
  - `inference.py` 当前已支持旧输出目录兼容查找，并会同时输出预测/真实变化 `id + delta + 电阻值`
- `fixed_change_recon/modelv3_new`：
  - 当前只保留 `fixed_2 / fixed_3` 两个纯回归任务
  - cache 与 outputs 现已自动按 `fixed_2 / fixed_3` 和 `dataset_tag` 双层拆分
  - 训练/推理会检查 cache 内 `fixed_k` 与 `source_csv` 是否和当前数据一致
  - `fixed_2` 当前已完成有效重跑，结果优于 `fixed_3`，说明固定 3 变化本身更难

## 二、目录结构
- `MLP_CLS/modelo1`
- `MLP_CLS/modelo2`
- `MLP_CLS/modelo3`
- `MLP_CLS/modelo4`
- `MLP_CLS/modelo5`
- `MLP_REG/modelo1`
- `MLP_REG/modelo2`
- `MLP_REG/modelo3`
- `MLP_REG/modelo4`
- `MLP_REG/modelo5`
- `MLP_FULL/modelo1_reg2prob`
- `MLP_FULL/modelo2_reg2prob`
- `MLP_FULL/modelv1_h_multitask`
- `MLP_FULL/modelv2_h_multitask`
- `MLP_FULL/modelv3_h_multitask`
- `MLP_FULL/modelv4_h_multitask`

每个版本均包含：
- `model/model.py`
- `train.py`
- `inference.py`
- `outputs/`（训练后生成）

## 三、版本说明

### 1) MLP_CLS / modelo1（基线）
- 任务：变化数量分类（0/1/2/3）。
- 架构：`896 -> 1024 -> 512 -> 256 -> 128 -> 3(CORAL logits)`。
- 每层：`Linear + BN + ReLU + Dropout(0.1)`。
- 损失：CORAL BCE + 类别权重。
- 解码：温度缩放 + 阈值搜索（步长 0.01，独立搜索）。

### 2) MLP_CLS / modelo2（o2）
- 在 `modelo1` 基础上新增：
- 带约束阈值搜索：强制 `t1 <= t2 <= t3`。
- 解释：3 个阈值对应 “是否大于 0/1/2 类”，若阈值乱序会导致解码不稳定。约束搜索保证类别边界单调，避免逻辑冲突。
- 2<->3 轻惩罚：对最后一个阈值分支（区分 2 和 3）增加附加 BCE，`lambda_adj` 默认 `0.15`。

### 3) MLP_REG / modelo1（基线）
- 任务：112 电阻 `dR` 回归。
- 架构：`896 -> 1024 -> 768 -> 512 -> 256 -> 112`。
- 输出约束：`tanh * 300`（对应 ±30% 先验范围）。
- 损失：
- 变化位置 `SmoothL1`
- 未变化位置 `L1`
- 未变化位置 `hinge(ReLU(|dR|-50)^2)`
- 稀疏正则 `mean(|dR|)`
- 指标：`mae_all`、`mae_changed`、`avg(|dR|>threshold)`、数量混淆矩阵（由回归阈值化得到）。

### 4) MLP_REG / modelo2（o2）
- 在 `modelo1` 基础上新增：
- 新参数起点：`w_change=2.0`，`w_unchange=1.4`。
- 分段稀疏调度：
- Epoch 1-20：弱稀疏（默认 `lambda_hinge=0.10`，`lambda_sparse=0.005`）。
- Epoch 21+：线性升至目标值（默认到 `lambda_hinge=0.30`，`lambda_sparse=0.015`）。
- 数量阈值不再固定 50：在验证集搜索 `40~70`（步长 1），选最优阈值用于测试集统计。

### 5) MLP_FULL / modelo1_reg2prob（基线）
- 任务：完整问题（回归 + 数量推断）。
- 核心：先回归 `dR`，再映射概率 `p = sigmoid((|dR|-50)/tau)`，最后 `k_pred = round(sum(p))`。
- 架构：同 `MLP_REG`。
- 损失：回归（变化/未变化/稀疏）+ 数量一致性 `L_count`。

### 6) MLP_FULL / modelo2_reg2prob（o2）
- 在 `modelo1_reg2prob` 基础上新增：
- 加入 REG 同款 hinge 项（未变化位置）；
- `w_unchange: 1.0 -> 1.3`；
- `lambda_count: 0.2 -> 0.5`；
- 默认新增 `lambda_hinge=0.3`，`hinge_threshold=50`。

### 7) MLP_FULL / modelv1_h_multitask（o2 新增）
- 结构：小写 “h” 多任务网络（共享主干 + 双头）。
- 主干：`896 -> 1024 -> 768 -> 512`。
- 分类头（CORAL）：`512 -> 256 -> 128 -> 3 logits`。
- 回归头：`512 -> 256 -> 112`（`tanh * max_abs`）。
- 损失：
- 分类损失：CORAL BCE（含类别权重）。
- 回归损失：变化 `SmoothL1` + 未变化 `L1` + hinge + sparse。
- 计数一致性：`L_count = |sum(sigmoid((|dR|-thr)/tau)) - k_true|`。
- 评估输出：
- 回归：`mae_all`、`mae_changed`、`avg(|dR|>50)`。
- 数量：分类头混淆矩阵（CORAL 阈值校准后）+ 回归阈值化的派生混淆矩阵。

## 四、运行建议（o2）
- 先跑 `MLP_CLS/modelo2` 看数量分类稳定性。
- 再跑 `MLP_REG/modelo2`，重点看 `mae_changed` 与 `avg(|dR|>threshold)` 的平衡。
- 完整任务并行对比：
- `MLP_FULL/modelo2_reg2prob`（单头，简单）
- `MLP_FULL/modelv1_h_multitask`（双头，表达更强）

## 五、o3 / v2 新版本（本轮新增）
### 8) MLP_CLS / modelo3
- 目标：继续压缩 `2/3` 误判，不削弱 `0/1` 稳定性。
- 架构：主干不变（`896 -> 1024 -> 512 -> 256 -> 128`），输出改为双头：
- 主头：CORAL `3 logits`（预测 `>0, >1, >2`）
- 辅头：`2-vs-3` 二分类头（仅在主头判为 2/3 时启用）
- 训练：
- 主损失：CORAL + 类别权重
- 辅损失：`2-vs-3 BCE`（warm-up 逐步增权）
- 保留最后阈值轻惩罚 `lambda_adj`
- 推理：`主头先判 0/1/2/3 -> 若为2或3则交给辅助头二次裁决`

### 9) MLP_REG / modelo3
- 目标：在保持 `mae_changed` 的同时，进一步减少“未变化电阻大幅漂移”。
- 架构：同 `modelo2`（回归主干不变）。
- 训练策略升级：
- 分段稀疏调度（早期弱稀疏，后期逐步加强）
- `w_change` 线性提升（先学幅值，再强调变化位）
- 新增 `L_count`：`|sum(sigmoid((|dR|-thr)/tau)) - k_true|`
- 未变化位 hard-negative hinge（对大假阳性加重惩罚）
- 早停依据改为组合指标：`val_mae_changed + alpha * val_avg(|dR|>50)`

### 10) MLP_FULL / modelv2_h_multitask
- 目标：缓解“分类头局部最优拖累回归头”的问题。
- 架构：仍是 `h` 型双头（不引入第三头）：
- 共享主干：`896 -> 1024 -> 768 -> 512`
- 分类头：CORAL
- 回归头：`dR` 回归（`tanh * max_abs`）
- 关键训练机制（两阶段）：
- Stage1（分类预热）：仅优化分类分支（含 `lambda_adj`），先把数量边界学稳
- Stage2（联合优化）：主优化回归相关损失；分类支路采用“特征梯度隔离”(`detach_cls`)避免反向牵引主干
- 联合损失：`L_cls + L_reg(change/unchange) + L_hinge + L_sparse + L_count + L_rank`
- 其中 `L_rank` 用于拉开“变化位/未变化位”绝对幅值差

### 推荐先跑顺序（本轮）
1. `MLP_CLS/modelo3`
2. `MLP_REG/modelo3`
3. `MLP_FULL/modelv2_h_multitask`

## 六、o4 新版本（分类线新增）
### 11) MLP_CLS / modelo4
- 目标：针对 `2/3` 边界继续优化，同时尽量不伤 `0/1`。
- 核心思想：把 `2/3` 细分改成“解耦训练”，不再与主分类头强耦合。
- 训练流程：
1. Stage1：只训练主 CORAL 头（0/1/2/3），并做 `t1<=t2<=t3` 的约束阈值搜索。
2. Stage2：冻结主干与主头，只训练 `2-vs-3` 辅头（仅用真实 2/3 样本监督）。
- 阈值策略：
- 主阈值搜索与辅助阈值搜索均加入加权目标，轻微惩罚 `3->2`，并提高 `class3 recall` 权重。
- 推理策略：
- `主头先判类别 -> 若主头判为2或3，则交给2v3辅头复判`。

### 12) MLP_REG / modelo4（已实现）
- 目标：回归主任务优先（MAE-first），计数约束退居辅助。
- 与 `modelo3` 的核心差异：
- `lambda_count` 下调为 `0.01`（弱化计数损失对主回归的牵引）
- 保留 `mae_changed` 主导的 early stopping 逻辑
- 推理输出增强：
- `inference.py` 现在同时输出预测与真实电阻值
- 输出字段包含：`pred_resistances`、`true_resistances`、`abs_error_resistance`
- 说明：数量混淆矩阵仍保留为“派生诊断指标”，不作为训练主导目标。

### 13) MLP_FULL / modelv3_h_multitask（已实现）
- 目标：在不明显牺牲计数稳定性的前提下，进一步强调重建主任务（保守降权版本）。
- 相对 `modelv2_h_multitask` 的改动：
- `lambda_cls_stage2_start/end`: `0.25/0.08 -> 0.20/0.06`（小幅下调）
- `lambda_count`: `0.60 -> 0.45`（保守下调，不做激进压缩）
- 推理输出增强（和 REG 保持一致风格）：
- 先输出预测变化个数与预测变化电阻 ID（head 与 reg-thr 两路）
- 再输出所有 112 个电阻的预测/真实变化量与电阻值对比
- JSON 字段包含：
- `pred_change_ids_head`、`pred_change_ids_reg_threshold`
- `pred_deltas`、`true_deltas`
- `pred_resistances`、`true_resistances`、`abs_error_resistance`

### 14) MLP_CLS / modelo5（已实现）
- 目标：在两阶段框架下抑制“Stage2 反向拖累”。
- 相对 `modelo4` 的改动：
- 新增回退机制：若 `Stage2_best < Stage1_best + 0.002`，最终回退到 Stage1 模型与阈值。
- Aux 训练改为难例采样，不再覆盖全部 `y>=2`：
- 样本条件：主头预测为 `2/3`，或满足边界条件 `|p3 - t3| < 0.15`。
- 其中：`p3` 为主头第三阈值概率，`t3` 为第三阈值，二者差值越小表示越靠近 2/3 分界。
- `aux_label_smoothing: 0.05 -> 0.03`
- 训练策略微调：
- `stage1_epochs=70`
- `stage2_epochs=30`
- `lr_aux=1e-4`
- `patience_aux=5`

### 15) MLP_REG / modelo5（已实现）
- 目标：继续压制 `2->3` 高估，同时维持重建优先。
- 相对 `modelo4` 的改动：
- 新增顺序约束损失 `L_order`（基于 `|dR|` 排序统计）
- 核心作用：约束第 `k_true+1` 大幅值不过大，减少“多报一个变化”。
- 早停评分改为更重建导向：
- `val_score = val_mae_changed + a*val_mae_all + b*val_avg(|dR|>50)`
- 阈值搜索目标加入 2 类加权（不再只看 macro-F1）：
- `score = macroF1 + w * F1_class2`
- inference 增强：
- 直接输出预测/真实变化电阻 `id + delta` 对照，便于样例直观分析。

### 16) MLP_FULL / modelv4_h_multitask（已实现）
- 目标：在 FULL 中引入更稳的重建先验，同时减少头间冲突。
- 相对 `modelv3_h_multitask` 的改动：
- Warm start：从 `MLP_REG/modelo4` 加载 trunk 对应层权重初始化。
- Stage2 权重调整（按你的要求保守修改）：
- `lambda_cls_stage2_end: 0.06 -> 0.10`（整体保持在 0.20->0.10 区间）
- `lambda_count: 0.45 -> 0.30`
- 新增头间一致性轻约束 `L_align`：
- 对齐 `count_cls_soft` 与 `count_reg_soft`，默认 `lambda_align=0.05`
- 推理延续 v3 的“先变化个数与ID，再全量电阻对照”输出模式。

## Fixed_change_recon 微项目（新增）
- 路径：`64Nodes/mlp/fixed_change_recon`
- 目标：固定每个样本变化数量，验证“已知变化数量”时位置与数值重构上限。
- 版本：`modelv1`、`modelv1_coord`、`modelv2`、`modelv2_coord`、`modelv1_new`、`modelv2_new`、`modelv3_new`
- `modelv2_new`：延续 `modelv1_new` 架构，但把损失改为“变化位加权回归 + 未变化位稀疏/hinge + 延后物理约束 + 第4假阳性抑制”，并补回 `avg(|dR|>50)` 诊断指标。
- `modelv3_new`：在 `modelv2_new` 上进一步做“简化 loss”对照，当前冻结为 `fixed_2 / fixed_3` 两个纯回归入口，并新增“第 k+1 大抑制 + 第 k/k+1 间隔”。
- 当前说明：
  - 这条线不再承担数量判断/分类功能
  - 当前主用途是作为 `2/3` 场景的重构诊断对照
- 详情见：`64Nodes/mlp/fixed_change_recon/README.md` 与 `64Nodes/mlp/fixed_change_recon/Log.md`

## 2026-03-25 新版实验入口：MLP_REG/modelo6 与 MLP_CLS/modelo6
### 17) MLP_REG / modelo6（MLP-Mixer Style REG）
- 目标：
  - 缓解“32 次激励全部展平后被混成一团”的问题。
  - 用显式门控回归和更直接的稀疏惩罚压制假阳性。
- 输入：
  - 保持 `(Batch, 32, 28)`，不再展平为 896。
  - 每个样本仍由 32 次激励下的 28 个外部节点电压差组成。
- 结构：
  - `SharedExcitationMLP`：对每次激励的 28 维响应做共享权重编码。
  - `ExcitationMixer`：在 32 次激励维度上做 token mixing，并叠加 channel mixing。
  - `trunk`：对混合后的全局特征做扁平化与进一步压缩。
  - 回归头：`mask_prob = sigmoid(head_mask)`，`value = tanh(head_value) * max_abs`，最终 `pred = mask_prob * value`。
- 损失：
  - `Loss = MSE(pred, true_delta) + lambda_mask_l1 * mean(mask_prob)`
  - 默认 `lambda_mask_l1=0.05`
- 关注指标：
  - `mae_changed`
  - `mae_all`
  - `avg(|dR|>threshold)`
  - `val_mask_mean`
- 默认数据：
  - 非 fixed-change 主线默认仍使用 `10mA`：`data/training_data64Nodes_2.csv`

### 18) MLP_CLS / modelo6（MLP-Mixer Style CLS）
- 目标：
  - 保留多激励结构，不再让 32 次激励在输入层直接丢失组织。
  - 用 supervised contrastive loss 强制拉开 `2/3` 类表征。
- 输入与骨干：
  - 与 `MLP_REG/modelo6` 相同，保持 `(Batch, 32, 28)`。
  - 共享 `SharedExcitationMLP + ExcitationMixer` 主干。
- 分类头：
  - 主头仍为 CORAL `3 logits`
  - 在分类头前加入 `contrast_proj`，用于 SupCon 表征学习
- 损失：
  - `Loss = CORAL + lambda_supcon * SupCon`
  - SupCon 当前只重点锚定真实 `2/3` 类样本
- 当前说明：
  - `modelo6` 是下一版实验入口，尚未形成训练结论。
  - `modelo5` 继续保留为当前已训练过的稳定基线。
- 本地验证：
  - 已通过 `py_compile`
  - 尚未在缺少 `torch` 的本地环境完成前向冒烟
## 2026-03-26 新版实验入口：MLP_CLS/modelo7 与 MLP_REG/modelo7
### 19) MLP_CLS / modelo7
- 基于 `modelo6` 的小步修正版本。
- 主要变化：
  - 概率计算改为 `scipy.special.expit`，修复 `np.exp` 溢出警告。
  - 默认 `weight_decay=1e-3`，抑制过大 logits。
  - `inference.py` 默认优先查看真实 `2/3` 变化样本。

### 20) MLP_REG / modelo7
- 结构仍沿用 `modelo6` 的 `MLP-Mixer + 门控回归`。
- 主要变化：
  - `lambda-mask-l1: 0.05 -> 0.01 -> 0.002`，与 `GNN_REG/modelo3` 同步，进一步减少过强稀疏惩罚对真实变化位的压制。
  - `val-sparse-alpha: 0.25 -> 0.05`，减轻选模阶段对“较多非零预测”的过强惩罚。
  - 常规 `inference.py` 默认优先查看真实 `2/3` 变化样本。
  - 新增 `inference_full.py`：用 `CLS` 先预测变化数 `K`，再取 `REG` 输出中绝对值最大的前 `K` 个电阻作为最终变化边。
- 说明：
  - `inference_full.py` 比固定阈值更贴近最终使用逻辑，尤其适合观察 `2/3` 场景的定位表现。

## 2026-03-27 最新结果
### 21) MLP_CLS / modelo7
- 当前结果：
  - `test_macro_f1=0.8735`
  - 最优验证轮次在 `epoch 30`
- 结论：
  - 不再出现数值溢出；
  - `0/1` 基本已稳定，`2/3` 仍有混淆，但作为“宏观数量判断”支线已经足够可靠。

### 22) MLP_REG / modelo7
- 当前结果：
  - `mae_all=1.5352`
  - `mae_changed=56.5487`
  - `avg(|dR|>40)=1.66`
- 结论：
  - 数值精度仍明显弱于 `GNN_REG/modelo3`
  - 但鲁棒性尚可，适合继续作为联合异构输出中的备份回归支线
## 2026-03-27 新增版本：MLP_CLS / modelo8
- 目标：
  - 在 `modelo7` 的基础上继续压缩 `2/3` 混淆
  - 保留 `MLP-Mixer` 主干，不回退到展平输入
- 结构：
  - 主干仍是 `SharedExcitationMLP + ExcitationMixer + trunk`
  - 主头仍为 CORAL `3 logits`
  - 在 `trunk_feat` 后新增辅助二分类头：
    - 任务：`>=2` vs `<=1`
- 损失：
  - `loss = coral_loss + lambda_aux * aux_bce + lambda_supcon * supcon`
  - `aux_bce` 正类为 `2/3`，负类为 `0/1`
  - `SupCon` 中 `2/3` 类 anchor 权重提高到 `1.5`
- 阈值：
  - 先根据验证集概率分布生成分位数候选
  - 再对三个 ordinal head 分别搜索最优阈值
- 训练：
  - 已加入 `CosineAnnealingWarmRestarts(T_0=10, T_mult=2)`
- 数据兼容：
  - 保持 `--data-path` 与 `--dataset-tag`
  - 当前默认主线数据仍为未筛选 `10mA`：`training_data64Nodes_2.csv`
- 当前实验结论：
  - 首轮结果未证明 `modelo8` 稳定优于 `modelo7`。
  - `screened` 数据当前不再推荐作为 `CLS` 主线默认数据：
    - 去除“内部正负抵消配对”后，训练分布被人为净化；
    - 模型在更简单分布上训练后，反而降低了处理复杂边缘特征的能力；
    - 对当前任务可视为一次不利的 Distribution Shift。
  - `modelo8` 的 Aux Head（`>=2` vs `<=1`）没有真正打开 `2/3` 边界：
    - 该辅助任务过于简单，`val_aux_acc` 很快达到约 `98%`；
    - 辅助 loss 随之迅速衰减，无法持续给主干提供区分 `2` 与 `3` 的有效梯度。
  - 因此当前 `MLP_CLS` 数量判断锚点仍保持为 `modelo7`，`modelo8` 暂不升为默认主线。

## 2026-03-28 公平 A/B 补充：MLP_CLS/modelo7 vs modelo8
- 当前已经完成真正同口径对照：
  - 数据一致：未筛选 `10mA`，即 `data/training_data64Nodes_2.csv`
  - seed 一致：`20260325`
- 结果：
  - `modelo7`: `test_macro_f1=0.9022`
  - `modelo8`: `test_macro_f1=0.8852`
- 从混淆矩阵看，`modelo8` 的问题不是简单“没收敛”：
  - `0/1` 类仍然很稳；
  - 真正恶化的是最难的 `2/3` 边界，尤其 `2->3` 与 `3->2` 双向混淆都比 `modelo7` 更重。
- 这进一步说明：
  - `>=2 vs <=1` 辅助头学到的是一个过于容易的粗分组任务；
  - 它没有把梯度集中投到真正需要区分的 `2 vs 3` 上。
- 当前路线结论：
  - `MLP_CLS` 下一版继续以 `modelo7` 为母版推进；
  - 若要继续加辅助任务，也应该直接服务于 `2 vs 3` 的细边界，而不是再强化 `>=2` 这种已经足够容易的信号。
