# MLP 运行日志（64Nodes）

说明：
- 本文件用于记录 `64Nodes/mlp` 各版本的关键训练结果与问题分析。
- 与 `64Nodes/mlp/README.md` 中的版本定义一一对应。
- 更新规则：仅新增，不覆盖旧记录。

记录模板（后续新增实验建议遵循）：
- 版本与命令
- 关键指标（分类：混淆矩阵；回归：mae_all/mae_changed/avg(|dR|>50)）
- 现象分析（例如 2/3 混淆、扩散、过拟合）
- 修改动作与下轮计划

## 2026-03-19 - MLP_CLS/modelo1
- 对应版本：`README -> 三、版本说明 -> 1) MLP_CLS / modelo1`
- 关键结果：
- 最佳验证宏平均F1（日志中观测）：约 `0.8045`
- 测试混淆矩阵：
```
[[ 66   0   0   0]
 [  0 290  22   5]
 [  0  37 172 112]
 [  0   5  74 217]]
```
- 现象分析：
- `0类` 识别稳定。
- `1类` 较好，但有少量被判到 `2/3`。
- `2/3` 混淆明显，主要是 `2->3` 偏多，说明阈值边界偏向高类别或模型对高变化更敏感。
- 初步结论：分类基线可用，但后续需重点优化 `2/3` 的分界策略。

## 2026-03-19 - MLP_REG/modelo1
- 对应版本：`README -> 三、版本说明 -> 2) MLP_REG / modelo1`
- 关键结果：
- `mae_all = 14.2580`
- `mae_changed = 44.6234`
- `avg(|dR|>50) = 5.27`
- 测试“变化数量”混淆矩阵（由回归结果阈值化后统计）：
```
[[ 66   0   0   0]
 [ 10 132  73 102]
 [  2  19  39 261]
 [  0   1   7 288]]
```
- 现象分析：
- 回归主指标优于 `MLP_FULL/modelo1_reg2prob`，说明纯回归头对幅值学习更稳定。
- `avg(|dR|>50)=5.27` 高于真实平均变化数量（约1.86），仍存在“扩散预测”。
- 数量混淆显示 `2/3` 类被高估，反映阈值判断后会偏向较大变化数量。
- 初步结论：回归能力较好，但稀疏性与数量一致性仍需加强。

## 2026-03-19 - MLP_FULL/modelo1_reg2prob
- 对应版本：`README -> 三、版本说明 -> 3) MLP_FULL / modelo1_reg2prob`
- 关键结果：
- `mae_all = 22.5167`
- `mae_changed = 29.0432`
- `avg(|dR|>50) = 14.09`
- 测试混淆矩阵：
```
[[  0   0  66   0]
 [  0   0   0 317]
 [  0   0   0 321]
 [  0   0   0 296]]
```
- 现象分析：
- `mae_changed` 低于纯回归模型，说明变化位置上的幅值拟合能力有提升。
- 但 `mae_all` 与 `avg(|dR|>50)` 明显恶化，表示模型对大量未变化电阻也输出了较大幅值。
- 混淆矩阵几乎全部塌缩到高类别（2/3），说明 `reg2prob` 的概率映射和计数约束当前设置不平衡。
- 初步结论：当前 `reg2prob` 更关注“有变化处”的拟合，牺牲了整体稀疏性与数量判定稳定性。

## 结论摘要（本轮）
- 当前最佳分类：`MLP_CLS/modelo1`（可用，需优化2/3分界）。
- 当前最佳回归：`MLP_REG/modelo1`（综合更稳）。
- 当前完整任务：`MLP_FULL/modelo1_reg2prob` 需要继续调参（重点控制扩散与数量塌缩）。

## 2026-03-20 - o2 改版（待训练）

### MLP_CLS/modelo2
- 变更点：
- 阈值搜索改为“带约束搜索”：`t1 <= t2 <= t3`。
- 新增 `2<->3` 轻惩罚项：`lambda_adj`（默认 0.15）。
- 预期影响：
- 减少阈值乱序导致的边界不稳定。
- 定向改善 `2` 与 `3` 的相互误判。

### MLP_REG/modelo2
- 变更点：
- 新参数起点：`w_change=2.0`，`w_unchange=1.4`。
- 分段稀疏调度：
- 1-20 epoch：弱稀疏（先学幅值）。
- 21+ epoch：线性提升 `lambda_hinge/lambda_sparse`（逐步收缩扩散）。
- 数量阈值改为验证集搜索：`40~70，step=1`。
- 预期影响：
- 前期降低“学不动”的风险，后期加强稀疏。
- 数量估计更贴近当前模型分布，减少固定阈值偏差。

### MLP_FULL/modelo2_reg2prob
- 变更点：
- 加入 REG 的 hinge 项。
- `w_unchange` 提升到 `1.3`。
- `lambda_count` 提升到 `0.5`。
- 预期影响：
- 抑制未变化位置的中等幅值漂移。
- 增强“数量一致性”约束力度。

### MLP_FULL/modelv1_h_multitask（新建）
- 变更点：
- 共享主干 + 双头（CORAL 分类头 + 回归头）。
- 损失联合优化：`L_cls + L_reg + L_count + L_sparse + L_hinge`。
- 输出新增两种数量评估：
- 分类头混淆矩阵（主数量输出）。
- 回归阈值派生混淆矩阵（辅助对照）。
- 预期影响：
- 比单头 reg2prob 更稳定地平衡“数量判别”和“幅值回归”。

## 2026-03-20 - o2 实际运行结果（已完成）

### MLP_CLS/modelo2
- 关键结果：
- 最终验证：`val_macro_f1=0.8083`
- 测试混淆矩阵：
```
[[ 66   0   0   0]
 [  0 289  28   0]
 [  0  36 206  79]
 [  0   5 114 177]]
```
- 现象分析：
- 0/1 类较稳，2/3 仍然明显混淆。
- 训练耗时偏慢，主要原因不是 batch 小，而是“约束阈值搜索（t1<=t2<=t3）”在每次验证时执行三重网格搜索，计算量显著增加。
- 结论：
- 分类可用性继续提升，但 2/3 边界仍是主要瓶颈。

### MLP_REG/modelo2
- 关键结果：
- `mae_all=14.0631`
- `mae_changed=44.2595`
- `best_count_threshold(val)=67.0`
- `avg(|dR|>67.0)=2.88`
- Derived Count CM：
```
[[ 66   0   0   0]
 [ 23 201  41  52]
 [  4  69  84 164]
 [  0  17  31 248]]
```
- 现象分析：
- 相比 modelo1，稀疏性明显变好（高幅值扩散减少）。
- 回归误差小幅改善，训练/验证曲线仍有轻微过拟合迹象（后期波动）。
- 结论：
- o2 的分段稀疏策略有效，是当前回归主线更稳版本。

### MLP_FULL/modelo2_reg2prob
- 关键结果：
- `mae_all=12.6730`
- `mae_changed=47.0277`
- `avg(|dR|>50)=4.20`
- Derived Count CM：
```
[[  0   0  66   0]
 [  0   0   0 317]
 [  0   0   0 321]
 [  0   0   0 296]]
```
- 现象分析：
- 幅值整体 MAE 不差，但数量推断完全塌缩到高类别（计数机制失效）。
- 结论：
- reg2prob 单头仍不可靠，不适合作为当前完整任务主方案。

### MLP_FULL/modelv1_h_multitask
- 关键结果：
- `mae_all=12.1640`
- `mae_changed=48.9125`
- `avg(|dR|>50)=3.89`
- 数量头 CM：
```
[[ 66   0   0   0]
 [  0 247  69   1]
 [  0  43 192  86]
 [  0   9 108 179]]
```
- Derived Count CM（由回归阈值得到）：
```
[[ 66   0   0   0]
 [ 18 197  51  51]
 [  3  86  81 151]
 [  0  28  44 224]]
```
- 现象分析：
- 相比 reg2prob，分类头显著恢复了数量判别能力。
- 但 2/3 仍存在较大混淆，且 changed MAE 尚未达到理想水平。
- 结论：
- 在完整任务上，`modelv1_h_multitask` 当前优于 `modelo2_reg2prob`，建议作为后续主分支继续优化。

## 本轮综合结论
- 分类主线：`MLP_CLS/modelo2`（可用，继续攻克 2/3 分界）。
- 回归主线：`MLP_REG/modelo2`（当前最稳，稀疏性改善明显）。
- 完整主线：优先 `MLP_FULL/modelv1_h_multitask`，暂停 `reg2prob` 作为主方案。

## 2026-03-20 - o3 / v2 改版（已实现，待训练）
### MLP_CLS/modelo3
- 改动摘要：
- 主头 CORAL + 辅助 `2-vs-3` 头（双头分类）
- 保留 `lambda_adj`，并加入辅助头 warm-up
- 推理改为“主头判 2/3 后再用辅助头复判”
- 目的：专门缓解 `2->3` 与 `3->2` 误判，同时尽量保持 0/1 类稳定。

### MLP_REG/modelo3
- 改动摘要：
- 恢复并增强分段训练：稀疏惩罚分段上调
- `w_change` 分段提升（从幅值学习过渡到变化位强调）
- 新增 `L_count`（数量一致性）
- 新增 hard-negative hinge（重点打压未变化位的大幅假阳性）
- 新增基于组合指标的 early stopping
- 目的：在 `mae_changed` 与稀疏性之间找到更稳定平衡。

### MLP_FULL/modelv2_h_multitask
- 改动摘要：
- 仍采用双头 `h` 架构，避免三头训练不稳定
- 两阶段训练：
- Stage1 先训分类（稳定数量边界）
- Stage2 联合训练时对分类分支做特征梯度隔离（`detach_cls`），降低分类头牵引主干导致局部最优
- 加入 `L_rank`（变化位与未变化位幅值分离约束）
- 保留 `L_count/L_hinge/L_sparse`
- 目的：减少“分类头先收敛后锁死”的负面影响，让回归主干继续可优化。

### 本轮执行说明
- 该版本为“代码已落地，待你在云端跑结果”状态。
- 运行后请把三条线的关键指标贴回（尤其是 `2/3` 混淆、`mae_changed`、`avg(|dR|>threshold)`），我再给你做下一轮定向调参。

## 2026-03-20 - MLP_CLS/modelo3 实测结果与分析（已回传）
- 对应版本：`README -> 五、o3 / v2 新版本 -> 8) MLP_CLS / modelo3`
- 关键结果（用户回传）：
- 最佳验证 `val_macro_f1=0.8024`（Epoch 85）
- 测试混淆矩阵：
```
[[ 66   0   0   0]
 [  0 276  37   4]
 [  0  41 189  91]
 [  0   5  98 193]]
```
- 测试集按该矩阵计算的各类 F1（约）：
- `class0=1.000`
- `class1=0.864`
- `class2=0.586`
- `class3=0.661`
- `macro-F1=0.778`

### 现象分析
- 结论一：仍存在明显的 `2/3` 双向混淆，尤其 `2->3=91`、`3->2=98`，说明模型仍未学到稳定边界形状。
- 结论二：`1类` 也出现一定误判（`1->2/3`），比理想状态偏多，说明辅助 2v3 分支在联合训练中对主边界有扰动。
- 结论三：训练损失持续下降，但验证损失整体上升，`val_macro_f1` 在 0.78~0.80 区间震荡，属于“损失层面过拟合 + 指标平台期”。

### 指标解释（给汇报/论文可直接用）
- `macro-F1` 是 4 个类别 F1 的算术平均，每一类权重相同。
- 它反映“每个类别都要顾及”的分类质量，尤其会放大 `2/3` 难类的影响。
- 当前长期卡在约 0.80，说明瓶颈不在 0/1，而在 2/3 边界可分性。

### 下一版（modelo4）建议
- 策略：把 `2/3` 细分从“联合训练”改为“解耦训练”，减少主任务被拖偏。
- 具体改动：
- 第1步：先按 `modelo2` 训练主模型（只做 0/1/2/3 CORAL，不加 aux 头梯度）。
- 第2步：冻结主干，仅用真实标签 `2/3` 样本训练独立二分类器（可用主干特征 + 小 MLP）。
- 推理：主模型先判 `0/1/2/3`，若为 `2或3` 再交给二分类器细分。
- 额外稳态手段：
- 对 2/3 二分类器使用轻量 label smoothing（如 0.05）
- 减小全局 dropout（如 0.1 -> 0.05）并配合 early stopping（以 val macro-F1）
- 目标：保持 `0/1` 不退化，同时专门降低 `2<->3` 互错。

## 2026-03-20 - MLP_CLS/modelo4 改版落地（待训练）
- 对应版本：`README -> 六、o4 新版本 -> 11) MLP_CLS / modelo4`
- 已完成代码改动：
- 两阶段训练：
1. Stage1 仅训练主 CORAL 头；
2. Stage2 冻结主干+主头，仅训练 2v3 辅头。
- 阈值与评分：
- 约束阈值搜索 `t1<=t2<=t3`；
- 引入加权评分（轻惩 `3->2`，轻奖 class3 recall），用于主阈值与 aux 阈值选择。
- 稳定性增强：
- 辅头加入 label smoothing；
- 梯度裁剪；
- 两阶段分别 early stopping。
- 预期效果：
- 降低 `2/3` 双向混淆，尤其减少 `3->2`。
- 尽量维持 `0/1` 分类稳定性。

## 2026-03-20 - MLP_REG/modelo3 实测结果与分析（已回传）
- 对应版本：`README -> 五、o3 / v2 新版本 -> 9) MLP_REG / modelo3`
- 关键结果（用户回传）：
- `mae_all=13.2610`
- `mae_changed=42.2229`
- `best_count_threshold(val)=63.0`
- `val_macro_f1(count)=0.6506`
- `avg(|dR|>63)=2.77`
- 测试 Derived Count 混淆矩阵：
```
[[ 68   0   0   0]
 [ 10 230  49  27]
 [  1  56  89 159]
 [  1  19  37 254]]
```

### 现象分析
- 相比 `modelo2`，核心回归指标有实质提升：
- `mae_all` 下降（约 `14.06 -> 13.26`）
- `mae_changed` 下降（约 `44.26 -> 42.22`）
- 稀疏性继续改善（`avg(|dR|>thr)` 小幅下降）
- 数量识别的主要短板仍是 `true=2`：
- `2->3` 误判非常高（159），说明模型倾向把“2变化”高估成“3变化”。
- 训练/验证 loss 分叉明显，但验证 `mae_changed` 在后期仍改善：
- 属于“总损失层面过拟合 + 目标指标继续优化”的混合状态，
- 原因是当前总损失包含多项正则，和我们最终关注指标并不完全同向。

### 结论
- `modelo3` 是目前 REG 线最优版本（就你回传结果看）。
- 问题已从“整体回归不准”转为“计数边界（尤其 2 vs 3）不准”。

### 下一版建议（MLP_REG/modelo4）
- 保持网络宽度不变（当前复杂度不是主矛盾），重点改损失与校准：
1. 新增“顺序统计约束”（针对 k 误判）：
- 设 `s` 为 `|dR|` 降序，真实变化数为 `k_true`。
- 约束：第 `k_true` 大应高于正阈值，第 `k_true+1` 大应低于负阈值（若存在）。
- 这会直接抑制 `true=2` 时第3个假阳性过大。
2. 阈值搜索目标改为“偏重2类”：
- 不再只最大化 macro-F1，加入 `class2 F1/recall` 权重，降低 `2->3`。
3. 早停分数加入 count 质量：
- `score = mae_changed + a*avg(|dR|>50) - b*val_count_f1`
- 避免只靠总损失或单一稀疏项决定停点。
4. 轻微降低后段正则增速（防止后段“压得过硬”）：
- 保持 `lambda_sparse` 上限不变，放缓 `lambda_hinge` 后半段爬升。

## 2026-03-20 - MLP_REG/modelo4 改版落地（待训练）
- 对应版本：`README -> 六、o4 新版本 -> 12) MLP_REG / modelo4`
- 按“重建优先（MAE-first）”执行的实际改动：
- `lambda_count` 从 `0.10` 下调至 `0.01`（弱化计数分支对回归主任务的牵引）
- 主回归损失与稀疏约束框架保持不变，便于与 `modelo3` 做干净对照
- 推理输出增强：
- 输出从“仅预测增量”扩展为“预测电阻值 vs 真实电阻值”对照
- `inference_samples.json` 现包含：
- `pred_deltas` / `true_deltas`
- `pred_resistances` / `true_resistances`
- `abs_error_resistance`
- 目的：让误差分析直接对齐“电阻重建”口径，便于汇报与论文制图。

## 2026-03-20 - MLP_FULL/modelv2_h_multitask 实测结果与分析（已回传）
- 对应版本：`README -> 五、o3 / v2 新版本 -> 10) MLP_FULL / modelv2_h_multitask`
- 关键结果（用户回传）：
- 回归：
- `mae_all=13.4150`
- `mae_changed=42.3895`
- `avg(|dR|>50)=4.85`
- 计数（数量头）：
- `val_macro_f1=0.6468`
- 测试混淆矩阵：
```
[[ 68   0   0   0]
 [  0 156 126  34]
 [  0  57 178  70]
 [  0   5 181 125]]
```
- 计数（由回归阈值派生）：
- `reg_count_threshold=60.0`, `val_macro_f1=0.6430`
- 测试混淆矩阵：
```
[[ 68   0   0   0]
 [  6 191  59  60]
 [  1  40  63 201]
 [  0  14  25 272]]
```

### 现象分析
- 回归层面：
- 与 `MLP_REG/modelo3` 已非常接近（仅略差），说明 `h-multitask` 没有严重破坏重建主任务。
- 计数层面：
- 数量头 偏向“中高类”判定（大量 `1->2`、`3->2`）。
- 回归派生计数偏向“高估”（大量 `2->3`）。
- 推理样例中 `count head` 与 `reg-thr` 对同一样本不一致，说明两条计数通道当前缺乏一致校准。

### 结论
- 若以“电阻重建”为主目标：`modelv2_h_multitask` 是可用版本（回归指标已到当前较优水平）。
- 若以“计数稳定性”为附加目标：当前 FULL 计数分支仍明显弱于理想状态，不建议作为主汇报指标。

### 下一版建议（MLP_FULL/modelv3_h_multitask，重建优先）
1. Stage2 进一步回归优先：
- `lambda_cls_stage2_start/end` 再下调（例如 `0.10 -> 0.02`）
- `lambda_count` 下调到 `0.05` 或 `0.01`
2. 保留 Stage1 分类预热，但 Stage2 分类头只做“轻校准”，不主导主干。
3. 输出汇报主指标固定为：
- `mae_all`、`mae_changed`、`avg(|dR|>50)`
- 计数矩阵仅作为附录诊断，不参与主模型选择。

## 2026-03-20 - MLP_FULL/modelv3_h_multitask 改版落地（待训练）
- 对应版本：`README -> 六、o4 新版本 -> 13) MLP_FULL / modelv3_h_multitask`
- 按用户要求采用“保守降权”策略（不大幅下调）：
- `lambda_cls_stage2_start/end`: `0.25/0.08 -> 0.20/0.06`
- `lambda_count`: `0.60 -> 0.45`
- 目标：先观察趋势改善，不做激进参数跳变，保持训练稳定可比。
- Inference 输出改版：
1. 先输出“预测变化个数与变化 ID”（head 与 reg-threshold 两路）
2. 再输出所有电阻的预测/真实变化对比（`pred_deltas` vs `true_deltas`）
3. 同步输出预测/真实电阻值与绝对误差（`pred_resistances`、`true_resistances`、`abs_error_resistance`）
- 用途：直接对齐“重建”汇报口径，便于做样例可视化与误差归因。

## 2026-03-21 - MLP 最新实测（CLS modelo4 / REG modelo4 / FULL modelv3）

### A) MLP_CLS/modelo4（用户回传）
- 训练现象：
- Stage1 最佳 `val_macro_f1=0.7923`（Epoch 70）
- Stage2 最高 `val_macro_f1=0.7826`（Epoch 35），未超过 Stage1 峰值
- 测试混淆矩阵：
```
[[ 68   0   0   0]
 [  0 275  25  16]
 [  1  33 167 104]
 [  0  10  87 214]]
```
- 由该矩阵计算（约）：
- `macro-F1=0.7739`
- 类别F1：`class0=0.993`、`class1=0.868`、`class2=0.572`、`class3=0.664`
- 分析：
- `2/3` 混淆仍是瓶颈（`2->3=104`, `3->2=87`）。
- `1` 类也存在外溢（`1->2/3=41`），说明边界仍有耦合误差。
- inference 单样本可“看起来全对”，但不代表总体分布，仍以整体验证/测试矩阵为准。
- 备注：`aux_2v3_prob` 的含义是“在进入 2/3 分支后，属于 3 类的概率”；对预测为 0/1 的样本不参与最终判决。

### B) MLP_REG/modelo4（用户回传）
- 测试指标：
- `mae_all=13.4571`
- `mae_changed=42.0631`
- `best_count_threshold(val)=69.0`
- `val_macro_f1(count)=0.6622`
- `avg(|dR|>69)=2.38`
- Derived Count 混淆矩阵：
```
[[ 68   0   0   0]
 [ 12 242  40  22]
 [  4  63 109 129]
 [  1  30  55 225]]
```
- 分析：
- 相比 `modelo3`，重建指标继续小幅改善（尤其 `mae_changed`、稀疏性）。
- 计数端仍存在 `2->3` 倾向，但整体可解释性提升（你的 JSON 样例中“邻近误报”较常见，且真实变化位幅值通常被明显抬高）。
- 结论：`modelo4` 当前是 REG 线更稳的“重建优先”版本。

### C) MLP_FULL/modelv3_h_multitask（用户回传）
- 测试指标：
- `mae_all=13.4896`
- `mae_changed=42.1225`
- `avg(|dR|>50)=4.95`
- 数量头（val）：`macro_f1=0.6457`
- 测试混淆矩阵：
```
[[ 68   0   0   0]
 [  0 142 121  53]
 [  0  37 144 124]
 [  0   8  88 215]]
```
- 由回归阈值派生（val）：`macro_f1=0.6288`
- 测试混淆矩阵：
```
[[ 68   0   0   0]
 [ 12 214  51  39]
 [  1  64  71 169]
 [  1  22  38 250]]
```
- 分析：
- 回归表现与 REG 主线已非常接近（说明 FULL 的“重建能力”可用）。
- 计数两路（head / reg-thr）都未达到理想，尤其 `2/3` 边界仍较混乱。
- 结论：FULL 目前建议继续“重建主任务优先”，计数作为附加诊断输出。

## 阶段性总评（MLP 线）
- 共同结论：
- 仅换架构难以突破，主要瓶颈在“输出建模 + 损失耦合 + 计数映射”。
- 当前可作为主汇报的优先模型：
1. 重建：`MLP_REG/modelo4`
2. 完整：`MLP_FULL/modelv3_h_multitask`（重建主指标可用，计数仅辅助）
3. 分类：`MLP_CLS/modelo4`（2/3 边界待进一步处理）

## 2026-03-21 - o5/v4 改版落地（待训练）

### 1) MLP_CLS/modelo5
- 已实现改动：
- 两阶段保留 + 回退机制：
- 若 `best_stage2_val_macro_f1 < best_stage1_val_macro_f1 + 0.002`，最终回退 Stage1。
- Aux 难例采样替代“全量 2/3”：
- 样本条件：`pred_main in {2,3}` 或 `|p3 - t3| < 0.15`。
- `aux_label_smoothing=0.03`
- 训练参数：`stage1=70`、`stage2=30`、`lr_aux=1e-4`、`patience_aux=5`
- 说明：`|p3 - t3|` 用于刻画“离 2/3 边界的距离”，越小越接近分界难例。

### 2) MLP_REG/modelo5
- 已实现改动：
- 新增 `L_order`（顺序约束），用于压制 `2->3` 高估。
- 早停评分改为更重建导向（加入 `val_mae_all` 项）。
- count 阈值搜索加入 2 类加权（`macroF1 + w*F1_class2`）。
- inference 增强：
- 输出预测/真实变化电阻 `id + delta` 对照；
- 同时保留全量电阻预测/真实数组。

### 3) MLP_FULL/modelv4_h_multitask
- 已实现改动：
- Warm start：从 `MLP_REG/modelo4/outputs/model_last.pt` 初始化 trunk 对应层。
- Stage2 权重按要求保守调整：
- `lambda_cls_stage2: 0.20 -> 0.10`
- `lambda_count: 0.45 -> 0.30`
- 新增头间一致性轻约束 `L_align`（默认 `0.05`）。

## 2026-03-21 change3_recon 初始化
- 新增验证子项目：`64Nodes/mlp/change3_recon`
- 数据：固定3变化（5000组合，32激励）
- 模型：
  - `modelv1`：重构基线 + 稀疏/排序 + 后期物理约束
  - `modelv2_coord`：在v1上增加坐标矩损失
- 目的：验证“已知变化数量”时的可达重构精度边界

## 2026-03-21 change3_recon v1（modelv3_sepcount 已实现，待训练）
- 新增版本目录：`64Nodes/mlp/change3_recon/modelv3_sepcount`
- 改动重点：
  - 保留 `modelv2_coord` 坐标矩约束
  - 新增固定计数软约束 `L_count3`（目标=3）
  - 新增 hardest 正负分离约束 `L_sep`（压制第4个假阳性）
  - 验证 score 显式加入 `top3_id_precision` 项，提升定位导向
- 当前状态：代码与文档已落地，等待训练结果回传

## 2026-03-21 - o5/v4 实测结果（用户回传）

### A) MLP_CLS/modelo5
- 训练现象（Stage1）：
  - 最佳 `val_macro_f1=0.7894`（Epoch 60）
- 训练现象（Stage2）：
  - 最佳 `val_macro_f1=0.7941`（Epoch 15）
  - 最终 Epoch 30：`val_macro_f1=0.7914`、`aux_thr=0.37`、`hard_samples=5343`
- 测试混淆矩阵：
```
[[ 68   0   0   0]
 [  0 278  38   0]
 [  1  42 178  84]
 [  0  12  90 209]]
```

### B) MLP_REG/modelo5
- 测试指标：
  - `mae_all=13.1408`
  - `mae_changed=42.1968`
  - `best_count_threshold(val)=65.0`
  - `val_macro_f1(count)=0.6634`
  - `val_score=0.8008`
  - `avg(|dR|>65.0)=2.56`
- Derived Count 混淆矩阵（由回归阈值得到）：
```
[[ 68   0   0   0]
 [ 12 238  44  22]
 [  2  59 100 144]
 [  0  19  57 235]]
```

### C) MLP_FULL/modelv4_h_multitask
- 测试指标：
  - `mae_all=13.5624`
  - `mae_changed=41.7580`
  - `avg(|dR|>50)=4.99`
  - `CORAL thresholds=[0.18000000000000005, 0.5100000000000001, 0.5200000000000001]`
  - `val_macro_f1(count_head)=0.6606`
  - `reg_count_threshold=67.0`
  - `val_macro_f1(reg_threshold)=0.6412`
- 混淆矩阵（数量头, 行=真实，列=预测）：
```
[[ 68   0   0   0]
 [  0 164 123  29]
 [  0  53 166  86]
 [  0  11 136 164]]
```
- Derived Count 混淆矩阵（由回归阈值得到）：
```
[[ 68   0   0   0]
 [  9 213  43  51]
 [  1  59  72 173]
 [  1  25  36 249]]
```

## 2026-03-22 - change3_recon 版本号重排
- 按项目整理需求，对 `change3_recon` 版本做重排：
  - 原 `modelv2_coord` 改名为 `modelv1_coord`
  - 原 `modelv3_sepcount` 改名为 `modelv2_coord`
  - 新增 `modelv2`（与 `modelv2_coord` 同一套增强损失，默认关闭坐标约束）
- 目的：统一“v1/v2 + coord”命名，便于对照实验管理与结果汇报。
- 详情见：`64Nodes/mlp/change3_recon/README.md` 与 `64Nodes/mlp/change3_recon/Log.md`

## 2026-03-22 - change3_recon modelv2 实测与文件核验
- 用户回传 `modelv2` 结果：
  - `mae_all=17.6353`
  - `mae_changed=67.9094`
  - `avg(|dR|>50)=5.67`
  - `top3_id_precision=0.6127`
- 对比 `modelv1 / modelv1_coord`，本轮未见提升（尤其 `mae_changed`）。
- 本地核验 `modelv2` 与 `modelv2_coord`：
  - 二者并非同文件（关键差异在 `lambda_coord` 与缓存路径）；
  - `modelv2` 默认 `lambda_coord=0.0`，`modelv2_coord` 默认 `lambda_coord=0.12`。
- 详细记录见：`64Nodes/mlp/change3_recon/Log.md`

## 2026-03-22 - change3_recon modelv2_coord 实测
- 用户回传 `modelv2_coord` 结果：
  - `mae_all=17.3997`
  - `mae_changed=66.5398`
  - `avg(|dR|>50)=5.59`
  - `top3_id_precision=0.6340`
- 与同轮 `modelv2` 对比，四项指标均有改善（尤其 top3 命中率）。
- 但 `avg(|dR|>50)` 仍显著高于目标 3，说明假阳性扩散问题仍未解决。
- 详细记录见：`64Nodes/mlp/change3_recon/Log.md`

## 2026-03-22 - change3_recon modelv1_new 已实现
- 新增版本目录：`64Nodes/mlp/change3_recon/modelv1_new`
- 架构：`896-1024-896-512-256`，含 `896` 残差连接，输出 `112`，`tanh * 310`
- 指标：`mae_all`、`mae_changed`、位置预测准确率（对0/1/2/3个）
- inference 输出精简为：
  - `pred_id`、`true_id`
  - `pred_id_delta`、`true_id_delta`
  - `pred_delta_all`
- 损失（权重优先级）：`MSE > ID(坐标) > Physics(4激励) > Sparse(L1)`
- 详情见：`64Nodes/mlp/change3_recon/README.md` 与 `64Nodes/mlp/change3_recon/Log.md`

## 2026-03-22 - change3_recon modelv1_new 首轮实测（用户回传）
- 测试指标：
  - `mae_all=5.8611`
  - `mae_changed=70.4115`
  - `位置准确率(对0/1/2/3个)=0.0040/0.1920/0.5140/0.2900`
- 阶段结论：
  - 新模型的全局误差与位置命中明显改善；
  - 但真实变化位的幅值误差仍高于 `modelv2_coord`，说明当前损失重心偏向“全局压误差”，还未真正转化为更好的变化值重构。
- 后续重点：
  - 保住位置命中增益；
  - 重新增强对真实变化位幅值的监督；
  - 补回假阳性诊断指标，继续围绕重构主任务调参。

## 2026-03-22 - change3_recon modelv2_new 已实现
- 新增版本目录：`64Nodes/mlp/change3_recon/modelv2_new`
- 相对 `modelv1_new` 的关键改动：
  - 回归主损失改为“变化位更重、未变化位较轻”的加权回归（默认 `w_change=7.0`，`w_unchange=1.0`）
  - 稀疏约束改为主要作用在未变化位，并新增未变化位 `hinge`
  - 新增第4假阳性分离约束 `L_sep`
  - 物理约束延后到 `epoch 25` 后逐步启用
  - 补回 `avg(|dR|>50)` 诊断指标，但不作为主目标
- 设计目标：
  - 保住 `modelv1_new` 的位置命中优势；
  - 重点拉低 `mae_changed` 与假阳性扩散。

## 2026-03-22 - change3_recon modelv2_new 首轮实测（用户回传）
- 测试指标：
  - `mae_all=18.2218`
  - `mae_changed=56.6768`
  - `avg(|dR|>50)=8.2640`
  - `位置准确率(对0/1/2/3个)=0.0040/0.2740/0.5380/0.1840`
- 阶段结论：
  - `modelv2_new` 显著改善了真实变化位的幅值重构，`mae_changed` 为当前 change3 主线最佳；
  - 但假阳性明显增多，`avg(|dR|>50)` 大幅恶化，且“3个位置全对”低于 `modelv1_new`。
- 样例现象：
  - 预测常呈现“2个真变化 + 1个高幅值假阳性”；
  - 局部邻边误报仍存在，说明模型更容易锁定区域，但第3个位置仍不够稳。
- 后续重点：
  - 不回退“变化位加权回归”主方向；
  - 下一轮重点加强未变化位抑制、提升第4假阳性压制强度，并重调 best checkpoint 选择口径。

## 2026-03-22 - change3_recon modelv3_new 已实现
- 新增版本目录：`64Nodes/mlp/change3_recon/modelv3_new`
- 设计定位：
  - 延续 `modelv2_new` 的加权回归主思路；
  - 但对 loss 做减法，只保留 4 项：
    - `L_reg`
    - `L_id`
    - `L_phys`
    - `L_fp4`
- 关键变化：
  - 删除 `L_sparse_unchange`、`L_hinge_unchange`、`L_sep`
  - 改为单一“第4大幅值抑制”项 `L_fp4`
  - best checkpoint 的评分加入对 `avg(|dR|>50)` 超标部分和 `pos3` 的更强惩罚
- 设计目标：
  - 观察“更少的 loss 项”能否在保住低 `mae_changed` 的同时，减少假阳性与位置退化。

## 2026-03-23 - 主线训练脚本支持按数据集切换与分目录输出
- 目的：
- 方便在云端并行训练 `5mA` 与 `10mA` 两套主数据，不再手动改代码路径。
- 本轮改动的主线入口：
- `MLP_CLS/modelo5/train.py`
- `MLP_REG/modelo5/train.py`
- `MLP_FULL/modelv4_h_multitask/train.py`
- 新增命令行参数：
- `--data-path`：切换主数据 CSV
- `--dataset-tag`：指定数据集标签；默认取 CSV 文件名
- `--dataset-subdir / --no-dataset-subdir`：控制是否按数据集拆分子目录
- 行为调整：
- 默认 cache 改为写入 `cache/<dataset_tag>/...`
- 默认输出改为写入 `outputs/<dataset_tag>/...`
- `MLP_FULL/modelv4_h_multitask` 的默认 warm start 路径也会自动跟随同一 `dataset_tag`
- 直接收益：
- 避免 `training_data64.csv` 与 `training_data64Nodes_2.csv` 共用同一个缓存文件，导致“看似换数据、实际没换”的风险
- 避免不同数据集实验覆盖同一个 `outputs/`

## 2026-03-23 - 主线 inference 脚本支持按数据集标签自动找模型
- 本轮补齐入口：
- `MLP_CLS/modelo5/inference.py`
- `MLP_REG/modelo5/inference.py`
- `MLP_FULL/modelv4_h_multitask/inference.py`
- 新增参数：
- `--data-path`
- `--dataset-tag`
- `--dataset-subdir / --no-dataset-subdir`
- 默认行为：
- 自动从 `cache/<dataset_tag>/...` 读取缓存
- 自动从 `outputs/<dataset_tag>/...` 读取 `model_last.pt / metrics.json / standardization.npz`
- `inference_samples.json` 也会写回对应数据集子目录

## 2026-03-24 - MLP_CLS/modelo5 三电流首轮对比（用户回传）
- `5mA`：Stage1 最佳 `val_macro_f1=0.7939`
- `10mA`：Stage1 最佳 `val_macro_f1=0.7929`
- `20mA`：Stage1 最佳 `val_macro_f1=0.7905`
- 现象分析：
- 三套数据差距很小，说明当前分类线的主要限制仍是 `2/3` 可分性，而不是信号幅值。
- 结论：
- 分类线没有充分证据支持把主数据默认切到更大电流。

## 2026-03-24 - MLP_REG/modelo5 三电流首轮对比（用户回传）
- `5mA`：
- `mae_all=12.9421`
- `mae_changed=42.8196`
- `best_count_threshold(val)=67.0`
- `avg(|dR|>67)=2.30`
- `10mA`：
- `mae_all=13.2653`
- `mae_changed=42.3156`
- `best_count_threshold(val)=70.0`
- `avg(|dR|>70)=2.28`
- `20mA`：
- `mae_all=13.1778`
- `mae_changed=42.4159`
- `best_count_threshold(val)=64.0`
- `avg(|dR|>64)=2.62`
- 现象分析：
- `10mA` 只在 `mae_changed` 上略优；
- `5mA` 的整体稳定性仍然最好；
- `20mA` 没有继续带来一致提升。
- 结论：
- 若只保留一套 MLP 回归主数据，建议先保留 `5mA`；若强调 `mae_changed`，可把 `10mA` 作为唯一并行对照。

## 2026-03-24 - MLP_FULL/modelv4_h_multitask 三电流首轮对比（用户回传）
- `5mA`：
- `mae_all=12.9463`
- `mae_changed=42.6742`
- `avg(|dR|>50)=4.56`
- `10mA`：
- `mae_all=13.3865`
- `mae_changed=41.8829`
- `avg(|dR|>50)=4.91`
- `20mA`：
- `mae_all=13.1761`
- `mae_changed=43.1798`
- `avg(|dR|>50)=4.62`
- 额外说明：
- 这三轮训练日志都出现了 `WarmStart Skip`；
- 原因不是模型权重缺失，而是默认 warm start 路径少了一层 `mlp/`。
- 当前处理：
- 默认路径已修正；
- 因此 0324 这三组 FULL 结果应按“无 warm start 版本”保守解读。
- 结论：
- FULL 目前仍未明显优于 REG 阈值化路线，数量头继续只建议作为诊断输出。

## 2026-03-24 - MLP 主线优先级与默认数据调整
- 当前 MLP 主推进顺序改为：
- 第一优先级：`MLP_REG/modelo5`
- 第二优先级：`mlp/change3_recon` fixed-change 重构主线
- 第三优先级：`MLP_CLS/modelo5`
- `MLP_FULL/modelv4_h_multitask` 暂不作为每轮同步改动对象，仅在回归线出现明确收益后补跑验证。
- 非 fixed-change MLP 主线默认数据统一切到 `10mA`：
- `data/training_data64Nodes_2.csv`
- 说明：
- 保留 `--data-path` 与 `--dataset-tag`，因此 `5mA / 20mA` 对照能力不受影响；
- 这次同时补稳了相对路径解析，默认入口与手动切换入口现在都可直接复用。

## 2026-03-24 - MLP 主线下一版优化（按当前优先级）
- `MLP_REG/modelo5`：
  - `L_order` 进一步强调真实 `2/3` 样本
  - 新增超参：
    - `order_margin=12.0`
    - `order_weight_k2=1.25`
    - `order_weight_k3=1.45`
  - 验证选模新增过预测惩罚：
    - `val_overpredict_alpha=0.60`
    - `val_overpredict_k23_alpha=1.10`
- `MLP_CLS/modelo5`：
  - 阈值搜索评分新增 `bonus_r2=0.05`
  - Stage2 的 aux hard mask 现在显式包含真实 `2/3` 样本，避免只围绕“主头边界附近样本”训练
- fixed-change 子项目：
  - `change3_recon` 已更名为 `fixed_change_recon`
  - 当前主线 `modelv3_new` 已支持 `fixed_2 / fixed_3`
- 当前专用数据已整理为：
  - `training_data64_fixed_3.csv`
  - `training_data64_fixed_2.csv`

## 2026-03-25 - MLP_CLS/modelo5（10mA，权重调整后首轮回传）
- 训练结果：
- Stage1 最佳验证 `val_macro_f1=0.7925`
- 测试混淆矩阵：
  - `[[68,0,0,0],[0,280,31,5],[1,31,173,100],[0,8,88,215]]`
- 现象分析：
- 与 0324 首轮相比没有明显提升；
- 主要错误仍集中在 `2 -> 3` 与 `3 -> 2`，说明本轮 Stage2 难例策略和评分微调不足以改变主瓶颈。
- 阶段结论：
- `MLP_CLS` 暂不建议继续大改主干，后续仅保留轻量级 `2/3` 难例与阈值校准尝试。

## 2026-03-25 - MLP_REG/modelo5（10mA，权重调整后首轮回传）
- 测试结果：
- `mae_all=12.8283`
- `mae_changed=44.0864`
- `best_count_threshold(val)=69.0`
- `val_macro_f1=0.6516`
- `val_score=0.7997`
- `avg(|dR|>69)=2.15`
- 计数混淆矩阵：
  - `[[68,0,0,0],[14,259,20,23],[5,89,115,96],[1,34,67,209]]`
- 与 0324 的 `10mA` 基线对比：
- `mae_changed: 42.3156 -> 44.0864`，回退；
- `avg(|dR|>threshold): 2.28 -> 2.15`，更稀疏。
- 现象分析：
- 本轮新增 `L_order` 权重与过预测惩罚，确实把伪峰数量压得更低；
- 但同时牺牲了真实变化位幅值恢复，说明当前约束强度偏大。
- 下一轮建议：
- 保留 `val_overpredict / val_overpredict_k23` 这类诊断日志；
- 但将新增排序项与选模惩罚回调到更保守区间，再看是否能找回 `mae_changed`。

## 2026-03-25 - fixed_change_recon/modelv3_new（fixed_3 首轮回传，fixed_2 暂不可信）
- `fixed_3` 测试结果：
- `mae_all=15.5451`
- `mae_changed=62.5130`
- `avg(|dR|>50)=5.0460`
- 位置准确率(对0/1/2/3个)=`0.0040/0.1960/0.5680/0.2320`
- 与上一轮 `fixed_3` 基线对比：
- `mae_changed: 64.0375 -> 62.5130`，略有改善；
- `avg(|dR|>50): 4.3280 -> 5.0460`，变差；
- `pos3: 0.2380 -> 0.2320`，略回退。
- 阶段判断：
- 当前不能把这轮视为稳定进步，更像“变化位幅值稍好，但假阳性又抬头”。
- `fixed_2` 本轮结果暂不可信：
- 训练记录先出现 `cache/change2_5mA/cache_change3_v3_new.npz` 的 `FileNotFoundError`；
- 随后 `fixed_2` 训练日志与 `fixed_3` 逐 epoch 完全一致，且仍显示 `位置准确率(0..3)`。
- 结论：
- `fixed_2` 当前存在 cache/路径隔离或 fixed_k 校验缺失问题；
- 在修复之前，不应使用这轮 `fixed_2` 指标做任何结论。

## 2026-03-25 - fixed_change_recon/modelv3_new（cache/输出隔离修正，并冻结为 fixed_2/fixed_3）
- 本轮代码修正：
- 当前只支持 `fixed_k in {2,3}`，不再把这条线当作通用 fixed-k 框架。
- 默认 cache 改为：
  - `cache/fixed_3/<dataset_tag>/cache_fixed_v3_new.npz`
  - `cache/fixed_2/<dataset_tag>/cache_fixed_v3_new.npz`
- 默认 outputs 改为：
  - `outputs/fixed_3/<dataset_tag>/...`
  - `outputs/fixed_2/<dataset_tag>/...`
- 训练/推理新增强校验：
  - cache 内 `fixed_k` 必须与当前数据一致
  - cache 内 `source_csv` 必须与当前数据文件一致
  - 首次运行新 tag 会先自动创建 cache 目录
- 阶段结论：
- `fixed_change_recon` 当前冻结为两个纯回归诊断子项目：
  - `fixed_3`
  - `fixed_2`
- 这条线暂不再作为当前主线持续改模对象。

## 2026-03-25 - MLP_REG/modelo5（默认值回退到 0324 更稳口径）
- 针对本轮 `mae_changed` 回退问题，已将 2026-03-24 追加的更强默认值回退：
  - `order_margin: 12.0 -> 0.0`
  - `order_weight_k2: 1.25 -> 1.0`
  - `order_weight_k3: 1.45 -> 1.0`
  - `val_overpredict_alpha: 0.60 -> 0.0`
  - `val_overpredict_k23_alpha: 1.10 -> 0.0`
- 保留项：
  - `L_order` 本身仍保留
  - `val_overpredict / val_overpredict_k23` 诊断日志仍保留
- 当前意图：
- 先恢复 0324 那版更好的 `mae_changed`，再决定是否重新逐项加回额外约束。

## 2026-03-25 - MLP_REG/modelo5（10mA，回退后首轮回传）
- 测试结果：
- `mae_all=12.9366`
- `mae_changed=43.2690`
- `best_count_threshold(val)=70.0`
- `val_macro_f1=0.6442`
- `val_score=0.7911`
- `avg(|dR|>70)=2.10`
- 与前一轮回退前对比：
- `mae_changed: 44.0864 -> 43.2690`，回升；
- `avg(|dR|>threshold): 2.15 -> 2.10`，仍保持较稀疏。
- 与 0324 的最佳 `10mA` 基线对比：
- `mae_changed: 42.3156 -> 43.2690`，仍略差。
- 阶段判断：
- 回退方向是正确的，但当前默认值还没有完全恢复到 0324 最佳状态；
- 下一步更适合从训练 seed / 早停点 / `w_change` 调度这类较小因素继续微调，而不是重新加重额外惩罚。

## 2026-03-25 - MLP_REG/modelo5 inference 修正
- 修正内容：
- 默认推理路径现在会按以下顺序自动兼容：
  - `outputs/<dataset_tag>/...`
  - `outputs/<data_path.stem>/...`
  - 旧版 `outputs/` 根目录文件
- 推理输出新增：
  - `pred_change_ids`
  - `pred_change_deltas`
  - `true_change_ids`
  - `true_change_deltas`
  - `true_deltas / pred_deltas`
  - `true_resistances / pred_resistances / abs_error_resistance`
- 目的：
- 解决“训练时未显式传 dataset-tag，推理却按别名 tag 查不到模型”问题；
- 同时让样例输出能够直接判断“预测到底对没对”。 

## 2026-03-25 - fixed_change_recon/modelv3_new（fixed_2 首轮有效结果）
- `fixed_2` 当前已完成有效重跑，运行口径正常：
  - 日志显示 `位置准确率(0..2)`
  - cache 路径落在 `cache/fixed_2/5mA/...`
  - outputs 路径落在 `outputs/fixed_2/5mA/...`
- 测试结果：
  - `mae_all=10.6534`
  - `mae_changed=54.1879`
  - `avg(|dR|>50)=2.3960`
  - 位置准确率(对0/1/2个)=`0.0080/0.4760/0.5160`
- 与 `fixed_3` 对比：
  - `fixed_2` 的 `mae_changed` 和假阳性都明显更好；
  - 说明固定 2 变化场景确实更容易，而固定 3 变化本身就更难，不只是分类边界问题。


## 2026-03-25 - MLP 新版实验目录建立（modelo6）
- 本轮新增：
  - `MLP_REG/modelo6`
  - `MLP_CLS/modelo6`
- `MLP_REG/modelo6` 结构要点：
  - 输入保持 `(Batch, 32, 28)`，不再展平为 896
  - 使用共享单激励编码 + 激励维 token mixing 的 MLP-Mixer 风格主干
  - 回归头改为 `mask(sigmoid) * value(tanh)` 门控输出
  - 损失改为 `MSE + lambda_mask_l1 * mean(mask_prob)`，默认 `lambda_mask_l1=0.05`
- `MLP_CLS/modelo6` 结构要点：
  - 主干同样改为 `(32, 28)` 的 MLP-Mixer 风格输入流
  - 分类头仍为 CORAL
  - 在分类头前新增 supervised contrastive 特征投影，重点拉开真实 `2/3` 类
- 当前状态：
  - `modelo6` 仅完成代码落地，尚未训练
  - `modelo5` 继续作为当前 MLP 主线稳定基线
- 本地验证说明：
  - 已通过 `python -m py_compile`
  - 由于当前本地环境缺少 `torch`，未完成真实前向冒烟测试
## 2026-03-26 - MLP 新版本首轮结果（modelo6）
- `MLP_CLS/modelo6`：
  - `test_macro_f1=0.8854`
  - 验证最优阈值：`[0.05, 0.05, 0.24]`
  - 测试混淆矩阵：
    - `[[71,0,0,0],[0,300,0,0],[0,11,224,77],[0,1,53,263]]`
- 与旧主线 `MLP_CLS/modelo5@10mA` 对比：
  - `0.7925 -> 0.8854`
  - 提升约 `+0.0929`
- 结论：
  - `modelo6` 在分类线上同样是明确成功的；
  - `MLP-Mixer` 风格输入流 + SupCon 对 `2/3` 类边界确实有实质帮助；
  - 当前 `MLP_CLS` 新主线可以切到 `modelo6`。

- `MLP_REG/modelo6`：
  - `mae_all=1.5010`
  - `mae_changed=56.7012`
  - `best_count_threshold(val)=40.0`
  - `val_macro_f1=0.7425`
  - `avg(|dR|>40)=1.64`
  - `avg(mask_prob)=0.0214`
  - 派生数量混淆矩阵：
    - `[[71,0,0,0],[5,295,0,0],[0,66,228,18],[0,24,164,129]]`
- 与旧主线 `MLP_REG/modelo5` 对比：
  - 优势：
    - `mae_all` 显著更低；
    - 输出更稀疏；
    - 未变化位抑制比旧主线更强。
  - 劣势：
    - `mae_changed` 仍明显高于旧主线；
    - `2/3` 变化样本出现系统性低估，尤其 `3 -> 2` 偏多。
- 结论：
  - `modelo6` 方向是对的，但当前回归头过于保守；
  - `mask(sigmoid) * value(tanh)` 与 `lambda_mask_l1=0.05` 的组合压缩了真实变化位输出；
  - 下一步宜优先减弱 `mask L1` 或改成分阶段/暖启动方式，而不是马上推翻 `Mixer` 结构。

- 本轮附带观察：
  - `MLP_CLS/modelo6` 验证阶段同样出现 `np.exp` overflow warning；
  - 属于数值稳定性问题，不影响这轮“分类大幅进步”的主结论。
## 2026-03-26 - MLP 下一版建立（modelo7）
- 基于 `modelo6` 首轮结果补充分析：
  - `MLP_CLS/modelo6` 已取得显著提升，但剩余误差主要集中在 `2/3` 边界；
  - 这与 GNN 分类线的混淆形态高度一致，说明当前残余难点更接近 EIT 逆问题本身，而不是单一架构缺陷。
  - 同时日志中的 `overflow encountered in exp` 说明分类 logits 已过大，模型存在过度自信风险。
  - `MLP_REG/modelo6` 则体现为“相对激进”：对 `1` 变化更敏感，但在 `3 -> 2` 低估上仍较明显。
- 本轮新建：
  - `MLP_CLS/modelo7`
  - `MLP_REG/modelo7`
- 目录说明：
  - 复制自 `modelo6`
  - 复制带来的旧 `outputs/cache/__pycache__` 已清理
- `MLP_CLS/modelo7` 本轮改动：
  - 验证/测试阶段的概率计算改为 `scipy.special.expit`
  - 默认 `weight_decay: 1e-4 -> 1e-3`
  - `inference.py` 现在默认优先抽样真实 `2/3` 变化样本
- `MLP_REG/modelo7` 本轮改动：
  - 保留 `modelo6` 的 `Mixer + 门控回归` 主干，先不推翻结构
  - `lambda-mask-l1: 0.05 -> 0.01 -> 0.002`，继续释放对非零变化的预测勇气，避免 `REG` 线整体仍被过强稀疏约束压住
  - `val-sparse-alpha: 0.25 -> 0.05`，减轻选模阶段对“较多非零预测”的惩罚
  - 常规 `inference.py` 默认优先抽样真实 `2/3` 变化样本
  - 新增 `inference_full.py`
- `MLP_REG/modelo7/inference_full.py` 逻辑：
  - 先用 `MLP_CLS/modelo7` 预测 `K`
  - 再取 `MLP_REG/modelo7` 输出中绝对值最大的前 `K` 个电阻作为最终变化边
  - 其余边强制置 0
- 兼容性修复：
  - 若运行环境缺少 `scipy`，`modelo7` 的 `train.py / inference.py / inference_full.py` 现在会自动回退到数值稳定的本地 `expit` 实现，不再因 `ModuleNotFoundError` 中断。
- 当前意图：
  - 先保留 `modelo6` 验证过的结构优势
  - 用更稳定的分类概率计算和更贴近最终用途的融合推理，继续观察 `2/3` 场景表现

## 2026-03-27 - MLP 0326 训练结果记录
- `MLP_CLS/modelo7`：
  - `test_macro_f1=0.8735`
  - `best_epoch=30`
  - 测试混淆矩阵：
    - `[[71,0,0,0],[0,300,0,0],[0,11,208,93],[0,1,51,265]]`
- 结论：
  - 当前 `MLP_CLS` 已足以承担“宏观数量判断”角色；
  - 主要剩余难点仍集中在 `2/3` 边界。
- `MLP_REG/modelo7`：
  - `mae_all=1.5352`
  - `mae_changed=56.5487`
  - `best_count_threshold(val)=40.0`
  - `val_macro_f1=0.7323`
  - `avg(|dR|>40)=1.66`
  - 测试派生数量混淆矩阵：
    - `[[71,0,0,0],[8,292,0,0],[1,68,215,28],[0,20,156,141]]`
- 结论：
  - 数值精度仍明显弱于 `GNN_REG/modelo3`
  - 但整体较稳，适合继续作为异构融合中的辅助回归分支
## 2026-03-27 - MLP_CLS / modelo8 新版本建立
- 新建目录：
  - `mlp/MLP_CLS/modelo8`
- 主要改动：
  - 在 `modelo7` 的 `MLP-Mixer` 主干后新增 `>=2 vs <=1` 辅助二分类头
  - 联合损失改为：
    - `CORAL + lambda_aux * BCE + lambda_supcon * SupCon`
  - `2/3` 类对比学习权重提高到 `1.5`
  - 阈值搜索改为基于验证集概率分布的分位数候选，再逐类搜索
  - 训练加入 `CosineAnnealingWarmRestarts(T_0=10, T_mult=2)`
- 当前定位：
  - `modelo8` 是下一轮 MLP 分类主线
  - 重点观察它是否能在不破坏 `0/1` 稳定性的前提下进一步改善 `2/3` 边界

## 2026-03-27 - MLP_CLS/modelo8 首轮结果回传与口径修正
- raw `10mA`（未筛选）：
  - `test_macro_f1=0.8909`
  - `test_aux_acc=0.9710`
  - `val_best_thresholds=[0.02, 0.02, 0.5]`
  - 测试混淆矩阵：
    - `[[71,0,0,0],[0,335,1,0],[0,18,201,58],[0,0,50,266]]`
- screened `10mA`：
  - `test_macro_f1=0.8801`
  - `test_aux_acc=0.9820`
  - `val_best_thresholds=[0.02, 0.02, 0.1]`
  - 测试混淆矩阵：
    - `[[71,0,0,0],[0,336,0,0],[0,12,189,76],[0,1,50,265]]`
- `MLP_CLS/modelo7@screened` 对照：
  - `test_macro_f1=0.8697`
  - 测试混淆矩阵：
    - `[[71,0,0,0],[0,300,0,0],[0,11,219,82],[0,0,69,248]]`
- 当前判断：
  - 首轮结果未证明 `modelo8` 相比 `modelo7` 有稳定优势。
  - `screened` 数据当前不再推荐作为 `CLS` 主线默认数据：
    - 去除“内部正负抵消配对”后，训练分布被人为净化；
    - 模型在更简单分布上训练后，反而削弱了对复杂边缘特征的鲁棒性；
    - 对当前任务可视为一次不利的 Distribution Shift。
  - `modelo8` 的 Aux Head 没有真正帮助 `2/3`：
    - `val_aux_acc` 很快达到约 `98.4%`；
    - 辅助任务过于简单，aux loss 迅速趋近于 0；
    - 因而无法持续给主干提供区分 `2` 与 `3` 的有效梯度。
- 当前取舍：
  - `MLP_CLS` 数量判断锚点仍保持为 `modelo7`
  - 主线默认数据继续维持为未筛选 `10mA`：`data/training_data64Nodes_2.csv`
  - `modelo8` 默认训练 / 推理 seed 已对齐为 `20260325`，与 `modelo7` 保持一致，便于重新做公平 A/B

## 2026-03-28 - 训练记录0327：MLP_CLS/modelo7 vs modelo8 公平 A/B
- 对照口径：
  - 数据：未筛选 `10mA`，即 `data/training_data64Nodes_2.csv`
  - seed：`20260325`
  - 目的：在完全相同的数据划分下，判断 `modelo8` 是否真的优于 `modelo7`

### MLP_CLS/modelo8
- 训练历程摘要：
  - `epoch 1`: `val_macro_f1=0.7823`，`train_aux_acc=0.8079`
  - `epoch 30`: `val_macro_f1=0.9011`，`train_aux_acc=0.9915`
  - `epoch 95`: `val_macro_f1=0.9105`，`train_aux_acc=0.9966`
  - `epoch 140`: `val_macro_f1=0.8998`，`train_aux_acc=1.0000`
  - `val_loss` 约在 `epoch 25` 左右降到最低区间，之后整体回升，呈现出“辅助头已饱和、主任务边界仍在震荡”的形态
- 最终结果：
  - `test_macro_f1=0.8852`
  - `test_aux_acc=0.9850`
  - `val_best_thresholds=[0.02, 0.02, 0.65]`
  - 测试混淆矩阵：
    - `[[71,0,0,0],[0,298,2,0],[0,10,231,71],[0,0,60,257]]`

### MLP_CLS/modelo7
- 训练历程摘要：
  - `epoch 1`: `val_macro_f1=0.7672`
  - `epoch 15`: `val_macro_f1=0.8958`
  - `epoch 30`: `val_macro_f1=0.9072`
  - `epoch 110`: early stopping，`best_epoch=30`
- 最终结果：
  - `test_macro_f1=0.9022`
  - `val_best_thresholds=[0.05, 0.05, 0.7500000000000002]`
  - 测试混淆矩阵：
    - `[[71,0,0,0],[0,296,4,0],[0,8,240,64],[0,0,46,271]]`

### 结论分析
- 公平 A/B 已经给出明确结论：
  - `modelo8` 不如 `modelo7`，差距约 `0.0170` macro-F1
- `modelo8` 的问题集中在真正困难的边界，而不是简单类别：
  - `2->3`: `64 -> 71`
  - `3->2`: `46 -> 60`
  - 说明 Aux Head 没有帮助 hardest boundary，反而让 `2/3` 类更容易互相吞并
- `aux_acc` 很高并不代表有效：
  - `test_aux_acc=0.9850` 只能说明 `>=2 vs <=1` 很容易学
  - 它没有转化成主任务收益，反而印证该辅助任务过于简单，无法持续提供有价值的梯度
- 阈值也给出了一致信号：
  - `modelo8` 的前两级阈值仍然贴近下界（`0.02, 0.02`）
  - 这说明 ordinal 边界本身并没有因为辅助头而变得更干净
- 当前路线取舍：
  - `MLP_CLS` 下一版继续从 `modelo7` 出发
  - `modelo8` 暂停，不再作为当前分类主线的默认母版
