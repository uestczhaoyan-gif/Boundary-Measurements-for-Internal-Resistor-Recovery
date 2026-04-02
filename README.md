# 64Nodes 项目总览

## 1. 目标
- 在 `8x8`（64 节点，112 电阻）网格上做方法验证。
- 输入：外部 28 节点在 32 组激励下的电压。
- 输出任务：
- 变化数量分类（0/1/2/3）
- 电阻变化量回归（112 维）
- 完整任务（数量 + 位置/数值）

## 2. 数据
- 主数据：`64Nodes/data/training_data64.csv`
- 主数据（10mA 版）：`64Nodes/data/training_data64Nodes_2.csv`
- 主数据（20mA 版）：`64Nodes/data/training_data64Nodes_3.csv`
- 元数据：`64Nodes/data/training_data64_meta.json`
- 元数据（10mA 版）：`64Nodes/data/training_data64Nodes_2_meta.json`
- 元数据（20mA 版）：`64Nodes/data/training_data64Nodes_3_meta.json`
- 生成脚本：`64Nodes/scripts/generate_training_data64.py`
- 当前主线训练/推理脚本已支持 `--data-path` 切换数据，并默认按数据集名拆分 `cache/` 与 `outputs/` 子目录，便于并行比较 `5mA` / `10mA` / `20mA` 三套实验。
- 自 `2026-03-24` 起，非 fixed-change 主线默认数据集统一设为 `10mA`：`64Nodes/data/training_data64Nodes_2.csv`；`5mA / 20mA` 仍保留为可选对照数据。

## 3. 方法目录（当前活跃）
- `64Nodes/mlp`
- `64Nodes/gnn`

## 3.0 当前优先级（2026-03-25）
1. `MLP_REG/modelo5` 与 `GNN_REG/modelo1`
2. `MLP_CLS/modelo5` 与 `GNN_CLS/modelo1`
- `mlp/fixed_change_recon` 当前冻结为固定 `2/3` 变化的纯回归诊断子项目，不作为当前主线持续改模对象。
- `FULL` 暂不作为当前主推进方向，仅在回归主线出现明确新收益后再补跑验证。

## 3.1 历史归档目录（history）
- `64Nodes/history/attention`
- `64Nodes/history/cnn`
- `64Nodes/history/cnn2d_mlp`
- `64Nodes/history/unet_mlp`

## 4. 模型演进导航（汇报入口）
- 全局演进总表与阶段结论：`64Nodes/Log.md`
- MLP 详细架构演进：`64Nodes/mlp/README.md`
- MLP 实验日志与问题分析：`64Nodes/mlp/Log.md`
- GNN 详细架构演进：`64Nodes/gnn/README.md`
- GNN 实验日志与问题分析：`64Nodes/gnn/Log.md`
- 历史方法归档：`64Nodes/history/*`

## 5. 文档职责（精简版）
- 根目录 `README.md`：项目说明、目录导航、读者入口。
- 根目录 `Log.md`：跨方法版本时间线与阶段性结论（摘要）。
- 子项目 `README.md`：该方法的详细架构、损失、参数、版本差异。
- 子项目 `Log.md`：训练结果、问题分析、改进建议、下一步动作。

## 6. 文档治理规则
- 长期规则见：`64Nodes/DOC_RULES.md`
- 目标：避免冗余、提升可追溯性、方便汇报与论文写作。

## 7. 汇报模板
- 一页式汇报模板：`64Nodes/ONE_PAGE_REPORT_TEMPLATE.md`

## 8. 最新版本提醒（2026-03-20）
- MLP 主线已新增：
- `64Nodes/mlp/MLP_CLS/modelo3`
- `64Nodes/mlp/MLP_CLS/modelo4`
- `64Nodes/mlp/MLP_CLS/modelo5`
- `64Nodes/mlp/MLP_REG/modelo3`
- `64Nodes/mlp/MLP_REG/modelo4`
- `64Nodes/mlp/MLP_REG/modelo5`
- `64Nodes/mlp/MLP_FULL/modelv2_h_multitask`
- `64Nodes/mlp/MLP_FULL/modelv3_h_multitask`
- `64Nodes/mlp/MLP_FULL/modelv4_h_multitask`
- 详细架构和参数请看：`64Nodes/mlp/README.md`
- 本轮实验日志入口：`64Nodes/mlp/Log.md`

## Fixed Change Recon（微型验证）
- 路径：`64Nodes/mlp/fixed_change_recon`
- 目标：针对固定变化数量场景做纯重构验证。
- 当前已冻结为两个固定子任务：
  - `fixed_3`：固定 3 个变化的纯回归重构
  - `fixed_2`：固定 2 个变化的纯回归重构
- 说明：这条线当前不再承担数量判断/分类功能，只保留为诊断与对照。
- 数据：直接基尔霍夫求解，当前已整理为专用数据目录 `data_fixed/`，主数据包括：
  - `training_data64_fixed_3.csv`
  - `training_data64_fixed_2.csv`
- 当前已支持按不同激励电流生成多套 fixed-change 专用数据。
- 模型：
  - `modelv1`：仅重构损失（稀疏 + top3 排序 + 后期物理约束）。
  - `modelv1_coord`：在 `modelv1` 上增加坐标约束（电阻 id -> 坐标，原点在左下）。
  - `modelv2`：增加固定计数软约束 + hardest 分离约束（默认不加坐标约束）。
  - `modelv2_coord`：在 `modelv2` 上叠加坐标相关约束。
  - `modelv1_new`：新重建模型（残差结构 + MSE/ID/Physics/Sparse 四项损失）。
  - `modelv2_new`：在 `modelv1_new` 上改为变化位加权回归，并新增未变化位稀疏/hinge、延后物理约束和第4假阳性抑制。
  - `modelv3_new`：当前冻结主线，仅保留 `fixed_2 / fixed_3` 两个纯回归入口，并新增“第 k+1 大抑制 + 第 k 与第 k+1 间隔”约束。
- 当前运行口径：
  - cache 自动按 `fixed_2 / fixed_3` 与 `dataset_tag` 双层拆分
  - outputs 自动按 `fixed_2 / fixed_3` 与 `dataset_tag` 双层拆分
  - 训练/推理会校验 cache 内 `fixed_k` 与数据源路径是否一致
- 详细文档：
  - `64Nodes/mlp/fixed_change_recon/README.md`
  - `64Nodes/mlp/fixed_change_recon/Log.md`

## 9. 当前提醒（2026-03-25）
- 主线 `5mA / 10mA / 20mA` 首轮对比已经完成；在主混合数据集上，增大激励带来的收益明显小于 inverse identifiability 中的“可检测性提升”结论。
- 对 MLP / GNN 主线而言，`10mA` 是当前最适合统一默认的数据选择：收益不算压倒性，但能兼顾 MLP 与 GNN 的表现，同时减少多数据并行带来的精力分散。
- 当前主精力集中在 `MLP/GNN` 两个架构的 `REG` 与 `CLS`。
- `fixed_change_recon` 目前已冻结为 `fixed_2 / fixed_3` 纯回归诊断子项目；其中 `fixed_2` 上一轮出现 cache 复用问题，本轮已补上隔离与校验，后续需要重新训练验证。

## 10. 2026-03-25 下一版模型入口（已建目录，待正式训练）
- MLP：
  - `64Nodes/mlp/MLP_REG/modelo6`
  - `64Nodes/mlp/MLP_CLS/modelo6`
- GNN：
  - `64Nodes/gnn/GNN_REG/modelo2`
  - `64Nodes/gnn/GNN_CLS/modelo2`
- 设计目标：
  - MLP 从“896 展平向量”改为保留 `(Batch, 32, 28)` 的 `MLP-Mixer` 风格输入流。
  - GNN 从“97 通道网格”改为 `(Batch, 32, 64, 4)` 的物理图输入，节点特征为 `[源掩码, 地掩码, 电压, 边界掩码]`。
  - REG 全部改为门控回归头：`mask(sigmoid) * value(tanh)`，并显式提高稀疏惩罚。
  - CLS 保留 CORAL，同时在分类头前加入 supervised contrastive loss，重点拉开 `2/3` 类表征。
- 当前定位：
  - `modelo5 / modelo1` 仍是已训练过的稳定基线。
  - `modelo6 / modelo2` 是本轮新建的下一版实验入口，后续优先在 `REG` 上跑首轮。
- 本地验证范围：
  - 已完成代码改写与 `py_compile` 语法检查。
  - 当前本地 Python 环境缺少 `torch`，尚未完成真实前向冒烟或训练验证。
## 11. 2026-03-25 GNN 新版补充说明
- `GNN_REG/modelo2` 与 `GNN_CLS/modelo2` 当前已经切换为 PyG 原生 `GATv2Conv`，不再使用本地手写注意力层。
- 为保证当前机器可直接运行，新版 GNN 入口已支持自动探测项目根目录下的本地依赖目录：`64Nodes/.vendor_torchpy311`。
- 当前本地已完成的验证包括：
  - `torch 2.11.0+cpu` 与 `torch-geometric 2.7.0` import 成功
  - 原生 `GATv2Conv` import 成功
  - `GNN_REG/modelo2` 与 `GNN_CLS/modelo2` 的最小前向通过

## 12. 2026-03-27 最新阶段判断
- 当前 `GNN_REG` 线的最佳 baseline 仍记为 `gnn/GNN_REG/modelo3` 的历史最好结果：
  - `mae_changed=25.0888`
  - 但这次结果在 2026-03-28 的 fresh cache / fresh outdir 复现实验中未能稳定重现，因此它当前更适合被视为“最佳 baseline / 历史可达上限”，用于提醒我们这条路线曾经达到过的精度，而不是可直接信赖的稳定现役模型。
- 当前最稳的数量判断锚点是 `mlp/MLP_CLS/modelo7`：
  - `test_macro_f1=0.8735`
  - `0/1` 几乎已稳定，残余误差主要仍集中在 `2/3` 边界。
- `mlp/MLP_REG/modelo7` 继续作为稳健后备：
  - `mae_changed=56.5487`
  - 数值精度仍落后于 GNN_REG，但鲁棒性较好，适合作为异构融合中的补充来源。
- `gnn/GNN_REG/model_tp1` 本轮失败：
  - `mae_changed=106.4545`
  - 说明“软物理损失 + 数据损失”仍存在明显冲突，单纯把 `KCL residual` 加进 loss 会诱导模型走向过保守解。

## 13. 2026-03-28 当前联合推理主入口
- 当前统一使用：
  - `64Nodes/gnn/inference_gnn_cmei.py`
- 默认组合：
  - `GNN_CLS/modelo3` 负责预测变化数量 `K`
  - `GNN_REG/modelo3` 负责输出 112 维电阻变化量
  - 再取 `|dR|` 最大的前 `K` 条边作为最终变化边
- 保留的有效机制：
  - 延续原 `CMEI v1` 中较稳的固定逻辑
  - 保留 `Near-Miss` 轻量后处理与完整测试集 `CMEI` 评估
- 不再使用：
  - `joint_inference/`
  - `MLP_CLS + GNN_REG (+ MLP_REG)` 异构联合推理链路

## 14. 2026-03-28 路线收敛补充
- `gnn/GNN_REG/model_tp1` 修正版结果：
  - `mae_changed=101.9642`
  - `avg(|dR|>40)=1.06`
  - 仍明显过于保守，因此这条线暂时停止继续扩展
- 当前更务实的整合方向：
  - `GNN_CLS/modelo3` 负责数量判断
  - `GNN_REG/modelo3` 负责定位与幅值回归
- 收敛原因：
  - `GNN_CLS` 与 `MLP_CLS` 的数量分类差异已经不大
  - `CLS/REG` 同时使用 GNN，更有利于形成统一的图结构主线与后续综合模型
  - 物理驱动与拓扑归纳偏置的潜力仍明显高于纯数据驱动的 `MLP` 路线

## 15. 2026-03-27 当前修正结论
- 数据口径修正：
  - `10mA` 筛选版数据集 `64Nodes/data/training_data64Nodes_2_screened.csv` 已完成首轮验证，但当前结论是不再作为默认主线数据。
  - 原因：去除“内部正负抵消配对”后，样本分布被人为净化，破坏了真实物理场景的连续分布；模型在更简单分布上训练后，反而削弱了处理复杂边缘特征的能力，属于当前任务下的分布偏移（Distribution Shift）。
  - 因此当前所有主线默认数据继续统一为未筛选 `10mA`：`64Nodes/data/training_data64Nodes_2.csv`。
- `MLP_CLS/modelo8` 当前结论：
  - 在 `modelo7` 的基础上新增了 `>=2` 辅助二分类头，但首轮结果未体现出稳定增益。
  - 当前判断：辅助头任务过于简单，`val_aux_acc` 很快升到约 `98%`，辅助 loss 迅速衰减，无法持续向主干回传对 `2/3` 边界真正有用的梯度。
  - 因此 `MLP_CLS` 的数量判断锚点仍保持 `modelo7`，`modelo8` 暂不升为默认主线。
- `GNN_REG/modelo4a / modelo4b` 当前结论：
  - 这两条分支已验证出明显过保守：`modelo4a` 的 `val_phys` 长时间维持在 `4600+`，`avg(|dR|>45)` 被卡在约 `1.15`。
  - 当前判断：KCL 软约束与主回归目标发生剧烈梯度冲突；同时强稀疏压力会把相邻真实变化边压缩成单边预测。
  - 因此 `modelo4a / modelo4b` 路线撤回，不再保留为当前主线；GNN 回归主线继续回到 `GNN_REG/modelo3`。
- `modelo3` 的下一步增强方向：
  - 不再另起 `modelo4b` 主干，而是基于 `modelo3` 直接追加候选集评估口径。
  - 目标是补充 `top3 / top4 / top5` 候选覆盖率与候选集 inference 输出，而不改变 `modelo3` 当前已验证有效的主干与训练逻辑。
  - 当前已落地的目录为：`64Nodes/gnn/GNN_REG/modelo3b`
- 联合推理补充说明：
  - 当前统一的 `gnn/inference_gnn_cmei.py` 默认仍使用未筛选 `10mA`：`64Nodes/data/training_data64Nodes_2.csv`
  - detail sample 默认使用固定 `split-seed` 与固定 `seed`，因此多次运行会看到相同样例，这是为了复现实验而非随机失效
  - 当前 `Near-Miss` 后处理会在相邻候选边之间做轻量替换，因此对“相邻双边同时保留”存在轻微抑制

## 16. 2026-03-28 训练记录0327补充
- `MLP_CLS/modelo7` 与 `MLP_CLS/modelo8` 已完成同口径公平对照：
  - 同一默认数据：未筛选 `10mA`，即 `64Nodes/data/training_data64Nodes_2.csv`
  - 同一默认 seed：`20260325`
  - 结果：
    - `modelo7`: `test_macro_f1=0.9022`
    - `modelo8`: `test_macro_f1=0.8852`
- 当前结论已明确：
  - `modelo8` 在公平 A/B 下确实不如 `modelo7`；
  - 虽然 `modelo8` 的 `val_aux_acc / test_aux_acc` 很高，但 `2/3` 边界反而更差，说明 `>=2 vs <=1` 的 Aux Head 学到的是一个过于容易的粗任务，没有真正帮助最难的 `2 vs 3`。
  - 因此 `MLP_CLS` 下一版应继续基于 `modelo7` 小步迭代，而不是以 `modelo8` 为母版继续扩展。
- `GNN_REG/modelo3b` 候选集推理已完成首轮验证：
  - `top3_candidate_cover=0.8300`
  - `top4_candidate_cover=0.8540`
  - `top5_candidate_cover=0.8610`
  - `changed_only` 口径下对应为 `0.8170 / 0.8428 / 0.8504`
- 这些数字的含义是：
  - `modelo3` 已经具备较强的“候选边排序”能力，很多样本里真实变化边虽然没全部进入最终预测集，但已经被排进前几名；
  - `top4` 相比 `top3` 有明显补救作用，说明不少错误属于“差一名”的排序误差；
  - `top5` 相比 `top4` 提升已经很小，说明剩余难样本并不只是阈值或 `K` 截断问题，而是真有一部分真实边被排到了前 5 之外。
- 阶段判断：
  - `modelo3b` 证明了 `GNN_REG/modelo3` 很适合作为候选生成器或下游二阶段筛选的前端；
  - 但它还不能被解读为“只要扩成 top5 就已经解决定位问题”，因为仍有约 `15%` 左右的变化样本至少会漏掉一条真实边。

## 17. 2026-03-28 `joint_method` 尝试、失败与回退
- 曾新建 `64Nodes/joint_method`，尝试把 `CLS/REG/联合推理` 的下一版集中到一个新目录中推进，减少 `mlp/` 与 `gnn/` 目录下 README/Log 的长度压力。
- 该尝试先后经历了：
  - 三模型版本：`MLP_CLS + GNN_REG + MLP_REG`
  - 双模型版本：`MLP_CLS + GNN_REG`
- 但在后续干净复现实验中，问题暴露得很明确：
  - `joint_method/n1/reg_n1` 没有带来回归改善；
  - 使用全新 `dataset_tag + cache_path` 重跑 `gnn/GNN_REG/modelo3` 后，结果也落回到过保守坏解，无法复现 0326 那次 `mae_changed=25.0888` 的历史好结果。
- 因此 `joint_method` 被正式判定为一次失败尝试并整体删除，不再作为后续主线目录继续维护。
- 放弃原因：
  - 它建立在一个“当前不可稳定复现”的 `GNN_REG/modelo3` 成功结果之上；
  - 在回归基线未重新站稳前，再叠加新的目录组织、损失修改和联合推理，只会增加变量、降低可定位性。

## 18. 2026-03-28 当前统一回到纯 GNN 主线
- 后续项目只保留并继续推进这两条 GNN 主线：
  - `gnn/GNN_CLS/modelo3`
  - `gnn/GNN_REG/modelo3`
- 不再继续以 `MLP_CLS`、`MLP_REG` 或 `joint_method` 作为最终方案的主开发方向。
- 这样做的原因是：
  - `GNN_CLS/modelo3` 的数量分类结果已经与当前最佳 MLP 路线同量级；
  - 纯 GNN 方案可以把“变化数量判断”和“变化边回归”统一到同一类输入表示与拓扑归纳偏置上；
  - 在 `REG` 可复现性尚未恢复前，先减少跨家族模型融合带来的额外不确定性。
- 当前 `gnn/GNN_REG/modelo3b` 仍保留为候选集能力诊断工具，但它不是新的训练主线。

## 19. 2026-03-28 GNN_REG `o4` 系列尝试
- 已建立以下 `GNN_REG` 新实验版本：
  - `gnn/GNN_REG/o4a`
  - `gnn/GNN_REG/o4a2`
  - `gnn/GNN_REG/o4b`
  - `gnn/GNN_REG/o4b2`
  - `gnn/GNN_REG/o4b3`
- 设计目的：
  - `o4a`：保持原始耦合输出，但用 `mask` 偏置初始化和 `lambda_mask_l1 warmup` 抵抗早期门控塌陷；
  - `o4a2`：继续保留耦合输出，但额外对 `mask` 头施加显式 `BCE` 监督，并把回归主损失改成更稳的 `SmoothL1`，试图在“不塌陷”和“不乱报”之间找到中间带；
  - `o4b`：彻底解耦 `mask/value`，验证“前向耦合 + 全局 MSE 梯度劫持”是否是核心问题；
  - `o4b2`：在 `o4b` 的基础上降低过报倾向，减小 `value` 分支权重并增加背景幅值抑制；
  - `o4b3`：进一步按 Loss 量级分析重平衡 `mask/value` 梯度，提升分类头的话语权。
- 当前已拿到的不完整结果还包括 `o4a`：
  - `epoch 1 -> 10` 中，`val_mask_mean` 从 `0.0340` 很快塌到 `0.0064`，`val_avg(|dR|>50)` 也回落到 `1.16~1.17`
  - 说明仅靠初始化和 `L1 warmup` 还不足以阻止门控重新掉进过保守坏解
- 当前已拿到的不完整结果来自 `o4b`：
  - `val_mae_changed` 从 `85.62` 降到 `29~33` 区间，说明解耦训练确实打破了 `modelo3` 近期复现中的门控塌陷；
  - 但 `val_avg(|dR|>50)` 同时冲到 `8~30`，明显严重过报，说明模型已经从“太保守”摆到了“太激进”的另一侧。
- 对 `val_mask_mean` 的解释：
  - 它表示验证集上 112 条边的平均 `mask_prob`；
  - 它不是越大越好，也不是越小越好；
  - 太小通常代表门控塌陷，太大通常代表过报，理想状态应与真实稀疏度同量级，并结合 `val_avg(|dR|>50)` 一起判断。
- 当前结论：
  - `o4` 系列尚未产出新的稳定基线；
  - 但它已经帮助定位出一个比“单次最好成绩”更重要的问题：当前 GNN 回归主线对 loss 量级和早期优化轨迹非常敏感。

## 20. 2026-03-28 统一切换为纯 GNN 联合推理
- 当前统一联合推理入口改为：
  - `64Nodes/gnn/inference_gnn_cmei.py`
- 默认组合改为：
  - `gnn/GNN_CLS/modelo3`
  - `gnn/GNN_REG/modelo3`
- 采用原因：
  - `GNN_CLS` 与当前最佳 `MLP_CLS` 的数量分类差距已经很小，继续维护跨架构联合链路的收益有限；
  - `CLS` 和 `REG` 同时使用 GNN，更有利于形成统一的图结构表示、拓扑归纳偏置和后续端到端综合模型；
  - 从项目长期方向看，物理驱动/拓扑驱动的建模潜力仍明显高于纯数据驱动的 `MLP` 路线。
- 因此当前正式停止：
  - `MLP_CLS + GNN_REG (+ MLP_REG)` 异构联合推理方案
  - `joint_inference/` 目录下的旧联合推理入口维护
- 说明：
  - 旧 `joint_inference` 中两版方案对比后，固定逻辑版整体比动态融合版更稳，因此新的纯 GNN 入口保留了 `Near-Miss + CMEI` 这类有效机制；
  - 但不再保留跨 `MLP/GNN` 的融合权重设计。

## 21. 2026-03-29 `GNN_REG` 新阶段判断
- 根目录 `GNN_REG训练记录.txt` 已补充 `o4a2 / o4b2 / o4b3` 的完整训练与测试结果。
- 当前最重要的新结论：
  - `gnn/GNN_REG/o4a2` 已成为当前最强的 `GNN_REG` 候选主线；
  - 它是第一条同时避开“门控塌陷”和“严重过报”的 `o4` 分支。
- `o4a2` 当前单次结果：
  - `mae_all=0.4854`
  - `mae_changed=24.2925`
  - `val_macro_f1=0.8683`
  - `avg(|dR|>40)=1.77`
- 这说明：
  - 它已经重新达到并略优于此前 `modelo3` 的历史最好上限；
  - 而且不是靠早期偶然冲高，而是在中后段持续停留在合理稀疏区。
- `o4b2 / o4b3` 的定位修正为：
  - 它们继续证明了解耦训练对降低 `mae_changed` 有帮助；
  - 但两者仍明显过报，更适合作为候选生成/高召回诊断支线，而不是最终主回归模型。
- 当前项目行动原则：
  - `modelo3` 继续保留为历史 baseline / 可达上限参考；
  - `o4a2` 作为后续优先复验对象；
  - 只有在 fresh cache / fresh outdir / 多 seed 下继续成功后，才正式替换 `modelo3`。

## 22. 2026-03-29 GNN 通用可视化工具
- 新增统一可视化脚本：
  - `64Nodes/gnn/visualize_gnn_results.py`
- 目标：
  - 为 `GNN_CLS / GNN_REG / 候选集 / 联合推理` 提供统一的展示入口；
  - 不只展示分数，还统一输出更适合汇报的总览图、样例图和版本演进对比图。
- 当前支持：
  - 单 run：生成 `overview + samples`
  - 多 run：额外生成 `comparison`
- 默认输出目录：
  - `64Nodes/gnn/outputs/visualizations`
- 当前建议用途：
  - 用于比较 `modelo3 / o4a2 / o4b2 / o4b3` 这类版本演进；
  - 用于联合推理结果的 `CMEI` 组成展示；
  - 用于挑选 4~5 个代表样例做图形化对比。
## 23. 2026-03-30 `GNN_REG/o4a2` 正式升级为当前最佳模型
- 根目录 `0330训练日志.txt` 已记录 `o4a2` 的 4 个 seed fresh-cache / fresh-outdir 复验结果。
- 当前阶段最重要的新结论：
  - `gnn/GNN_REG/o4a2` 已不再只是“单次最好结果”，而是第一条通过多 seed 验证、证明具备实际可复现性的 `GNN_REG` 主线。
  - 因此当前 `GNN` 统一主线更新为：
    - `GNN_CLS/modelo3`
    - `GNN_REG/o4a2`
- 4-seed 复验摘要：
  - `mae_all` 均值约 `0.5215`，标准差约 `0.0328`
  - `mae_changed` 均值约 `26.2952`，标准差约 `0.8000`
  - `val_count_macro_f1` 均值约 `0.8564`，标准差约 `0.0056`
  - `avg(|dR|>thr)` 均值约 `1.7268`，标准差约 `0.0247`
- 当前最佳单 checkpoint 仍来自 `gnn/GNN_REG/o4a2/outputs/training_data64Nodes_2/`：
  - `mae_all=0.4854`
  - `mae_changed=24.2925`
  - `val_macro_f1=0.8683`
  - `avg(|dR|>40)=1.77`
- 若强调“最稳妥可复验锚点”，推荐优先参考：
  - `gnn/GNN_REG/o4a2/outputs/o4a2_seed20260326/`
- 这批下载回本地的 outputs 已构成完整可复用模型包：
  - `model_last.pt`
  - `metrics.json`
  - `standardization.npz`
  - 因此它们已经可以直接用于后续推理、复现实验和结果归档。
- 当前仍需保留的现象判断：
  - 训练后期存在轻微过拟合/平台期震荡，但脚本保存的是验证最优 checkpoint，因此不影响当前可用性判断。
  - 模型仍存在较明显的保守预测倾向，误差仍主要集中在高变化数样本。
  - `mask_l1=0.002` 在 warmup 结束后按设计保持常数；它并不是“失效”，但说明当前后期稀疏控制已不再依赖继续升高的 `L1`，后续若再优化，应更多从门控监督与保守偏置修正入手。

## 24. 2026-03-30 新增 `o5a / o5b / GNN_FULL/Mv1`
- 指标口径确认：
  - `o4a2` 训练历史里显示的 `mae_all / mae_changed` 没有被重新定义；
  - 变化的是训练损失，不是这两个展示指标的计算方式。
- 新增 `GNN_REG/o5a`
  - 这是基于 `o4a2` 的最小改版；
  - 核心只针对当前最明显的“保守预测倾向”；
  - 新增对真实变化边的幅值下限约束，目标是减少“明明找到了，但幅值被压到阈值以下”的漏报。
- 新增 `GNN_REG/o5b`
  - 按 `GNN_REG优化.txt` 落地；
  - 核心改动是：
    - 给 112 条边加入绝对位置 embedding
    - 用 relaxed top-k sparsity 替代原始全局 `mask_prob.mean()` 稀疏惩罚
- 新增 `GNN_FULL/Mv1`
  - 这是第一版联合 `GNN_CLS/modelo3` 与 `GNN_REG/o4a2` 的融合目录；
  - 目录内保留两条训练入口和一个完整推理入口：
    - `train_cls.py`
    - `train_reg.py`
    - `inference.py`
  - 训练输出统一落在：
    - `outputs/cls/`
    - `outputs/reg/`
    - `outputs/inference/`
- `Mv1` 当前已用现有权重完成一次本地冒烟，结果与当前统一 GNN 推理一致，说明它已经可以作为后续联合实验的正式容器继续迭代。

## 25. 2026-03-31 `o5a / o5b` 首轮结果与主线判断
- 根目录 `0331训练记录.txt` 已完成沉淀，原始文本记录可删除。
- `o5a` 首轮结果：
  - `mae_all=0.7770`
  - `mae_changed=38.8351`
  - `val_macro_f1=0.8524`
  - `avg(|dR|>40)=1.77`
- `o5b` 首轮结果：
  - `mae_all=0.5119`
  - `mae_changed=25.4532`
  - `val_macro_f1=0.8564`
  - `avg(|dR|>40)=1.74`
- 阶段结论：
  - `o5a` 明显弱于 `o4a2`，说明单纯给真实变化边增加幅值下限约束，会破坏原有幅值校准，不适合继续作为主线。
  - `o5b` 明显强于 `o5a`，并且已经接近 `o4a2` 的多 seed 均值水平；但它仍未超过当前最佳单 checkpoint `o4a2`，因此暂时不替代现有主线。
  - 当前正式主线继续保持：
    - `GNN_CLS/modelo3`
    - `GNN_REG/o4a2`
- `o5b` 已完成一次与 `GNN_CLS/modelo3` 的联合推理验证：
  - `CMEI=93.20`
  - `macro_f1=0.9075`
  - `id_recall=0.9120`
  - `mse_all_edges=56.4333`
- 这说明：
  - `o5b` 单独作为回归器是有潜力的；
  - 但放入当前统一 GNN 联合推理链路后，整体效果仍弱于 `modelo3 + o4a2` 的默认组合，因此还不适合升为正式默认。
- 用户已把 `o5a / o5b` 的完整 outputs 下载回本地；这些输出目录中包含 `model_last.pt / metrics.json / standardization.npz`，已具备直接复用和复现实验的条件。
- `GNN_FULL/Mv1` 已新增下一版联合推理文件：
  - `gnn/GNN_FULL/Mv1/inference_v2.py`
- 这版推理新增了两类后处理：
  - 对 near-miss 增加“高置信相邻边保护”，避免误杀 `o5b` 找到的相邻真实损坏簇；
  - 加入基于 `|ΔR|` 的物理死区截断与高幅值补漏规则。
- 实测 `GNN_CLS/modelo3 + GNN_REG/o5b` 在 `inference_v2` 下得到：
  - `CMEI=91.38`
  - `macro_f1=0.8587`
  - `id_recall=0.9019`
  - `mse_all_edges=56.7826`
- 这比旧版统一推理对同一组模型的结果更差，说明：
  - “保护高置信相邻边”这个方向是合理的；
  - 但当前这版把物理死区截断和 REG 倒逼 CLS 的规则加得过重，已经开始伤害整体计数与联合分数；
  - 因此 `inference_v2` 目前只保留为实验文件，不升级为默认联合推理入口。
- 后续已基于 `o5b` 再新增一个更小的训练分支：
  - `gnn/GNN_REG/o5b1`
  - 唯一改动：`mask_bce_weight: 25 -> 20`
- 同时 `GNN_FULL/Mv1/inference_v2.py` 已继续简化：
  - 删除物理死区截断与补漏机制；
  - 只保留“高置信相邻边保护”版 near-miss。
- 简化后再次用 `GNN_CLS/modelo3 + GNN_REG/o5b` 实测：
  - `CMEI=93.17`
  - `macro_f1=0.9075`
  - `id_recall=0.9115`
  - `mse_all_edges=57.6391`
- 解释：
  - 这说明上一版 `91.38` 的退化确实主要来自过硬的物理死区/补漏；
  - 但“只保留保护墙”的版本依旧没有超过原始统一推理的 `93.20`；
  - 因此当前默认联合推理入口仍不变，`inference_v2` 继续作为实验对照文件保留。

## 26. 2026-03-31 `GNN_FULL/Mv1` 路径修复与 `Noise_test` 步骤 A
- `gnn/GNN_FULL/Mv1/inference.py` 与 `gnn/GNN_FULL/Mv1/inference_v2.py` 已补齐两层兼容：
  - 从根目录传入 `gnn/GNN_FULL/Mv1/...` 这类相对路径时，不再重复拼出 `gnn/GNN_FULL/Mv1/gnn/GNN_FULL/Mv1/...`
  - 当 `Mv1/cache/<dataset_tag>/cache_dataset_{cls,reg}_graphattn.npz` 缺失时，会自动调用 `train_cls.py / train_reg.py` 按训练同口径重建 cache，再继续推理
- 本地已用 `Mv1` 自己训练得到的 `cls/reg` 权重完成一次正式复跑：
  - `Mv1 CLS`: `test_macro_f1=0.8975`
  - `Mv1 REG`: `mae_all=0.4718`，`mae_changed=24.8125`，`val_count_macro_f1=0.8530`，`avg(|dR|>40)=1.71`
  - `Mv1 inference`: `CMEI=93.11`，`num_accuracy=0.8740`，`macro_f1=0.8975`，`id_recall=0.9173`，`mse_all_edges=53.3761`
- 基于根目录 `Noise_test.txt` 与当前默认 `10mA` 数据 `training_data64Nodes_2`，已完成步骤 A（`noise_std=0.1`，仅测试集注入 standardized voltage noise）：
  - `GNN_CLS/modelo3` 单模型：`macro_f1 0.9075 -> 0.1203`
  - `GNN_REG/o4a2` 单模型：`mae_all 0.4854 -> 23.2065`，`mae_changed 24.2925 -> 70.3696`，`count_macro_f1 0.1203`，`avg(|dR|>40) 1.771 -> 17.668`
  - `modelo3 + o4a2` 联合推理：`CMEI 93.73 -> 41.79`，`num_accuracy 0.8850 -> 0.3170`，`macro_f1 0.9075 -> 0.1203`，`id_recall 0.9248 -> 0.2608`，`mse_all_edges 49.7686 -> 1735.1177`
- 阶段判断：
  - 当前最佳 clean GNN 链条对这组 `20dB` 噪声设定非常敏感；
  - `Noise_test` 的步骤 B（训练期噪声增强）已经有充分必要性；
  - 但在带噪训练结果出来前，默认 clean 主线仍保持 `GNN_CLS/modelo3 + GNN_REG/o4a2`。

## 27. 2026-03-31 新增 `GNN_NOISE`
- 已在 `gnn/` 下正式新增噪声增强子目录：
  - `gnn/GNN_NOISE`
- 目录作用：
  - 用于执行 `Noise_test` 的策略 B
  - 即“在 clean 最优权重上做动态随机噪声增强微调”
- 当前已落地两条线：
  - `gnn/GNN_NOISE/CLS_modelo3_ft`
  - `gnn/GNN_NOISE/REG_o4a2_ft`
- 共同策略：
  - 不从头训练
  - 默认 warm start 到当前 clean 最优 `modelo3 / o4a2`
  - 默认 `lr=5e-5`
  - 默认 `epochs=30`
  - 仅训练集注入随机噪声
  - 默认 `noise_mode=gaussian`
  - 默认 `noise_std_max=0.1`
- 本轮同时新增根目录诊断文件：
  - `首轮20dB噪声诊断记录.md`（现已迁移到 `gnn/GNN_NOISE/首轮20dB噪声诊断记录.md`）
- 当前判断：
  - 这条线是当前最高优先级的新实验主线；
  - 后续如果带噪微调有效，比较重点将从 clean 分数转向 `20dB` 下的恢复程度。

## 28. 2026-04-01 `Noise_test` 收编进 `GNN_NOISE`
- 已重新比对根目录原 `Noise_test.txt` 的步骤 B 与当前 `GNN_NOISE` 默认实现：
  - 两者不完全相同
  - 当前默认实现是 `warm start + random noise + boundary-only`
  - 原始步骤 B 更接近 `fixed 20dB gaussian + all-voltage-channel`
- 因此没有直接把两者视为同一方案，而是把原始步骤 B 也正式收编进：
  - `gnn/GNN_NOISE/原始步骤B_fixed20dB.md`
- 同时在噪声训练脚本里补齐了复现原始步骤 B 的参数开关：
  - `--noise-schedule {random,fixed}`
  - `--fixed-noise-std`
  - `--noise-scope {boundary,all}`
- 根目录旧文件 `Noise_test.txt` 已删除，不再作为唯一说明入口保留。

## 29. 2026-04-01 `o5b1` 训练记录吸收与判断
- 已查看根目录 `o5b1训练记录.txt`，并与本地 `o5b1 / o5b / o4a2` 输出指标交叉核对。
- `o5b1`（`mask_bce_weight: 25 -> 20`）结果：
  - `mae_all=0.5097`
  - `mae_changed=25.5355`
  - `val_count_macro_f1=0.8422`
  - `avg(|dR|>40)=1.73`
  - `avg(mask_prob)=0.0213`
- 对比 `o5b`：
  - `mae_all` 仅从 `0.5119 -> 0.5097`，变化极小；
  - `mae_changed` 从 `25.4532 -> 25.5355`，略有回退；
  - `val_count_macro_f1` 从 `0.8564 -> 0.8422`，出现小幅下降；
  - 说明放松这一档 `mask BCE` 约束后，模型没有崩，但也没有带来实质提升。
- 对比当前最佳 `o4a2`：
  - `o5b1` 仍全面略弱于 `o4a2`
  - 因此当前默认回归主线仍保持 `GNN_REG/o4a2`
- 当前判断：
  - “约束放松后没有引发假阳性爆炸”这个分析是成立的；
  - 但把它直接上升为“已撞到信息论上限”仍然偏强；
  - 更准确的表述应是：在当前 `64Nodes + clean 10mA + 现有观测口径` 下，`o4a2 / o5b / o5b1` 已表现出明显的经验平台期，继续做 clean-only 小超参微调的收益很可能已经很低。

## 30. 2026-04-01 根目录实验记录清理
- `首轮20dB噪声诊断记录.md` 已从根目录迁移到：
  - `gnn/GNN_NOISE/首轮20dB噪声诊断记录.md`
- 迁移原因：
  - 该文件本质上属于 `Noise_test / GNN_NOISE` 实验链条；
  - 放回 `gnn/GNN_NOISE` 后，噪声诊断、原始步骤 B 保留版与带噪微调说明都集中在同一目录。
- 已确认以下原始训练记录内容都已吸收进正式文档，因此根目录原文件已删除：
  - `o5b1训练记录.txt`
  - `Mv1训练记录.txt`
  - `Noise_test.txt`
- 当前根目录不再保留这些一次性实验草稿文件，后续统一以 `README.md / Log.md / gnn/README.md / gnn/Log.md / gnn/GNN_NOISE/*` 为准。

## 31. 2026-04-01 `0401训练记录` 吸收与判断
- 已吸收根目录 `0401训练记录.txt` 的 clean / noisy 训练与推理结果。

### clean 复训结果
- `GNN_CLS/modelo3`：
  - `dataset_tag=training_data64Nodes_2_clean_20260401`
  - `test_macro_f1=0.9027`
- `GNN_REG/o4a2`：
  - `dataset_tag=training_data64Nodes_2_clean_20260401`
  - `mae_all=0.4679`
  - `mae_changed=23.5724`
  - `val_count_macro_f1=0.8628`
- `modelo3 + o4a2` 联合：
  - `CMEI=93.53`
  - `num_accuracy=0.8800`
  - `macro_f1=0.9027`
  - `id_recall=0.9237`
  - `mse_all_edges=53.2930`
- 判断：
  - 这轮 clean 复训没有超过历史最好联合结果 `93.73`
  - 但再次验证了当前主线 `modelo3 + o4a2` 的稳定性

### `noiseft_rand_boundary` 分支
- `REG_o4a2_ft` 已完成：
  - `mae_all=0.5900`
  - `mae_changed=27.5311`
  - `count_macro_f1=0.8140`
- `CLS_modelo3_ft` 没有真正训练完成：
  - 对应输出目录只有 `standardization.npz`
  - 没有 `model_last.pt`
- 因此后续 `CLS_modelo3_ft/inference.py --dataset-tag training_data64Nodes_2_noiseft_rand_boundary_20260401` 报错并不是推理脚本坏了，而是该 tag 下根本没有训练好的分类权重。
- 当前最合理解释：
  - 命令执行时实际只训练了 `REG rand_boundary`
  - 分类侧对应 tag 没有成功落盘

### `noiseft_fixed20db_all` 分支
- `CLS_modelo3_ft` 使用同一 tag 连续训练了两次，因此后一次结果覆盖前一次；
- 当前目录中真正保留下来的最终分类结果是：
  - `test_macro_f1=0.7275`
- `REG_o4a2_ft` 最终结果：
  - `mae_all=0.9201`
  - `mae_changed=48.8259`
  - `count_macro_f1=0.6178`
- clean 测试上的联合结果：
  - `CMEI=83.39`
  - `num_accuracy=0.7110`
  - `macro_f1=0.7275`
  - `id_recall=0.8048`
  - `mse_all_edges=124.9065`
- `20dB` 带噪单模型结果：
  - `CLS macro_f1=0.7121`
  - `REG mae_all=1.1800`
  - `REG mae_changed=54.2884`
  - `REG count_macro_f1=0.5829`
- 判断：
  - 相比 zero-shot `20dB` 崩塌，这条 fixed-all 带噪线确实恢复了不少噪声鲁棒性；
  - 但代价是 clean 精度和 clean 联合分数明显下降；
  - 说明“固定 20dB + 全图电压通道加噪”过于激进，当前不适合作为默认带噪训练方案。

### 额外说明
- 当前每条模型线通常会看到两类 inference：
  - 各自目录下的 `inference.py`：单模型评估
  - `gnn/inference_gnn_cmei.py`：联合 `CLS + REG` 的最终 CMEI 评估
- `只有 standardization.npz 没有 model_last.pt` 的原因是：
  - `standardization.npz` 在训练循环前就会写出
  - `model_last.pt` 则要到训练完成后才会保存
- 云端 `gnn/inference_gnn_cmei.py` 对 `--noise-std/--noise-seed` 报“unrecognized arguments”，说明当时云端运行的脚本版本还不是当前本地这版，后续若要跑 noisy CMEI，需要先同步最新文件。

## 32. 2026-04-01 中期汇报数据可视化重做
- 已按“只保留图、不保留配套说明图文板”的要求清理旧产物：
  - 删除 `midterm_assets/20260401_data_figures`
  - 删除 `中期汇报_数据可视化说明.md`
  - 删除 `tools/generate_midterm_figures.py`
- 新图统一重生成为仅包含图形元素的最终结果目录：
  - `midterm_assets/20260401_visuals/01_topology_boundary_nodes.svg`
  - `midterm_assets/20260401_visuals/02_dataset_composition.svg`
  - `midterm_assets/20260401_visuals/03_changed_edge_frequency.svg`
  - `midterm_assets/20260401_visuals/04_boundary_response_heatmaps.svg`
- 这 4 张图分别覆盖：
  - `8x8` 拓扑与 `28` 个边界测量节点
  - `10000` 个 combo 的 `change_count` 组成与 `|ΔR|` 分布
  - `112` 条电阻边在数据集中被抽到作为变化边的频次热力图
  - `32` 次激励下的 clean 边界电压均值与 damaged 相对 clean 的 `|ΔV|` 热力图
- 当前工程判断同步更新：
  - 若云端的 `gnn/inference_gnn_cmei.py` 仍不识别 `--noise-std / --noise-seed`，则需要重新上传当前本地最新版
  - 现阶段真正还缺的补训项不是全部重来，而是只补 `GNN_NOISE/CLS_modelo3_ft` 的 `training_data64Nodes_2_noiseft_rand_boundary_20260401`
  - `REG_o4a2_ft` 的 `rand_boundary`、以及 `fixed20db_all` 分支都已经有完整落盘结果，不必默认重训

## 33. 2026-04-01 `0401补充训练` 吸收与本地 outputs 校验
- 已读取根目录 `0401补充训练.txt`，确认这次补的是推荐噪声线 `noiseft_rand_boundary_20260401` 的分类侧闭环与 noisy 联合推理。

### 补充训练新增结果
- `GNN_NOISE/CLS_modelo3_ft` with `training_data64Nodes_2_noiseft_rand_boundary_20260401`
  - warm start 成功加载 clean `modelo3`
  - `test_macro_f1=0.8750`
  - `best_thresholds=[0.05, 0.17, 0.37]`
- 同 tag 的 `20dB` noisy 分类评估：
  - `test_macro_f1=0.7780`
- 同 tag 的 noisy 联合结果：
  - `CMEI=82.56`
  - `num_accuracy=0.7360`
  - `macro_f1=0.7780`
  - `id_recall=0.7579`
  - `mse_all_edges=154.4499`

### 本地 outputs 下载校验
- `gnn/GNN_NOISE/CLS_modelo3_ft/outputs/training_data64Nodes_2_noiseft_rand_boundary_20260401`
  - 本地已正确包含 `model_last.pt / metrics.json / noise_eval.json / standardization.npz`
- `gnn/GNN_NOISE/REG_o4a2_ft/outputs/training_data64Nodes_2_noiseft_rand_boundary_20260401`
  - 本地已有 `model_last.pt / metrics.json`
  - 但当前未看到 `noise_eval.json`
- `gnn/outputs`
  - 当前本地仅看到早期 `gnn_cmei / gnn_cmei_noise20db / gnn_cmei_o4a2_smoke / gnn_cmei_o5b_eval`
  - 未看到这两轮带噪训练对应的联合输出目录，例如：
    - `gnn_cmei_noiseft_rand_boundary_20db_20260401`
    - `gnn_cmei_noiseft_fixed20db_all_clean_20260401`
    - `gnn_cmei_noiseft_fixed20db_all_20db_20260401`
- 因此当前判断是：
  - 本地下载回来的单模型 outputs 基本正确
  - 但联合 `gnn/outputs` 仍不完整，后续若要本地留档，建议把以上 joint 输出目录再同步一次

## 34. 2026-04-01 `0401补充训练` 文件删除与鲁棒性曲线脚本
- 因为 `0401补充训练.txt` 的内容已经正式吸收到根目录与 `gnn/GNN_NOISE` 日志，现已删除该文件。
- 新增鲁棒性曲线脚本：
  - `gnn/GNN_NOISE/plot_noise_robustness.py`
- 脚本用途：
  - 读取多组归档后的 `json` 指标文件
  - 在同一张图上绘制不同方法在不同 `SNR(dB)` 下的鲁棒性曲线
  - 默认输出 `svg`，不依赖本仓库额外保存中间数据

## 35. 2026-04-01 `modelo3` 两阶段细阈值搜索脚本
- 已在 `gnn/GNN_CLS/modelo3` 新增独立脚本：
  - `two_stage_threshold_search.py`
- 设计原则：
  - 不修改原 `train.py / inference.py`
  - 先做原口径 `0.01` 粗搜索
  - 再围绕粗最优阈值做局部细搜索（默认 `radius=0.03, step=0.002`）
  - 可用于 `modelo3` 本体，也可用于兼容的 `GNN_NOISE/CLS_modelo3_ft`
- 已对当前正式主线 `rand_boundary` 做一次本地校准验证：
  - 原阈值：`[0.05, 0.17, 0.37]`
  - 细化后阈值：`[0.05, 0.164, 0.368]`
  - `val_macro_f1`：`0.8976 -> 0.8976`
  - `test_macro_f1`：`0.8750 -> 0.8749`
- 当前判断：
  - 现有 `0.01` 阈值粗搜已经非常接近局部最优
  - 眼下主瓶颈并不在阈值离散度，而更可能在 noisy 数据分布本身与鲁棒特征学习

## 36. 2026-04-02 `0402补充日志` 吸收
- 已读取根目录 `0402补充日志.txt`，确认 `rand_boundary / fixed20db_all` 的 `20dB` 与 clean 联合推理已补齐。

### `rand_boundary`
- `REG` 20dB noisy 单模型：
  - `mae_all=1.2692`
  - `mae_changed=54.1729`
  - `count_macro_f1=0.5844`
- clean joint：
  - `CMEI=91.01`
  - `num_accuracy=0.8460`
  - `macro_f1=0.8750`
  - `id_recall=0.8848`
  - `mse_all_edges=66.2160`
- `20dB` joint：
  - `CMEI=82.56`
  - `num_accuracy=0.7360`
  - `macro_f1=0.7780`
  - `id_recall=0.7579`
  - `mse_all_edges=154.4499`

### `fixed20db_all`
- clean joint：
  - `CMEI=83.39`
  - `num_accuracy=0.7110`
  - `macro_f1=0.7275`
  - `id_recall=0.8048`
  - `mse_all_edges=124.9065`
- `20dB` joint：
  - `CMEI=81.79`
  - `num_accuracy=0.6510`
  - `macro_f1=0.7121`
  - `id_recall=0.7947`
  - `mse_all_edges=146.5452`

### 当前结论
- `rand_boundary` 仍然是正式主线：
  - clean 远强于 `fixed20db_all`
  - `20dB` joint 也略高于 `fixed20db_all`
- 因此后续真正还值得继续跑的，不再是补 20dB，而是扩展到 `30/40dB` 做鲁棒性曲线

## 37. 2026-04-02 `rand_boundary` 大范围噪声鲁棒性曲线
- 已读取根目录 `0402大范围噪声训练.txt`，并确认 `rand_boundary` 的 `40dB / 30dB / 20dB / clean` joint 结果已经齐全。
- 已生成最终鲁棒性曲线图：
  - `gnn/GNN_NOISE/rand_boundary_robustness_curve.svg`
- 对应 Python 脚本：
  - `gnn/GNN_NOISE/plot_rand_boundary_robustness.py`

### `rand_boundary` joint 结果
- clean：
  - `CMEI=91.01`
  - `num_accuracy=0.8460`
  - `macro_f1=0.8750`
  - `id_recall=0.8848`
  - `mse_all_edges=66.2160`
- `40dB`：
  - `CMEI=90.83`
  - `num_accuracy=0.8420`
  - `macro_f1=0.8721`
  - `id_recall=0.8832`
  - `mse_all_edges=69.5486`
- `30dB`：
  - `CMEI=89.62`
  - `num_accuracy=0.8310`
  - `macro_f1=0.8636`
  - `id_recall=0.8613`
  - `mse_all_edges=85.7003`
- `20dB`：
  - `CMEI=82.56`
  - `num_accuracy=0.7360`
  - `macro_f1=0.7780`
  - `id_recall=0.7579`
  - `mse_all_edges=154.4499`

### 最终比较结论
- 当前正式主线应明确固定为 `rand_boundary`
- 从 clean 到 `40dB / 30dB`，曲线下降非常平缓，说明带噪训练已经在中等噪声区间建立起稳定鲁棒性
- 到 `20dB` 时性能出现明显下降，但依然远好于最初 zero-shot 噪声崩塌结果
- 与早期 zero-shot `20dB` baseline 对比：
  - `CMEI: 41.79 -> 82.56`
  - `macro_f1: 0.1203 -> 0.7780`
  - `num_accuracy: 0.3170 -> 0.7360`
  - `id_recall: 0.2608 -> 0.7579`
  - `mse_all_edges: 1735.1177 -> 154.4499`
- 与 clean-only 历史最佳主线对比：
  - clean `CMEI` 从 `93.53` 降到 `91.01`
  - 代价约 `2.52` 分
  - 但换来了 `20dB` 条件下超过 `40` 分的 `CMEI` 提升
- 这说明当前最优工程策略不是继续抠 clean 极限，而是接受少量 clean 损失，换取真实噪声环境下的可用性

### 文件清理
- `0402补充日志.txt` 与 `0402大范围噪声训练.txt` 的内容已经正式吸收，现已删除。

### 目录收敛与联合推理迁移（2026-04-02）
- 当前只继续推进 `gnn` 主线。
- 为减少路径分散，已新建：
  - `gnn/GNN_CMEI_INFERENCE`
- 已迁移：
  - `gnn/inference_gnn_cmei.py -> gnn/GNN_CMEI_INFERENCE/inference_gnn_cmei.py`
  - `gnn/outputs -> gnn/GNN_CMEI_INFERENCE/outputs`
- 同时保留了兼容入口：
  - `gnn/inference_gnn_cmei.py`
  - 作用仅为转发到新目录，避免旧命令失效。
- 因此后续文件同步口径收敛为两块：
  - `gnn/GNN_NOISE/**/outputs`
  - `gnn/GNN_CMEI_INFERENCE/outputs`
- 下一版带噪训练如继续迭代，建议统一新建在：
  - `gnn/GNN_NOISE/CLS_modelo3_ft_v2`
  - `gnn/GNN_NOISE/REG_o4a2_ft_v2`
  并保持各自 `cache/` 与 `outputs/` 都留在本目录内部。
- 已读取并吸收根目录 `GNN_联合优化.txt`，其中真正适合直接做成推理层改动的内容只有：
  - `near-miss` 高置信保护
  - `REG` 证据驱动的动态 `K`
- 其余建议如 `Absolute Edge Embedding / Relaxed Sparsity Loss / Focal-CORAL / Pseudo-Edge Pooling` 属于训练或结构层，不纳入 `CMEI inference v2`。
- 已新建：
  - `gnn/GNN_CMEI_INFERENCE/inference_gnn_cmei_v2.py`
- 根目录 `GNN_联合优化.txt` 已删除。
- 已本地实跑 `v2` 在当前正式主线 `rand_boundary` 上的结果：
  - `v1 clean CMEI=91.01`
  - `v2(guard_only) clean CMEI=90.85`
  - `v2(full arbitration) clean CMEI=90.08`
  - `v1 20dB CMEI=82.56`
  - `v2(guard_only) 20dB CMEI=82.40`
  - `v2(full arbitration) 20dB CMEI=79.40`
- 结论：
  - `GNN_联合优化.txt` 中的推理层想法具备实验价值
  - 但在当前 `rand_boundary` 主线上并未超过现有 `v1`
- 因此 `inference_gnn_cmei_v2.py` 暂不转正，当前正式联合推理入口仍保持 `v1`

### `GNN_NOISE v2`（2026-04-02）
- 已新建：
  - `gnn/GNN_NOISE/CLS_modelo3_ft_v2`
  - `gnn/GNN_NOISE/REG_o4a2_ft_v2`
- 这一版仍然遵循：
  - 单模型 outputs 留在 `GNN_NOISE`
  - joint outputs 留在 `GNN_CMEI_INFERENCE`
- `v2` 不是结构换代，而是噪声建模升级：
  - `curriculum` 噪声强度采样
  - `structured boundary noise`
  - 少量 clean mixing

## 2026-04-02 GNN_EXPAND 拓扑与规模扩展容器
- 已正式新建：
  - `gnn/GNN_EXPAND`
- 目标：
  - 在不修改原有主线 `GNN_CLS / GNN_REG / GNN_CMEI_INFERENCE` 程序的前提下
  - 把当前 clean 最佳方法扩展到不同拓扑和节点规模
- 当前四阶段：
  - `stage1_square_10x10`
  - `stage2_rect_6x10`
  - `stage3_honeycomb_63`
  - `stage4_transfer_circlecut_69`
- 每个阶段均已建立：
  - `cls`
  - `reg`
  - `joint_inference`
- 关键实现口径：
  - 继续使用原始无噪声主数据 `training_data64Nodes_2.csv`
  - 28 个边界电压通道按顺时针电极顺序映射到目标拓扑边界节点
  - 原始 `8x8 / 112` 电阻标签按几何位置重映射到目标拓扑电阻边
  - 因此 `6x10 / 蜂窝 / 角点裁切` 阶段也能直接承接 clean 数据训练
- 默认 warm start：
  - `stage1/2/3` 继续对齐当前主线最优 `modelo3 + o4a2`
  - `stage4` 默认尝试承接 `stage1_square_10x10` 权重，体现 transfer 设定
- 文档入口：
  - `gnn/GNN_EXPAND/README.md`
  - `gnn/GNN_EXPAND/Log.md`
- `模型扩展路径.txt` 的内容已吸收入正式文档，完成校验后删除

### GNN_EXPAND 原生数据
- 已新增通用拓扑数据生成器：
  - `gnn/GNN_EXPAND/generate_expand_datasets.py`
- 已在 `gnn/GNN_EXPAND/data` 下生成四套原生 clean 数据：
  - `square_10x10.csv`
  - `rect_6x10.csv`
  - `honeycomb_63.csv`
  - `circlecut_69.csv`
- 每套都附带对应 `*_meta.json`
- 当前已确认：
  - 激励只使用外部节点
  - 测量只输出外部节点电压
  - 每套数据均固定为 `28` 个外部节点、`32` 组边界激励

## 2026-04-02 GNN_EXPAND 数据口径更正
- 先前记录里把 `GNN_EXPAND` 的四套原生数据统一写成 `28` 个外部节点、`32` 组激励，这个说法不对。
- `GNN_EXPAND` 与原始 `8x8` 主线不同，数据生成时会按目标拓扑自己的外边界节点数构造激励与测量。
- 统一约束保持不变：
  - 激励只使用外部节点
  - 测量只输出外部节点电压
- 当前四套原生数据真实规模为：
  - `square_10x10`: `36` 个外部节点，`40` 组激励
  - `rect_6x10`: `28` 个外部节点，`32` 组激励
  - `honeycomb_63`: `28` 个外部节点，`32` 组激励
  - `circlecut_69`: `24` 个外部节点，`28` 组激励
- 其中 `square_10x10` 的 `40` 组激励来自：
  - `36` 组顺时针相邻边界节点激励
  - `4` 组额外跨边界激励

## 2026-04-02 DOC_RULES 升级
- 根目录 `DOC_RULES.md` 已从“文档治理规则”升级为项目长期接手文件。
- 新定位同时包含：
  - 文档与日志管理规则
  - 必须遵守的项目原则
  - 当前正式主线、当前最佳路线、当前重点工作
- 后续新对话窗口默认应先读：
  - `DOC_RULES.md`
  - 再读根目录与 `gnn` 对应 `README.md / Log.md`
- 本次明确固化的关键原则包括：
  - 数据生成时激励只使用外部节点
  - 数据生成时测量只使用外部节点
  - 当前所有主线推进默认在 `gnn` 内完成
  - `GNN_NOISE` 只负责带噪训练与单模型推理
  - `GNN_CMEI_INFERENCE` 统一负责联合推理与 joint 输出
  - `GNN_EXPAND` 内的扩展工作不得回写原 clean 主线程序

## 2026-04-02 RULES 改名与规则补充
- 根目录长期规则主文件已由 `DOC_RULES.md` 更名为：
  - `RULES.md`
- 为避免旧日志和旧说明中的引用失效，当前保留了一个轻量兼容入口：
  - `DOC_RULES.md`
  - 但后续实际维护与新窗口接手都以 `RULES.md` 为准
- 本次新增两条长期执行规则：
  - 修改代码前，先向用户呈现修改思路，再开始动手
  - 一切模型更新优先在复制后的新版本目录上进行，不直接改旧模型
- 同时再次明确：
  - `README.md / Log.md` 原则上不删除旧内容
  - 文档更新以追加记录和更正说明为主

## 2026-04-02 Git 轻量方案 A
- 项目根目录已增加 Git 管理能力，用于保护当前代码、文档、规则文件和最佳路线说明。
- 当前采用轻量方案 A：
  - 跟踪源码、脚本、文档、规则、元数据
  - 不直接跟踪 `outputs / cache / 权重 / 可重建 csv`
- 已新增：
  - `RULES.md`
    - 作为长期规则与当前主线入口
  - `CURRENT_BEST.md`
    - 作为当前最佳路线与本地产物锚点清单
  - `.gitignore`
    - 统一排除训练产物、缓存、权重、本地依赖和可重建数据
- 这套方案的目标是：
  - 保持仓库轻量
  - 保护当前正式路线与工程结构
  - 让后续窗口能快速恢复当前最佳状态
