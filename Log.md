# 64Nodes 项目日志（全局简版）

说明：
- 本文件仅记录全局信息与版本迭代摘要。
- 各方法的详细架构、参数、问题分析，请查看对应子目录 README/Log。
- 更新规则：只新增版本记录，不覆盖历史版本。

## 全局演进索引（便于汇报）
- 数据基线版本：`v0`
- MLP 路线：
- 起始版本：`v1`（`MLP_CLS/modelo1`、`MLP_REG/modelo1`、`MLP_FULL/modelo1_reg2prob`）
- 当前迭代：`v2`（o2，含 `modelv1_h_multitask`）
- 详情：`64Nodes/mlp/README.md`、`64Nodes/mlp/Log.md`
- CNN2D_MLP 路线：
- 起始版本：`v3`（`CNN2D_CLS/modelo1`、`CNN2D_REG/modelo1`、`CNN2D_FULL/modelo1_h_multitask`）
- 当前迭代：`v7`（输入增强 + 残差主干 + 分阶段训练）
- 详情：`64Nodes/cnn2d_mlp/README.md`、`64Nodes/cnn2d_mlp/Log.md`
- Attention 路线：
- 起始版本：`v8`（`ATTN_CLS/REG/FULL` 骨架）
- 详情：`64Nodes/attention/README.md`、`64Nodes/attention/Log.md`
- CNN 路线：
- 起始版本：`v8`（`CNN_CLS/REG/FULL` 骨架）
- 详情：`64Nodes/cnn/README.md`、`64Nodes/cnn/Log.md`
- U-Net_MLP 路线：
- 起始版本：`v8`（`UNET_MLP_CLS/REG/FULL` 骨架）
- 详情：`64Nodes/unet_mlp/README.md`、`64Nodes/unet_mlp/Log.md`
- GNN 路线：
- 起始版本：`v8`（`GNN_CLS/REG/FULL` 骨架）
- 详情：`64Nodes/gnn/README.md`、`64Nodes/gnn/Log.md`
- 文档规则固化：`v4`（`DOC_RULES.md`）

## 2026-03-19 v0
- 完成 8x8（64节点）数据生成脚本：
- `scripts/generate_training_data64.py`
- 采用严格基尔霍夫线性方程直接求解，不使用低秩加速。
- 生成正式数据：
- `data/training_data64.csv`（10000 组合，320000 行）
- `data/training_data64_meta.json`
- 建立方法目录：
- `mlp`、`cnn2d_mlp`、`cnn`、`unet_mlp`、`attention`、`gnn`

## 2026-03-19 v1（MLP 三子项目）
- 完成 `mlp/MLP_CLS/modelo1`（分类）
- 完成 `mlp/MLP_REG/modelo1`（回归）
- 完成 `mlp/MLP_FULL/modelo1_reg2prob`（完整任务）
- 补充 `mlp/README.md`：
- 统一数据与训练规范
- 三个版本的模型架构与训练思路
- `h` 网络（多任务分叉）后续版本规划

## 2026-03-20 v2（MLP o2 更新）
- 分类 `mlp/MLP_CLS/modelo2`：
- 增加“带约束阈值搜索”，强制 `t1 <= t2 <= t3`。
- 增加 `2<->3` 轻惩罚项（`lambda_adj`，默认 0.15）。
- 回归 `mlp/MLP_REG/modelo2`：
- 参数起点更新：`w_change=2.0`，`w_unchange=1.4`。
- 新增分段稀疏调度：前 20 epoch 弱稀疏，后续逐步增强到目标值。
- 数量阈值由验证集搜索（`40~70, step=1`），不再固定 50。
- 完整任务：
- 新增 `mlp/MLP_FULL/modelo2_reg2prob`（加入 hinge、提高未变化约束和计数损失权重）。
- 新增 `mlp/MLP_FULL/modelv1_h_multitask`（共享主干 + 分类头 + 回归头）。
- 文档同步更新：
- `mlp/README.md`
- `mlp/Log.md`

## 2026-03-20 v3（CNN2D_MLP 工程搭建）
- 新增目录与版本：
- `cnn2d_mlp/CNN2D_CLS/modelo1`
- `cnn2d_mlp/CNN2D_REG/modelo1`
- `cnn2d_mlp/CNN2D_FULL/modelo1_h_multitask`
- 输入方案落地：
- 32 激励通道映射到 `8x8` 边界，内部置 0
- 附加 1 个边界掩码通道
- 关键机制：
- 分类：CORAL + 约束阈值搜索
- 回归：分段稀疏调度 + 验证集阈值搜索
- 完整：共享主干双头（分类头 + 回归头）
- 文档同步更新：
- `cnn2d_mlp/README.md`
- `cnn2d_mlp/Log.md`

## 2026-03-20 v4（文档治理规则固化）
- 新增文档规则文件：`DOC_RULES.md`
- 明确根目录与子项目目录的职责边界：
- 根目录保留全局摘要与导航
- 子项目保留详细架构、实验日志、问题分析与改进建议
- 目标：最大化支撑对外汇报与论文材料整理，减少重复维护成本。

## 2026-03-20 v5（MLP o2 首轮实测）
- 分类：
- `MLP_CLS/modelo2` 验证 `macro-F1=0.8083`，0/1 类稳定，2/3 仍混淆。
- 回归：
- `MLP_REG/modelo2` 达到 `mae_all=14.0631`、`mae_changed=44.2595`，稀疏性优于上一版。
- 完整任务：
- `MLP_FULL/modelo2_reg2prob` 数量预测塌缩（几乎全判高类别）。
- `MLP_FULL/modelv1_h_multitask` 明显优于 reg2prob，当前作为完整任务主分支。
- 详细指标与分析：`64Nodes/mlp/Log.md`

## 2026-03-20 v6（CNN2D_MLP 首轮实测）
- `CNN2D_CLS/modelo1`：
- 最高验证 `macro-F1` 约 `0.7852`，测试集中 `2/3` 类混淆较明显。
- `CNN2D_REG/modelo1`：
- `mae_all=18.7488`、`mae_changed=53.4642`，低于 MLP 回归主线。
- `CNN2D_FULL/modelo1_h_multitask`：
- `mae_all=17.1720`、`mae_changed=75.6614`，完整任务效果仍弱于 MLP_FULL 主线。
- 阶段结论：
- 现有 CNN2D 输入表示（边界有值、内部置零）尚未发挥优势，进入 v2 输入增强与训练策略优化阶段。
- 详细指标与分析：`64Nodes/cnn2d_mlp/Log.md`

## 2026-03-20 v7（CNN2D_MLP v2 改版完成）
- 三条改动全部落地到 CLS/REG/FULL：
- 输入增强：加入 `src_map/gnd_map`，输入由 33 通道升级为 97 通道。
- 主干升级：改为保分辨率残差卷积主干，减少池化信息损失。
- 训练策略升级：
- CLS：early stopping + 可选 `fixed_t2` 阈值稳定；
- REG：early stopping；
- FULL：两阶段训练（先分类后联合）+ 早停。
- 代码改版已完成，等待新一轮训练对比。
- 详细实现与参数：`64Nodes/cnn2d_mlp/README.md`、`64Nodes/cnn2d_mlp/Log.md`

## 2026-03-20 v8（Attention/CNN/GNN/U-Net_MLP 工程骨架）
- 按统一格式新增 4 条方法线的基准骨架（每条均含 CLS/REG/FULL）：
- `attention/ATTN_CLS/modelo1`、`attention/ATTN_REG/modelo1`、`attention/ATTN_FULL/modelo1_h_multitask`
- `cnn/CNN_CLS/modelo1`、`cnn/CNN_REG/modelo1`、`cnn/CNN_FULL/modelo1_h_multitask`
- `gnn/GNN_CLS/modelo1`、`gnn/GNN_REG/modelo1`、`gnn/GNN_FULL/modelo1_h_multitask`
- `unet_mlp/UNET_MLP_CLS/modelo1`、`unet_mlp/UNET_MLP_REG/modelo1`、`unet_mlp/UNET_MLP_FULL/modelo1_h_multitask`
- 每条方法线均补充：
- 子项目 `README.md`（架构与参数）
- 子项目 `Log.md`（实验记录模板与初始化记录）
- 当前状态：等待首轮 baseline 运行结果。

## 2026-03-20 v9（README 补充“适配性与优势”说明）
- 为各子项目 README 统一新增开头段落：
- `mlp/README.md`
- `cnn2d_mlp/README.md`
- `attention/README.md`
- `cnn/README.md`
- `gnn/README.md`
- `unet_mlp/README.md`
- 补充内容：为什么该架构可能适合本课题、优势与后续扩展价值。

## 2026-03-20 v10（可尝试方法清单，待后续验证）
- 新增可尝试方向（先记录，不立即改代码）：
- `DeepSets / Set Transformer`：按激励集合建模，弱化激励顺序伪相关。
- `Graph U-Net / Hierarchical GNN`：在图拓扑上做多尺度聚合，兼顾全局与局部异常。
- `Edge-centric / Line-Graph GNN`：直接在电阻边空间建模，减少“节点到边”解码损失。
- `MoE（专家混合）`：按变化数量/幅值分配专家，缓解单模型折中。
- `Physics-informed loss（非PINN）`：KCL 残差、边界一致性、互易性约束，结合 `tanh` 范围先验。
- 说明：
- 当前先完成各主线 baseline 运行；后续按结果优先级逐项落地。

## 2026-03-20 v11（MLP o3 / v2 代码落地）
- 新增并完成实现（待跑实验）：
- `mlp/MLP_CLS/modelo3`
- `mlp/MLP_REG/modelo3`
- `mlp/MLP_FULL/modelv2_h_multitask`
- 设计目标：
- CLS：强化 `2/3` 细分（主 CORAL + 2v3 辅头）
- REG：在 `mae_changed` 与稀疏性之间更稳平衡（分段调度 + count consistency + hard-negative）
- FULL：两阶段 + 分类梯度隔离，缓解“分类头局部最优拖累回归”
- 详细结构与参数：`64Nodes/mlp/README.md`
- 详细改动记录：`64Nodes/mlp/Log.md`

## 2026-03-20 v12（MLP_CLS/modelo3 首轮实测）
- 用户已回传 `MLP_CLS/modelo3` 结果：
- 最佳验证 `val_macro_f1=0.8024`
- 测试集仍有明显 `2/3` 混淆（双向均较高）
- 阶段结论：
- o3 未突破长期约 0.80 的分类平台
- 下轮建议转入“2/3 解耦训练”路线（主模型与 2v3 细分器分阶段训练）
- 详细分析与下一版建议：`64Nodes/mlp/Log.md`

## 2026-03-20 v13（MLP_CLS/modelo4 已实现）
- 按用户同意方案落地 `mlp/MLP_CLS/modelo4`：
- Stage1 主 CORAL 训练 + Stage2 独立 2v3 细分器训练（冻结主干）
- 阈值搜索加入“3->2 惩罚 + class3 召回加权”目标
- 目标是在不破坏 0/1 的前提下继续压缩 `2/3` 混淆
- 详细实现与后续测试记录入口：`64Nodes/mlp/README.md`、`64Nodes/mlp/Log.md`

## 2026-03-20 v14（MLP_REG/modelo3 首轮实测）
- 用户已回传 `mlp/MLP_REG/modelo3`：
- `mae_all=13.2610`、`mae_changed=42.2229`（较上一版继续改善）
- 稀疏性指标继续改善：`avg(|dR|>63)=2.77`
- 主要瓶颈转为计数边界：`true=2` 大量被判为 `3`
- 阶段结论：
- REG 主线当前版本有效，但下一轮应重点做“2 vs 3 计数边界”优化，而非盲目加深网络
- 详细分析与 `modelo4` 建议：`64Nodes/mlp/Log.md`

## 2026-03-20 v15（MLP_REG/modelo4 已实现）
- 按“重建优先”原则新增 `mlp/MLP_REG/modelo4`：
- `lambda_count` 调整为 `0.01`，降低计数分支对回归主任务干扰
- 推理脚本升级为“预测电阻 vs 真实电阻”直接对照输出（JSON）
- 用途：强化重建结果解释性，便于对比误差分布与样例分析
- 详细改动说明：`64Nodes/mlp/README.md`、`64Nodes/mlp/Log.md`

## 2026-03-20 v16（MLP_FULL/modelv2_h_multitask 首轮实测）
- 用户已回传 `mlp/MLP_FULL/modelv2_h_multitask`：
- 回归指标：`mae_all=13.4150`、`mae_changed=42.3895`、`avg(|dR|>50)=4.85`
- 计数指标：数量头 与 Reg-threshold 两路均在 `2/3` 边界存在明显偏差
- 阶段结论：
- FULL 版本在“重建主任务”上表现可接受，接近 REG 主线
- 计数分支暂不适合主导训练，下一轮应继续执行“重建优先”策略
- 详细分析与后续建议：`64Nodes/mlp/Log.md`

## 2026-03-20 v17（MLP_FULL/modelv3_h_multitask 已实现）
- 新增 `mlp/MLP_FULL/modelv3_h_multitask`（保守改版）：
- Stage2 分类权重与计数权重做“小幅下调”，避免激进改动导致不可比
- `inference.py` 升级为重建对比输出：
- 先输出预测变化个数与预测 ID
- 再输出全电阻预测/真实变化与电阻值对照（JSON）
- 目标：优先观察“重建指标是否继续改善”的趋势
- 详细改动说明：`64Nodes/mlp/README.md`、`64Nodes/mlp/Log.md`

## 2026-03-21 v18（MLP 最新三线实测汇总）
- 用户回传最新结果：
- `MLP_CLS/modelo4`：Stage2 未超过 Stage1 峰值，测试 `2/3` 混淆仍显著
- `MLP_REG/modelo4`：重建指标继续小幅提升（`mae_changed` 更优），当前 REG 主线最稳
- `MLP_FULL/modelv3_h_multitask`：重建指标接近 REG 主线，计数分支仍为短板
- 阶段结论：
- 64Nodes 当前瓶颈更偏“输出模式与损失耦合”，而非骨干架构本身
- 后续以“重建优先（MAE-first）+ 计数诊断化”继续推进
- 详细指标与分析：`64Nodes/mlp/Log.md`

## 2026-03-21 v19（MLP o5/v4 代码落地）
- 按用户指定完成三条线改版：
- `mlp/MLP_CLS/modelo5`
- `mlp/MLP_REG/modelo5`
- `mlp/MLP_FULL/modelv4_h_multitask`
- 关键改动摘要：
- CLS：两阶段回退机制 + Aux 难例采样 + `label_smoothing=0.03`
- REG：新增 `L_order` + 重建导向早停 + 2类加权阈值搜索 + inference 对照增强
- FULL：从 REG warm start + Stage2 权重保守下调 + 新增头间一致性 `L_align`
- 详细实现说明与后续实验入口：`64Nodes/mlp/README.md`、`64Nodes/mlp/Log.md`

## 2026-03-21 v20（Change3 微项目）
- 新增微项目：`64Nodes/mlp/change3_recon`
- 目的：在固定变化数（恰好 3 个电阻变化）设定下，验证位置与数值重构能否进一步提升。
- 数据脚本：`64Nodes/mlp/change3_recon/scripts/generate_data_change3.py`
  - 8x8、64 节点、112 电阻、32 组激励
  - 每个组合固定 3 个变化电阻
  - 共 5000 个不重复组合
- 新增两条模型线：
  - `modelv1`：重构基线 + 稀疏/top3 排序 + 可选后期基尔霍夫损失（随机 4/8 组激励）
  - `modelv2_coord`：在 `modelv1` 上增加基于 `id -> (x,y)` 的坐标矩损失
- 坐标映射输出：
  - `64Nodes/mlp/change3_recon/data_change3/resistor_coords_bl_origin.json`
- change3 数据已生成：
  - `64Nodes/mlp/change3_recon/data_change3/training_data64_change3.csv`
  - `64Nodes/mlp/change3_recon/data_change3/training_data64_change3_meta.json`
  - `64Nodes/mlp/change3_recon/data_change3/resistor_coords_bl_origin.json`

## 2026-03-21 v21（Change3 增强模型落地）
- 新增 change3 模型版本：`64Nodes/mlp/change3_recon/modelv3_sepcount`
- 核心思路：保留坐标损失，并加入固定计数=3 的软约束 + hardest 正负分离损失。
- 验证评分显式纳入 top3 id 命中率，使模型选择更贴合重构目标。
- 详情与命令：`64Nodes/mlp/change3_recon/README.md`、`64Nodes/mlp/change3_recon/Log.md`

## 2026-03-21 v22（MLP o5/v4 与 change3 首轮结果）
- 用户回传以下结果：
  - `mlp/MLP_CLS/modelo5`
  - `mlp/MLP_REG/modelo5`
  - `mlp/MLP_FULL/modelv4_h_multitask`
  - `mlp/change3_recon/modelv1`
  - `mlp/change3_recon/modelv2_coord`
- 结果明细已追加至：
  - `64Nodes/mlp/Log.md`
  - `64Nodes/mlp/change3_recon/Log.md`
- 对 `change3_recon` 的后续重点：
  - 去除分类导向的输出与指标
  - 将日志/指标/损失/inference 全部切换为重构优先口径

## 2026-03-22 v23（项目精简为 MLP + GNN）
- 根目录保留活跃方法线：
  - `64Nodes/mlp`
  - `64Nodes/gnn`
- 历史方法线归档至：
  - `64Nodes/history/attention`
  - `64Nodes/history/cnn`
  - `64Nodes/history/cnn2d_mlp`
  - `64Nodes/history/unet_mlp`
- 根 README 导航已同步更新。

## 2026-03-22 v24（change3 版本重命名）
- `change3_recon` 版本命名重排：
  - `modelv2_coord` -> `modelv1_coord`
  - `modelv3_sepcount` -> `modelv2_coord`
  - 新增 `modelv2`（同一增强损失族，默认无坐标约束）
- 已同步更新相关 README/Log：
  - `64Nodes/mlp/change3_recon/README.md`
  - `64Nodes/mlp/change3_recon/Log.md`
  - `64Nodes/mlp/README.md`
  - `64Nodes/mlp/Log.md`

## 2026-03-22 v25（change3 modelv2 实测回传）
- 用户回传 `modelv2` 测试结果（固定3变化）：
  - `mae_all=17.6353`
  - `mae_changed=67.9094`
  - `avg(|dR|>50)=5.67`
  - `top3_id_precision=0.6127`
- 同步完成本地文件核验：
  - `modelv2` 与 `modelv2_coord` 非同文件（`lambda_coord` 与缓存名不同）。
- 详情见：`64Nodes/mlp/change3_recon/Log.md`

## 2026-03-22 v26（change3 modelv2_coord 实测回传）
- 用户回传 `modelv2_coord` 测试结果（固定3变化）：
  - `mae_all=17.3997`
  - `mae_changed=66.5398`
  - `avg(|dR|>50)=5.59`
  - `top3_id_precision=0.6340`
- 相比同轮 `modelv2`，四项指标均改善，但假阳性扩散仍存在。
- 详情见：`64Nodes/mlp/change3_recon/Log.md`

## 2026-03-22 v27（change3 modelv1_new 架构与损失重构）
- 新增 `change3_recon/modelv1_new`：
  - 架构：`896-1024-896-512-256`（含 896 残差），输出 `112`，`tanh*310`
  - 指标：`mae_all`、`mae_changed`、位置准确率（对0/1/2/3个）
  - inference 字段精简：`pred_id`、`true_id`、`pred_id_delta`、`true_id_delta`、`pred_delta_all`
  - 损失优先级：`MSE > ID(坐标) > Physics(每批随机4激励) > Sparse`
- 详情见：`64Nodes/mlp/change3_recon/README.md` 与 `64Nodes/mlp/change3_recon/Log.md`

## 2026-03-22 v28（change3 modelv1_new 首轮实测）
- 用户回传 `change3_recon/modelv1_new` 首轮结果：
  - `mae_all=5.8611`
  - `mae_changed=70.4115`
  - `位置准确率(对0/1/2/3个)=0.0040/0.1920/0.5140/0.2900`
- 阶段结论：
  - 新模型在全局误差与位置命中上有明显提升；
  - 但真实变化位幅值重构仍弱于 `modelv2_coord`，下一轮应继续做“重构优先”损失再平衡，而不是回到数量预测主导。

## 2026-03-22 v29（change3 modelv2_new 代码落地）
- 新增 `change3_recon/modelv2_new`：
  - 基于 `modelv1_new` 保留残差 MLP 架构
  - 主损失改为变化位加权回归
  - 稀疏约束聚焦未变化位，并新增第4假阳性分离约束
  - 物理约束延后启用
  - 诊断指标补回 `avg(|dR|>50)`
- 目标：
  - 在保住位置命中的同时，重点改善 `mae_changed` 与假阳性扩散。

## 2026-03-22 v30（change3 modelv2_new 首轮实测）
- 用户回传 `change3_recon/modelv2_new` 首轮结果：
  - `mae_all=18.2218`
  - `mae_changed=56.6768`
  - `avg(|dR|>50)=8.2640`
  - `位置准确率(对0/1/2/3个)=0.0040/0.2740/0.5380/0.1840`
- 阶段结论：
  - 变化位加权回归有效，`mae_changed` 明显优于此前各版本；
  - 但假阳性扩散显著加重，位置“全对3个”的比例也低于 `modelv1_new`；
  - 下一轮应继续围绕“保住低 `mae_changed`，同时压低假阳性”调损失与模型选择策略。

## 2026-03-22 v31（change3 modelv3_new 代码落地）
- 新增 `change3_recon/modelv3_new`：
  - 保持 `modelv2_new` 的残差 MLP 架构
  - loss 简化为 `L_reg + L_id + L_phys + L_fp4`
  - 删除重叠较多的未变化位稀疏/hinge/分离项
  - 选模评分更强调 `avg(|dR|>50)` 超标与“3个位置全对”
- 目标：
  - 通过“减法”验证更少的 loss 项是否更利于稳定学习与重构平衡。

## 2026-03-23 v32（主线训练脚本支持 5mA/10mA 数据切换）
- 已生成 `10mA` 主数据：
  - `64Nodes/data/training_data64Nodes_2.csv`
  - `64Nodes/data/training_data64Nodes_2_meta.json`
- 当前主线 MLP / GNN 训练脚本已支持通过 `--data-path` 切换主数据
- 默认 cache 与 outputs 会按数据集标签自动拆分子目录，便于云端并行跑 `5mA` / `10mA`
- 详情见：
  - `64Nodes/mlp/README.md`
  - `64Nodes/mlp/Log.md`
  - `64Nodes/gnn/README.md`
  - `64Nodes/gnn/Log.md`

## 2026-03-23 v33（主线 inference 脚本支持 dataset-tag 自动寻址）
- 当前主线 MLP / GNN 的 6 个 `inference.py` 已补齐 `dataset-tag` 机制
- 现在推理默认会自动读取对应数据集的：
- `cache/<dataset_tag>/...`
- `outputs/<dataset_tag>/model_last.pt`
- `outputs/<dataset_tag>/metrics.json`
- `outputs/<dataset_tag>/standardization.npz`
- 这样训练与推理两侧都能无缝切换 `5mA` / `10mA`

## 2026-03-24 v34（0324 三电流首轮结果回传）
- 主线 MLP / GNN 已完成 `5mA / 10mA / 20mA` 首轮对比。
- 主结论：
- inverse identifiability 中“更大电流提升可检测性”的结论依然成立；
- 但落到当前主混合数据训练指标上，收益明显被稀释，`20mA` 没有稳定优于 `5mA/10mA`。
- 当前更值得继续跟进的数据选择：
- MLP 主线：`5mA` 仍可作为稳定基线，`10mA` 仅在 `mae_changed` 上略优；
- GNN 主线：`10mA` 当前最值得优先继续。
- `change3_recon/modelv3_new` 已完成首轮正式实测：
- `mae_all=14.7789`
- `mae_changed=64.0375`
- `avg(|dR|>50)=4.3280`
- 位置准确率(对0/1/2/3个)=`0.0040/0.1860/0.5720/0.2380`
- 阶段判断：`modelv3_new` 当前更像“幅值/假阳性/定位”的平衡版本，下一步更适合在 fixed-3 场景继续比较 `5mA / 10mA / 20mA`。

## 2026-03-24 v35（训练入口修正与 change3 多激励支持）
- 修正 `gnn/GNN_FULL/modelo1_h_multitask/train.py`：
- 原因：训练脚本错误访问 `model.reg_head`，而模型实际回归头名为 `edge_mlp`。
- 影响：`GNN_FULL` 之前在 Stage1 前即报错，当前已可重新补跑。
- 修正 `mlp/MLP_FULL/modelv4_h_multitask/train.py` 默认 warm start 路径：
- 原路径少了一层 `mlp/`，导致 0324 三套 FULL 训练实际都未加载到 `MLP_REG/modelo4` 权重。
- 说明：0324 的 FULL 结果应按“无 warm start 版本”保守解读。
- `change3_recon` 已补充多激励支持：
- 数据生成脚本支持 `--current-a` + `--dataset-tag`；
- `modelv2_coord`、`modelv3_new` 训练/推理支持按 `dataset_tag` 自动拆分 cache 与 outputs；
- 训练脚本会优先从对应 meta 同步 `current_source_a`，避免数据电流与物理损失电流不一致。

## 2026-03-24 v36（主线优先级收缩与默认数据统一）
- 项目当前主推进顺序调整为：
- 第一优先级：`MLP_REG/modelo5`、`GNN_REG/modelo1`
- 第二优先级：fixed-change 重构主线（当前以 `mlp/change3_recon/modelv3_new` 为主）
- 第三优先级：`MLP_CLS/modelo5`、`GNN_CLS/modelo1`
- `FULL` 暂时降为验证支线，不再作为每轮同步推进对象。
- 非 fixed-change 主线训练/推理入口默认数据统一改为 `10mA`：
- `data/training_data64Nodes_2.csv`
- 多数据集功能保留；仍可通过 `--data-path` 和 `--dataset-tag` 切回 `5mA / 20mA`。
- 同步修正主线入口的相对路径解析：
- 现在会优先兼容绝对路径、项目根相对路径和脚本相对路径，避免“不传参数时默认路径指错目录”的问题。

## 2026-03-24 v37（五条主线下一版优化与 fixed-change 子项目重命名）
- 按当前优先级，对五个活跃子项目做了小步优化：
- `MLP_REG/modelo5`：
  - `L_order` 进一步强调真实 `2/3` 样本，新增 `order_margin=12`、`order_weight_k2=1.25`、`order_weight_k3=1.45`
  - 验证选模新增过预测惩罚：`val_overpredict_alpha=0.60`、`val_overpredict_k23_alpha=1.10`
- `GNN_REG/modelo1`：
  - 阈值搜索改为 `macroF1 + class2_F1` 加权评分，默认 `count_cls2_weight=0.35`
  - 新增 `L_fp_next` 抑制真实变化数之后的第一个伪峰，默认 `lambda_fp_next=0.12`
  - early stopping 从纯 `val_loss` 改为更重建导向的 `val_score`
- `MLP_CLS/modelo5`：
  - 阈值搜索评分新增 `bonus_r2=0.05`
  - Stage2 的 aux 难例集合显式包含真实 `2/3` 样本
- `GNN_CLS/modelo1`：
  - 阈值搜索改为 weighted score，新增 `bonus_r2=0.05`
- fixed-change 子项目：
  - 目录由 `mlp/change3_recon` 更名为 `mlp/fixed_change_recon`
  - 原专用数据重命名为 `training_data64_fixed_3.csv`
  - 新生成 `training_data64_fixed_2.csv`
- 当前主线 `modelv3_new` 已支持 `fixed_2 / fixed_3`
  - 新增通用的“第 k+1 大抑制 + 第 k/k+1 间隔”约束

## 2026-03-25 v38（权重调整后首轮回传与 fixed-change 缓存问题暴露）
- 用户已完成一轮“权重/评分调整后”的 10mA 主线训练与 fixed-change 训练。
- 主线分类结果：
- `MLP_CLS/modelo5`：最佳验证 `val_macro_f1=0.7925`
- `GNN_CLS/modelo1`：最佳验证 `val_macro_f1=0.7855`
- 结论：
- 分类线整体没有出现实质性突破，`2/3` 类双向混淆依然是主瓶颈。
- 主线回归结果：
- `MLP_REG/modelo5`（10mA）：
  - `mae_all=12.8283`
  - `mae_changed=44.0864`
  - `best_count_threshold(val)=69.0`
  - `avg(|dR|>69)=2.15`
- `GNN_REG/modelo1`（10mA）：
  - `mae_all=4.6721`
  - `mae_changed=70.9448`
  - `best_count_threshold(val)=45.0`
  - `avg(|dR|>45)=2.22`
- 结论：
- `MLP_REG` 本轮更稀疏，但 `mae_changed` 较 0324 的 `42.3156` 回退，说明新增“排序/过预测抑制”暂时压过了真实变化位幅值学习；
- `GNN_REG` 本轮出现明显退化，当前新增 `L_fp_next + class2 加权阈值 + val_score` 组合不适合作为继续主线。
- fixed-change 结果：
- `fixed_3 / modelv3_new`：
  - `mae_all=15.5451`
  - `mae_changed=62.5130`
  - `avg(|dR|>50)=5.0460`
  - 位置准确率(对0/1/2/3个)=`0.0040/0.1960/0.5680/0.2320`
- 阶段判断：
- 相比上一轮 `modelv3_new`，本轮 `mae_changed` 略有改善，但假阳性和 `pos3` 没有同步改善，因此不能判定为稳定进步。
- fixed_2 当前结果无效：
- 训练记录中先出现：
  - `FileNotFoundError: .../cache/change2_5mA/cache_change3_v3_new.npz`
- 随后的 `fixed_2` 训练日志与 `fixed_3` 全程逐 epoch 完全一致，且仍输出 `位置准确率(0..3)`；
- 因此当前 `fixed_2` 支持虽然已在代码层接入，但本轮实际运行结果不可信，不能用于模型结论。
- 当前更可能的原因：
- fixed-change 的 cache 文件名和目录隔离仍保留旧 `change3` 痕迹；
- 新 dataset tag 首次运行时，`cache/<dataset_tag>/` 没有统一保证先创建；
- 训练/推理入口缺少“cache 中 fixed_k 与当前数据/参数是否一致”的显式校验。
- 下一轮修改优先级建议：
- 第一优先级先修 `fixed_change_recon/modelv3_new` 的 cache/输出隔离与 fixed_k 一致性检查；
- 第二优先级回退或显著减弱 `GNN_REG` 本轮新增约束；
- 第三优先级保留 `MLP_REG` 的日志诊断项，但放松新增排序/过预测压力。

## 2026-03-25 v39（fixed-change 修正并冻结；REG 回退到更稳默认）
- `fixed_change_recon/modelv3_new` 已完成运行口径修正：
  - 当前只支持 `fixed_2 / fixed_3`
  - cache 默认改为 `cache_fixed_v3_new.npz`
  - cache 自动拆到 `cache/fixed_2/<dataset_tag>/...` 与 `cache/fixed_3/<dataset_tag>/...`
  - outputs 自动拆到 `outputs/fixed_2/<dataset_tag>/...` 与 `outputs/fixed_3/<dataset_tag>/...`
  - 训练/推理会校验 cache 内 `fixed_k` 与 `source_csv` 是否和当前数据一致
  - 首次运行新 tag 时会自动创建 cache 目录，并打印最终运行路径
- `fixed_change_recon` 当前状态调整为：
  - 冻结为固定 `2/3` 变化的纯回归诊断子项目
  - 不再作为当前主线持续改模对象
  - 不再承担数量判断/分类功能
- `MLP_REG/modelo5` 已回退到更稳默认：
  - 关闭本轮新增的更强排序间隔和过预测惩罚默认值
  - 目的：优先恢复 0324 那一版更好的 `mae_changed`
- `GNN_REG/modelo1` 已回退到更稳默认：
  - `lambda_fp_next=0.0`
  - `count_cls2_weight=0.0`
  - early stopping 恢复按 `val_loss`
- 当前项目主重点更新为：
  - `MLP_REG/modelo5`
  - `GNN_REG/modelo1`
  - `MLP_CLS/modelo5`
  - `GNN_CLS/modelo1`
- `fixed_change_recon` 保留，但当前只作为冻结的诊断对照线。

## 2026-03-25 v40（回退后首轮结果与 REG inference 修正）
- 用户已完成回退后的首轮训练与 fixed-change 重新训练。
- `fixed_change_recon`：
- `fixed_3 / modelv3_new` 结果与上一轮一致：
  - `mae_all=15.5451`
  - `mae_changed=62.5130`
  - `avg(|dR|>50)=5.0460`
- `fixed_2 / modelv3_new` 当前已确认运行正常，结果有效：
  - `mae_all=10.6534`
  - `mae_changed=54.1879`
  - `avg(|dR|>50)=2.3960`
  - 位置准确率(对0/1/2个)=`0.0080/0.4760/0.5160`
- 结论：
- `fixed_2` 明显比 `fixed_3` 更容易，说明当前 `2/3` 困难不只是数量判断边界问题，`fixed_3` 本身的定位/幅值恢复也更难。
- 主线回归结果：
- `MLP_REG/modelo5`（10mA，回退后）：
  - `mae_all=12.9366`
  - `mae_changed=43.2690`
  - `best_count_threshold(val)=70.0`
  - `avg(|dR|>70)=2.10`
- `GNN_REG/modelo1`（10mA，回退后）：
  - `mae_all=3.2785`
  - `mae_changed=43.0275`
  - `best_count_threshold(val)=67.0`
  - `avg(|dR|>67)=2.17`
- 阶段判断：
- `MLP_REG` 已比上一轮回升，但仍未完全回到 0324 的最好 `mae_changed=42.3156`；
- `GNN_REG` 已基本恢复到有效区间，当前比退化版明显更可信。
- 本轮额外修正：
- `MLP_REG/GNN_REG` 的 `inference.py` 已补上旧输出目录兼容：
  - 当 `outputs/<dataset_tag>/...` 不存在时，会自动尝试 `outputs/<data_path.stem>/...`
  - 若仍不存在，会继续尝试旧的 `outputs/` 根目录文件
- `MLP_REG/GNN_REG` 的推理输出已补齐：
  - `true_change_ids`
  - `true_change_deltas`
  - `pred_change_ids`
  - `pred_change_deltas`
  - 全量 `true_deltas / pred_deltas / true_resistances / pred_resistances / abs_error_resistance`


## 2026-03-25 v41（四个主线子项目下一版模型目录已建立）
- 本轮新建四个下一版实验目录：
  - `mlp/MLP_REG/modelo6`
  - `mlp/MLP_CLS/modelo6`
  - `gnn/GNN_REG/modelo2`
  - `gnn/GNN_CLS/modelo2`
- MLP 下一版架构：
  - 输入保持 `(Batch, 32, 28)`，不再展平为 896。
  - 先对单次激励的 28 维响应做共享 MLP 编码，再在 32 次激励维度上做 MLP-Mixer 式 token mixing。
  - REG 头改为 `mask(sigmoid) * value(tanh)` 门控回归。
  - CLS 头在 CORAL 前加入 supervised contrastive 特征投影，重点拉开 `2/3` 类。
- GNN 下一版架构：
  - 输入改为 `(Batch, 32, 64, 4)`，每次激励展开成单独图，节点特征为 `[源掩码, 地掩码, 电压, 边界掩码]`。
  - 主干改为 3 层残差式 GATv2 风格注意力层，并加入跨激励 attention pooling。
  - REG 边解码器输入为 `cat([H_u, H_v, |H_u-H_v|])`，输出同样为门控回归。
  - CLS 在图级特征后加入 supervised contrastive loss。
- 损失函数口径：
  - REG 当前统一为 `MSE(pred, true) + lambda_mask_l1 * mean(mask_prob)`，默认 `lambda_mask_l1=0.05`，显式抑制“预测过多变化”。
  - CLS 当前统一为 `CORAL + lambda_supcon * SupCon`，其中 SupCon 主要锚定真实 `2/3` 类样本。
- 项目状态更新：
  - 当前主重点已转为 `MLP_REG/modelo6`、`GNN_REG/modelo2` 及对应的 `CLS` 新版本。
  - `modelo5 / modelo1` 继续保留为已训练过的稳定基线，便于与新版本首轮结果对比。
- 本地验证说明：
  - 已完成 `py_compile` 语法检查。
  - 由于当前本地 Python 环境缺少 `torch`，尚未完成真实前向冒烟测试；新版本的 shape/训练行为需以云端首轮运行为准。
## 2026-03-25 v42（GNN modelo2 切换为 PyG 原生 GATv2Conv，并完成最小验证）
- 按用户要求，已通过清华源下载并在项目根目录建立本地 vendor：`64Nodes/.vendor_torchpy311`
- 当前 vendor 内已安装：
  - `torch 2.11.0+cpu`
  - `torch-geometric 2.7.0`
- `GNN_REG/modelo2` 与 `GNN_CLS/modelo2` 已从“自实现 GATv2 风格注意力层”切换为 PyG 原生 `GATv2Conv`
- 当前 GNN 新版的关键实现状态：
  - 输入仍为 `(Batch, 32, 64, 4)`
  - 单次激励图先经过 3 层残差式 `GATv2Conv`
  - 再做跨激励 attention pooling
  - REG 边解码器保持 `cat([H_u, H_v, |H_u-H_v|]) + mask/value` 门控输出
- 本地兼容处理：
  - `gnn/GNN_REG/modelo2` 与 `gnn/GNN_CLS/modelo2` 的 `train.py / inference.py / model.py` 已支持自动探测项目根目录下的 `.vendor_torchpy311`
  - 因此在当前机器上可直接运行新版 GNN 脚本，无需额外手动设置 `PYTHONPATH`
- 最小验证结果：
  - 原生 `GATv2Conv` import 成功
  - `PhysicsInformedGNNRegressor` 前向通过：输出 shape 为 `(2, 112)`
  - `PhysicsInformedGNNClassifier` 前向通过：输出 shape 为 `(2, 3)`
  - `python gnn/GNN_REG/modelo2/train.py --help` 启动正常
## 2026-03-25 v43（GNN modelo2 显存修复：按 excitation 分块编码）
- 用户在云端运行 `GNN_CLS/modelo2` 时出现 `torch.OutOfMemoryError`：
  - 触发位置在原生 `GATv2Conv` 的 `edge_update`
  - 根因是默认把 `(Batch, 32, 64, 4)` 一次性展平成 `Batch*32` 张图并行送入 3 层注意力主干，默认 batch 下峰值显存过高
- 本轮修复：
  - `GNN_REG/modelo2` 与 `GNN_CLS/modelo2` 新增 `excitation_chunk_size`
  - 32 次激励现在按 excitation 分块过图主干，再在块之间拼回 `(Batch, 32, 64, Hidden)`
  - 结构语义不变，只降低峰值显存
- 默认训练参数同步收紧：
  - `GNN_REG/modelo2`: `batch_size 64 -> 8`
  - `GNN_CLS/modelo2`: `batch_size 48 -> 8`
  - 新增默认 `excitation_chunk_size=4`
- 本地验证：
  - 分块版与整批版前向输出保持一致，仅有浮点级误差：
    - REG 最大差异约 `3.8e-06`
    - CLS 最大差异约 `1.5e-08`
- 当前建议云端先用：
  - `python gnn/GNN_CLS/modelo2/train.py --dataset-tag 10mA --batch-size 8 --excitation-chunk-size 4`
  - `python gnn/GNN_REG/modelo2/train.py --dataset-tag 10mA --batch-size 8 --excitation-chunk-size 4`
- 若仍显存紧张，可继续降到：
  - `--batch-size 4 --excitation-chunk-size 2`
## 2026-03-26 v44（四个主线新版本首轮结果：CLS 明显跃升，REG 稀疏显著但 changed-MAE 仍偏高）
- 用户已完成四个新版本首轮训练：
  - `gnn/GNN_CLS/modelo2`
  - `gnn/GNN_REG/modelo2`
  - `mlp/MLP_CLS/modelo6`
  - `mlp/MLP_REG/modelo6`
- GPU 运行情况：
  - 经过 `modelo2` 的 excitation 分块显存修复后，云端训练可稳定运行，训练期 GPU 利用率约 `80%`，说明新图主干虽更重，但当前显存路径已基本可用。
- 分类线（当前最积极的结果）：
  - `GNN_CLS/modelo2`：`test_macro_f1=0.8871`
  - `MLP_CLS/modelo6`：`test_macro_f1=0.8854`
  - 相比旧主线：
    - `GNN_CLS/modelo1@10mA`: `0.7855 -> 0.8871`，提升约 `+0.1016`
    - `MLP_CLS/modelo5@10mA`: `0.7925 -> 0.8854`，提升约 `+0.0929`
  - 阶段结论：
    - 两条 `CLS` 新架构都取得了实质性进步，不再只是小幅波动；
    - 当前 `2/3` 混淆仍存在，但已经从“核心瓶颈完全卡住”下降到“仍需继续精修的残余问题”。
- 回归线（当前最需要继续调整的结果）：
  - `GNN_REG/modelo2`：
    - `mae_all=0.8814`
    - `mae_changed=50.9660`
    - `avg(|dR|>40)=1.40`
  - `MLP_REG/modelo6`：
    - `mae_all=1.5010`
    - `mae_changed=56.7012`
    - `avg(|dR|>40)=1.64`
  - 相比旧主线：
    - `GNN_REG/modelo1@10mA` 最稳参考约 `mae_changed=43.0275`，新版本 `mae_changed` 仍更差；
    - `MLP_REG/modelo5@10mA` 最稳参考约 `mae_changed=43.2690`，新版本 `mae_changed` 同样更差。
  - 但新版本的共同优点也非常明显：
    - `mae_all` 大幅下降；
    - 预测显著更稀疏，`avg(|dR|>threshold)` 明显更低；
    - 推理样例中 `0/1/2` 变化样本已能出现较干净的正确定位。
  - 阶段结论：
    - 新 `REG` 架构不是无效，而是“明显偏保守”：
      - 对未变化位的抑制和全局误差控制非常强；
      - 但对真实变化位的幅值恢复和 `2/3` 样本召回仍不够，导致 `mae_changed` 和数量派生统计偏差较大。
- 当前最合理的整体判断：
  - 这轮最大的明确进步来自 `CLS` 新架构，属于“可以确认的跨档提升”；
  - `REG` 新架构已经学到了更干净的稀疏先验，但门控回归 + L1 目前压得过重，尚未达到替代旧 `modelo5/modelo1` 基线的程度；
  - 因此下一阶段适合：
    - 分类线保留当前新版本，继续小步精修 `2/3` 边界；
    - 回归线继续沿新版本迭代，但优先解决“过度保守、低估真实变化数和变化幅值”的问题。
- 本轮还暴露了一个较小技术问题：
  - `MLP_CLS/modelo6` 与 `GNN_CLS/modelo2` 在验证/测试时出现 `np.exp(-logits)` overflow warning；
  - 这不影响当前分类结果判断，但后续应改为更稳定的 sigmoid 计算方式（如 clip 或 `scipy.special.expit`）。
## 2026-03-26 v45（四个主线子项目下一版已建立：修复数值溢出、放松 GNN_REG、加入 CLS+REG 融合推理）
- 基于 0326 首轮结果补充的阶段分析：
  - `CLS` 线表现已明显跃升，但 `GNN_CLS/modelo2` 与 `MLP_CLS/modelo6` 的混淆矩阵高度相似：
    - `0` 变化与 `1` 变化几乎已接近完美；
    - 主要误差集中在 `2 <-> 3`。
  - 这说明当前剩余瓶颈已更接近 EIT 逆问题本身的物理病态性：
    - `3` 个较小扰动的边界响应，确实可能与 `2` 个中等扰动相似；
    - 因此 `2/3` 混淆仍会残留，但不再像旧版本那样成为压倒性主问题。
  - 另一方面，分类训练日志中的 `overflow encountered in exp` 说明 logits 已过大，模型出现过度自信倾向；
    - 虽然不影响这轮主结论，但会带来数值稳定性和泛化风险。
  - `REG` 线的对比则体现出“性格互补”：
    - `GNN_REG/modelo2` 偏保守，漏报多，但一旦预测非零，数值更稳；
    - `MLP_REG/modelo6` 相对激进，对 `1` 变化更敏感，但在 `3 -> 2` 低估上更明显。
- 本轮新建下一版目录：
  - `mlp/MLP_CLS/modelo7`
  - `mlp/MLP_REG/modelo7`
  - `gnn/GNN_CLS/modelo3`
  - `gnn/GNN_REG/modelo3`
- 新版目录处理：
  - 由上一版复制而来，但复制带来的 `outputs/`、`cache/`、`__pycache__/` 等内容已清空，避免旧结果污染。
- 本轮代码改动重点：
  - `CLS`：
    - 用 `scipy.special.expit` 替换手写 `1 / (1 + exp(-logits))`
    - `MLP_CLS/modelo7` 与 `GNN_CLS/modelo3` 默认 `weight_decay` 提高到 `1e-3`
  - `GNN_REG`：
    - `lambda_mask_l1: 0.05 -> 0.01`
    - 目标是释放模型对非零变化的预测勇气，减少“缩头乌龟式”漏报
  - `MLP_REG`：
    - 后续同步修正 `MLP_REG/modelo7` 的 `lambda_mask_l1: 0.05 -> 0.01`
    - 保持与 `GNN_REG/modelo3` 一致的放松方向，避免两条回归线比较时口径不一致
  - `Inference`：
    - `CLS/REG` 常规 `inference.py` 现在默认优先抽取真实 `2/3` 变化样本，减少样例过多落在 `0/1`
    - 新增 `inference_full.py`：
      - `MLP_REG/modelo7/inference_full.py`
      - `GNN_REG/modelo3/inference_full.py`
    - 新逻辑为：
      - 先用 `CLS` 预测变化数 `K`
      - 再用 `REG` 输出的 `|dR|` 排序，取前 `K` 个边作为最终扰动边
      - 不再依赖死板阈值截断
  - 兼容性：
    - 若云端环境缺少 `scipy`，新版脚本会自动回退到数值稳定的本地 `expit` 实现，避免因 `ModuleNotFoundError` 无法启动训练或推理
- 阶段判断：
  - 分类线当前更适合继续精修，而不是推翻重来；
  - 回归线当前最该测试的是“放松稀疏惩罚后，`mae_changed` 能否回升而不显著牺牲稀疏性”；
  - 新增的 `inference_full` 更贴近最终使用逻辑，后续应优先观察它在 `2/3` 样本上的位置命中与误报情况。

## 2026-03-26 v46
- 按根目录 `新架构.md`，在 `gnn/GNN_REG` 下新建物理迭代回归分支：
  - `gnn/GNN_REG/model_tp1`
- 该分支不再以 `GATv2` 为主干，而改为：
  - 共享边电导参数；
  - KCL 残差迭代传播；
  - 边界节点每步固定；
  - 最终对 112 条电阻边做门控回归。
- 与现有项目保持一致：
  - 仍直接读取主 CSV 数据；
  - 仍支持 `--data-path`、`--dataset-tag`；
  - 仍按 `cache/<dataset_tag>/` 与 `outputs/<dataset_tag>/` 管理结果。
- 本地已完成：
  - `py_compile`
  - `train.py --help`
  - `inference.py --help`
  - 最小前向验证 `(2, 112)`

## 2026-03-26 v47
- `GNN_CLS/modelo3` 首轮正式结果：
  - `test_macro_f1=0.9075`
  - 说明分类新架构继续有效，`0/1` 基本稳定，误差仍主要集中在 `2/3`。
- `GNN_REG/modelo3` 中途观察到：
  - `val_avg(|dR|>50)` 长时间停在约 `1.18`
  - 与主数据真实平均变化数 `1.86` 相比明显偏低
  - 说明当前 `REG` 线的稀疏压力仍然偏大。
- 因此同步再放松两条 `REG` 主线：
  - `gnn/GNN_REG/modelo3`
    - `lambda_mask_l1: 0.01 -> 0.002`
    - `val_sparse_alpha: 0.20 -> 0.05`
  - `mlp/MLP_REG/modelo7`
    - `lambda_mask_l1: 0.01 -> 0.002`
    - `val_sparse_alpha: 0.25 -> 0.05`

## 2026-03-27 v48
- 检查 `训练记录0325.txt`：
  - 其中 `GNN_CLS/modelo2`、`GNN_REG/modelo2`、`MLP_CLS/modelo6`、`MLP_REG/modelo6` 的关键结果与阶段分析均已记入 `Log.md / gnn/Log.md / mlp/Log.md`
  - 因此该文件可删除
- 记录 `训练记录0326.txt` 的关键结果：
  - `GNN_REG/modelo3`：`mae_changed=25.0888`，记为当前项目 `REG` 线最佳 baseline
  - `MLP_CLS/modelo7`：`test_macro_f1=0.8735`，继续作为稳健数量判断锚点
  - `MLP_REG/modelo7`：`mae_changed=56.5487`，作为稳定备份回归线保留
  - `GNN_REG/model_tp1`：`mae_changed=106.4545`，当前方案失败
- 补充阶段分析：
  - `GNN_REG/modelo3` 的成功说明，放松稀疏惩罚后，图结构对逆问题重构的帮助被真正释放出来了
  - `MLP_CLS/modelo7` 当前已经足以承担“先判断大致变化数量”的角色
  - `model_tp1` 的失败再次说明，软物理损失若过早、过重地介入，会和数据驱动损失发生明显冲突
  - `MLP_REG/modelo7` 仍不够精，但作为融合支线有保留价值
- 新增根目录联合异构推理入口：
  - `inference_hetero_cmei.py`
  - 默认逻辑：
    - `MLP_CLS/modelo7` 预测 `K`
    - `GNN_REG/modelo3` 与 `MLP_REG/modelo7` 按 `0.7 / 0.3` 融合
    - 再取前 `K` 大绝对值作为最终边
  - 脚本会在完整测试集上计算 `CMEI`，并输出 2/3 变化样例图

## 2026-03-27 v49
- 记录 `GNN_REG/model_tp1` 修正版训练结果：
  - `mae_all=2.0972`
  - `mae_changed=101.9642`
  - `best_count_threshold(val)=40.0`
  - `val_macro_f1=0.3097`
  - `avg(|dR|>40)=1.06`
  - `avg(mask_prob)=0.0068`
  - `avg(kcl_residual)=0.021538`
  - 派生数量混淆矩阵：
    - `[[73,0,0,0],[58,203,46,16],[51,189,48,9],[20,201,68,18]]`
- 结论修正：
  - 虽然比上一版 `model_tp1` 有轻微回升，但仍明显过于保守，`2/3` 变化大量塌缩到 `0/1`
  - 这条物理总约束支线暂时停止继续展开，不再作为当前主推进方向
  - 当前主目标转为把 `MLP_CLS/modelo7 + GNN_REG/modelo3 + MLP_REG/modelo7` 整合成可实际使用的联合方案
- 更新根目录联合推理脚本 `inference_hetero_cmei.py`：
  - 终端会先分别输出 `S_num / S_F1 / S_id / S_mse`，再输出加权总分 `CMEI`
  - 终端会同步打印 3 到 4 个 `2/3` 变化样例的 `true_k / pred_k / true_ids / pred_ids`
  - 输出目录只保留聚合评分、少量样例图片和小型样例摘要，不再导出完整测试集逐样本明细
  - 重建图中的数值标签已移出粗边位置，避免遮挡

## 2026-03-27 v50
- 新建筛选版 10mA 数据集：
  - `data/training_data64Nodes_2_screened.csv`
  - `data/training_data64Nodes_2_screened_meta.json`
  - 生成脚本：
    - `scripts/generate_training_data64_screened.py`
  - 规则：
    - 若内部电阻变化集合同时出现正负号，则跳过该组合并重采样
    - 总组合数仍保持 `10000`
  - 本次生成结果：
    - `Generated combos: 10000`
    - `Skipped ambiguous internal opposite-sign combos: 279`
- 新建 `MLP_CLS/modelo8`：
  - 主干延续 `MLP-Mixer`
  - 在骨干后增加 `>=2 vs <=1` 辅助二分类头
  - 训练损失改为：
    - `CORAL + lambda_aux * BCE + lambda_supcon * SupCon`
  - `2/3` 类对比学习权重提高到 `1.5`
  - 阈值搜索改为按验证集概率分布生成自适应分位数候选，再做逐类阈值搜索
  - 加入 `CosineAnnealingWarmRestarts`
- 新建 `GNN_REG/modelo4a`：
  - 在 `modelo3` 的 `GATv2 + cross-excitation pooling` 基础上加入 `resistor_embedding`
  - 新增 `top-K` 位置值损失
  - 新增小权重电压重投影物理损失 `lambda_physics=0.01`
  - 加入 `CosineAnnealingWarmRestarts`
- 新建 `GNN_REG/modelo4b`：
  - 结构与 `modelo4a` 一致
  - 额外输出 `top3 / top4 / top5` 候选覆盖率，用于“候选集包含真实变化边即可判对”的评估口径
- 新建联合推理目录：
  - `joint_inference/`
  - `joint_inference/inference_hetero_cmei.py`
  - `joint_inference/inference_hetero_cmeiv2.py`
  - 根目录 `inference_hetero_cmei.py` 改为薄包装入口
- `CMEIv2` 说明：
  - 使用 `MLP_CLS` 的分类熵估计分类置信度
  - 使用 `GNN_REG / MLP_REG` 的 mask 概率估计回归置信度
  - 对每个样本动态分配 `GNN/MLP` 融合权重，高置信样本更偏向高置信模型，低置信样本更接近均值融合
- 兼容性补充：
  - 新脚本已增加 `inverse_identifiability/.vendor` 的自动依赖回退，避免本地缺 `numpy` 时直接中断

## 2026-03-27 v51
- `MLP_CLS/modelo8` 首轮回传结果：
  - raw `10mA` 与 screened `10mA` 均未体现出对 `modelo7` 的稳定增益；
  - 当前判断：Aux Head（`>=2` vs `<=1`）任务过于简单，`val_aux_acc` 很快升到约 `98%`，辅助 loss 迅速衰减，未能持续帮助主干区分 `2/3`。
- 筛选版 `10mA` 数据口径修正：
  - 去除“内部正负抵消配对”后，数据分布被人为净化，破坏了真实物理场景的连续分布；
  - 模型在更简单数据上训练后，反而削弱了处理复杂边缘特征的能力；
  - 因此当前判断这属于一次不利的 Distribution Shift，主线默认数据明确退回未筛选 `10mA`：`data/training_data64Nodes_2.csv`。
- `GNN_REG/modelo4a` 首轮部分结果显示：
  - `val_phys` 长时间维持在 `4600+`
  - `avg(|dR|>45)` 被卡在约 `1.15`
  - 当前判断：KCL 软约束与主回归目标发生剧烈梯度冲突，模型退回保守安全区；同时强稀疏压力会把相邻真实变化边压缩成单边预测。
- 路线取舍修正：
  - 撤回 `GNN_REG/modelo4a / modelo4b`
  - GNN 回归主线继续维持为 `GNN_REG/modelo3`
  - 若后续需要候选集评估口径，则直接基于 `modelo3` 追加 `top3 / top4 / top5` 候选覆盖率，而不再另起 `modelo4b` 主干。
- `GNN_REG/modelo3b` 已建立：
  - 作为 `modelo3` 的推理级候选集版本存在
  - 不新增全局联合推理入口
  - 仅单独输出 `top3 / top4 / top5` 候选覆盖率与样例候选集结果
- 联合推理补充：
  - `CMEIv2` 与固定权重 `CMEI v1` 当前基本打平，默认推荐继续保留 `v1`
  - detail sample 默认固定 `split-seed` 与 `seed`，因此多次运行看到相同样例是设计如此
  - `Near-Miss` 后处理会对相邻候选边做轻量替换，因此对相邻双边同时保留存在轻微抑制
- `MLP_CLS/modelo8` 默认 seed 已对齐到 `20260325`
  - 与 `MLP_CLS/modelo7` 保持同一默认划分口径，便于重新做公平 A/B

## 2026-03-28 v52
- 记录根目录 `训练记录0327.txt` 的公平对照结论：
  - `MLP_CLS/modelo7` 与 `MLP_CLS/modelo8` 均在未筛选 `10mA`、同一 `seed=20260325` 下训练
  - `modelo7`: `test_macro_f1=0.9022`
  - `modelo8`: `test_macro_f1=0.8852`
- 结论收敛：
  - `modelo8` 在同口径 A/B 下确实不如 `modelo7`
  - `modelo8` 的 Aux Head 虽然把 `aux_acc` 拉到约 `98.5%`，但并没有改善真正困难的 `2/3` 边界，反而使 `2->3` 与 `3->2` 混淆进一步加重
  - 因此 `MLP_CLS` 下一版继续以 `modelo7` 为母版推进，`modelo8` 不再作为当前默认演进方向
- `GNN_REG/modelo3b` 首轮候选集推理结果：
  - `top3_candidate_cover=0.8300`
  - `top4_candidate_cover=0.8540`
  - `top5_candidate_cover=0.8610`
  - `changed_only` 口径下为 `0.8170 / 0.8428 / 0.8504`
- 含义解释：
  - `GNN_REG/modelo3` 已具备较强的候选排序能力，适合承担“候选生成”角色
  - `top4` 对 `top3` 有实质补救，说明不少错误只差一名
  - `top5` 提升有限，说明剩余难例里存在真实边被排到前五之外的情况，候选扩容不能替代主模型本身的进一步改进

## 2026-03-28 v53
- 新建 `joint_method/`，作为最终组合模型的新主优化目录。
- 本次迁移基于两条已经收敛的阶段结论：
  - `MLP_CLS/modelo7` 在同数据同 seed 的公平 A/B 下优于 `modelo8`，因此 `CLS` 下一版回到 `modelo7` 继续演进。
  - `GNN_REG/modelo3` 仍是当前最可信的 GNN 回归锚点，`modelo3b` 的候选集结果说明它适合承担候选生成与主回归来源。
- 依据根目录 `改进方案.txt`，在新目录中建立：
  - `joint_method/cls_n1`
  - `joint_method/reg_n1/mlp_reg`
  - `joint_method/reg_n1/gnn_reg`
  - `joint_method/inference_n1.py`
- 新主线的默认改动如下：
  - `cls_n1`：基于 `modelo7` 增加只针对真实 `2/3` 样本的 Aux Head。
  - `reg_n1`：MLP/GNN 回归器都改为 `Adaptive Relaxed Sparsity`，默认放行 Top-5 候选边。
  - `inference_n1.py`：联合推理回退到静态融合，默认 `0.85 * GNN + 0.15 * MLP`。
- 默认数据继续统一为未筛选 `10mA`：`data/training_data64Nodes_2.csv`。
- 文档迁移完成：
  - 根目录 `README.md / Log.md` 已记录目录转换。
  - 详细说明转入 `joint_method/README.md / joint_method/Log.md`。
- `改进方案.txt` 与 `训练记录0327.txt` 的关键内容已经沉淀进正式文档，后续可删除原始 txt 记录，避免重复维护。

## 2026-03-28 v54
- `joint_method` 已由三模型过渡版本收敛为纯双模型主线：`MLP_CLS + GNN_REG`。
- 当前保留入口：
  - `joint_method/cls_n1`
  - `joint_method/reg_n1`
  - `joint_method/inference_n1.py`
- 已删除 `joint_method` 中不再需要的回归支线目录：
  - `joint_method/reg_n1/mlp_reg`
  - `joint_method/reg_n1/gnn_reg`
- 新版 `inference_n1.py` 只保留：
  - `MLP_CLS` 预测变化数量 `K`
  - `GNN_REG` 输出 112 维变化量
  - 取 `|dR|` 前 `K` 大边作为最终预测
- 本地已完成纯双模型推理冒烟：
  - 使用旧稳定组合 `MLP_CLS/modelo7 + GNN_REG/modelo3`
  - `CMEI=90.84`
  - `num_accuracy=0.8440`
  - `macro_f1=0.8735`
  - `id_recall=0.8827`
  - `mse_all_edges=75.1218`
- 说明：
  - 这次分数只是为了验证新的双模型推理链路可运行，不代表 `cls_n1 / reg_n1` 的正式结果。
  - 后续正式比较，应以云端训练出的 `joint_method/cls_n1` 与 `joint_method/reg_n1` 新权重为准。

## 2026-03-28 v55
- 用户使用全新路径重新复现 `gnn/GNN_REG/modelo3`：
  - `--dataset-tag o3_repro_0328`
  - `--cache-path gnn/GNN_REG/modelo3/cache/o3_repro_0328/cache_dataset_reg_graphattn.npz`
- 复现实验前 15 个 epoch 仍停留在明显过保守区域：
  - `val_mae_changed` 约 `87.25 -> 71.15`
  - `val_avg(|dR|>50)` 约 `1.16 ~ 1.20`
  - 与 `0326` 历史记录中那次快速进入 `1.6+` 激活区、最终达到 `mae_changed=25.0888` 的结果不一致。
- 同期 `joint_method/n1/reg_n1` 也表现出几乎相同的坏轨迹，说明问题不在目录切换本身，而在当前 `REG` 主干已经稳定落入过保守坏解。
- 阶段结论更新：
  - `0326` 那次 `GNN_REG/modelo3` 的好结果目前只能视为“历史上曾出现过的优异运行”，不能再视为已验证、可稳定复现的现役基线；
  - `joint_method` 路线正式放弃并整体删除；
  - 项目后续简化为纯 GNN 主线，只保留 `gnn/GNN_CLS/modelo3` 与 `gnn/GNN_REG/modelo3` 继续推进。
- 经验教训：
  - 逆问题 `REG` 的门控稀疏回归对优化初始轨迹极其敏感；
  - 以后任何“最佳结果”都必须先经过 fresh cache、fresh outdir、多 seed 复现实验，才允许升级为正式主线结论。

## 2026-03-28 v56
- 新建 `GNN_REG` 的 `o4` 实验系列：
  - `gnn/GNN_REG/o4a`
  - `gnn/GNN_REG/o4a2`
  - `gnn/GNN_REG/o4b`
  - `gnn/GNN_REG/o4b2`
  - `gnn/GNN_REG/o4b3`
- 设计意图：
  - `o4a`：在保留耦合输出的前提下，用 `mask` 偏置初始化和 `lambda_mask_l1 warmup` 尝试逃离门控塌陷；
  - `o4a2`：保留耦合输出，但用 `BCE(mask_logits)` 直接监督门控，并把回归主损失改成 `SmoothL1`，试图在不过度激活的前提下阻止门控塌陷；
  - `o4b`：彻底解耦 `mask/value`，验证“耦合前向 + 全局 MSE 梯度劫持”分析；
  - `o4b2`：针对 `o4b` 的过激活现象，降低 `mask/value` 激进度并加入背景值抑制；
  - `o4b3`：进一步根据 Loss 量级分析，把总损失重平衡为“显式抬高 BCE 话语权”的版本。
- `o4b` 已取得一轮不完整结果（`dataset_tag=o4b_0328_try1`）：
  - `epoch 1`：`val_mae_changed=85.6182`，`val_avg(|dR|>50)=11.18`，`val_mask_mean=0.3417`
  - `epoch 10`：`val_mae_changed=39.9807`，`val_avg(|dR|>50)=14.91`
  - `epoch 20`：`val_mae_changed=36.1730`，`val_avg(|dR|>50)=11.93`
  - `epoch 35`：`val_mae_changed=29.6492`，`val_avg(|dR|>50)=8.48`
  - `epoch 45`：`val_mae_changed=30.3320`，`val_avg(|dR|>50)=9.40`，`val_mask_mean=0.1170`
- 阶段分析：
  - `o4b` 的 `val_mae_changed` 明显优于最近复现失败时的 `modelo3`，说明解耦训练确实打破了“门控直接塌陷到几乎全关”的坏局部最优；
  - 但 `val_avg(|dR|>50)` 与 `val_mask_mean` 同时明显偏大，说明问题从“过保守”切换成了“严重过报”；
  - 根因判断是：虽然结构解耦了，但总 Loss 里 `masked MSE` 的量级仍远大于 `BCE`，导致优化器仍主要服务于回归头，分类头的话语权不足。
- 记录解释：
  - `val_mask_mean` 是验证集上 112 条边平均 `mask_prob` 的均值；
  - 它不是越大越好或越小越好，而是用于判断系统处于“塌陷”还是“过报”哪一侧；
  - 对当前任务来说，`0.10+` 这种量级已经意味着显著过活跃。

## 2026-03-28 v57
- 用户追加回传 `o4a` 的不完整训练结果：
  - `epoch 1`：`val_mae_changed=86.2339`，`val_avg(|dR|>50)=1.32`，`val_mask_mean=0.0340`
  - `epoch 5`：`val_mae_changed=73.5238`，`val_avg(|dR|>50)=1.16`，`val_mask_mean=0.0069`
  - `epoch 10`：`val_mae_changed=71.5638`，`val_avg(|dR|>50)=1.17`，`val_mask_mean=0.0064`
- 分析：
  - `o4a` 虽然加入了 `mask` 偏置初始化和 `lambda_mask_l1 warmup`，但门控仍在前 5~10 个 epoch 内迅速塌回低激活区；
  - 这说明问题已经不只是 `L1` 稀疏项过早施加，而是“耦合输出 + 仅靠回归损失”本身无法持续给门控头提供足够直接的监督。
- 基于这一判断，新增 `gnn/GNN_REG/o4a2`：
  - 模型结构仍保持 `o4a` 的耦合输出 `pred = mask_prob * value`
  - 训练侧新增 `BCEWithLogits(mask_logits, y_change)` 直接监督门控
  - 回归主损失由 `MSE` 改为更稳的 `SmoothL1`
  - 保留 `lambda_mask_l1 warmup`，但不再把“只靠初始化与 warmup”视为足够方案

## 2026-03-28 v58
- 基于原 `joint_inference` 两版方案的对比结果，正式新建统一 GNN 联合推理入口：
  - `gnn/inference_gnn_cmei.py`
- 新入口默认组合为：
  - `gnn/GNN_CLS/modelo3`
  - `gnn/GNN_REG/modelo3`
- 取舍依据：
  - 原 `joint_inference` 中，固定逻辑版整体比动态融合版更稳，未观察到动态权重带来明确收益；
  - `GNN_CLS` 与当前最佳 `MLP_CLS` 的数量分类效果差距已经很小；
  - `CLS + REG` 同时使用 GNN，更有利于形成统一的图结构主线，并保留物理驱动/拓扑归纳偏置的后续扩展空间。
- 因此路线正式切换为：
  - 停止维护 `MLP_CLS + GNN_REG (+ MLP_REG)` 的异构联合推理链路；
  - 停止维护 `joint_inference/` 目录；
  - 统一改为纯 `GNN_CLS + GNN_REG` 的 `CMEI` 推理与评估。
- 新入口保留的有效机制：
  - 延续旧固定逻辑版里较稳的 `Near-Miss` 轻量后处理；
  - 保留完整测试集 `CMEI` 评分、混淆矩阵和 detail samples 输出；
  - 删除跨 `MLP/GNN` 的动态融合权重设计，避免继续增加无效变量。

## 2026-03-29 v59
- 已把根目录 `GNN_REG训练记录.txt` 中的 `o4a2 / o4b2 / o4b3` 完整结果沉淀进正式日志。
- 当前阶段最重要的新判断：
  - `o4a2` 是第一条真正回到“中间带”的 `GNN_REG` 新支线；
  - 它同时避免了 `modelo3` 近期复现中的门控塌陷，也避免了 `o4b` 系列早期那种严重过报。
- `o4a2` 当前单次结果：
  - `mae_all=0.4854`
  - `mae_changed=24.2925`
  - `best_count_threshold(val)=40.0`
  - `val_macro_f1=0.8683`
  - `avg(|dR|>40)=1.77`
  - `avg(mask_prob)=0.0192`
- 这组数字的含义：
  - 已经重新达到并略好于此前 `GNN_REG/modelo3` 的历史最好上限；
  - 更重要的是，整条训练曲线在后半段稳定停留在合理稀疏区，而不是靠单次偶然冲到好结果。
- `o4b2 / o4b3` 的阶段结论：
  - 两者都继续证明“解耦训练”能明显降低 `mae_changed`；
  - 但它们仍存在明显过报，表现为：
    - `o4b2`：`mae_all=2.5001`，`val_macro_f1=0.6904`，`avg(|dR|>56)=2.82`
    - `o4b3`：`mae_all=2.2490`，`val_macro_f1=0.6990`，`avg(|dR|>67)=2.27`
  - 说明完全解耦路线当前更像“候选生成/高召回支线”，还不适合直接作为主回归模型。
- 当前项目结论更新为：
  - `gnn/GNN_REG/o4a2` 升级为当前最强的 `GNN_REG` 候选主线；
  - `gnn/GNN_REG/modelo3` 继续保留为历史 baseline / 可达上限参考；
  - `o4b2 / o4b3` 先保留为诊断和候选集思路验证分支，不升为主线。
- 后续最优先工作：
  - 对 `o4a2` 做 fresh cache / fresh outdir / 多 seed 复验；
  - 只有在重复成功后，才把它正式替换 `modelo3` 成为稳定 baseline。

## 2026-03-29 v60
- 新增 GNN 通用可视化脚本：
  - `gnn/visualize_gnn_results.py`
- 设计目标：
  - 用统一风格展示 `GNN_CLS / GNN_REG / 候选集 / 联合推理` 四类结果；
  - 除了分数，还强调版本演进、混淆矩阵、样例拓扑对比和 `CMEI` 组成。
- 当前脚本能力：
  - 自动识别 `metrics.json / cmei_metrics.json / candidate_metrics.json`
  - 自动读取 `inference_samples / inference_full_samples / detail_samples / candidate_samples`
  - 单 run 生成 `overview.png + samples.png`
  - 多 run 生成 `comparison.png`
- 已完成本地冒烟：
  - `joint=gnn/outputs/gnn_cmei/training_data64Nodes_2`
  - `cls2=gnn/GNN_CLS/modelo2/outputs/training_data64Nodes_2`
  - `cls3=gnn/GNN_CLS/modelo3/outputs/training_data64Nodes_2`
- 当前默认可视化输出目录：
  - `gnn/outputs/visualizations`
## 2026-03-30 v61
- 根目录 `0330训练日志.txt` 已纳入正式结论：
  - `gnn/GNN_REG/o4a2` 已完成 4 个 seed 的 fresh-cache / fresh-outdir 复验；
  - 结果区间稳定，证明这条线已具备实际可复现性，不再只是单次幸运运行。
- 4-seed 摘要：
  - `o4a2_seed20260325`: `mae_all=0.4806`, `mae_changed=25.9977`, `count_f1=0.8508`, `avg_gt=1.713`
  - `o4a2_seed20260326`: `mae_all=0.5013`, `mae_changed=25.3102`, `count_f1=0.8655`, `avg_gt=1.722`
  - `o4a2_seed20260327`: `mae_all=0.5387`, `mae_changed=27.5191`, `count_f1=0.8535`, `avg_gt=1.704`
  - `o4a2_seed20260328`: `mae_all=0.5654`, `mae_changed=26.3538`, `count_f1=0.8558`, `avg_gt=1.768`
- 均值/波动：
  - `mae_all mean=0.5215, std=0.0328`
  - `mae_changed mean=26.2952, std=0.8000`
  - `count_f1 mean=0.8564, std=0.0056`
  - `avg_gt mean=1.7268, std=0.0247`
- 阶段结论正式更新：
  - `GNN_REG/o4a2` 取代 `GNN_REG/modelo3`，升级为当前最佳且可复验的正式主线；
  - `modelo3` 保留为历史 baseline / 早期可达上限参考；
  - `o4b2 / o4b3` 继续停留在诊断/高召回支线定位。
- 当前最佳单 checkpoint 说明：
  - 本地 `gnn/GNN_REG/o4a2/outputs/training_data64Nodes_2/` 仍保存着更强的单次结果：
    - `mae_all=0.4854`
    - `mae_changed=24.2925`
    - `val_macro_f1=0.8683`
    - `avg(|dR|>40)=1.77`
  - 它可以继续作为“当前最好可用权重”保存和使用；
  - 若强调复验锚点，则优先参考 `o4a2_seed20260326`。
- 对用户观察的三点分析确认如下：
  - 训练末期确有轻微过拟合/平台期震荡，但由于脚本保存的是验证最优权重，因此当前可用性仍成立；
  - 模型仍有保守预测倾向，主要体现在高变化数样本上偏向少报；
  - `mask_l1=0.002000` 后期不再变化是设计如此：warmup 结束后它固定为常数，说明后期稀疏约束没有继续增强；后续若优化 `o4a2`，应优先考虑修正保守偏置，而不是单纯继续拉大 `mask_l1`。
- 默认 unified GNN 推理也已完成一次本地冒烟：
  - 当前默认组合为 `GNN_CLS/modelo3 + GNN_REG/o4a2`
  - `CMEI=93.73`
  - `num_accuracy=0.8850`
  - `macro_f1=0.9075`
  - `id_recall=0.9248`
  - `mse_all_edges=49.7686`
  - 说明 `o4a2` 下载回本地后的 outputs 不仅能单独复用，也已经可以直接接入当前统一推理链路。

## 2026-03-30 v62
- 已确认一个容易混淆的点：
  - `o4a2` 训练历史里每个 epoch 打印的 `val_mae_all / val_mae_changed` 口径没有被“改小”；
  - 它们仍然分别是：
    - 所有 112 条边上的绝对误差平均值；
    - 真实发生变化边上的绝对误差平均值。
- 真正改动的是：
  - 训练主损失从旧版 `MSE(+L1)` 改成了 `SmoothL1 + BCE(mask) + warmup(L1)`；
  - 因此数值变好是训练效果变化，不是展示指标被重新定义。
- `0330训练日志.txt` 的内容已经沉淀进正式文档，原始 txt 记录可删除，避免继续维护两份来源。
- 基于 `o4a2` 已新增两条新支线：
  - `gnn/GNN_REG/o5a`
    - 定位：只针对“保守预测倾向”的小改版；
    - 做法：在 `o4a2` 原有 `SmoothL1 + BCE(mask) + warmup(L1)` 之上，新增“真实变化边幅值下限”约束；
    - 目标：减少真实变化边被压到计数阈值以下、从而导致少报的情况。
  - `gnn/GNN_REG/o5b`
    - 定位：按根目录 `GNN_REG优化.txt` 的方案改版；
    - 做法：
      - 为 112 条边引入绝对位置 `edge embedding`
      - 把全局 `mask_prob.mean()` 稀疏惩罚改成 relaxed top-k sparsity
    - 目标：补足绝对位置先验，并减少对相邻真实变化边的过度抑制。
- 新建联合目录：
  - `gnn/GNN_FULL/Mv1`
  - 当前结构：
    - `model/model_cls.py`
    - `model/model_reg.py`
    - `train_cls.py`
    - `train_reg.py`
    - `inference.py`
    - `outputs/`
  - 定位：
    - 这是第一版把 `GNN_CLS/modelo3` 与 `GNN_REG/o4a2` 放进同一联合目录下管理的方案；
    - 两条训练线仍分开训练，但联合推理入口统一为一个 `inference.py`。
- `GNN_FULL/Mv1/inference.py` 已完成一次借用现有权重的本地冒烟：
  - 使用：
    - `GNN_CLS/modelo3`
    - `GNN_REG/o4a2`
  - 结果与当前统一 GNN 推理一致：
    - `CMEI=93.73`
    - `num_accuracy=0.8850`
    - `macro_f1=0.9075`
    - `id_recall=0.9248`
    - `mse_all_edges=49.7686`
  - 说明 `Mv1` 的联合推理壳层已经可运行，后续只需把 `Mv1` 自己训练得到的 `cls/reg` 结果接进去即可。

## 2026-03-31 v63
- 根目录 `0331训练记录.txt` 已吸收进正式文档，记录内容覆盖 `o5a / o5b` 首轮训练与测试结果。
- `o5a` 首轮完整结果：
  - `mae_all=0.7770`
  - `mae_changed=38.8351`
  - `best_count_threshold=40.0`
  - `val_count_macro_f1=0.8524`
  - `avg_abs_gt_threshold=1.772`
  - `avg_mask_prob=0.0200`
- 对 `o5a` 的判断：
  - 它没有修复 `o4a2` 的保守倾向，反而显著破坏了幅值回归质量；
  - 说明“真实变化边幅值下限”这条约束过于直接，会把数值头拉离原本较好的校准区；
  - `o5a` 作为失败消融保留即可，不再继续扩展。
- `o5b` 首轮完整结果：
  - `mae_all=0.5119`
  - `mae_changed=25.4532`
  - `best_count_threshold=40.0`
  - `val_count_macro_f1=0.8564`
  - `avg_abs_gt_threshold=1.737`
  - `avg_mask_prob=0.0207`
- 对 `o5b` 的判断：
  - `edge embedding + relaxed top-k sparsity` 的方向是对的；
  - 它明显优于 `o5a`，并已达到或略优于 `o4a2` 多 seed 均值；
  - 但它仍略弱于当前最佳单 checkpoint `o4a2`，因此还不能直接接替 `o4a2` 成为默认回归器。
- 已完成一次 `GNN_CLS/modelo3 + GNN_REG/o5b` 的统一联合推理验证：
  - 输出目录：`gnn/outputs/gnn_cmei_o5b_eval/o5b_10mA/`
  - `CMEI=93.20`
  - `num_accuracy=0.8850`
  - `macro_f1=0.9075`
  - `id_recall=0.9120`
  - `mse_all_edges=56.4333`
- 与当前默认 `modelo3 + o4a2` 比较：
  - 分类侧基本持平；
  - 但 `id_recall` 和 `mse_all_edges` 都更差；
  - 因此 `o5b` 还不适合替换默认联合推理中的 `o4a2`。
- 当前后续建议：
  - 主线继续保持 `GNN_CLS/modelo3 + GNN_REG/o4a2`
  - `o5b` 下一步应先做多 seed 复验，确认均值和方差是否整体优于 `o4a2`
  - 在没有多 seed 证据前，不再继续沿 `o5a` 方向投入训练成本

## 2026-03-31 v64
- 已在 `gnn/GNN_FULL/Mv1/` 下新增实验推理文件：
  - `inference_v2.py`
- 设计动机：
  - 修复旧 near-miss 对 `o5b` 相邻真实损坏簇的误杀；
  - 同时加入基于 `|ΔR|` 的物理死区截断与高幅值补漏，尝试用 REG 反向校正 CLS 的计数误判。
- 关键规则包括：
  - `weakest` 相邻边若 `reg_prob > 0.85` 或 `|ΔR| > 45`，near-miss 不再替换；
  - top-k 先经过 `35Ω` deadband 过滤；
  - 当第 `k+1` 条边 `|ΔR| >= 45Ω` 且 `k < 3` 时，允许 REG 抢救回一条高置信边。
- 已完成本地实测：
  - 组合：`GNN_CLS/modelo3 + GNN_REG/o5b`
  - 输出：`gnn/GNN_FULL/Mv1/outputs/inference_v2_o5b_eval/o5b_10mA/`
  - 结果：
    - `CMEI=91.38`
    - `raw_cls_num_accuracy=0.8850`
    - `num_accuracy=0.8400`
    - `macro_f1=0.8587`
    - `id_recall=0.9019`
    - `mse_all_edges=56.7826`
- 结论：
  - 新版规则没有把 `o5b` 救回来，反而进一步拉低了联合推理表现；
  - 问题不在“相邻边保护”本身，而在于当前 deadband + rescue 组合过于刚性，已经开始过度改写 CLS 的原始计数；
  - 因此 `inference_v2.py` 暂时只作为实验文件保留，不替换当前默认的统一 GNN 推理入口。

## 2026-03-31 v65
- 已新增 `GNN_REG/o5b1`
  - 来源：复制 `o5b`
  - 唯一改动：`mask_bce_weight` 从 `25.0` 下调到 `20.0`
- 设计目的：
  - 温和放松 `mask` 监督强度；
  - 在不直接跳到 `18` 的前提下，先看是否能缓解 `o5b` 仍然存在的轻微保守倾向。
- `GNN_FULL/Mv1/inference_v2.py` 已按最新判断继续简化：
  - 删除了 `35Ω deadband` 与 `45Ω rescue` 规则；
  - 只保留“高置信相邻边保护”版 near-miss。
- 简化后再次实测组合：
  - `GNN_CLS/modelo3 + GNN_REG/o5b`
  - 输出目录：`gnn/GNN_FULL/Mv1/outputs/inference_v2_guard_only_o5b_eval/o5b_10mA/`
  - 结果：
    - `CMEI=93.17`
    - `raw_cls_num_accuracy=0.8850`
    - `num_accuracy=0.8850`
    - `macro_f1=0.9075`
    - `id_recall=0.9115`
    - `mse_all_edges=57.6391`
- 对比前一版 `inference_v2`（含物理死区/补漏）：
  - `93.17 > 91.38`
  - 说明此前的明显退化主要是物理死区 + rescue 规则过硬造成的
- 对比原始统一 GNN 推理中同一组模型：
  - 旧结果：`CMEI=93.20`
  - 新 guard-only 结果：`CMEI=93.17`
  - 说明单独保留“高置信相邻边保护”后，整体已基本回到原始水平，但仍未形成明确增益。
- 当前判断：
  - `inference_v2.py` 可保留为更合理的实验版；
  - 但默认联合推理入口仍不应切换；
  - 下一步更值得看的仍是 `o5b1` 训练本身，而不是继续堆后处理规则。

## 2026-03-31 v66
- `GNN_FULL/Mv1` 路径问题已做成正式兼容修复：
  - `gnn/GNN_FULL/Mv1/inference.py`
  - `gnn/GNN_FULL/Mv1/inference_v2.py`
- 本次修复不仅消除了根目录传参时重复拼出的 `gnn/GNN_FULL/Mv1/gnn/GNN_FULL/Mv1/...`，还额外补上了：
  - 当 `Mv1/cache/<dataset_tag>/cache_dataset_cls_graphattn.npz` 或 `cache_dataset_reg_graphattn.npz` 缺失时，自动按 `train_cls.py / train_reg.py` 的同口径数据流程重建 cache
  - 因而现在即使只下载了 `outputs/` 与原始 CSV，也可以直接复跑 `Mv1` 联合推理
- 已用 `Mv1` 自己训练得到的 `cls/reg` 权重在本地完成一次正式复跑：
  - 输出目录：`gnn/GNN_FULL/Mv1/outputs/inference/training_data64Nodes_2/`
  - `CMEI=93.11`
  - `num_accuracy=0.8740`
  - `macro_f1=0.8975`
  - `id_recall=0.9173`
  - `mse_all_edges=53.3761`
- `Mv1` 当前定位更新为：
  - 已不只是“壳层可运行”
  - 而是“在本地缺 cache 条件下也能独立复跑的正式联合实验目录”
  - 但它的当前联合结果仍弱于默认 `modelo3 + o4a2` 组合的 `CMEI=93.73`
- 已按根目录 `Noise_test.txt` 完成步骤 A（`20dB`，即 standardized voltage noise `std=0.1`，仅测试集加噪）：
  - `GNN_CLS/modelo3`：`macro_f1 0.9075 -> 0.1203`
  - `GNN_REG/o4a2`：`mae_all 0.4854 -> 23.2065`，`mae_changed 24.2925 -> 70.3696`，`count_macro_f1 0.1203`
  - `modelo3 + o4a2`：`CMEI 93.73 -> 41.79`，`num_accuracy 0.8850 -> 0.3170`，`id_recall 0.9248 -> 0.2608`，`mse_all_edges 49.7686 -> 1735.1177`
- 噪声现象判断：
  - 三条链路在这组 zero-shot 噪声下都出现了明显塌缩；
  - `CLS` 与 `REG` 的派生计数混淆矩阵都几乎退化为“全部判为 3”
  - 这说明当前最佳 clean 模型虽然在无噪声 10mA 上已达到较好结果，但物理泛化到 `20dB` 噪声场景仍明显不足
- 当前后续建议：
  - 默认 clean 主线不变，继续保持 `GNN_CLS/modelo3 + GNN_REG/o4a2`
  - `Noise_test` 的步骤 B（训练期噪声增强）应进入高优先级
  - 若后续要对外汇报鲁棒性，这一版步骤 A 结果已经足够作为“zero-shot robustness baseline”

## 2026-03-31 v67
- 根目录已新增专门的首轮噪声诊断文件：
  - `首轮20dB噪声诊断记录.md`（现已迁移到 `gnn/GNN_NOISE/首轮20dB噪声诊断记录.md`）
- 该文件沉淀了三类内容：
  - `modelo3 / o4a2 / modelo3+o4a2` 的 clean vs noise 核心数值
  - 对 zero-shot `20dB` 崩塌现象的统一诊断
  - 为什么这不是局部架构 bug，而是 EIT 逆问题病态性的直接体现
- 同时已在 `gnn/` 下新建 `GNN_NOISE` 子目录，正式开启策略 B：
  - `gnn/GNN_NOISE/CLS_modelo3_ft`
  - `gnn/GNN_NOISE/REG_o4a2_ft`
  - `gnn/GNN_NOISE/README.md`
  - `gnn/GNN_NOISE/Log.md`
- `GNN_NOISE` 首版实现原则：
  - 不从头训练，而是默认 warm start 到 clean 最优权重
  - 用较小学习率 `5e-5` 做 `30 epoch` 左右微调
  - 仅训练集做动态随机噪声注入，验证/测试仍保持 clean
- `CLS_modelo3_ft/train.py` 已落地的关键改动：
  - 新增 `--pretrained-model-path`
  - 新增 `--add-noise / --no-add-noise`
  - 新增 `--noise-mode {gaussian,uniform}`
  - 新增 `--noise-std-max`
  - 训练集 `__getitem__` 改为每次动态采样噪声强度
- `REG_o4a2_ft/train.py` 已落地的关键改动：
  - 同样新增 warm start 与动态随机噪声增强入口
  - 保留 `o4a2` 原始的 `SmoothL1 + mask BCE + mask L1 warmup` 主损失逻辑
- 当前状态：
  - 代码已完成
  - 还未开始云端真实 fine-tune
  - 下一步最关键的是拿这两条带噪微调线去复跑 `20dB` 评估，检查 `CMEI / macro_f1 / id_recall / mse_all_edges` 是否实质恢复

## 2026-04-01 v68
- 已核对根目录原 `Noise_test.txt` 的步骤 B 与当前 `GNN_NOISE` 默认方案：
  - 结论：不完全相同
  - 当前默认 `GNN_NOISE` 是 `warm start + random noise + boundary-only`
  - 原始步骤 B 更接近 `fixed 20dB gaussian + all-voltage-channel`
- 为避免根目录继续保留分散说明，已把原始步骤 B 收编到：
  - `gnn/GNN_NOISE/原始步骤B_fixed20dB.md`
- 同时补齐代码入口：
  - `CLS_modelo3_ft/train.py` 与 `REG_o4a2_ft/train.py` 新增 `--noise-schedule`
  - 新增 `--fixed-noise-std`
  - 新增 `--noise-scope`
- 根目录旧文件 `Noise_test.txt` 已删除。
- 后续无论跑增强版还是原始 fixed-20dB 版，都统一从 `gnn/GNN_NOISE` 管理。

## 2026-04-01 v69
- 已读取根目录 `o5b1训练记录.txt`，并与以下正式输出交叉核对：
  - `gnn/GNN_REG/o5b1/outputs/o5b1_10mA/metrics.json`
  - `gnn/GNN_REG/o5b/outputs/o5b_10mA/metrics.json`
  - `gnn/GNN_REG/o4a2/outputs/training_data64Nodes_2/metrics.json`
- `o5b1` 正式指标：
  - `mae_all=0.5097`
  - `mae_changed=25.5355`
  - `val_count_macro_f1=0.8422`
  - `avg(|dR|>40)=1.73`
  - `avg(mask_prob)=0.0213`
- 对比 `o5b`：
  - `mae_all: 0.5119 -> 0.5097`
  - `mae_changed: 25.4532 -> 25.5355`
  - `val_count_macro_f1: 0.8564 -> 0.8422`
  - `avg(mask_prob): 0.0207 -> 0.0213`
- 因此本轮更准确的判断是：
  - 下调 `mask_bce_weight` 没有造成假阳性失控，说明 `o5b` 系列本身具有较强稳定性；
  - 但 `o5b1` 没有超越 `o5b`，更没有超过当前最佳 `o4a2`
  - 所以这次试探不足以支持“已严格撞到信息论上限”的强结论
- 当前更稳妥的说法：
  - 在当前 clean 观测设置下，`o4a2 / o5b / o5b1` 很可能已经进入经验平台区；
  - 继续做 clean-only 小幅 loss 超参微调，预期收益较低；
  - 下一步优先级仍应放在鲁棒性与带噪训练，而不是继续抠 clean 极限小数点。

## 2026-04-01 v70
- 已完成根目录实验文件清理与归档：
  - `首轮20dB噪声诊断记录.md` 已迁移到 `gnn/GNN_NOISE/首轮20dB噪声诊断记录.md`
  - `o5b1训练记录.txt` 已删除
  - `Mv1训练记录.txt` 已删除
  - `Noise_test.txt` 已删除
- 删除依据：
  - `o5b1训练记录.txt` 的关键指标与判断已吸收进 `README.md / Log.md / gnn/README.md / gnn/Log.md`
  - `Mv1训练记录.txt` 的训练结果、路径问题与修复结论已吸收进根目录与 `gnn/` 日志
  - `Noise_test.txt` 的步骤 A/B 已分别沉淀为：
    - `gnn/GNN_NOISE/首轮20dB噪声诊断记录.md`
    - `gnn/GNN_NOISE/原始步骤B_fixed20dB.md`
- 当前统一约定：
  - 噪声实验说明统一放在 `gnn/GNN_NOISE`
  - 一次性训练草稿不再在根目录长期保留

## 2026-04-01 v71
- 已吸收根目录 `0401训练记录.txt`。

### clean 主线复训
- `modelo3`：
  - `dataset_tag=training_data64Nodes_2_clean_20260401`
  - `test_macro_f1=0.9027`
- `o4a2`：
  - `dataset_tag=training_data64Nodes_2_clean_20260401`
  - `mae_all=0.4679`
  - `mae_changed=23.5724`
  - `val_count_macro_f1=0.8628`
- 联合结果：
  - `CMEI=93.53`
  - `num_accuracy=0.8800`
  - `macro_f1=0.9027`
  - `id_recall=0.9237`
  - `mse_all_edges=53.2930`
- 判断：
  - clean 复训再次证明主线稳定；
  - 但没有超过历史最好 `93.73`

### `noiseft_rand_boundary_20260401`
- `REG_o4a2_ft` 已完成：
  - `mae_all=0.5900`
  - `mae_changed=27.5311`
  - `count_macro_f1=0.8140`
- `CLS_modelo3_ft` 对应输出目录只有 `standardization.npz`
  - 没有 `model_last.pt`
  - 因此后续 inference 报错本质上是“权重文件不存在”
- 说明：
  - 这条分支实际没有形成完整的 `CLS + REG` 成对结果；
  - 更像是云端执行时只把 `REG rand_boundary` 训练完成了

### `noiseft_fixed20db_all_20260401`
- `CLS_modelo3_ft` 在同一 tag 下被连续训练两次：
  - 先得到 `test_macro_f1=0.7439`
  - 后一次覆盖为 `test_macro_f1=0.7275`
- 当前保留下来的最终结果应以目录内最后落盘的 `0.7275` 为准。
- `REG_o4a2_ft` 最终结果：
  - `mae_all=0.9201`
  - `mae_changed=48.8259`
  - `count_macro_f1=0.6178`
- clean 联合结果：
  - `CMEI=83.39`
- `20dB` 单模型带噪结果：
  - `CLS macro_f1=0.7121`
  - `REG mae_all=1.1800`
  - `REG mae_changed=54.2884`
  - `REG count_macro_f1=0.5829`
- 判断：
  - 固定 20dB 全图加噪确实恢复了部分鲁棒性；
  - 但 clean 性能牺牲过大；
  - 这条线可保留为强增强对照，不适合直接转正为默认方案

### 工程问题解释
- `GNN_NOISE/*/inference.py` 是单模型评估入口；
- `gnn/inference_gnn_cmei.py` 是最终联合 CMEI 评估入口；
- `只有 standardization.npz 没有 model_last.pt` 的目录，代表训练流程没有完整结束；
- 云端 `gnn/inference_gnn_cmei.py` 对 `--noise-std/--noise-seed` 报错，说明当时云端脚本版本落后于当前本地版本。

## 2026-04-01 v72
- 已重做中期汇报数据可视化，并按用户要求删除旧说明与临时程序：
  - 删除 `midterm_assets/20260401_data_figures`
  - 删除 `中期汇报_数据可视化说明.md`
  - 删除 `tools/generate_midterm_figures.py`
- 新图输出目录：
  - `midterm_assets/20260401_visuals`
- 新图文件：
  - `01_topology_boundary_nodes.svg`
  - `02_dataset_composition.svg`
  - `03_changed_edge_frequency.svg`
  - `04_boundary_response_heatmaps.svg`
- 当前云端同步与补训判断：
  - 若云端 `gnn/inference_gnn_cmei.py` 仍不支持 `--noise-std/--noise-seed`，需要先同步当前本地最新版
  - 当前真正缺失的训练闭环仅为 `CLS_modelo3_ft` 的 `training_data64Nodes_2_noiseft_rand_boundary_20260401`
  - 其它 `0401` 已落盘结果可直接用于后续推理与汇报

## 2026-04-01 v73
- 已吸收根目录 `0401补充训练.txt`。
- 本次补充训练实质上完成了推荐噪声线 `noiseft_rand_boundary_20260401` 的分类侧闭环：
  - `CLS clean test_macro_f1=0.8750`
  - `CLS noisy(20dB) test_macro_f1=0.7780`
  - noisy joint `CMEI=82.56`
  - noisy joint `num_accuracy=0.7360`
  - noisy joint `id_recall=0.7579`
  - noisy joint `mse_all_edges=154.4499`
- 本地 outputs 校验结果：
  - `CLS rand_boundary` 目录已正确包含 `model_last.pt / metrics.json / noise_eval.json`
  - `REG rand_boundary` 目录当前未看到 `noise_eval.json`
  - `gnn/outputs` 当前未看到带噪训练对应的 joint 输出目录
- 当前更准确的工程判断：
  - 本地单模型输出基本正确
  - 但联合输出下载不完整，建议后续把 `gnn_cmei_noiseft_*` 目录补同步到本地

## 2026-04-01 v74
- 已删除 `0401补充训练.txt`
  - 删除依据：其内容已吸收进正式文档
- 已新增 `gnn/GNN_NOISE/plot_noise_robustness.py`
  - 用于把不同方法在 `20/30/40dB` 下的指标画到同一张鲁棒性对比图中
  - 输出格式为 `svg`

## 2026-04-01 v75
- 已新增 `gnn/GNN_CLS/modelo3/two_stage_threshold_search.py`
  - 不改原训练代码
  - 支持对 `modelo3` 兼容分类器做“两阶段细阈值搜索”
- 已在本地对 `rand_boundary` 主线分类器实跑验证：
  - 原阈值：`[0.05, 0.17, 0.37]`
  - 细化后：`[0.05, 0.164, 0.368]`
  - `val_macro_f1` 持平：`0.8976 -> 0.8976`
  - `test_macro_f1` 轻微回落：`0.8750 -> 0.8749`
- 结论：
  - 当前 `CLS` 的阈值步长不是主要瓶颈
  - 正式主线仍应聚焦 `rand_boundary` 的鲁棒性增强，而不是继续深挖阈值小数点

## 2026-04-02 v76
- 已吸收根目录 `0402补充日志.txt`
- 当前 `20dB` 补推理与 joint 输出已完整跑通：
  - `rand_boundary` clean joint：`CMEI=91.01`
  - `rand_boundary` 20dB joint：`CMEI=82.56`
  - `fixed20db_all` clean joint：`CMEI=83.39`
  - `fixed20db_all` 20dB joint：`CMEI=81.79`
- 额外补齐：
  - `REG rand_boundary` 20dB noisy 单模型：`mae_all=1.2692`，`mae_changed=54.1729`，`count_macro_f1=0.5844`
- 结论更新：
  - `rand_boundary` 继续保持为正式主线
  - 目前剩余最有价值的工作只剩 `30/40dB` 鲁棒性曲线扩展

## 2026-04-02 v77
- 已吸收根目录 `0402大范围噪声训练.txt`
- 已生成最终鲁棒性曲线图：
  - `gnn/GNN_NOISE/rand_boundary_robustness_curve.svg`
- 已新增作图脚本：
  - `gnn/GNN_NOISE/plot_rand_boundary_robustness.py`
- `rand_boundary` joint 曲线结果：
  - clean `CMEI=91.01`
  - `40dB CMEI=90.83`
  - `30dB CMEI=89.62`
  - `20dB CMEI=82.56`
- 最终判断：
  - `40/30dB` 下性能下降平缓，鲁棒性已基本建立
  - `20dB` 下虽有明显衰减，但仍远优于 zero-shot baseline
  - 正式主线继续固定为 `rand_boundary`
- 已删除：
  - `0402补充日志.txt`
  - `0402大范围噪声训练.txt`

## 2026-04-02 v78
- 已进行联合推理目录收敛：
  - `gnn/inference_gnn_cmei.py -> gnn/GNN_CMEI_INFERENCE/inference_gnn_cmei.py`
  - `gnn/outputs -> gnn/GNN_CMEI_INFERENCE/outputs`
- 已新增：
  - `gnn/GNN_CMEI_INFERENCE/inference_gnn_cmei_v2.py`
  - `gnn/GNN_CMEI_INFERENCE/README.md`
  - `gnn/GNN_CMEI_INFERENCE/Log.md`
- `v2` 吸收自 `GNN_联合优化.txt` 的有效推理层建议：
  - `near-miss` 高置信保护
  - `REG` 数量证据仲裁 / 动态 `K`
- 未纳入 `v2` 的条目：
  - `Absolute Edge Embedding`
  - `Relaxed Sparsity Loss`
  - `Focal-CORAL`
  - `Pseudo-Edge Pooling`
- 原因：
  - 它们属于训练或结构层，不是纯推理层迁移项
- 已删除：
  - `GNN_联合优化.txt`
- 已补跑 `v2` 本地验证：
  - `v1 clean CMEI=91.01`
  - `v2(guard_only) clean CMEI=90.85`
  - `v2(full arbitration) clean CMEI=90.08`
  - `v1 20dB CMEI=82.56`
  - `v2(guard_only) 20dB CMEI=82.40`
  - `v2(full arbitration) 20dB CMEI=79.40`
- 判断：
  - `v2` 当前没有超过 `v1`
  - `REG` 数量仲裁默认打开会过度降 `K`
  - 因此已把 `v2` 调整为更保守的实验入口，但不作为正式主线

## 2026-04-02 v79
- 已新建下一版带噪训练分支：
  - `gnn/GNN_NOISE/CLS_modelo3_ft_v2`
  - `gnn/GNN_NOISE/REG_o4a2_ft_v2`
- 这一版继续遵循目录收敛原则：
  - 单模型 outputs 留在 `GNN_NOISE/*_v2/outputs`
  - joint outputs 留在 `GNN_CMEI_INFERENCE/outputs`
- `v2` 的核心变化不在模型结构，而在噪声建模：
  - `noise_schedule=curriculum`
  - `noise_mode=structured`
  - `noise_scope=boundary`
  - 引入 `clean_mix_prob`
  - 引入结构化边界噪声的 `iid / drift / common / bad electrode`
- warm start 默认来源：
  - `v1 rand_boundary`
- 已完成本地校验：
  - `py_compile`
  - `--help`

## 2026-04-02 v80
- 已正式建立 `gnn/GNN_EXPAND` 拓扑与规模扩展容器。
- 约束原则：
  - 不修改原有 clean 主线程序
  - 所有扩展代码仅落在 `GNN_EXPAND` 内部
  - 每个阶段都在自己的 `cls/reg/joint_inference` 下闭环保存 `cache/outputs`
- 已落地四个阶段：
  - `stage1_square_10x10`
  - `stage2_rect_6x10`
  - `stage3_honeycomb_63`
  - `stage4_transfer_circlecut_69`
- 已补齐共享抽象层：
  - 拓扑注册表
  - 通用 `CLS/REG/joint` 训练与推理脚本
  - 阶段包装入口
- 为兼容“继续使用原始 clean 数据”的要求，新增两层适配：
  - 28 个边界电压通道按电极顺序映射到目标拓扑边界节点
  - 原始 8x8 `r*_id` 按归一化边中点位置与方向映射到目标拓扑电阻边
- `stage4` 的默认 warm start 已切到 `stage1_square_10x10`，用于承接 transfer 设定。
- 新增文档：
  - `gnn/GNN_EXPAND/README.md`
  - `gnn/GNN_EXPAND/Log.md`
  - 各阶段 `README.md / Log.md`
- `模型扩展路径.txt` 的内容已完成正式吸收，待本地验证后删除。

## 2026-04-02 v81
- 已在 `gnn/GNN_EXPAND` 内补充通用拓扑数据生成器：
  - `gnn/GNN_EXPAND/generate_expand_datasets.py`
- 该脚本直接在目标拓扑上做正演，不再依赖把 `8x8` 根目录数据重映射成伪标签数据。
- 已生成四套原生数据到：
  - `gnn/GNN_EXPAND/data/square_10x10.csv`
  - `gnn/GNN_EXPAND/data/rect_6x10.csv`
  - `gnn/GNN_EXPAND/data/honeycomb_63.csv`
  - `gnn/GNN_EXPAND/data/circlecut_69.csv`
- 每套数据都附带对应 meta 文件，并已切换为各阶段默认输入数据。
- 本轮已重点确认：
  - 激励只使用外部节点
  - 测量只输出外部节点电压
  - 四套数据均为 `28` 个外部节点、`32` 组激励
- `GNN_EXPAND/common` 仍保留旧的 8x8 兼容重映射逻辑，但当前默认路径已优先使用原生拓扑数据。

## 2026-04-02 v82
- 更正 `GNN_EXPAND` 原生数据口径：
  - 先前写成“四套数据均为 `28` 个外部节点、`32` 组激励”不正确
  - `EXPAND` 生成器始终遵守“激励只使用外部节点、测量只输出外部节点电压”
  - 但外部节点数与激励组数应由目标拓扑自己的边界规模决定
- 当前已核对 `gnn/GNN_EXPAND/data/*_meta.json`，真实结果为：
  - `square_10x10`: `36` 个外部节点，`40` 组激励
  - `rect_6x10`: `28` 个外部节点，`32` 组激励
  - `honeycomb_63`: `28` 个外部节点，`32` 组激励
  - `circlecut_69`: `24` 个外部节点，`28` 组激励
- 其中 `square_10x10` 对应 `10x10` 方格外边界，边界节点数为 `2*(10+10)-4 = 36`，因此不是 `28/32`。

## 2026-04-02 v83
- 已升级根目录 `DOC_RULES.md`，使其从单纯的文档治理规则文件，变为新窗口优先读取的项目接手文件。
- 本次升级统一固化了三层内容：
  - 文档、日志管理规则
  - 必须遵守的项目原则与禁忌
  - 当前正式主线、当前最佳路线、当前重点工作
- 当前在 `DOC_RULES.md` 中明确写死的项目关键口径包括：
  - 数据生成时激励只使用外部节点
  - 数据生成时测量只使用外部节点
  - 当前主要开发工作都在 `gnn`
  - clean 正式主线为 `GNN_CLS/modelo3 + GNN_REG/o4a2 + inference_gnn_cmei.py`
  - noisy 正式主线为 `rand_boundary`
  - 当前重点工作为 `GNN_NOISE v2` 与 `GNN_EXPAND`
- 后续若“当前正式路线 / 当前最佳模型 / 当前重点工作”发生变化，除更新常规 `README.md / Log.md` 外，还必须同步更新 `DOC_RULES.md`。

## 2026-04-02 v84
- 根目录长期规则主文件已正式更名为 `RULES.md`。
- 为兼容旧文档引用，新增保留：
  - `DOC_RULES.md`
  - 其作用仅为历史兼容入口
- 本轮新增并写死到 `RULES.md` 的长期规则包括：
  - 修改代码前，必须先向用户呈现修改思路，不能直接先改
  - 每一版模型更新原则上都应基于复制后的新目录推进，不直接改原模型
  - `README.md / Log.md` 原则上不删旧内容，只追加新内容或更正说明
- 后续新窗口接手项目时，默认优先读取：
  - `RULES.md`
  - 然后再读根目录和 `gnn` 的 `README.md / Log.md`

## 2026-04-02 v85
- 已在根目录初始化 Git 仓库，用于保护当前项目代码、文档、规则文件与主线路线。
- 当前采用 Git 轻量方案 A：
  - Git 跟踪源码、脚本、文档、规则、元数据
  - Git 不跟踪 `outputs / cache / *.pt / *.pth / *.npz / 可重建 csv / 本地依赖目录`
- 本轮新增文件：
  - `.gitignore`
  - `CURRENT_BEST.md`
  - `DOC_RULES.md` 保留为兼容入口，正式规则主文件为 `RULES.md`
- 其中：
  - `CURRENT_BEST.md` 用于记录当前最佳 clean/noisy/joint/expand 路线与本地产物锚点
  - `RULES.md` 已新增 Git 规则，明确当前仓库的跟踪范围与更新时机
- 这意味着后续可以通过 Git 稳定保存：
  - 当前正式工程结构
  - 当前规则文件
  - 当前最佳路线说明
  - 各轮代码与文档演进历史

## 2026-04-02 v86
- 已补充项目 Git 操作规范，用于固定“实验分支如何开、当前最佳版本如何升级与打 tag”。
- 当前正式规则已写入：
  - `RULES.md`
- 固定口径如下：
  - `main` 只保存稳定、可回退、可作为正式路线的状态
  - 日常实验默认不直接在 `main` 上推进
  - 新实验默认新建 `codex/` 前缀分支
  - 推荐分支命名格式为 `codex/<模块>-<版本>-<目的>`
- 当某次实验准备升级为“当前最佳版本”时，要求同步完成：
  - 稳定代码进入 `main`
  - 更新 `CURRENT_BEST.md`
  - 追加更新根目录与对应子目录 `README.md / Log.md`
  - 新增可读 Git tag 作为恢复锚点
- 推荐 tag 命名格式固定为：
  - `best-<模块>-<版本>-<日期>`
- 同时明确：
  - tag 只新增，不覆盖旧 tag
  - 未形成正式结论的版本停留在实验分支，不进入 `main`

## 2026-04-03 v87
- 已读取并吸收根目录 `0402噪声v2训练日志.txt`，并按正式文档口径提炼记录，未直接转抄整段终端日志。
- `GNN_NOISE v2` 本轮云端已确认结果：
  - clean `CLS test_macro_f1=0.9149`
  - clean `REG mae_all=0.4664`，`mae_changed=24.2457`，`count_macro_f1=0.8349`
  - clean joint `CMEI=93.49`
  - `40dB CLS test_macro_f1=0.9078`
  - `40dB REG mae_all=0.5317`，`mae_changed=25.3754`，`count_macro_f1=0.8342`
- `30dB / 20dB` 本轮未产出有效结果，已定位为云端命令中的 `--dataset-tag ${TAG}` 未展开，属于验证命令链问题，不是模型结构错误。
- 为避免重复出错，已新增：
  - `gnn/GNN_NOISE/run_noise_eval_suite.py`
- 该脚本负责：
  - 统一串行执行 clean / `40dB` / `30dB` / `20dB` 的 `CLS / REG / joint` 评估
  - 自动按噪声等级保存单模型评估 `json`
  - 支持 `--dry-run` 先打印完整命令
- 同时已在以下入口补充 `dataset-tag` 空值与未展开占位符的明确报错：
  - `gnn/GNN_NOISE/CLS_modelo3_ft_v2/train.py`
  - `gnn/GNN_NOISE/CLS_modelo3_ft_v2/inference.py`
  - `gnn/GNN_NOISE/REG_o4a2_ft_v2/train.py`
  - `gnn/GNN_NOISE/REG_o4a2_ft_v2/inference.py`
  - `gnn/GNN_CMEI_INFERENCE/inference_gnn_cmei.py`

## 2026-04-03 v88
- 已吸收根目录 `拓展训练日志.txt`，并与本地 `GNN_EXPAND` 各阶段真实输出文件交叉核对。
- 当前四阶段结果为：
  - `stage1_square_10x10`: `CLS macro_f1=0.8581`，`REG mae_changed=37.0220`，`joint CMEI=89.30`
  - `stage2_rect_6x10`: `CLS macro_f1=0.9018`，`REG mae_changed=16.9012`，`joint CMEI=94.38`
  - `stage3_honeycomb_63`: `CLS macro_f1=0.8671`，`REG mae_changed=31.3267`，`joint CMEI=91.05`
  - `stage4_transfer_circlecut_69`: `CLS macro_f1=0.8818`，`REG mae_changed=43.2324`，`joint CMEI=88.42`
- 当前判断：
  - `stage2_rect_6x10` 是本轮扩展表现最好的阶段
  - `stage4_circlecut_69` 的主要瓶颈在回归与联合推理，不规则拓扑仍最难
- 已发现并修正一处真实代码问题：
  - `stage4 transfer` 的默认 warm start 路径原先错误指向 `training_data64Nodes_2`
  - 现已改为正确的 `stage1_square_10x10/.../outputs/square_10x10/model_last.pt`
- 因此本轮已记录的 `stage4` 结果不能直接当作最终 transfer 结论，而应视为当前不规则拓扑基线。
- 为便于后续汇报与比较，已新增：
  - `gnn/GNN_EXPAND/plot_expand_summary.py`
  - `gnn/GNN_EXPAND/expand_summary_metrics.json`
  - `gnn/GNN_EXPAND/expand_summary.svg`
- v89
  - 将 `GNN_EXPAND` 汇总图从手写 `svg` 版切换为 `matplotlib` 正式版。
  - 新增正式输出：
    - `gnn/GNN_EXPAND/expand_summary.png`
    - `gnn/GNN_EXPAND/expand_summary.pdf`
  - 保留 `expand_summary_metrics.json` 作为配套汇总底表。
  - 删除旧 `gnn/GNN_EXPAND/expand_summary.svg`。
- v90
  - `GNN_EXPAND` 新增 `Figure` 目录，统一承接汇总图与拓扑示意图。
  - 汇总图位置调整为：
    - `gnn/GNN_EXPAND/Figure/expand_summary.png`
    - `gnn/GNN_EXPAND/Figure/expand_summary.pdf`
  - 新增四张拓扑结构图：
    - `topology_square_10x10.png`
    - `topology_rect_6x10.png`
    - `topology_honeycomb_63.png`
    - `topology_circlecut_69.png`

## 2026-04-04 v91
- 已读取并吸收根目录 `0404训练日志.txt`，但正式记录继续以本地真实 `outputs/*.json` 为准，不直接转抄原始终端日志。

### `GNN_NOISE v2` 完整曲线补齐
- 这次真正补齐了 `clean / 40dB / 30dB / 20dB` 全部 `CLS / REG / joint` 结果：
  - clean：`CLS macro_f1=0.9149`，`REG mae_changed=24.2457`，`joint CMEI=93.49`
  - `40dB`：`CLS macro_f1=0.9078`，`REG mae_changed=25.3754`，`joint CMEI=92.81`
  - `30dB`：`CLS macro_f1=0.8903`，`REG mae_changed=34.0454`，`joint CMEI=90.44`
  - `20dB`：`CLS macro_f1=0.7582`，`REG mae_changed=58.1169`，`joint CMEI=80.42`
- 与上一轮 `rand_boundary` 相比：
  - clean：`91.01 -> 93.49`
  - `40dB`：`90.83 -> 92.81`
  - `30dB`：`89.62 -> 90.44`
  - `20dB`：`82.56 -> 80.42`
- 当前更准确的判断是：
  - `structured boundary v2` 已经把优势稳定推进到 clean 与中等噪声区间
  - 但在最重的 `20dB` 上，它还没有超过 `rand_boundary`
  - 因此 noisy 路线当前不再适合被简单概括为“单条全区间绝对最优”，而更像是“`v2` 主打 clean~中噪声，`rand_boundary` 保留最强 `20dB` 端点”
- 已新增并重做正式汇总图：
  - `gnn/GNN_NOISE/plot_noise_v2_summary.py`
  - `gnn/GNN_NOISE/noise_v2_summary_metrics.json`
  - `gnn/GNN_NOISE/Figure/noise_v2_summary.png`
  - `gnn/GNN_NOISE/Figure/noise_v2_summary.pdf`
- 已删除旧噪声图：
  - `gnn/GNN_NOISE/rand_boundary_robustness_curve.svg`

### `GNN_EXPAND` 真正 transfer 结果补齐
- 本轮 `stage4_transfer_circlecut_69` 已按修正后的默认路径成功加载 `stage1` 权重：
  - `CLS warm_start.loaded=36`
  - `REG warm_start.loaded=36`
- 最新真实结果为：
  - `CLS macro_f1=0.8928`
  - `REG mae_changed=36.1173`
  - `joint CMEI=91.14`
- 相比 2026-04-03 记录的未成功 transfer 基线：
  - `macro_f1: 0.8818 -> 0.8928`
  - `mae_changed: 43.2324 -> 36.1173`
  - `CMEI: 88.42 -> 91.14`
- 当前扩展线判断修正为：
  - `stage2_rect_6x10` 仍是整体最强阶段
  - 但 `stage4_circlecut_69` 在真实 transfer 下已经回到可用区间，不再应被描述为“明显失败”
  - 当前最主要难点从“transfer 是否生效”转为“不规则拓扑下怎样进一步压低回归误差”
- `EXPAND` 汇总图与拓扑图已按最新数据重生成：
  - `gnn/GNN_EXPAND/plot_expand_summary.py`
  - `gnn/GNN_EXPAND/plot_expand_topologies.py`
  - `gnn/GNN_EXPAND/expand_summary_metrics.json`
  - `gnn/GNN_EXPAND/Figure/expand_summary.png`
  - `gnn/GNN_EXPAND/Figure/expand_summary.pdf`
  - `gnn/GNN_EXPAND/Figure/topology_square_10x10.png`
  - `gnn/GNN_EXPAND/Figure/topology_rect_6x10.png`
  - `gnn/GNN_EXPAND/Figure/topology_honeycomb_63.png`
  - `gnn/GNN_EXPAND/Figure/topology_circlecut_69.png`
- 旧版 `EXPAND` 同名图片已直接被最新版本覆盖。

### 文件清理
- `0404训练日志.txt` 的内容已吸收到正式文档，现已删除。

## 2026-04-04 v92
- 已在 `main` 上创建当前“初级阶段收敛”快照提交：
  - commit: `0763917`
  - message: `Finalize initial GNN convergence snapshot`
- 同时新增恢复锚点 tag：
  - `best-project-2026-04-04`
- 这次 Git 快照对应的工程含义是：
  - clean 主线已经稳定收敛到 `modelo3 + o4a2 + inference_gnn_cmei.py`
  - noisy 路线已经明确为“双锚点”格局：`v2` 主打 clean~中噪声，`rand_boundary` 保留最强 `20dB`
  - expand 路线已经完成四阶段首轮闭环，并且 `stage4` 真正 transfer 已验证有效
- 因此当前可以把这次版本视为：
  - 64Nodes 项目“初级阶段结束 / 下一阶段起点”的正式恢复点
