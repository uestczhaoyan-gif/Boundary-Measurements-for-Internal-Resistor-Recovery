# GNN 子项目说明（64Nodes）

说明：
- 本文件记录 GNN 系列模型的架构与版本定义。
- 训练结果与问题分析请看：`64Nodes/gnn/Log.md`。

## 为什么 GNN 可能适合本课题
- 电阻网络天然是图结构，GNN 与物理拓扑一一对应，归纳偏置最贴合问题本质。
- 节点消息传递可模拟“电压/电流在网络中传播”的机制，适合定位内部变化。
- 边解码器直接对电阻边输出，天然对应回归目标（112 电阻）。
- 便于后续注入物理一致性约束（KCL残差、等效导纳约束等）。

## 一、目录结构
- `GNN_CLS/modelo1`
- `GNN_REG/modelo1`
- `GNN_FULL/modelo1_h_multitask`

每个版本包含：
- `model/model.py`
- `train.py`
- `inference.py`
- `outputs/`（训练后生成）

## 二、输入与图结构
- 数据：`64Nodes/data/training_data64.csv`
- 可切换数据：`64Nodes/data/training_data64Nodes_2.csv`（10mA 激励版）
- 可切换数据：`64Nodes/data/training_data64Nodes_3.csv`（20mA 激励版）
- 当前非 fixed-change GNN 主线默认数据：`64Nodes/data/training_data64Nodes_2.csv`（10mA）
- 训练脚本输入仍是 97 通道网格，模型内部转换为 64 个节点 token。
- 图邻接：`8x8` 四邻域 + 自环，使用归一化邻接做消息传递。
- 电阻边解码顺序：与数据生成规则一致（行内横向后纵向，112边）。
- 当前主线训练/推理入口支持命令行切换：
- `--data-path`：切换主数据 CSV
- `--dataset-tag`：指定数据集标签；默认取 CSV 文件名
- 默认会把 cache 写到 `cache/<dataset_tag>/`，输出写到 `outputs/<dataset_tag>/`
- 当前已支持该机制的版本：
- `GNN_CLS/modelo1`
- `GNN_REG/modelo1`
- `GNN_FULL/modelo1_h_multitask`

## 三、基准架构（modelo1）
- 图主干：GraphSAGE 风格层（`hidden=128, depth=5, dropout=0.1`）
- CLS：全图池化 -> CORAL 分类头
- REG：节点对拼接 -> 边解码器 -> 112 维 `dR`
- FULL：共享图主干 + 分类头 + 边回归头

## 四、训练建议（2026-03-25）
1. 先跑 `GNN_REG/modelo1`
2. 再跑 `GNN_CLS/modelo1`
- fixed-change 当前冻结在 `MLP/fixed_change_recon` 里做诊断，不作为 GNN 主线当前迭代对象。
- `GNN_FULL/modelo1_h_multitask` 当前不作为主推进对象，等回归线有明确新收益后再补跑验证。

## 五、0324 首轮结论
- `GNN_CLS/modelo1` 在 `5mA / 10mA / 20mA` 上差距不大，`10mA` 的验证 `macro-F1` 略高，但 `2/3` 混淆仍是主问题。
- `GNN_REG/modelo1` 在 `10mA` 上取得当前最优 `mae_changed=41.7682`，比 `5mA` 与 `20mA` 更值得优先继续。
- `GNN_FULL/modelo1_h_multitask` 在 2026-03-24 首轮训练时因回归头属性名不一致报错，已修复训练入口，后续结果需要重新补跑再判断。
- 考虑到当前实验资源希望尽量集中，GNN 主线默认数据统一改为 `10mA`，其余数据集仅在需要对照时显式传入。

## 六、当前主线小步优化（2026-03-24）
- `GNN_REG/modelo1`：
  - 数量阈值搜索改为 `macroF1 + class2_F1` 的加权评分，默认 `count_cls2_weight=0.35`
  - 新增 `L_fp_next`，直接压制真实变化数之后的第一个伪峰，默认 `lambda_fp_next=0.12`
  - early stopping 从纯 `val_loss` 改为更重建导向的 `val_score`
- `GNN_CLS/modelo1`：
  - 阈值搜索改为 weighted score，新增 `bonus_r2=0.05`、`bonus_r3=0.06`
  - 目标是把分类线继续限制在“2/3 边界校准”层面，不把大量资源重新投入大版本改模

## 七、当前状态修正（2026-03-25）
- `GNN_REG/modelo1`：
  - 2026-03-24 新增的 `L_fp_next + class2 加权阈值 + val_score` 组合在实测中明显拖慢了回归主任务；
  - 当前默认已回退到更稳的口径：
    - `lambda_fp_next=0.0`
    - `count_cls2_weight=0.0`
    - early stopping 恢复按 `val_loss`
  - 目的：优先恢复 0324 的更优 `mae_changed`
  - `inference.py` 当前已支持旧输出目录兼容查找，并会同时输出预测/真实变化 `id + delta + 电阻值`
- 当前主精力集中在 `GNN_REG/modelo1` 与 `GNN_CLS/modelo1` 两条线，fixed-change 对照暂不继续扩线。

## 八、2026-03-25 新版实验入口：GNN_REG/modelo2 与 GNN_CLS/modelo2
### 1) GNN_REG / modelo2（Physics-Informed GNN REG）
- 目标：
  - 缓解旧版 97 通道网格输入导致的多激励混杂与图主干过平滑问题。
  - 通过跨激励池化与门控边回归增强稀疏重构能力。
- 输入：
  - 改为 `(Batch, 32, 64, 4)`
  - 每次激励展开为一张 64 节点图
  - 节点特征为 `[源掩码, 地掩码, 电压, 边界掩码]`
- 主干：
  - 3 层残差式 GATv2 风格注意力层
  - 每层都保留残差与归一化，避免过平滑
  - 之后对 32 次激励做 cross-excitation attention pooling，恢复到 `(Batch, 64, Hidden)`
- 边解码器：
  - 对每条电阻边 `(u, v)` 构造 `cat([H_u, H_v, |H_u-H_v|])`
  - 输出 `mask_prob(sigmoid)` 与 `value(tanh)`，最终 `pred = mask_prob * value`
- 损失：
  - `Loss = MSE(pred, true_delta) + lambda_mask_l1 * mean(mask_prob)`
  - 默认 `lambda_mask_l1=0.05`

### 2) GNN_CLS / modelo2（Physics-Informed GNN CLS）
- 目标：
  - 让每次激励先独立传播，再在跨激励层面聚合，减少分类边界被混掉的问题。
  - 用 supervised contrastive loss 强化 `2/3` 类分离。
- 输入与图主干：
  - 与 `GNN_REG/modelo2` 相同，使用 `(Batch, 32, 64, 4)` 的图输入与 3 层残差注意力主干
- 图级聚合：
  - 节点级 cross-excitation pooling 后，再做全图 mean/max 聚合形成图级特征
- 分类头：
  - 主头仍为 CORAL `3 logits`
  - 在分类头前新增 `contrast_proj` 用于 supervised contrastive loss
- 当前说明：
  - `modelo2` 是下一版实验目录，尚未形成训练结论
  - `modelo1` 仍保留为当前已训练过的稳定基线
- 本地验证：
  - 已通过 `py_compile`
  - 当前本地环境缺少 `torch`，尚未完成真实前向冒烟
## 九、2026-03-25 PyG 原生 GATv2Conv 补充说明
- `GNN_REG/modelo2` 与 `GNN_CLS/modelo2` 当前已明确切换到 PyG 原生 `GATv2Conv`。
- 先前用于占位验证的“自实现 GATv2 风格注意力层”不再是当前版本定义。
- 当前本地项目内额外建立了依赖目录：`64Nodes/.vendor_torchpy311`
- `modelo2` 的 `train.py / inference.py / model.py` 已支持自动探测该目录，以便在当前机器的基础 Python 环境缺少 `torch` 时仍可直接运行。
- 已完成最小验证：
  - 原生 `GATv2Conv` import 成功
  - `GNN_REG/modelo2` 前向输出 shape：`(2, 112)`
  - `GNN_CLS/modelo2` 前向输出 shape：`(2, 3)`
## 十、2026-03-25 GNN modelo2 显存说明
- `modelo2` 当前采用“按 excitation 分块过图主干”的实现，以降低原生 `GATv2Conv` 的峰值显存。
- 新增参数：
  - `--excitation-chunk-size`
- 当前推荐默认：
  - `batch_size=8`
  - `excitation_chunk_size=4`
- 适用场景：
  - 尤其适合 `8~12GB` 显存的单卡环境
- 若仍 OOM，可按顺序继续收紧：
  - `--batch-size 4`
  - `--excitation-chunk-size 2`
  - `--gat-heads 2`
## 十一、2026-03-26 新版实验入口：GNN_CLS/modelo3 与 GNN_REG/modelo3
### 3) GNN_CLS / modelo3
- 基于 `modelo2` 的小步修正版本。
- 主要变化：
  - 验证/测试阶段概率计算改为 `scipy.special.expit`。
  - 默认 `weight_decay=1e-3`，缓解 logits 过大与过度自信。
  - `inference.py` 默认优先查看真实 `2/3` 变化样本。

### 4) GNN_REG / modelo3
- 结构仍沿用 `modelo2` 的 `Physics-Informed GNN + gated REG`。
- 主要变化：
  - `lambda_mask_l1` 进一步下调到 `0.002`，继续释放模型对非零变化的预测勇气。
  - `val_sparse_alpha` 进一步下调到 `0.05`，减轻选模阶段对“较多非零预测”的过强惩罚。
  - 常规 `inference.py` 默认优先查看真实 `2/3` 变化样本。
  - 新增 `inference_full.py`：用 `CLS` 先给出预测变化数 `K`，再在 `REG` 输出里取前 `K` 大绝对值作为最终变化边。
- 说明：
  - `inference_full.py` 是比固定阈值更贴近实际使用的联合推理入口。

### 5) GNN_REG / model_tp1
- 目标：
  - 按根目录 `新架构.md` 的思路，建立更贴近物理过程的回归版本。
  - 不再以 `GATv2` 主干为核心，而改用“共享边电导 + KCL 迭代”的物理传播模块。
- 输入与数据适配：
  - 保持使用当前主项目数据与 `dataset_tag/cache/outputs` 规则。
  - 每个样本组织为一个 `PyG Data` 图，`x` 形状为 `(64, 32)`：
    - 32 列对应 32 次激励；
    - 边界 28 个节点填入电压差值；
    - 内部节点初值为 0。
- 物理传播：
  - 112 条电阻边各自拥有一个可学习电导参数；
  - 所有迭代层共享这组电导参数；
  - 使用稳定的 KCL 残差更新 `V <- V - alpha * L_g(V)`；
  - 每次迭代后重新固定边界节点电压，只更新内部节点；
  - `alpha` 默认可学习，初值 `0.1`，上界 `0.25`。
- 边解码器：
  - 先拿到最终节点电压 `V(L)`；
  - 对每条边 `(u, v)` 统计跨 32 次激励的 `Vu / Vv / |Vu-Vv| / avg(Vu,Vv)` 的 `mean/max/std`；
  - 再拼接当前边的共享电导，送入共享 MLP；
  - 输出仍采用 `mask(sigmoid) * value(tanh)` 的门控回归形式。
- 损失：
  - `MSE(pred, true_delta)`
  - `+ lambda_mask_l1 * mean(mask_prob)`
  - `+ lambda_kcl * mean(interior_residual^2)`
- 当前定位：
  - `model_tp1` 是一条新的物理先验回归支线；
  - 重点观察它是否能在不明显放大假阳性的情况下改善 `2/3` 场景下的定位与 `mae_changed`。

## 十二、2026-03-27 最新结果与修正方向
### 6) GNN_REG / modelo3
- 当前最佳 baseline（历史可达上限）：
  - `mae_all=0.5573`
  - `mae_changed=25.0888`
  - `best_count_threshold(val)=45.0`
  - `val_macro_f1=0.8314`
  - `avg(|dR|>45)=1.75`
- 结论：
  - 这是当前 `GNN_REG` 线最值得保留的 baseline，也是目前已知可达到的精度上限提示；
  - 但它在 2026-03-28 的 fresh cache / fresh outdir 复现实验中未能稳定重现，因此不能再直接视为稳定现役模型。

### 7) GNN_CLS / modelo3
- 当前结果：
  - `test_macro_f1=0.9075`
- 结论：
  - `0/1` 已很稳；
  - 残余误差仍主要集中在 `2/3`，更接近逆问题本身的物理病态性，而不是单纯架构缺陷。

### 8) GNN_REG / model_tp1（修正版）
- 0326 首轮结果：
  - `mae_changed=106.4545`
  - `avg(|dR|>40)=0.93`
  - 对 `2/3` 变化样本明显过保守。
- 当前修正思路：
  - 不再只做“共享静态电导 + KCL 残差”；
  - 改为 `node-edge-global` 的 GN 更新块：
    - 节点状态：电压
    - 边状态：电导
    - 全局状态：总电流与全局统计
  - 每轮更新：
    - 边更新：根据 `vi/vj/eij/u` 调整边电导
    - 节点更新：根据物理电流聚合与全局状态更新电压
    - 全局更新：聚合节点与边统计更新全局状态
- 同时放松约束：
  - `lambda_mask_l1` 降到更轻量级
  - `lambda_kcl` 降到 `0.005`
  - 增加 `kcl` warmup，避免一开始就和 `MSE` 正面打架

### 9) 2026-03-27 当前结论补充
- `GNN_REG/model_tp1` 修正版实测：
  - `mae_all=2.0972`
  - `mae_changed=101.9642`
  - `val_macro_f1=0.3097`
  - `avg(|dR|>40)=1.06`
- 说明：
  - 修正版虽然比最差点略有回升，但仍远落后于 `GNN_REG/modelo3`
  - 这表明“总物理约束主导”的路线暂时还没有找到有效平衡点
  - 当前这条线先暂停，不再作为 GNN 主推进方向
- 当前建议：
  - `GNN_REG` 主线继续以 `modelo3` 为准
  - 更优先把 `GNN_REG/modelo3` 接入根目录异构联合推理方案，与 `MLP_CLS/modelo7`、`MLP_REG/modelo7` 配合
## 2026-03-27 路线修正：撤回 GNN_REG / modelo4a 与 modelo4b
- `modelo4a / modelo4b` 已完成首轮验证，但当前判断是不再继续保留这两条分支。
- 这两条分支原本尝试的增强包括：
  - `resistor_embedding`
  - `top-K` 位置值损失
  - 轻量电压重投影物理损失
  - `modelo4b` 的 `top3 / top4 / top5` 候选覆盖率评估
- 当前暴露出的核心问题：
  - `val_phys` 长时间维持在 `4600+`，说明 KCL 软约束与主回归 `MSE` 之间存在明显梯度冲突；
  - `avg(|dR|>45)` 被卡在约 `1.15`，远低于当前主数据真实平均变化数；
  - 在强门控 / 强稀疏压力下，相邻真实变化边容易被压缩成单边预测。
- 因此当前 GNN 回归主线继续维持为 `GNN_REG/modelo3`。
- 若后续需要候选集口径：
  - 不再另起 `modelo4b` 主干；
  - 而是直接基于 `modelo3` 追加 `top3 / top4 / top5` 候选覆盖率和候选集 inference 输出；
  - 当前已落地的目录为 `gnn/GNN_REG/modelo3b`，它是独立推理版，不进入全局 `joint_inference`。
- 当前默认主线数据：
  - 未筛选 `10mA`：`64Nodes/data/training_data64Nodes_2.csv`

## 2026-03-28 modelo3b 候选集结果解释
- `gnn/GNN_REG/modelo3b` 已完成首轮独立推理：
  - `top3_candidate_cover=0.8300`
  - `top4_candidate_cover=0.8540`
  - `top5_candidate_cover=0.8610`
  - `changed_only` 口径为 `0.8170 / 0.8428 / 0.8504`
- 这些结果说明：
  - `modelo3` 的边级排序能力已经比较强，很多失败样本并不是“完全没找到区域”，而是“漏掉的真实边排在第 4 或第 5 名附近”；
  - `top4` 相比 `top3` 的提升比较明显，因此它是当前更有性价比的候选集规模；
  - `top5` 继续提升但幅度已经很小，说明剩余错误中有一部分不是简单扩候选集就能补回，而是模型本身把真实边排得过低。
- 当前定位：
  - `modelo3b` 适合作为候选生成与上限诊断工具；
  - 它证明 `modelo3` 已具备较强的粗定位能力；
  - 但它不代表主回归问题已经解决，因此仍不替代 `modelo3` 主线本身的继续优化。

## 2026-03-28 GNN_REG `o4` 系列排障记录
- 在确认 `modelo3` 的历史好结果当前不可稳定复现后，新增了 5 条有明确针对性的 `REG` 实验分支：
  - `gnn/GNN_REG/o4a`
  - `gnn/GNN_REG/o4a2`
  - `gnn/GNN_REG/o4b`
  - `gnn/GNN_REG/o4b2`
  - `gnn/GNN_REG/o4b3`

### `o4a`
- 保留原始 `pred = mask_prob * value` 的耦合输出；
- 只针对“前几轮门控过早塌陷”做轻量修正：
  - `mask_head` 正偏置初始化；
  - `lambda_mask_l1` 采用 warmup，而不是从第 1 个 step 就满额施加。
- 第一轮不完整结果表明：
  - `val_mask_mean` 很快从 `0.0340` 掉到 `0.0064`
  - `val_avg(|dR|>50)` 也回到 `1.16~1.17`
- 说明仅靠初始化和 `L1 warmup` 还不够，门控依旧会重新塌陷。

### `o4a2`
- 保留 `o4a` 的耦合结构，不走 `o4b` 的完全解耦路线；
- 但在训练上引入两项直接修正：
  - 给 `mask_logits` 加显式 `BCEWithLogitsLoss` 监督；
  - 把回归主损失从 `MSE` 改成 `SmoothL1`，降低极端误差对总梯度的绑架。
- 它的目标很明确：
  - 不像 `o4a` 那样无声塌陷；
  - 也尽量避免像 `o4b` 那样一下子冲到严重过报。

### `o4b`
- 彻底解耦输出：
  - 前向只返回 `value`
  - `mask_logits` 用 `BCEWithLogitsLoss`
  - `value` 只在真实变化边上做 masked MSE
  - 验证和推理时才组合 `pred = mask_prob * value`
- 第一轮不完整结果表明：
  - `val_mae_changed` 已降到 `29~33`，明显优于最近复现失败的 `modelo3`
  - 但 `val_avg(|dR|>50)` 冲到 `8~30`，说明严重过报
- 这说明“耦合前向 + 全局 MSE 劫持”分析方向是对的，但原始 `o4b` 走得太猛，从“塌陷”直接摆到了“过激活”。

### `o4b2`
- 在 `o4b` 的基础上做第一次刹车：
  - 降低 `mask_pos_weight`
  - 降低 `value` 正样本损失权重
  - 额外加入负样本边的背景幅值惩罚
- 它的定位是“过报修正版”，但目前还没有完整训练结果。

### `o4b3`
- 基于新的 Loss 量级分析继续修正：
  - 将 `mask_pos_weight` 默认进一步降到 `10.0`
  - 把总损失显式改成 `50 * BCE(mask) + 0.05 * masked_MSE(value)`
- 目标不是再去改结构，而是直接抬高分类头在总梯度中的地位，避免 `BCE` 被 `MSE` 在量级上“降维打击”。

### `val_mask_mean` 的解释
- `val_mask_mean` 是验证集上 112 条边平均 `mask_prob` 的均值。
- 它不是越大越好，也不是越小越好：
  - 太小：常见于门控塌陷，模型几乎不敢报变化；
  - 太大：常见于严重过报，模型觉得很多边都在变。
- 它必须和 `val_avg(|dR|>50)`、`val_mae_changed` 一起看。
- 就当前任务而言，`0.10+` 已经偏大，通常对应明显过活跃。
## 2026-03-28 统一 GNN 联合推理入口
- 当前 `gnn` 主线统一联合推理脚本为：
  - `64Nodes/gnn/inference_gnn_cmei.py`
- 默认组合：
  - `GNN_CLS/modelo3` 负责预测变化数量 `K`
  - `GNN_REG/modelo3` 负责输出 112 维电阻变化量
  - 最终取 `|dR|` 前 `K` 大边作为预测变化边
- 方案来源：
  - 旧 `joint_inference` 的固定逻辑版与动态融合版已经完成对比；
  - 当前结论是固定逻辑版更稳，动态融合没有带来明确收益，因此新入口继承固定逻辑版中有效的 `Near-Miss + CMEI` 评估机制。
- 路线切换原因：
  - `GNN_CLS` 与 `MLP_CLS` 的数量分类差距已很小；
  - `CLS` 与 `REG` 同时使用 GNN，更有利于统一图表示、拓扑归纳偏置和后续综合模型建设；
  - 物理驱动/拓扑驱动方向的长期潜力大于纯数据驱动的 `MLP` 联合推理路线。
- 因此：
  - 停止维护 `MLP_CLS + GNN_REG (+ MLP_REG)` 的异构联合推理；
  - `modelo3b` 仍保留为候选集诊断工具，但不再进入任何旧 `joint_inference` 流程；
  - `joint_inference/` 目录整体退役。

## 2026-03-29 `o4a2 / o4b2 / o4b3` 的最新判断
- `o4a2`
  - 当前单次测试：`mae_all=0.4854`，`mae_changed=24.2925`，`val_macro_f1=0.8683`，`avg(|dR|>40)=1.77`
  - 训练后半段 `val_avg(|dR|>50)` 稳定在 `1.66 ~ 1.71`，`val_mask_mean` 稳定在 `0.019 ~ 0.023`
  - 说明它第一次真正把系统拉回了“既不塌、也不过报”的中间带
- `o4b2`
  - 当前单次测试：`mae_all=2.5001`，`mae_changed=22.4532`，`val_macro_f1=0.6904`，`avg(|dR|>56)=2.82`
  - 说明完全解耦路线虽然让真实变化边上的数值拟合更强，但仍会明显过报
- `o4b3`
  - 当前单次测试：`mae_all=2.2490`，`mae_changed=21.2190`，`val_macro_f1=0.6990`，`avg(|dR|>67)=2.27`
  - 说明提高 `BCE` 权重后，过报相较 `o4b2` 有所收敛，但仍未回到理想稀疏区
- 当前阶段结论：
  - `o4a2` 是当前最值得继续复验和推进的 `GNN_REG` 主线候选
  - `o4b2 / o4b3` 更适合被理解为“候选生成器/高召回诊断支线”，暂不作为最终主回归模型
- 后续优先级：
  - 先对 `o4a2` 做 fresh cache / fresh outdir / 多 seed 复验
  - 在 `o4a2` 稳住之前，不再继续扩散太多新的 `REG` 架构分支

## 2026-03-29 GNN 通用可视化脚本
- 新增：
  - `64Nodes/gnn/visualize_gnn_results.py`
- 支持的输入类型：
  - `GNN_CLS` 输出目录
  - `GNN_REG` 输出目录
  - `GNN_REG/modelo3b` 候选集输出目录
  - `gnn/inference_gnn_cmei.py` 的联合推理输出目录
- 支持的产物：
  - 单 run：`overview` 总览图 + `samples` 样例图
  - 多 run：额外输出 `comparison` 演进对比图
- 默认输出位置：
  - `64Nodes/gnn/outputs/visualizations`
- 当前建议的展示套路：
  - 版本演进：`modelo3 -> o4a2 -> o4b2 -> o4b3`
  - 联合推理：展示 `CMEI / S_num / S_F1 / S_id / S_mse`
  - 样例展示：固定 4~5 个代表样例，用拓扑图对比 `true/pred` 边
## 2026-03-30 `GNN_REG/o4a2` 正式转正
- `o4a2` 已完成 4 个 seed 的 fresh-cache / fresh-outdir 复验，当前应视为 `GNN_REG` 的正式主线，而不是仅靠单次好结果支撑的候选分支。
- 4-seed 结果区间：
  - `mae_all`: `0.4806 ~ 0.5654`
  - `mae_changed`: `25.3102 ~ 27.5191`
  - `val_count_macro_f1`: `0.8508 ~ 0.8655`
  - `avg_abs_gt_threshold`: `1.704 ~ 1.768`
- 这说明：
  - `o4a2` 已经跨过“能不能偶然跑出来”的阶段；
  - 它现在具备了作为后续持续优化母版的稳定性基础。
- 当前推荐口径分成两层：
  - 当前最佳单 checkpoint：
    - `64Nodes/gnn/GNN_REG/o4a2/outputs/training_data64Nodes_2/`
  - 当前最佳复验锚点：
    - `64Nodes/gnn/GNN_REG/o4a2/outputs/o4a2_seed20260326/`
- 当前统一 GNN 联合推理默认组合更新为：
  - `GNN_CLS/modelo3`
  - `GNN_REG/o4a2`
- 本地下载回来的 `o4a2` 输出目录已具备完整复用条件：
  - `model_last.pt`
  - `metrics.json`
  - `standardization.npz`
  - 因此可直接用于推理、可视化和后续结果复核。
- 仍需保留的问题判断：
  - `o4a2` 训练后段有轻微过拟合/震荡；
  - 对高变化数样本仍有保守预测倾向；
  - `mask_l1=0.002` 在 warmup 后固定不再变化，这说明当前后期稀疏控制更多依赖门控头已学出的分布，而不是继续增强的 `L1` 惩罚。

## 2026-03-30 新增 `o5a / o5b / GNN_FULL/Mv1`
- 指标口径补充说明：
  - `o4a2` 训练历史里的 `mae_all / mae_changed` 计算口径没有改；
  - 它们仍然是绝对误差意义上的 MAE；
  - 因此这次变好不是“改了展示方式”，而是模型本身真的更好了。
- `GNN_REG/o5a`
  - 是围绕 `o4a2` 保守预测倾向做的最小修正版；
  - 仅新增对真实变化边的幅值下限约束，不改主骨架；
  - 目标是优先修正高变化数样本里的少报问题。
- `GNN_REG/o5b`
  - 按 `GNN_REG优化.txt` 落地；
  - 结构上补入 112 条边的绝对位置 embedding；
  - 损失上改成 relaxed top-k sparsity，更少惩罚最可疑的前 k 条边。
- `GNN_FULL/Mv1`
  - 作为第一版联合 `GNN_CLS/modelo3 + GNN_REG/o4a2` 的本地融合目录建立；
  - 目录结构为：
    - `model/model_cls.py`
    - `model/model_reg.py`
    - `train_cls.py`
    - `train_reg.py`
    - `inference.py`
    - `outputs/`
  - 它不是新的端到端单模型，而是把当前最优 `CLS` 与 `REG` 主线放进同一实验容器中统一管理。

## 2026-03-31 `o5a / o5b` 首轮结果
### `o5a`
- 最终测试：
  - `mae_all=0.7770`
  - `mae_changed=38.8351`
  - `best_count_threshold(val)=40.0`
  - `val_macro_f1=0.8524`
  - `avg(|dR|>40)=1.77`
  - `avg(mask_prob)=0.0200`
- 分析：
  - `o5a` 的目标是缓解 `o4a2` 在高变化数样本上的保守少报；
  - 但结果表明，新增的“真实变化边幅值下限”约束会明显破坏整体幅值校准；
  - 虽然平均激活边数仍在合理区，但 `mae_all / mae_changed` 都大幅劣化；
  - 因此这条路线应视为失败消融，不再继续扩展。

### `o5b`
- 最终测试：
  - `mae_all=0.5119`
  - `mae_changed=25.4532`
  - `best_count_threshold(val)=40.0`
  - `val_macro_f1=0.8564`
  - `avg(|dR|>40)=1.74`
  - `avg(mask_prob)=0.0207`
- 分析：
  - `o5b` 明显优于 `o5a`，说明给边引入绝对位置先验、并把稀疏惩罚改成 relaxed top-k 是正确方向；
  - 从单次结果看，它已经接近 `o4a2` 的多 seed 均值；
  - 但它仍略弱于当前最佳单 checkpoint `o4a2`，所以暂时不能替代主线。
- 当前建议定位：
  - `o4a2` 继续作为默认主回归器；
  - `o5b` 作为下一阶段最值得继续复验的候选分支；
  - 如果 `o5b` 多 seed 均值优于 `o4a2`，再考虑正式切线。

### `modelo3 + o5b` 联合推理结论
- 已使用统一入口完成一次纯 GNN 联合推理验证：
  - `GNN_CLS/modelo3 + GNN_REG/o5b`
- 结果：
  - `CMEI=93.20`
  - `num_accuracy=0.8850`
  - `macro_f1=0.9075`
  - `id_recall=0.9120`
  - `mse_all_edges=56.4333`
- 解释：
  - 分类能力基本不受影响；
  - 但相较当前默认 `modelo3 + o4a2`，`o5b` 在联合链路上的 `id_recall` 与 `mse` 都更差；
  - 所以它还不是更好的最终组合。

### `GNN_FULL/Mv1/inference_v2.py`
- 新增目的：
  - 保护 `o5b` 预测出的高置信相邻损坏边，不再被旧 near-miss 误杀；
  - 同时引入基于 `35Ω / 45Ω` 的物理死区与补漏规则，让 REG 能在一定程度上倒逼 CLS。
- 实测结果（`GNN_CLS/modelo3 + GNN_REG/o5b`）：
  - `CMEI=91.38`
  - `num_accuracy=0.8400`
  - `macro_f1=0.8587`
  - `id_recall=0.9019`
  - `mse_all_edges=56.7826`
- 阶段判断：
  - 单独看“高置信相邻边保护”是合理方向；
  - 但把物理死区和补漏机制同时加进来后，当前规则过硬，反而开始损伤计数质量；
  - 因此 `inference_v2.py` 目前只保留为实验版，不作为默认联合推理。

### `o5b1`
- 新增位置：
  - `gnn/GNN_REG/o5b1`
- 改动非常小：
  - 仅把 `mask_bce_weight` 从 `25` 降到 `20`
- 目的：
  - 在不大幅放松监督的前提下，先测试更温和的 `mask` 约束是否能缓解 `o5b` 的轻微保守倾向。

### 简化后的 `inference_v2`
- 当前 `gnn/GNN_FULL/Mv1/inference_v2.py` 已删除物理死区与补漏机制；
- 现在只保留：
  - “高置信相邻边保护”版 near-miss
- 对 `GNN_CLS/modelo3 + GNN_REG/o5b` 的实测结果：
  - `CMEI=93.17`
  - `num_accuracy=0.8850`
  - `macro_f1=0.9075`
  - `id_recall=0.9115`
  - `mse_all_edges=57.6391`
- 解释：
  - 这说明物理死区/补漏是上一版退化的主因；
  - 但只保留保护墙后，结果仍然没有超过原始统一推理的 `93.20`
  - 因此 `inference_v2` 继续仅作为实验版保留。

## 2026-03-31 `GNN_FULL/Mv1` 路径修复与 `Noise_test` 步骤 A
### `GNN_FULL/Mv1` 路径修复
- 修复文件：
  - `gnn/GNN_FULL/Mv1/inference.py`
  - `gnn/GNN_FULL/Mv1/inference_v2.py`
- 本次修复覆盖两类问题：
  - 根目录相对路径传参时，不再把 `gnn/GNN_FULL/Mv1/...` 再次拼到脚本目录下面，避免重复前缀
  - 若 `Mv1/cache/<dataset_tag>/cache_dataset_cls_graphattn.npz` 或 `cache_dataset_reg_graphattn.npz` 不存在，则自动调用 `train_cls.py / train_reg.py` 中的 `build_dataset(...)` 同口径重建 cache
- 这意味着当前 `Mv1` 推理入口不再依赖“cache 先手工下载齐全”这一前提；只要原始 CSV、模型权重与 `standardization.npz` 在，就可以直接从根目录复跑。

### 当前 `Mv1` 已训练模型结果
- `Mv1/outputs/cls/training_data64Nodes_2/`
  - `test_macro_f1=0.8975`
  - `best_thresholds=[0.05, 0.05, 0.23]`
- `Mv1/outputs/reg/training_data64Nodes_2/`
  - `mae_all=0.4718`
  - `mae_changed=24.8125`
  - `best_count_threshold(val)=40.0`
  - `val_count_macro_f1=0.8530`
  - `avg(|dR|>40)=1.71`
- `Mv1/outputs/inference/training_data64Nodes_2/`
  - `CMEI=93.11`
  - `num_accuracy=0.8740`
  - `macro_f1=0.8975`
  - `id_recall=0.9173`
  - `mse_all_edges=53.3761`
- 当前解释：
  - `Mv1 REG` 的幅值回归已经与 `o4a2` 同量级
  - 但 `Mv1 CLS` 仍弱于当前正式 `modelo3`
  - 因此 `Mv1` 现在更适合作为“联合实验容器”而非正式替代默认主线

### `Noise_test` 步骤 A（20dB zero-shot noise）
- 执行口径：
  - 数据：当前默认 `10mA` 数据 `training_data64Nodes_2`
  - 噪声：只在测试集 standardized voltage 通道注入高斯白噪声，`noise_std=0.1`
  - 解释：对应 `Noise_test.txt` 里给出的 `20dB` 设定
- `GNN_CLS/modelo3`
  - clean：`test_macro_f1=0.9075`
  - noise：`test_macro_f1=0.1203`
  - 现象：测试集混淆矩阵几乎完全塌到类别 `3`
- `GNN_REG/o4a2`
  - clean：`mae_all=0.4854`，`mae_changed=24.2925`，`count_macro_f1=0.8683`，`avg(|dR|>40)=1.771`
  - noise：`mae_all=23.2065`，`mae_changed=70.3696`，`count_macro_f1=0.1203`，`avg(|dR|>40)=17.668`，`avg(mask_prob)=0.1927`
  - 现象：派生计数同样几乎全部塌到 `3`
- `modelo3 + o4a2` 统一联合推理
  - clean：`CMEI=93.73`，`num_accuracy=0.8850`，`macro_f1=0.9075`，`id_recall=0.9248`，`mse_all_edges=49.7686`
  - noise：`CMEI=41.79`，`num_accuracy=0.3170`，`macro_f1=0.1203`，`id_recall=0.2608`，`mse_all_edges=1735.1177`
- 当前判断：
  - 这次步骤 A 已经说明当前最佳 clean GNN 链条对 standardized `20dB` 噪声非常敏感；
  - 下一步最合理的动作不是继续堆后处理，而是进入 `Noise_test` 的步骤 B，直接做带噪训练与鲁棒性复验。

## 2026-03-31 新增 `GNN_NOISE`
### 目录定位
- 新增目录：
  - `gnn/GNN_NOISE`
- 作用：
  - 专门承接 `Noise_test` 的策略 B
  - 即在 clean 最优权重上做带噪数据增强与低学习率微调

### 当前已建立的两条线
- `gnn/GNN_NOISE/CLS_modelo3_ft`
  - 继承 `GNN_CLS/modelo3`
- `gnn/GNN_NOISE/REG_o4a2_ft`
  - 继承 `GNN_REG/o4a2`

### 共同设计原则
- 默认 warm start：
  - `CLS` 默认加载 `GNN_CLS/modelo3/outputs/training_data64Nodes_2/model_last.pt`
  - `REG` 默认加载 `GNN_REG/o4a2/outputs/training_data64Nodes_2/model_last.pt`
- 默认 fine-tune 配置：
  - `epochs=30`
  - `lr=5e-5`
  - `batch_size=8`
- 训练集动态随机噪声：
  - 默认 `add_noise=True`
  - 默认 `noise_schedule=random`
  - 默认 `noise_mode=gaussian`
  - 默认 `noise_std_max=0.1`
  - 默认 `fixed_noise_std=0.1`
  - 默认 `noise_scope=boundary`
  - 每次 `__getitem__` 都重新采样 `noise_std = noise_std_max * rand()`
- 物理口径保持一致：
  - 只在边界节点的 `voltage_delta` 通道注入噪声
  - 不改原始 `modelo3 / o4a2` 的主干结构与损失定义

### 推荐用途
- 先完成带噪微调
- 再用以下链路做评估：
  - 单模型：`GNN_NOISE/*/inference.py --noise-std 0.1`
  - 联合：`gnn/inference_gnn_cmei.py --cls-dir GNN_NOISE/CLS_modelo3_ft --reg-dir GNN_NOISE/REG_o4a2_ft --noise-std 0.1`
- 真正要比较的重点不再是 clean 分数，而是：
  - `20dB` 下的 `macro_f1`
  - `id_recall`
  - `mse_all_edges`
  - `CMEI`

### 原始步骤 B 保留版
- 已确认根目录原 `Noise_test` 步骤 B 不等于当前默认增强版。
- 原始 fixed-20dB 版本现已统一迁移到：
  - `gnn/GNN_NOISE/原始步骤B_fixed20dB.md`
- 对应复现参数为：
  - `noise_schedule=fixed`
  - `fixed_noise_std=0.1`
  - `noise_mode=gaussian`
  - `noise_scope=all`

## 2026-04-01 `o5b1` 训练结果与平台期判断
### 训练记录核对
- 已读取根目录 `o5b1训练记录.txt`，并用本地输出文件核对一致。
- `o5b1` 最终结果：
  - `mae_all=0.5097`
  - `mae_changed=25.5355`
  - `val_count_macro_f1=0.8422`
  - `avg(|dR|>40)=1.73`
  - `avg(mask_prob)=0.0213`

### 与 `o5b` 的对比
- `o5b1` 相比 `o5b`：
  - `mae_all` 仅有极小改善：`0.5119 -> 0.5097`
  - `mae_changed` 轻微回退：`25.4532 -> 25.5355`
  - `val_count_macro_f1` 明显回退：`0.8564 -> 0.8422`
  - `avg(mask_prob)` 略升：`0.0207 -> 0.0213`
- 结论：
  - 把 `mask_bce_weight` 从 `25` 放到 `20` 没有把模型推向假阳性爆炸；
  - 但这次放松也没有带来可用的精度红利。

### 当前判断
- 用户给出的“模型没有炸，说明结构本身很稳”这部分判断是合理的。
- 但“已经撞到信息论上限”这个表述目前证据还不够强。
- 更合适的工程表述是：
  - 在当前 `64Nodes / clean 10mA / 单一观测口径` 下，`o4a2 / o5b / o5b1` 已出现明显平台期；
  - 继续做 clean-only 小超参微调，大概率只是在平台边缘抖动；
  - 因此接下来不应把主要时间继续投入到 clean 精度抠小数点，而应转向 noisy robustness 与更高信息量的训练设定。

## 2026-04-01 根目录记录清理完成
- `Noise_test` 相关诊断文件已集中到：
  - `gnn/GNN_NOISE/首轮20dB噪声诊断记录.md`
  - `gnn/GNN_NOISE/原始步骤B_fixed20dB.md`
- 已删除根目录原始草稿：
  - `o5b1训练记录.txt`
  - `Mv1训练记录.txt`
  - `Noise_test.txt`
- 理由：
  - 对应信息都已经被吸收进当前正式文档；
  - 后续查阅与复现实验时，以 `gnn/README.md / gnn/Log.md / gnn/GNN_NOISE/*` 为准即可。

## 2026-04-01 `0401训练记录` 结果整理
### clean 主线
- `modelo3`：
  - `test_macro_f1=0.9027`
- `o4a2`：
  - `mae_all=0.4679`
  - `mae_changed=23.5724`
  - `val_count_macro_f1=0.8628`
- `modelo3 + o4a2`：
  - `CMEI=93.53`
  - `num_accuracy=0.8800`
  - `macro_f1=0.9027`
  - `id_recall=0.9237`
  - `mse_all_edges=53.2930`
- 结论：
  - clean 复训仍然稳定，但没有超过历史最好 `93.73`

### `noiseft_rand_boundary`
- `REG_o4a2_ft` 成功训练并推理：
  - `mae_all=0.5900`
  - `mae_changed=27.5311`
  - `count_macro_f1=0.8140`
- `CLS_modelo3_ft` 对应 tag 下没有 `model_last.pt`
- 这说明：
  - 这条推荐增强版分支在云端这次并没有完整训练成对结果；
  - 当前只能把 `REG` 结果当作单边观察，不能拿来做完整联合比较

### `noiseft_fixed20db_all`
- 这是原始步骤 B 保留版的实际云端结果。
- 最终保留下来的分类结果是：
  - `test_macro_f1=0.7275`
- 回归结果是：
  - `mae_all=0.9201`
  - `mae_changed=48.8259`
  - `count_macro_f1=0.6178`
- clean 联合：
  - `CMEI=83.39`
- `20dB` 单模型：
  - `CLS macro_f1=0.7121`
  - `REG count_macro_f1=0.5829`
- 结论：
  - 强增强确实恢复了噪声鲁棒性；
  - 但 clean 主线被破坏得太明显；
  - 当前更值得继续推进的仍是 `random + boundary-only` 那条推荐增强版，而不是 fixed-all。

## 2026-04-01 数据可视化图集替换与当前执行口径
- 已按新的汇报需求删除旧图集与说明性材料：
  - `midterm_assets/20260401_data_figures`
  - `中期汇报_数据可视化说明.md`
  - `tools/generate_midterm_figures.py`
- 重新生成的最终图集位于：
  - `midterm_assets/20260401_visuals/01_topology_boundary_nodes.svg`
  - `midterm_assets/20260401_visuals/02_dataset_composition.svg`
  - `midterm_assets/20260401_visuals/03_changed_edge_frequency.svg`
  - `midterm_assets/20260401_visuals/04_boundary_response_heatmaps.svg`
- 关于联合推理脚本：
  - 当前本地最新版路径为 `gnn/inference_gnn_cmei.py`
  - 若云端那份脚本仍然不识别 `--noise-std` 与 `--noise-seed`，则需要重新上传这一个文件
- 关于当前还要不要重训：
  - clean 主线 `modelo3 + o4a2` 已有 `0401` 闭环，不是当前必补项
  - `noiseft_fixed20db_all_20260401` 已有完整 `CLS + REG` 落盘，不是当前必补项
  - 仅 `noiseft_rand_boundary_20260401` 的 `CLS_modelo3_ft` 缺失 `model_last.pt`，应优先补齐

## 2026-04-01 `0401补充训练` 与下载输出核对
- 已读取根目录 `0401补充训练.txt`，确认补训的是推荐增强版：
  - `training_data64Nodes_2_noiseft_rand_boundary_20260401`
- 新增确认结果：
  - `CLS clean test_macro_f1=0.8750`
  - `CLS noisy test_macro_f1=0.7780`
  - noisy joint `CMEI=82.56`
  - `num_accuracy=0.7360`
  - `id_recall=0.7579`
  - `mse_all_edges=154.4499`
- 本地下载输出检查：
  - `GNN_NOISE/CLS_modelo3_ft/outputs/...rand_boundary...` 已经完整到可复现实验
  - `GNN_NOISE/REG_o4a2_ft/outputs/...rand_boundary...` 当前缺少 `noise_eval.json`
  - `gnn/outputs` 当前未包含补充训练对应的 joint 输出目录
- 因此这批下载结果的判断是：
  - 单模型目录大体正确
  - 联合输出目录不完整，需要后续补下载

## 2026-04-01 鲁棒性曲线脚本与记录清理
- 根目录 `0401补充训练.txt` 已删除，因为内容已经完成正式归档。
- 新增：
  - `gnn/GNN_NOISE/plot_noise_robustness.py`
- 当前用途：
  - 读取归档后的 `20/30/40dB` 评估 `json`
  - 在同一张图上对比 `fixed20db_all / rand_boundary / zero-shot` 等多条曲线

## 2026-04-01 `modelo3` 两阶段细阈值搜索
- 新增独立脚本：
  - `gnn/GNN_CLS/modelo3/two_stage_threshold_search.py`
- 作用：
  - 复用现有数据构建、划分和模型定义
  - 对 `CLS` 先粗搜、再局部细搜阈值
  - 不需要修改原 `train.py`
- 对当前正式主线 `rand_boundary` 的本地实测：
  - 原阈值 `best_thresholds=[0.05, 0.17, 0.37]`
  - 细化后 `best_thresholds=[0.05, 0.164, 0.368]`
  - `val_macro_f1` 不变
  - `test_macro_f1` 由 `0.8750` 轻微变为 `0.8749`
- 当前解释：
  - 阈值确实可以再细，但当前收益几乎为零
  - 模型下一步更应该增强 noisy 表征稳定性，而不是继续做阈值层面的微调

## 2026-04-02 `0402补充日志` 结果整理
- 已确认以下 `gnn/outputs` 目录现已补齐：
  - `gnn_cmei_noiseft_rand_boundary_clean_20260401`
  - `gnn_cmei_noiseft_rand_boundary_20db_20260401`
  - `gnn_cmei_noiseft_fixed20db_all_clean_20260401`
  - `gnn_cmei_noiseft_fixed20db_all_20db_20260401`
- `rand_boundary`：
  - `REG 20dB`: `mae_all=1.2692`，`mae_changed=54.1729`，`count_macro_f1=0.5844`
  - clean joint: `CMEI=91.01`
  - 20dB joint: `CMEI=82.56`
- `fixed20db_all`：
  - clean joint: `CMEI=83.39`
  - 20dB joint: `CMEI=81.79`
- 结论：
  - 20dB 这批补推理已经完成
  - 后续如继续推进，只需补 `30/40dB`

## 2026-04-02 `rand_boundary` 最终鲁棒性曲线与结论
- 已吸收 `0402大范围噪声训练.txt`
- 已生成：
  - `gnn/GNN_NOISE/rand_boundary_robustness_curve.svg`
- 已新增脚本：
  - `gnn/GNN_NOISE/plot_rand_boundary_robustness.py`
- 当前正式主线 `rand_boundary` 的 joint 指标：
  - clean: `CMEI=91.01`
  - `40dB`: `CMEI=90.83`
  - `30dB`: `CMEI=89.62`
  - `20dB`: `CMEI=82.56`
- 结合 `macro_f1 / num_accuracy / id_recall` 曲线可见：
  - `40dB` 几乎不损失性能
  - `30dB` 仍然保持稳定
  - `20dB` 才出现明显衰减
- 与 zero-shot `20dB` 相比，当前 `rand_boundary` 已实现决定性恢复，因此不再需要继续保留 `fixed20db_all` 作为主线候选
- `0402补充日志.txt` 与 `0402大范围噪声训练.txt` 均已删除

## 2026-04-02 联合推理目录重构
- 已新建：
  - `gnn/GNN_CMEI_INFERENCE`
- 本目录职责重新划分为：
  - `GNN_CLS`：clean 分类训练与单模型推理
  - `GNN_REG`：clean 回归训练与单模型推理
  - `GNN_NOISE`：带噪训练与单模型推理
  - `GNN_CMEI_INFERENCE`：`CLS + REG` 联合推理与 `CMEI` 输出
- 已迁移：
  - `inference_gnn_cmei.py`
  - `outputs/`
- 后续 joint 推理默认建议改为：
  - `python gnn/GNN_CMEI_INFERENCE/inference_gnn_cmei.py ...`
- `gnn/inference_gnn_cmei.py` 目前仅保留为兼容转发入口。
- 如继续做下一版带噪训练，目录命名建议统一为：
  - `GNN_NOISE/CLS_modelo3_ft_v2`
  - `GNN_NOISE/REG_o4a2_ft_v2`
  以保证训练、单模型推理、输出都在各自分支内闭环。
- 已基于 `GNN_联合优化.txt` 新建实验推理版：
  - `GNN_CMEI_INFERENCE/inference_gnn_cmei_v2.py`
- 其中保留的有效思路只有：
  - `near-miss` 高置信保护
  - `REG` 动态 `K` / 数量仲裁
- 训练侧结构建议不放进 `CMEI v2`，避免推理层职责混乱。
- 已完成本地实跑验证：
  - `v1 clean CMEI=91.01`
  - `v2(guard_only) clean CMEI=90.85`
  - `v2(full arbitration) clean CMEI=90.08`
  - `v1 20dB CMEI=82.56`
  - `v2(guard_only) 20dB CMEI=82.40`
  - `v2(full arbitration) 20dB CMEI=79.40`
- 因此当前正式 joint 入口仍保持：
  - `GNN_CMEI_INFERENCE/inference_gnn_cmei.py`
- `v2` 只保留为实验脚本，不进入默认流程。

## 2026-04-02 `GNN_NOISE v2` 建立
- 已新建下一版带噪训练分支：
  - `GNN_NOISE/CLS_modelo3_ft_v2`
  - `GNN_NOISE/REG_o4a2_ft_v2`
- 这两个目录继续遵循同一原则：
  - 训练、单模型推理、单模型 outputs 都留在 `GNN_NOISE`
  - 只有 joint `CMEI` 输出写入 `GNN_CMEI_INFERENCE/outputs`
- `v2` 的核心新增不是改模型结构，而是改噪声建模：
  - 从单纯随机白噪声升级到结构化边界噪声
  - 同时引入 `curriculum` 式噪声强度采样

## 2026-04-02 - `GNN_EXPAND` 拓扑与规模扩展目录
- 已新建：
  - `gnn/GNN_EXPAND`
- 本目录职责：
  - 在不改原 `GNN_CLS / GNN_REG / GNN_CMEI_INFERENCE` 程序的前提下
  - 建立面向不同拓扑与节点规模的独立扩展容器
- 当前四阶段：
  - `stage1_square_10x10`
  - `stage2_rect_6x10`
  - `stage3_honeycomb_63`
  - `stage4_transfer_circlecut_69`
- 每个阶段均包含：
  - `cls/train.py`
  - `cls/inference.py`
  - `reg/train.py`
  - `reg/inference.py`
  - `joint_inference/inference.py`
- 当前沿用的最佳方法：
  - `CLS` 沿用 `modelo3`
  - `REG` 沿用 `o4a2`
  - `joint` 沿用统一 `CMEI` 逻辑
- 为满足“继续使用原始无噪声数据”的要求，`GNN_EXPAND` 新增：
  - 边界节点顺时针映射
  - 原始 `8x8 / 112` 电阻 id 到目标拓扑电阻边的几何重映射
- 其中 `stage4` 默认改为优先承接 `stage1_square_10x10` 权重，保留 transfer / zero-shot 试验入口
- 详细文档入口：
  - `gnn/GNN_EXPAND/README.md`
  - `gnn/GNN_EXPAND/Log.md`

## 2026-04-02 - `GNN_EXPAND` 原生数据生成
- 已新增：
  - `gnn/GNN_EXPAND/generate_expand_datasets.py`
- 该脚本用于直接在目标拓扑上做 clean 正演数据生成。
- 当前已生成：
  - `gnn/GNN_EXPAND/data/square_10x10.csv`
  - `gnn/GNN_EXPAND/data/rect_6x10.csv`
  - `gnn/GNN_EXPAND/data/honeycomb_63.csv`
  - `gnn/GNN_EXPAND/data/circlecut_69.csv`
- 每套数据都配有：
  - `*_meta.json`
- 当前统一约束：
  - 激励只使用外部节点
  - 测量只输出外部节点电压
  - 每套数据均固定 `28` 个外部节点、`32` 组激励
- 各阶段默认数据入口已切换为读取本目录内对应的原生 CSV。

## 2026-04-02 GNN_EXPAND 数据口径更正
- 先前把 `GNN_EXPAND` 四套原生数据统一写成 `28` 个外部节点、`32` 组激励，这里补充更正。
- `GNN_EXPAND` 的原生数据不是沿用主线 `8x8` 的固定边界规模，而是按目标拓扑真实外边界生成：
  - 激励只使用外部节点
  - 测量只输出外部节点电压
- 当前四套数据的真实规模：
  - `square_10x10`: `36` 个外部节点，`40` 组激励
  - `rect_6x10`: `28` 个外部节点，`32` 组激励
  - `honeycomb_63`: `28` 个外部节点，`32` 组激励
  - `circlecut_69`: `24` 个外部节点，`28` 组激励

## 2026-04-02 `DOC_RULES` 作为接手入口
- 根目录 `DOC_RULES.md` 已升级为后续新窗口的优先接手文件。
- 对 `gnn` 线而言，当前应优先从该文件获取：
  - 正式 clean 主线
  - 正式 noisy 主线
  - 正式 joint inference 入口
  - `GNN_EXPAND` 的边界节点硬约束与当前阶段任务
- 后续如果 `gnn` 主线发生切换，除了更新本文件和 `gnn/Log.md`，也要同步更新根目录 `DOC_RULES.md`。

## 2026-04-02 `RULES.md` 命名切换
- 根目录长期规则主文件已从 `DOC_RULES.md` 更名为 `RULES.md`。
- 对 `gnn` 线来说，后续新窗口应优先从 `RULES.md` 获取：
  - 当前正式 clean/noisy/joint/expand 路线
  - 代码修改前先呈现思路的执行规则
  - 版本更新优先复制原模型、不直接改旧模型的原则
- 旧的 `DOC_RULES.md` 仅保留为历史兼容入口，不再作为主维护文件。
