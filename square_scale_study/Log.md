# 正方形规模研究日志

## 说明

本文件用于记录 `square_scale_study` 的实际实验过程、阶段结论和后续交接信息。

- `PLAN.md` 负责固定问题定义、指标和实验规则
- `Log.md` 负责记录已经做过的事、观察到的现象和当前结论

建议每一轮实验至少记录：

- 日期
- 规模范围与 `K` 范围
- 数据规模
- 激励设置
- 实际运行内容
- 核心观察
- 下一步动作

## 2026-04-10：子项目初始化

- 新建 `square_scale_study/` 子项目，与旧 `gnn` 主线解耦
- 搭建固定 `K` 数据生成、最简回归训练、推理导出、规模扫描和多激励入口
- 清理初始化时产生的 smoke 产物，避免正式实验前目录混乱

## 2026-04-10：正式数据批次

- 新增 `generate_dataset_grid.py`，支持批量生成 `(N, K)` 数据集
- 完成 `N = 3, 4, 5`、`K = 1..6` 的正式数据集生成
- 每个 `(N, K)` 使用统一数据规模：
  - train: `8000`
  - val: `1000`
  - test: `1000`
- 数据为直接解基尔霍夫方程得到的无噪声正演结果
- 默认激励方式为顺时针边界相邻循环激励，激励数等于端口数 `P`

## 2026-04-11：`modelv1` 全量基线

### 运行范围

- 规模：`N = 3, 4, 5`
- 变化数量：`K = 1..6`
- 模型：`modelv1`
- 目标：建立固定 `K` 下的最简纯回归基线

### 关键结果

- `P = 8 (N=3)`：`K_max = 1`
- `P = 12 (N=4)`：`K_max = 1`
- `P = 16 (N=5)`：在严格 `id >= 0.98` 条件下，`K_max = 0`

主结果说明：

- `value_accuracy` 在大多数点上仍维持较高水平
- 真正的瓶颈是 `id_exact_rate`，且会随着 `K` 增大和规模增大明显下降
- 说明 `modelv1` 更像一个“数值回归基线”，而不是最终可用方案

### 关键误差分析

首个失败点分析结果：

- `N3x3, K=2`：
  - `id_exact_rate = 0.938`
  - `value_accuracy = 0.9748`
  - `mean_true_rank = 1.643`
  - `near_miss_rate_on_failures = 0.323`
- `N4x4, K=2`：
  - `id_exact_rate = 0.892`
  - `value_accuracy = 0.9604`
  - `mean_true_rank = 1.939`
  - `near_miss_rate_on_failures = 0.389`
- `N5x5, K=1`：
  - `id_exact_rate = 0.975`
  - `value_accuracy = 0.9643`
  - `mean_true_rank = 1.325`
  - `near_miss_rate_on_failures = 0.120`

结论：

- `modelv1` 失败的主要原因不是数值回归崩坏，而是 support 排序不够稳定
- 真实变化边通常已经排在较靠前位置，但仍会被少量假阳性边挤出 top-`K`
- 后续模型修改应优先围绕 support 恢复而不是单纯继续压低回归误差

### 记录归档

- `0410modelv1日志.txt` 的重要内容已由 `outputs/`、汇总表和本日志归档
- 后续不再保留该长日志文本

## 2026-04-11：`modelv2` 全量结果

### 运行范围

- 规模：`N = 3, 4, 5`
- 变化数量：`K = 1..6`
- 模型：`modelv2`
- 结构：score head + value head
- 损失：ranking loss + 带变化边/未变化边权重的两部分 SmoothL1

### 关键结果

- `P = 8 (N=3)`：`K_max = 1`
- `P = 12 (N=4)`：`K_max = 0`
- `P = 16 (N=5)`：`K_max = 1`

与 `modelv1` 对比后的整体结论：

- `modelv2` 在多数点上进一步提高了 `value_accuracy`
- 但 `id_exact_rate` 并没有整体改善，很多点反而更低
- 因而它并没有把主线的 `K_max` 曲线整体抬起来

### 关键误差分析

首个关键失败点分析结果：

- `N3x3, K=2`：
  - `id_exact_rate = 0.943`
  - `value_accuracy = 0.9833`
  - `mean_true_rank = 1.583`
  - `near_miss_rate_on_failures = 0.579`
- `N4x4, K=1`：
  - `id_exact_rate = 0.978`
  - `value_accuracy = 0.9705`
  - `mean_true_rank = 1.157`
  - `near_miss_rate_on_failures = 0.409`
- `N5x5, K=2`：
  - `id_exact_rate = 0.809`
  - `value_accuracy = 0.9622`
  - `mean_true_rank = 2.087`
  - `near_miss_rate_on_failures = 0.497`

结论：

- `score head + ranking loss` 并没有自动转化为更高的样本级精确 support 恢复率
- 当前 `modelv2` 更像是“数值更稳、排序仍不够强”的版本
- 这意味着继续直接增加复杂度不一定能解决问题，下一步应优先考虑更轻量、更聚焦的结构修改

### 记录归档

- `0411modelv2日志.txt` 的重要内容已由 `outputs_modelv2/`、汇总图表和本日志归档
- 后续不再保留该长日志文本

## 当前下一步

- 后续模型改进优先在 `3x3` 网络上快速迭代
- 只有当 `3x3` 上出现明显提升后，才扩展到更大规模
- 下一轮应优先尝试更简单、更直接对准 support 恢复的结构，而不是继续堆深层注意力模块

## 2026-04-11：`modelv3` 首轮 `3x3` 小规模验证

### 运行范围

- 规模固定为 `N = 3`
- 变化数量测试为 `K = 2, 3, 4`
- 模型为 `modelv3`
- 输出目录为 `outputs_modelv3/`
- 图表目录为 `Figure/metric_dropoff_by_port_modelv3/`
- 目标是验证“更轻量、更浅层、直接面向 support 恢复”的 `modelv3` 是否优于 `modelv1/modelv2`

### 模型设置

- 结构采用单层浅层 GATv2 主干，`hidden_dim = 64`
- 不再使用较重的多层注意力堆叠，也不做跨激励注意力
- 跨激励融合改为 `mean + max` 的简单聚合
- 保留 `score head + value head`
- 损失为：
  - `score BCE`
  - `ranking hinge loss`
  - 带 changed / unchanged 权重的两部分 `SmoothL1`

### 关键测试结果

- `N3x3, K=2`
  - `test_id_exact_rate = 0.925`
  - `test_value_accuracy = 0.9823`
  - `test_mae_changed = 3.5486`
- `N3x3, K=3`
  - `test_id_exact_rate = 0.810`
  - `test_value_accuracy = 0.9779`
  - `test_mae_changed = 4.4226`
- `N3x3, K=4`
  - `test_id_exact_rate = 0.668`
  - `test_value_accuracy = 0.9701`
  - `test_mae_changed = 5.9830`

最佳验证轮次：

- `K=2`：`best_epoch = 50`
- `K=3`：`best_epoch = 46`
- `K=4`：`best_epoch = 47`

### 与前两版对比

同样只看 `N=3` 上的测试集结果：

- `K=2`
  - `modelv1`: `id = 0.938`, `value = 0.9748`
  - `modelv2`: `id = 0.943`, `value = 0.9833`
  - `modelv3`: `id = 0.925`, `value = 0.9823`
- `K=3`
  - `modelv1`: `id = 0.884`, `value = 0.9702`
  - `modelv2`: `id = 0.852`, `value = 0.9823`
  - `modelv3`: `id = 0.810`, `value = 0.9779`
- `K=4`
  - `modelv1`: `id = 0.810`, `value = 0.9707`
  - `modelv2`: `id = 0.778`, `value = 0.9789`
  - `modelv3`: `id = 0.668`, `value = 0.9701`

结论：

- `modelv3` 没有在 `3x3` 上超过 `modelv1` 或 `modelv2`
- 尤其在 `K=3,4` 时，`id_exact_rate` 明显落后
- 这说明“简单化结构”本身并不会自动带来更好的 exact support recovery

### 误差分析

- `K=2`
  - `mean_true_rank = 1.623`
  - `near_miss_rate_on_failures = 0.573`
- `K=3`
  - `mean_true_rank = 2.206`
  - `near_miss_rate_on_failures = 0.516`
- `K=4`
  - `mean_true_rank = 2.753`
  - `near_miss_rate_on_failures = 0.491`

可以看到：

- 随着 `K` 增大，真实变化边在排序中的平均名次持续变差
- 失败样本里约有一半仍属于“邻近边/近似边混淆”
- 主瓶颈依旧不是回归值本身，而是 top-`K` support 排序不够稳

### 本轮结论

- `modelv3` 证明了一个重要事实：当前问题的核心矛盾不在“主干太重”，而在“训练目标没有真正把 exact support 恢复推起来”
- 仅靠浅层化 + 双头 + ranking loss，仍然不能把排序优势稳定转化为样本级 `id_exact_rate`
- 因而下一步不应继续沿着“更深或更浅的 GAT 结构微调”反复试，而应直接修改支持集恢复机制

### 下一步重点

- 继续只在 `3x3` 上做快速迭代，不扩规模
- 保留 `K=1..4` 的小实验矩阵，先把 `K=2,3` 的 `id_exact_rate` 做出实质提升
- 下一轮模型应优先探索：
  - 更直接的 support 监督方式
  - 更贴近 top-`K` 选边过程的训练目标
  - 明确区分“选边”与“数值回归”的两阶段机制

### 结果解释备注

- 当前 `outputs_modelv3/port_vs_kmax_summary.csv` 中显示的 `K_max = 0`，仅仅是因为这一轮只跑了 `K = 2, 3, 4`
- 该汇总表缺少 `K = 1` 的结果，因此不能被解释为“`modelv3` 在 `K=1` 也失败”
- 若后续需要正式比较 `K_max`，必须补跑 `N3x3, K=1`

## 2026-04-12：关于“物理识别上限”判据的补充

### 可采纳的核心观点

- 当前问题应明确区分两类瓶颈：
  - 算法瓶颈：模型没有学好
  - 物理瓶颈：边界测量本身不包含足够信息
- 在无噪声条件下，单纯继续比较不同神经网络精度并不能直接回答“是否达到该规模下的物理识别上限”
- 更合理的思路是增加一套“极限可拟合性”测试，用来判断失败到底来自模型能力不足，还是来自信息本身不可分

### 需要修正的地方

- 不能简单把“训练集也过拟合不上”直接等同于“已经达到物理极限”
- 这只能说明：在当前模型族、当前优化方式和当前训练目标下，系统已经很难再被拟合
- 若要更接近“物理极限”结论，还需要同时满足：
  - 已经使用了更宽松、明显偏向记忆能力的无噪声测试设置
  - 多种模型或求解方式都失败
  - 物理侧分析也显示有效信息维度受限或样本间存在强歧义

### 当前项目下更合适的判断标准

对于固定规模 `N` 和固定变化数量 `K`，建议建立一套“物理上限判据”：

1. 在无噪声条件下，使用专门的 limit-test 设置，而不是直接沿用当前偏稳健的主线模型
2. 去掉或显著减弱不必要的保守项，例如较强的 dropout、过强的排序约束、过强的未变化边压制
3. 允许模型以“训练集能否被记住”为目标，而不是以泛化为首要目标
4. 如果在这种放宽条件下，训练集 support 仍长期无法接近 100%，再把它视为“接近物理极限”的强信号

### 建议的验证流程

- 第一步：做一个专门的 `limit-test` 版本
  - 固定无噪声
  - 固定单一 `N`
  - 固定单一 `K`
  - 去掉不必要的稳健化设计
- 第二步：先只看训练集拟合能力
  - 如果训练集都记不住，说明问题首先不是泛化，而是信息与表示本身不足
- 第三步：再结合物理侧指标
  - 响应矩阵有效秩
  - 奇异值谱衰减
  - 样本间响应最小间隔
  - 是否存在大量近重复响应
- 第四步：只有当“算法侧记不住”与“物理侧确实低可分”同时出现时，才汇报为“接近该规模下的物理识别上限”

### 对当前工作的指导意义

- Gemini 的分析方向整体是有道理的，尤其是“不要把所有失败都归因于模型不够强”这一点非常重要
- 但在我们的项目里，下一步不是直接宣称已经到达物理极限，而是先新增一条更干净的无噪声 `limit-test` 支线
- 这条支线的目标不是发表最终结果，而是回答一个更基础的问题：
  - 当前失败究竟主要来自 support 学习机制，还是来自边界测量的信息天花板

## 2026-04-12：外部模型分析意见的吸收与修正

### 总体判断

- Gemini 与 Deepseek 的外部分析，方向上提供了两个有价值的提醒：
  - 不要把当前失败全部理解成“模型还不够大”
  - 在无噪声条件下，应专门区分“同规模泛化目标”与“物理上限诊断目标”
- 这与当前项目的新主线是相容的
- 但两份外部分析里也夹杂了一些对现有 `modelv3` 代码的错误判断，因此只能把它们当作思路参考，不能直接当作代码诊断结论

### 可以吸收的经验

- 对当前课题而言，可以放弃“跨规模泛化”作为近期目标
- 更合理的目标是：
  - 固定一个规模 `N`
  - 只研究该规模内部、随机样本之间的识别能力
- 因而保留 `8:1:1` 的 train / val / test 划分仍然是有意义的：
  - train：检验模型是否能真正学会该规模下的映射
  - val：选模型、看是否稳定
  - test：看同规模随机样本上的泛化能力
- 在无噪声条件下，应单独增加一条高容量 `limit-test` 支线，专门判断“当前失败更像算法瓶颈还是信息瓶颈”
- 若后续 GNN 与 MLP 两类大模型都在同一个 `(N, K)` 上训练集也无法逼近完美拟合，那么“接近物理上限”的判断才会更有说服力

### 需要修正的地方

- 外部分析中提到的某些具体代码问题并不适用于当前仓库中的 `modelv3`
- 例如：
  - 并不存在它所说的固定 `112` 条边的硬编码 edge embedding
  - 当前 `modelv3` 也不存在它提到的那种 `loss_sparse` 项
- 因此，外部分析里关于“某行代码导致当前模型失败”的结论不能直接采纳
- 另外，对 `N=3` 而言，盲目把 GNN 叠到很多层并不一定更好，因为图直径本来就小，过深更容易出现过平滑
- 在 `N=3` 上，更合理的是：
  - 宽度做大
  - 残差做强
  - 跳连做清楚
  - 支持集监督做直接

### 对当前主线的重新定位

- 当前主线不再追求“一套模型跨多个规模都通用”
- 当前主线改为：
  - 先固定 `N=3`
  - 在 `N=3` 内部把 `K=1..4` 的识别问题做扎实
  - 等到 `N=3` 上结果稳定后，再迁移到更大规模

## 2026-04-12：`limit-test` 的目的与方法说明

### 目的

`limit-test` 不是新的最终模型路线，而是一条诊断路线。

它要回答的问题是：

- 某个固定 `(N, K)` 失败，到底是因为当前模型没学会
- 还是因为边界电压里本来就没有足够信息来区分这些变化模式

### 核心思想

- 主线模型关心的是“同规模下能否稳健泛化”
- `limit-test` 关心的是“在明显放宽限制、明显提高容量后，训练集还能不能被学会”

如果一个高容量模型在无噪声、弱正则、固定规模、固定 `K` 的条件下，训练集仍然长期无法逼近几乎完美的 support 恢复，那么这才是“接近物理上限”的强信号

### 推荐做法

- 数据仍保留 `8:1:1`
- 但在 `limit-test` 中，训练集指标是第一优先级
- 同时保留 val / test，是为了防止把单次优化失败误认为物理极限

判断逻辑建议如下：

1. 若 train 很高，val / test 也高：该 `(N, K)` 明显可识别
2. 若 train 很高，但 val / test 明显差：说明信息足够，但模型只会记忆，泛化还没做好
3. 若 train 也始终上不去，且 GNN / MLP 两条线都失败：说明该 `(N, K)` 很可能已经接近物理识别上限

### 当前阶段的使用方式

- `limit-test` 先只在 `N=3` 上做
- 只测 `K=2, 3, 4`
- 不直接拿来汇报“最终最优模型”
- 它的作用是帮我们先判断：
  - 是该换训练目标
  - 还是已经撞到了信息天花板

## 2026-04-12：下一版双路线详细计划

### 共同原则

- 近期不研究跨规模泛化
- 近期只研究固定规模内部的随机样本泛化
- 保留 `8:1:1` 划分
- 先固定 `N=3`
- 默认测试 `K = 1, 2, 3, 4`
- 每条路线都保留三组指标：
  - train
  - val
  - test
- 重点先看：
  - `id_exact_rate`
  - `value_accuracy`
  - train 是否能逼近满分

### 路线 A：大容量 GNN

目标：

- 在保留图结构先验的前提下，做一个真正高容量、但适合 `N=3` 的 support-first GNN

结构建议：

- 输入仍沿用当前 excitation-conditioned 图表示
- 节点编码器宽度提升到 `128` 或 `256`
- 图主干采用 `3~4` 层残差图块，不建议在 `N=3` 上盲目堆到更深
- 每层后加入：
  - residual skip
  - layer norm 或 batch norm
  - GELU / LeakyReLU
- 采用 jump knowledge 或多层特征拼接，避免只用最后一层表示
- 跨激励融合不建议再做太复杂的注意力，优先试：
  - mean + max
  - 或轻量 attention pooling
- 输出继续拆成两头：
  - support score head
  - value regression head

训练建议：

- 第一阶段先把 support 头训稳：
  - BCE / focal 类损失
  - top-`K` ranking / margin loss
- 第二阶段再做联合训练：
  - support loss
  - changed-edge value loss
  - unchanged-edge 小权重 value loss
- 对 `N=3` 的主实验，建议默认关闭 dropout 或只保留极小 dropout
- `weight_decay` 设得很小
- 每个 `(K)` 至少跑 `3` 个 seed

这一条线的判断目标：

- 如果大容量 GNN 仍然在 `N=3, K=2/3` 上 train 都上不去，就说明问题不只是“主干太浅”

### 路线 B：大容量残差 MLP

目标：

- 去掉图网络的归纳偏置，直接测试“边界电压向量本身是否足够支持学习”
- 它同时也是对 GNN 结论的一个重要对照

结构建议：

- 输入直接展平为固定长度向量
- 做深宽残差 MLP：
  - 宽度 `1024` 或 `2048`
  - 深度 `6~8` 个残差块
  - 每块包含：
    - Linear
    - LayerNorm
    - GELU
    - Linear
    - residual
- 输出维度直接对应全部边
- 同样使用双头：
  - support head
  - value head

训练建议：

- 仍保留 `8:1:1`
- 同样优先看 train 指标，但必须同时记录 val / test
- 主损失与 GNN 线尽量统一，便于横向比较：
  - support BCE / focal
  - ranking loss
  - weighted value regression
- 默认也做 `3` 个 seed

这一条线的判断目标：

- 如果 MLP 能把 train 拟合得更高，而 GNN 不行，说明图结构实现或图归纳偏置可能限制了效果
- 如果 GNN 和 MLP 都卡在同一个 `(N, K)`，则更接近“信息本身受限”的解释

### 实验推进顺序

建议按这个顺序推进：

1. 先做 `N=3, K=2`
2. 若明显优于当前 `modelv3`，再测 `K=3`
3. 若 `K=3` 仍有进步，再测 `K=4`
4. 只有当 `N=3` 路线稳定后，再考虑上 `N=4`

### 当前最推荐的执行方式

- 不要先急着二选一
- 最合理的是：
  - GNN 作为主线
  - MLP 作为对照线
- 因为这能最快回答两个问题：
  - 图结构先验到底有没有帮上忙
  - 当前失败到底更像模型问题，还是信息问题
## 2026-04-12：modelo1 系列模型已落地

本轮已新增三条正式模型线，并统一记录到 `MODEL_DESIGN.md`：

- `modelo1_gnn`：大容量 support-aware GNN 主线
- `modelo1_mlp1`：带 score head 的高容量残差 MLP 对照线
- `modelo1_mlp2`：仅用纯 MSE 的高容量残差 MLP 对照线

本轮记录重点包括：

- 三个模型的功能分工
- 不同结果组合的解释方式
- 每个模型的架构与默认超参数
- 各模型 loss 的精确定义
- `N=3, K=2~4` 的推荐训练与推理命令
## 2026-04-13：N=3 上 modelo1 系列结果整理

本轮在固定 `N=3` 下，完成了四条 `modelo1` 线路的完整对照：

- `modelo1_gnn`
- `modelo1_gnn_mse`
- `modelo1_mlp1`
- `modelo1_mlp2`

统一测试范围为：

- `K = 2, 3, 4`
- 无噪声
- 固定 `8:1:1` 的 `train / val / test`

### 一、测试集结果总表

`modelo1_gnn`

- `K=2`：`test_id_exact_rate = 0.932`，`test_value_accuracy = 0.9876`
- `K=3`：`test_id_exact_rate = 0.893`，`test_value_accuracy = 0.9906`
- `K=4`：`test_id_exact_rate = 0.842`，`test_value_accuracy = 0.9901`

`modelo1_gnn_mse`

- `K=2`：`test_id_exact_rate = 0.977`，`test_value_accuracy = 0.9920`
- `K=3`：`test_id_exact_rate = 0.941`，`test_value_accuracy = 0.9915`
- `K=4`：`test_id_exact_rate = 0.905`，`test_value_accuracy = 0.9860`

`modelo1_mlp1`

- `K=2`：`test_id_exact_rate = 0.941`，`test_value_accuracy = 0.9848`
- `K=3`：`test_id_exact_rate = 0.873`，`test_value_accuracy = 0.9866`
- `K=4`：`test_id_exact_rate = 0.793`，`test_value_accuracy = 0.9848`

`modelo1_mlp2`

- `K=2`：`test_id_exact_rate = 0.979`，`test_value_accuracy = 0.9880`
- `K=3`：`test_id_exact_rate = 0.936`，`test_value_accuracy = 0.9884`
- `K=4`：`test_id_exact_rate = 0.889`，`test_value_accuracy = 0.9871`

### 二、与上一轮 `modelv3` 的对比

`modelv3` 的对应测试集结果为：

- `K=2`：`ID = 0.925`，`Value = 0.9823`
- `K=3`：`ID = 0.810`，`Value = 0.9779`
- `K=4`：`ID = 0.668`，`Value = 0.9701`

对比可见：

- 四条 `modelo1` 新线都整体优于 `modelv3`
- 提升最明显的是 `K=3,4` 时的 support 识别能力
- 尤其是新增的 `modelo1_gnn_mse`，把 GNN 线在 `K=3,4` 上的结果明显抬高了
- 说明这轮“先固定 `N=3`，再做结构与 loss 的正交对照”方向是有效的

### 三、当前最重要的实验结论

1. 当前最重要的结论不是“MLP 一定比 GNN 好”，而是“纯 `MSE` 版本整体优于 support-aware 版本”

- 在 MLP 组中：
  - `modelo1_mlp2 > modelo1_mlp1`
- 在 GNN 组中：
  - `modelo1_gnn_mse > modelo1_gnn`
- 这说明当前 `N=3` 设置下，`score head + ranking loss` 更像是在干扰最终的 top-`K` 恢复，而不是帮助它

2. 在加入 `modelo1_gnn_mse` 之后，GNN 线的判断发生了更新

- 之前如果只看 `modelo1_gnn`，会得出“GNN 不如 MLP”的结论
- 但加入同容量纯 `MSE` 的 `modelo1_gnn_mse` 后可以看到：
  - `K=2`：`modelo1_mlp2 = 0.979`，`modelo1_gnn_mse = 0.977`
  - `K=3`：`modelo1_gnn_mse = 0.941`，`modelo1_mlp2 = 0.936`
  - `K=4`：`modelo1_gnn_mse = 0.905`，`modelo1_mlp2 = 0.889`
- 这说明图结构先验并不是无效，而是**在不叠加额外 score/ranking 约束时，GNN 反而在更难的 `K=3,4` 上表现更强**

3. 当前瓶颈主要仍然体现在 `ID`，而不是数值回归

- 五条模型在 `K=2,3,4` 上的 `value_accuracy` 都已经稳定在 `0.982+`
- 真正拉开差距的是 `id_exact_rate`
- 说明当前问题依然不是“数值回归学不会”，而是“如何让 top-`K` support 排序更稳”

4. 当前严格阈值下，`N=3` 仍未正式通过

- 项目当前通过标准是：
  - `id_exact_rate >= 0.98`
  - `value_accuracy >= 0.90`
- 本轮最佳结果是 `modelo1_mlp2, K=2` 的 `id_exact_rate = 0.979`
- 它已经非常接近阈值，但在当前严格定义下仍然记为 `pass = False`

### 四、对后续工作的直接指导

当前阶段的最优先级结论是：

- `N=3` 下不应继续优先堆复杂的 score/ranking 设计
- 当前最值得继续推进的是两条纯 `MSE` 主线：
  - `modelo1_gnn_mse`
  - `modelo1_mlp2`
- 如果目标是继续冲击 `id_exact_rate >= 0.98`，那么当前最合理的做法不是再加复杂监督，而是围绕纯回归主线继续优化容量、正则和训练稳定性
- 下一步需要认真讨论的是：
  - `id_exact_rate` 的阈值是否必须固定为 `0.98`
  - 对每一个 `N`，是否都应单独寻找最合适的模型结构
  - 若目标是“估计该规模的最大可识别 `K`”，那么是否需要针对每个 `N` 先做充分的 per-`N` 模型优化，再宣布该 `N` 的 `K_max`

### 五、当前一句话总结

在固定 `N=3` 的完整五线对照中，新增的 `modelo1_gnn_mse` 证明了“GNN 并不一定弱，真正的问题更可能是 support-aware loss 干扰了学习”；当前最强路线已经从“更复杂的监督”转向“更干净的纯 `MSE` 回归”，其中 `K=2` 最优为 `modelo1_mlp2`，而 `K=3,4` 最优为 `modelo1_gnn_mse`。
