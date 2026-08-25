# square_scale_study 模型设计记录

## 1. 当前阶段的研究口径

当前阶段不再把“跨规模泛化”作为主目标，而是先把**同一规模内的可识别性问题**研究清楚。
因此，我们对近期实验做如下统一约束：

- 只在固定规模内训练与测试，不要求一个模型同时适配多种 `N`
- 保留 `8:1:1` 的 `train / val / test` 划分
- 近期所有新模型优先在 `N = 3` 上验证
- 当 `N = 3` 上出现明确提升后，再扩展到更大规模
- 评价仍然同时保留两类核心指标：
  - `id_exact_rate`
  - `value_accuracy`

这样做的目的，是把“模型表达能力不足”和“问题本身接近物理识别极限”这两类原因分开。

## 2. 三条模型线分别在回答什么问题

本轮新增三条并行模型线：

- `modelo1_gnn`
- `modelo1_mlp1`
- `modelo1_mlp2`

它们不是简单重复，而是承担不同诊断作用。

### 2.1 `modelo1_gnn` 的作用

`modelo1_gnn` 是新的主线候选模型。它保留图结构先验，显式利用节点、边、激励之间的关系，同时使用 support 分数头和数值回归头联合训练。

它主要回答：

- 在固定方形网络上，图结构先验是否真的有帮助？
- 当我们显式优化 support 排序后，`id_exact_rate` 能否明显优于纯回归模型？
- 更大、更深的图模型，能否把 `N = 3` 下的训练与测试表现同时拉高？

### 2.2 `modelo1_mlp1` 的作用

`modelo1_mlp1` 是与 GNN 对照的高容量残差 MLP。它不显式做图消息传递，而是直接对展平后的输入做深层残差建模，但仍然保留 score head 和 value head。

它主要回答：

- 即使不使用图结构，只靠边界电压和激励编码，模型能否把 support 学出来？
- 如果 `MLP` 与 `GNN` 接近，说明当前收益可能主要来自“大容量拟合”，而不是图归纳偏置本身
- 如果 `GNN` 明显更好，说明图结构先验是有价值的

### 2.3 `modelo1_mlp2` 的作用

`modelo1_mlp2` 是最“纯”的对照线。它只做一个大容量回归器，不加 score head，不加 ranking loss，只用纯 `MSE`。

它主要回答：

- 如果只优化数值误差，模型能否自然学出正确的 support 排序？
- “回归做得好但 support 不准”的问题是否真实存在？
- score head 和 ranking loss 到底有没有实际作用

## 3. 三个结果出来以后应该怎么分析

这三条线的联合分析比单看一个最好结果更重要。建议按下面的逻辑解释：

### 情况 A：`modelo1_gnn > modelo1_mlp1 > modelo1_mlp2`

说明：

- 图结构先验有效
- 显式 support 监督有效
- 纯回归不足以稳定恢复变化边集合

这时后续应继续沿 `GNN + support-aware loss` 深化。

### 情况 B：`modelo1_gnn ≈ modelo1_mlp1 > modelo1_mlp2`

说明：

- support-aware 训练目标是关键
- 图结构先验有帮助，但不是决定性因素
- 当前问题的主要瓶颈更像“输出目标设计”而不是“主干类型”

这时可以继续沿更简单、更稳定的架构推进，未必需要非常复杂的 GNN。

### 情况 C：`modelo1_mlp1 > modelo1_gnn`，且两者都明显好于 `modelo1_mlp2`

说明：

- 当前 GNN 的归纳偏置或者实现方式可能反而束缚了模型
- 更直接的大容量映射反而更容易学到 support
- 接下来应优先检查 GNN 主干是否过度设计、过深、或融合方式不合适

### 情况 D：三个模型训练集都上不去

说明：

- 这就不是“换个 loss 或换个主干”能轻易解决的问题
- 需要开始怀疑当前 `(N, K)` 已经接近信息极限，或者输入表征本身还不够

这时应进入 `limit-test` / 物理可辨识性分析，而不是继续盲目堆模型。

## 4. 为什么要保留 train / val / test 三套指标

虽然当前阶段更关注“同一规模内是否能学会”，但 `val / test` 仍然有保留价值。

- `train` 用来判断模型有没有学会
- `val` 用来做 checkpoint 选择，防止只看某个偶然的训练高点
- `test` 用来确认同规模内随机样本上的稳定性

因此，当前阶段的核心口径不是“放弃泛化”，而是：

- **不再追求跨规模泛化**
- **仍然保留同规模内随机样本的泛化检验**

## 5. `limit-test` 的目的和使用方式

`limit-test` 不是最终模型，而是一条诊断路径。

它要回答的问题是：

- 当前失败，到底是模型还不够强？
- 还是边界观测本身已经不足以把这些变化模式区分开？

推荐用法如下：

1. 固定一个具体 `(N, K)`，例如 `N = 3, K = 4`
2. 使用明显更高容量、明显更弱正则的模型
3. 重点观察训练集 `id_exact_rate`
4. 如果训练集长期也上不去，再结合物理响应相似性分析，才有理由怀疑接近识别上限

所以，`limit-test` 的核心不是追求最终泛化结果，而是判断问题到底“卡在模型”还是“卡在信息”。

## 6. 本轮架构记录

本节用于记录每次正式落地的模型架构，作为后续比较基准。

### 6.1 `modelo1_gnn`

定位：

- 主线候选模型
- 保留图结构先验
- 显式优化 support 排序和数值回归

输入：

- 继续使用现有 `excitation-conditioned` 图表示
- 每个激励对应一张图
- 节点特征沿用当前主线：源点掩码、汇点掩码、电压扰动、边界节点掩码

主干：

- 节点编码器：两层 MLP，把节点特征映射到高维隐藏空间
- 图消息传递：`4` 层残差 `GATv2Conv`
- 每层后接：
  - residual skip
  - LayerNorm
  - 两层前馈网络
- 使用 jump-style 多层特征拼接，再投影回统一隐藏维度

跨激励融合：

- 对每个激励先得到边级候选表征
- 再对激励维度做轻量 attention pooling
- 同时保留 mean / max 信息，避免只依赖单一聚合方式

输出头：

- `score_head`：输出每条边是变化边的 support score
- `value_head`：输出每条边的电阻变化量回归值

当前默认规模：

- `hidden_dim = 256`
- `edge_hidden = 512`
- `heads = 8`
- `num_layers = 4`
- `dropout = 0.02`

设计动机：

- 比 `modelv3` 更大、更深
- 但跨激励部分仍保持相对克制，避免再次落入“融合太复杂、但不一定有效”的问题

### 6.2 `modelo1_mlp1`

定位：

- 高容量非图结构对照组
- 保留 support-aware 训练目标

输入：

- 将图输入整体展平为固定长度向量

主干：

- 深层残差 MLP
- 由输入投影层加 `8` 个残差块组成
- 每个残差块内部是两层全连接 + LayerNorm + GELU

输出头：

- `score_head`
- `value_head`

当前默认规模：

- `hidden_dim = 1536`
- `num_blocks = 8`
- `ff_multiplier = 2.0`
- `dropout = 0.02`

设计动机：

- 故意做成高容量、深层、宽层
- 让它足以成为 GNN 的强对照，而不是一个过弱的陪跑模型

### 6.3 `modelo1_mlp2`

定位：

- 纯回归对照组
- 用来检验“只优化数值误差是否足够”

输入与主干：

- 与 `modelo1_mlp1` 相同

输出头：

- 仅 `value_head`

当前默认规模：

- `hidden_dim = 1536`
- `num_blocks = 8`
- `ff_multiplier = 2.0`
- `dropout = 0.02`

设计动机：

- 保持主干容量不变
- 只去掉 score head 与 support-aware loss
- 这样可以把“架构容量”和“训练目标设计”的影响拆开分析

## 7. 各模型 loss 的精确定义

下面给出本轮三个模型的正式 loss 定义。

记：

- 共有 `M` 条边
- 第 `i` 条边的真实变化量为 `y_i`
- 第 `i` 条边的回归预测为 `v_i`
- 第 `i` 条边的 support score logit 为 `s_i`
- 定义变化边指示量

`m_i = 1(|y_i| > \epsilon)`

其中 `\epsilon` 为一个很小的判定阈值。

### 7.1 `modelo1_gnn` 与 `modelo1_mlp1`

这两个模型都采用三部分联合损失。

#### 7.1.1 Support BCE 损失

将变化边看作正类，不变边看作负类。

`L_bce = w_pos * mean_{i: m_i = 1} BCEWithLogits(s_i, 1) + w_neg * mean_{i: m_i = 0} BCEWithLogits(s_i, 0)`

其中：

- `w_pos` 是变化边权重
- `w_neg` 是未变化边权重

当前默认取值：

- `w_pos = 1.0`
- `w_neg = 0.15`

#### 7.1.2 Pairwise Ranking 损失

希望所有真实变化边的 score 都高于真实未变化边。

对于任意一对正负样本边 `(p, n)``：

`L_rank(p, n) = max(0, margin - (s_p - s_n))`

总体 ranking loss 取所有可用正负边对的平均：

`L_rank = mean_{p in P, n in N} max(0, margin - (s_p - s_n))`

当前默认取值：

- `margin = 1.0`

它的作用是：

- 不只要求“变化边分高一些”
- 而是直接要求“变化边分数必须显著压过未变化边”

这对 top-`K` support 恢复更直接。

#### 7.1.3 加权数值回归损失

在 value head 上使用分区加权 `MSE`：

`L_value = w_changed * mean_{i: m_i = 1} (v_i - y_i)^2 + w_unchanged * mean_{i: m_i = 0} (v_i - y_i)^2`

当前默认取值：

- `w_changed = 3.0`
- `w_unchanged = 0.05`

这样做的原因是：

- 我们最关心变化边上的数值是否预测准
- 未变化边的回归值也要压小，但不应该主导总损失

#### 7.1.4 总损失

总损失为：

`L_total = lambda_bce * L_bce + lambda_rank * L_rank + lambda_value * L_value`

当前默认取值：

- `lambda_bce = 1.0`
- `lambda_rank = 1.0`
- `lambda_value = 1.0`

### 7.2 `modelo1_mlp2`

`modelo1_mlp2` 只做纯回归，使用全边 `MSE`：

`L_total = (1 / M) * sum_{i=1}^{M} (v_i - y_i)^2`

它没有：

- score head
- BCE loss
- ranking loss

推理时直接按 `|v_i|` 取 top-`K` 作为变化边预测。

## 8. 推荐的近期实验顺序

建议优先做最短闭环，而不是一次性全面铺开。

### 第一阶段

- 固定 `N = 3`
- 先做 `K = 2`
- 三个模型都跑 `3` 个 seeds

目标：

- 看谁最容易把 `train / val / test` 同时拉起来

### 第二阶段

- 若第一阶段有明确提升，再做 `K = 3`

目标：

- 看提升是否只停留在较简单场景

### 第三阶段

- 若 `K = 3` 仍有提升，再做 `K = 4`

目标：

- 看 support-aware 训练在更困难设置下是否真正扩大了可识别范围

## 9. 训练与推理命令

下面给出可以直接在 PowerShell 中运行的命令模板。

### 9.1 `modelo1_gnn`

训练：

```powershell
python square_scale_study\models\modelo1_gnn\train.py `
  --meta-path square_scale_study\data\N3x3\square_N3x3_K2_meta.json `
  --out-dir square_scale_study\outputs_modelo1_gnn\N3x3_K2
```

推理：

```powershell
python square_scale_study\models\modelo1_gnn\inference.py `
  --meta-path square_scale_study\data\N3x3\square_N3x3_K2_meta.json `
  --out-dir square_scale_study\outputs_modelo1_gnn\N3x3_K2 `
  --split test
```

### 9.2 `modelo1_mlp1`

训练：

```powershell
python square_scale_study\models\modelo1_mlp1\train.py `
  --meta-path square_scale_study\data\N3x3\square_N3x3_K2_meta.json `
  --out-dir square_scale_study\outputs_modelo1_mlp1\N3x3_K2
```

推理：

```powershell
python square_scale_study\models\modelo1_mlp1\inference.py `
  --meta-path square_scale_study\data\N3x3\square_N3x3_K2_meta.json `
  --out-dir square_scale_study\outputs_modelo1_mlp1\N3x3_K2 `
  --split test
```

### 9.3 `modelo1_mlp2`

训练：

```powershell
python square_scale_study\models\modelo1_mlp2\train.py `
  --meta-path square_scale_study\data\N3x3\square_N3x3_K2_meta.json `
  --out-dir square_scale_study\outputs_modelo1_mlp2\N3x3_K2
```

推理：

```powershell
python square_scale_study\models\modelo1_mlp2\inference.py `
  --meta-path square_scale_study\data\N3x3\square_N3x3_K2_meta.json `
  --out-dir square_scale_study\outputs_modelo1_mlp2\N3x3_K2 `
  --split test
```

### 9.4 一次跑 `K = 2, 3, 4`

#### `modelo1_gnn`

```powershell
$jobs = @(
  @{ Meta = "square_scale_study\data\N3x3\square_N3x3_K2_meta.json"; Out = "square_scale_study\outputs_modelo1_gnn\N3x3_K2" },
  @{ Meta = "square_scale_study\data\N3x3\square_N3x3_K3_meta.json"; Out = "square_scale_study\outputs_modelo1_gnn\N3x3_K3" },
  @{ Meta = "square_scale_study\data\N3x3\square_N3x3_K4_meta.json"; Out = "square_scale_study\outputs_modelo1_gnn\N3x3_K4" }
)

foreach ($job in $jobs) {
  python square_scale_study\models\modelo1_gnn\train.py --meta-path $job.Meta --out-dir $job.Out
  python square_scale_study\models\modelo1_gnn\inference.py --meta-path $job.Meta --out-dir $job.Out --split test
}
```

#### `modelo1_mlp1`

```powershell
$jobs = @(
  @{ Meta = "square_scale_study\data\N3x3\square_N3x3_K2_meta.json"; Out = "square_scale_study\outputs_modelo1_mlp1\N3x3_K2" },
  @{ Meta = "square_scale_study\data\N3x3\square_N3x3_K3_meta.json"; Out = "square_scale_study\outputs_modelo1_mlp1\N3x3_K3" },
  @{ Meta = "square_scale_study\data\N3x3\square_N3x3_K4_meta.json"; Out = "square_scale_study\outputs_modelo1_mlp1\N3x3_K4" }
)

foreach ($job in $jobs) {
  python square_scale_study\models\modelo1_mlp1\train.py --meta-path $job.Meta --out-dir $job.Out
  python square_scale_study\models\modelo1_mlp1\inference.py --meta-path $job.Meta --out-dir $job.Out --split test
}
```

#### `modelo1_mlp2`

```powershell
$jobs = @(
  @{ Meta = "square_scale_study\data\N3x3\square_N3x3_K2_meta.json"; Out = "square_scale_study\outputs_modelo1_mlp2\N3x3_K2" },
  @{ Meta = "square_scale_study\data\N3x3\square_N3x3_K3_meta.json"; Out = "square_scale_study\outputs_modelo1_mlp2\N3x3_K3" },
  @{ Meta = "square_scale_study\data\N3x3\square_N3x3_K4_meta.json"; Out = "square_scale_study\outputs_modelo1_mlp2\N3x3_K4" }
)

foreach ($job in $jobs) {
  python square_scale_study\models\modelo1_mlp2\train.py --meta-path $job.Meta --out-dir $job.Out
  python square_scale_study\models\modelo1_mlp2\inference.py --meta-path $job.Meta --out-dir $job.Out --split test
}
```

## 10. 当前推荐执行策略

当前最推荐的推进顺序是：

1. 先跑 `modelo1_gnn`
2. 再跑 `modelo1_mlp1`
3. 最后跑 `modelo1_mlp2`

原因是：

- `modelo1_gnn` 是主线候选
- `modelo1_mlp1` 是最关键的强对照
- `modelo1_mlp2` 用来验证“只做回归是否足够”

如果三者都跑完，我们就能比较清楚地区分：

- 是图结构有用
- 还是 support-aware loss 有用
- 还是单纯增大容量就足够

## 11. 2026-04-13 补充：`modelo1_gnn_mse`

为了把“模型类型”和“loss 设计”的影响彻底拆开，本轮补充一条新的 GNN 对照线：

- `modelo1_gnn_mse`

它的定位是：

- 与 `modelo1_gnn` 使用同等容量、同等深度、同等跨激励融合方式
- 只移除 `score_head`
- 只保留 `value_head`
- 训练时只使用纯 `MSE`

这样就形成了两组完全对称的对照：

- GNN 组：
  - `modelo1_gnn`：GNN + score/value 双头 + support-aware loss
  - `modelo1_gnn_mse`：GNN + 纯 value 头 + 纯 `MSE`
- MLP 组：
  - `modelo1_mlp1`：MLP + score/value 双头 + support-aware loss
  - `modelo1_mlp2`：MLP + 纯 value 头 + 纯 `MSE`

### 11.1 这个模型主要回答什么问题

`modelo1_gnn_mse` 的作用不是为了直接替代主线，而是为了回答：

- 对 GNN 而言，性能差异究竟主要来自“图结构主干”，还是主要来自“loss 设计”？
- 如果 `modelo1_gnn_mse > modelo1_gnn`，说明 support-aware loss 在当前 `N=3` 上可能确实干扰了训练
- 如果 `modelo1_gnn_mse ≈ modelo1_gnn`，说明 loss 不是决定性因素，瓶颈更像是主干结构本身
- 如果 `modelo1_gnn_mse < modelo1_gnn`，说明对 GNN 来说，score/ranking 仍可能有正面价值

### 11.2 架构记录

`modelo1_gnn_mse` 与 `modelo1_gnn` 的主干完全一致：

- 节点编码器：两层 MLP
- 图消息传递：`4` 层残差 `GATv2Conv`
- 多层特征拼接：jump-style aggregation
- 跨激励融合：attention pooling + mean pool + max pool
- 边级主干：与 `modelo1_gnn` 相同的高容量 edge trunk

唯一差别是输出头：

- `modelo1_gnn`：`score_head + value_head`
- `modelo1_gnn_mse`：仅 `value_head`

默认超参数与 `modelo1_gnn` 保持一致：

- `hidden_dim = 256`
- `edge_hidden = 512`
- `heads = 8`
- `num_layers = 4`
- `dropout = 0.02`

### 11.3 损失定义

`modelo1_gnn_mse` 只使用纯回归 `MSE`：

`L_total = (1 / M) * sum_{i=1}^{M} (v_i - y_i)^2`

其中：

- `M` 是总边数
- `v_i` 是第 `i` 条边的预测变化量
- `y_i` 是第 `i` 条边的真实变化量

推理时按 `|v_i|` 取 top-`K` 作为变化边预测。

### 11.4 运行命令

训练：

```powershell
python square_scale_study\models\modelo1_gnn_mse\train.py `
  --meta-path square_scale_study\data\N3x3\square_N3x3_K2_meta.json `
  --out-dir square_scale_study\outputs_modelo1_gnn_mse\N3x3_K2
```

推理：

```powershell
python square_scale_study\models\modelo1_gnn_mse\inference.py `
  --meta-path square_scale_study\data\N3x3\square_N3x3_K2_meta.json `
  --out-dir square_scale_study\outputs_modelo1_gnn_mse\N3x3_K2 `
  --split test
```

一次跑 `K = 2, 3, 4`：

```powershell
$jobs = @(
  @{ Meta = "square_scale_study\data\N3x3\square_N3x3_K2_meta.json"; Out = "square_scale_study\outputs_modelo1_gnn_mse\N3x3_K2" },
  @{ Meta = "square_scale_study\data\N3x3\square_N3x3_K3_meta.json"; Out = "square_scale_study\outputs_modelo1_gnn_mse\N3x3_K3" },
  @{ Meta = "square_scale_study\data\N3x3\square_N3x3_K4_meta.json"; Out = "square_scale_study\outputs_modelo1_gnn_mse\N3x3_K4" }
)

foreach ($job in $jobs) {
  python square_scale_study\models\modelo1_gnn_mse\train.py --meta-path $job.Meta --out-dir $job.Out
  python square_scale_study\models\modelo1_gnn_mse\inference.py --meta-path $job.Meta --out-dir $job.Out --split test
}
```

## 12. 2026-04-14 补充：`modelo2_gnn`

### 12.1 定位

`modelo2_gnn` 是在 `modelo1_gnn_mse` 基础上的更大规模版本，目标不是改变任务口径，而是让纯回归 GNN 更适合 `6x6` 到 `10x10` 的规模扩展。

它继续回答同一个问题：

- 在固定 `K`、无噪声、边界激励和边界测量条件下，纯回归 GNN 能否在更大规模上保持较高的 `id_exact_rate`

### 12.2 架构变化

相对于 `modelo1_gnn_mse`，本轮做了四个明确扩展：

1. **优先增大宽度而不是盲目加深**
- 默认 `hidden_dim` 提升到 `384`
- 默认 `edge_hidden` 提升到 `1024`
- 默认层数为 `6`

2. **加入二阶消息边**
- 除原有相邻节点消息边外，加入距离为 `2` 的二阶节点连接
- 目的不是完全改图，而是在不过度加深网络的情况下扩大感受野

3. **把原先的线性 JK 升级为 attention-based JK**
- 每层节点表示都会保留
- 通过学习到的层权重，对不同深度的表示做自适应融合
- 这样在更大规模上可以避免只依赖最后一层表示

4. **加入静态拓扑特征编码**
- 节点侧使用：
  - 归一化坐标
  - 边界节点标记
  - 归一化度数
  - 到中心的归一化距离
- 边侧使用：
  - 边中点坐标
  - 方向增量
  - 边长度
  - 两端是否为边界节点
  - 中点径向位置

### 12.3 损失定义

`modelo2_gnn` 仍然坚持纯回归：

`L_total = (1 / M) * sum_{i=1}^{M} (v_i - y_i)^2`

其中：

- `M` 是总电阻边数
- `v_i` 是第 `i` 条边预测的电阻变化量
- `y_i` 是第 `i` 条边真实的电阻变化量

推理时仍按 `|v_i|` 取 top-`K` 作为变化边识别结果。

### 12.4 默认超参数

- `hidden_dim = 384`
- `edge_hidden = 1024`
- `gat_heads = 8`
- `num_layers = 6`
- `ff_multiplier = 2.5`
- `dropout = 0.03`
- `excitation_chunk_size = 12`

## 13. 2026-04-14 补充：`modelo2_mlp`

### 13.1 定位

`modelo2_mlp` 是在 `modelo1_mlp2` 基础上的更大容量纯回归 MLP，对照目标很明确：

- 若更强 MLP 仍明显落后于更强 GNN，则说明拓扑归纳偏置在大规模上确实更有价值
- 若更强 MLP 追平或超过更强 GNN，则说明边界响应本身已经足够支撑高容量全连接建模

### 13.2 架构变化

本轮主要扩大了宽度、深度和前馈块表达能力：

- `hidden_dim` 提升到 `3072`
- `num_blocks` 提升到 `12`
- `ff_multiplier` 提升到 `3.0`
- 每个残差块由普通两层前馈改为 `GEGLU` 风格门控前馈
- 激活函数统一为 `GELU / GEGLU`

它仍然保持：

- 输入展平
- 深残差 MLP 主干
- 单 value 头输出全边回归值

### 13.3 损失定义

`modelo2_mlp` 同样只使用纯回归：

`L_total = (1 / M) * sum_{i=1}^{M} (v_i - y_i)^2`

推理时同样按 `|v_i|` 取 top-`K` 作为变化边预测。

### 13.4 默认超参数

- `hidden_dim = 3072`
- `num_blocks = 12`
- `ff_multiplier = 3.0`
- `dropout = 0.03`

## 14. 2026-04-14 大规模数据生成口径

`N=6~10` 的数据补齐继续沿用当前主线的统一口径，不另起新规则：

- 拓扑：正方形网格 `N x N`
- 每个 `(N, K)` 单独生成数据
- 当前默认 `K = 1..6`
- 每个 `(N, K)` 共 `10000` 个样本
  - train: `8000`
  - val: `1000`
  - test: `1000`
- 激励与测量限制：
  - 只使用边缘节点作为激励端口
  - 只记录边缘节点电压响应
- 正演方法：
  - 直接构建电导矩阵
  - 直接解基尔霍夫方程组
  - 不使用额外近似技巧
- 电阻变化规则：
  - 基准电阻 `R0 = 1000 Ω`
  - 每个样本固定 `K` 条变化边
  - 变化幅值在 `±20%` 内连续采样

### 14.1 数据文件类型

每个 `(N, K)` 至少包含：

- `square_N6x6_K1_train.csv`
- `square_N6x6_K1_val.csv`
- `square_N6x6_K1_test.csv`
- `square_N6x6_K1_meta.json`

其中：

- `csv` 保存逐激励展开后的边界电压响应与标签
- `meta.json` 保存拓扑结构、端口定义、激励集合、数据划分和生成参数

### 14.2 训练时使用的数据类型

从 `csv` 读入后，训练脚本会整理成：

- 输入 `x`：
  - 形状为 `[batch, excitation, node, feature]`
  - 节点特征仍为四类：
    - source mask
    - ground mask
    - boundary voltage perturbation
    - boundary node mask
- 标签 `y`：
  - 形状为 `[batch, num_resistors]`
  - 表示每条电阻边相对于基准值的回归目标

## 15. 2026-04-16 拟议：统一中小规模 `GNN` 主线重构

### 15.1 为什么需要新一版统一 `GNN`

当前已有结果说明：

- `MLP` 不适合作为后续三项主线任务的 sweep 模型
- `modelo2_gnn` 虽然在大规模上比 `MLP` 更有希望，但它过重、过慢，不适合做大量 `(设置, K)` 扫描
- 接下来真正需要的是一个：
  - 足够轻，能在 `3x3~6x6` 和多种协议上稳定批量训练
  - 足够统一，能同时支持三项子项目
  - 仍然保留图结构先验的中等容量 `GNN`

因此下一版建议不再沿 `modelo2_gnn` 的超重注意力结构继续堆，而是重构为一个更适合 sweep 的统一 `GNN` 主模型。

### 15.2 设计目标

这套统一 `GNN` 主模型要同时支持三种输入协议：

1. 全电阻可变 + 全端口可测
2. 候选可变电阻集合受限
3. 可测端口集合受限

因此它必须同时理解两类额外信息：

- 哪些边允许变化
- 哪些边界端口实际可测

### 15.3 输入接口重构计划

#### 节点动态特征

在当前四类动态节点特征基础上，扩展为五类：

- source mask
- ground mask
- boundary voltage perturbation
- boundary node mask
- measured port mask

其中：

- `boundary node mask` 表示“这个节点是否位于边界”
- `measured port mask` 表示“这个边界节点当前是否真的参与测量”

这样在子项目 3 中，即使只减少可测端口而不改变拓扑，模型也能显式知道哪些端口信息是可用的。

#### 节点静态特征

为了提高中小规模泛化与位置辨识能力，建议追加固定静态节点特征：

- 归一化 `x` 坐标
- 归一化 `y` 坐标
- 是否 corner
- 所属边界侧别（上/下/左/右，可做 one-hot）

#### 边静态特征

统一主模型后续应显式使用边特征，至少包括：

- `candidate_change_mask`
- orientation（horizontal / vertical）
- `boundary_edge_flag`
- 边中点归一化 `x`
- 边中点归一化 `y`

其中 `candidate_change_mask` 是子项目 2 的关键。

### 15.4 主干结构建议

#### 总体原则

- 放弃当前过重的多头 `GATv2` 深堆叠方案
- 使用更适合 sweep 的中等容量残差图网络
- 保留 excitation-conditioned 图表示
- 但跨激励融合尽量简化，避免再次把主要算力花在注意力上

#### 建议结构

- hidden width：`192` 或 `256`
- 图主干层数：`4`
- 每层采用：
  - 残差连接
  - 归一化
  - 边特征参与消息传递
- 更推荐的卷积类型：
  - `GINEConv` 或同类可直接接收 edge attribute 的残差图块
- 不再默认启用：
  - 二阶消息边
  - 深层多头注意力
  - 过重的跨激励 attention

#### 多层融合

- 保留轻量版 `JK`
- 建议使用：
  - learned weighted sum
  - 或 concat 后接线性压缩

#### 跨激励融合

建议优先从轻量方案开始：

- `mean + max` 融合
- 或简单 gate pooling

而不是再次引入大规模 attention pooling。

### 15.5 输出头建议

当前三项子项目的正式评价仍然基于 top-`K` support 恢复，因此主输出仍然使用边级回归值：

- 主输出：每条边的 `ΔR` 回归值

第一阶段建议继续以“纯回归主线”为核心，不把复杂 support 头重新作为默认主线。

但为了后续必要时增强，可以预留一个轻量辅助头接口：

- 可选辅助输出：support score head

默认策略是：

- 第一轮先不开启 support head
- 如果 `N=3~6` 上仍然明显卡在 support 排序，再考虑以很小权重引入辅助监督

### 15.6 损失函数计划

#### 第一阶段默认主损失：加权纯回归

建议从“加权纯回归”开始，而不是直接回到 ranking loss。

定义：

\[
L_{\text{value}} =
\alpha \cdot \frac{1}{|E_c|} \sum_{e \in E_c} (\hat{\Delta R}_e - \Delta R_e)^2
+
\beta \cdot \frac{1}{|E_u|} \sum_{e \in E_u} (\hat{\Delta R}_e - 0)^2
\]

其中：

- `E_c`：真实变化边集合
- `E_u`：真实未变化边集合
- 默认建议：
  - `alpha = 6`
  - `beta = 1`

这样做的目的不是改任务定义，而是在不打乱纯回归口径的情况下，缓解“稀疏目标被大量未变化边淹没”的问题。

#### 第二阶段备选：轻量辅助 support loss

只有当第一阶段仍然不够时，再考虑加入一个很小权重的辅助项：

\[
L = L_{\text{value}} + \lambda_{\text{sup}} L_{\text{sup}}
\]

其中：

- `L_sup` 优先使用简单 `BCE`
- 不建议第一时间重新引入更强的 ranking loss
- 默认建议：
  - `lambda_sup = 0.05 ~ 0.10`

### 15.7 推理规则

推理规则保持不变：

- 按 `|\hat{\Delta R}|` 从大到小选择前 `K` 条边
- 作为 `id` 预测
- 数值指标只在真实变化边上统计

这样可以保证：

- 三项子项目之间评价口径完全一致
- 新模型与旧主线结果仍可直接比较

### 15.8 训练策略建议

#### 环境

- 继续使用现有 `5090` 环境
- 不需要更换软件环境

#### 训练策略

- 混合精度默认开启
- 先控制 batch 在中小范围：
  - `batch_size = 8~16`
  - `eval_batch_size = 16~32`
- gradient clip：
  - `1.0`
- 不必每个 epoch 都做完整 train/val 全量评估
- 建议：
  - 训练日志每个 epoch 打印 loss
  - 完整 train/val 指标每 `3` 或 `5` 个 epoch 评估一次

这样做的原因是：

- 这三项子项目后续是大批量扫表
- 评估开销本身已经会明显拖慢总周期

### 15.9 这套统一 `GNN` 为什么适合三项子项目

它的优点在于：

- 对子项目 1：
  - 能作为新的中小规模主线模型
- 对子项目 2：
  - 借助 `candidate_change_mask` 直接支持“候选可变边集合受限”
- 对子项目 3：
  - 借助 `active_port_mask` 直接支持“可用端口受限但拓扑不变”

也就是说，后续三项工作可以尽量在同一个模型骨架下完成，而不是每次换协议就重写一套模型。

### 15.10 推荐的近期落地顺序

1. 先实现这套统一中小规模 `GNN`
2. 先在子项目 1 上验证它能否稳定得到 `3x3~6x6` 主曲线
3. 若子项目 1 可行，再进入子项目 2 与子项目 3
4. 只有当子项目 1 已经证明主线稳定时，才考虑额外 support 辅助头

## 16. 2026-04-16 拟议：`modelg1`（子项目 1 首版四跳中小规模 GNN）

### 16.1 使用范围

`modelg1` 的首版目标只针对子项目 1：

- 规模：`N = 3, 4, 5, 6`
- 任务：固定 `K` 的纯回归逆识别
- 数据：直接复用现有正式数据，不重新生成

因此 `modelg1` 第一版的设计原则是：

- 先不为子项目 2 和 3 引入额外数据依赖
- 但结构上为后续扩展预留空间

### 16.2 从 `GNN优化.txt` 中吸收的有效建议

本轮决定实际吸收的点是：

1. 保留多层信息，使用 `JK`
2. 激活函数统一改为 `GELU` 或 `SiLU`
3. 纯回归场景下允许引入很轻的输出稀疏正则
4. 学习率策略采用余弦退火重启

本轮暂不吸收的点是：

1. `score head`
2. `ranking loss`
3. Graph-NMS 后处理
4. 复杂相对误差回归头

原因是：

- `modelg1` 的第一目标是建立一个干净、稳定、可批量扫表的中小规模 `GNN`
- 不是重新引入过多不可控因素

### 16.3 输入与表征

#### 动态节点输入

首版 `modelg1` 直接复用当前数据，因此动态节点输入仍为四类：

- source mask
- ground mask
- boundary voltage perturbation
- boundary node mask

#### 静态节点特征

这部分不依赖数据文件新增字段，可由拓扑直接构造：

- 归一化 `x`
- 归一化 `y`
- 是否边界节点
- 是否 corner

#### 静态边特征

同样直接由拓扑构造：

- horizontal / vertical orientation
- 是否 boundary edge
- 边中点归一化 `x`
- 边中点归一化 `y`

### 16.4 主干结构

#### 总体定位

- 这是一个“四跳”中小规模 `GNN`
- “四跳”在实现上对应 `4` 个消息传递图块
- 目标是在 `N<=6` 的范围内既保留结构表达能力，又控制训练与评估开销

#### 具体建议

- 图块数：`4`
- hidden width：`256`
- edge hidden：`128`
- 每层结构：
  - edge-conditioned message passing
  - residual connection
  - normalization
  - `GELU`
- 卷积类型优先：
  - `GINEConv`
  - 或等价的可接收边特征的残差消息传递块

#### 不采用的设计

- 不使用二阶消息边
- 不使用深层多头 `GATv2`
- 不使用重型跨激励 attention

### 16.5 多层与跨激励融合

#### 多层融合

为了抑制过平滑并保留局部突变信息，`modelg1` 使用：

- `JK concat`

也就是：

- 把每一层图块输出都保留下来
- 在 excitation 内部先拼接
- 再通过线性层压回统一维度

这一步是本轮从 `GNN优化.txt` 中吸收的最重要结构建议。

#### 跨激励融合

首版不再采用复杂 attention，而采用轻量融合：

- `mean pooling`
- `max pooling`
- 二者拼接后线性压缩

理由是：

- 对子项目 1，先把主干稳住比继续堆复杂激励注意力更重要

### 16.6 边级回归头

边级预测仍按标准流程进行：

- 取边两端节点嵌入
- 拼接边静态特征
- 送入边回归头
- 输出每条边的归一化变化量预测

建议：

- edge readout hidden：`512`
- 激活：`GELU`
- 末层不加激活，直接输出连续值

### 16.7 训练目标

#### 主目标：归一化后的加权 MSE

首先把目标改写成无量纲形式：

\[
z = \frac{\Delta R}{0.2R_0}
\]

这样真实目标大致落在 `[-1, 1]`，数值条件明显更好。

首版主损失定义为：

\[
L_{\text{mse}} =
\alpha \cdot \frac{1}{|E_c|} \sum_{e \in E_c} (\hat z_e - z_e)^2
+
\beta \cdot \frac{1}{|E_u|} \sum_{e \in E_u} (\hat z_e - 0)^2
\]

建议默认：

- `alpha = 6`
- `beta = 1`

#### 轻量稀疏引导

在不引入分类头的前提下，加入一个很小的输出 `L1` 正则：

\[
L = L_{\text{mse}} + \lambda_{1}\|\hat z\|_1
\]

建议默认：

- `lambda_1 = 1e-4`

这样做的目的不是追求严格稀疏解，而是稍微压低未变化边上的伪响应，帮助 top-`K` 排序更稳。

### 16.8 优化器与学习率

结合当前项目经验和 `GNN优化.txt` 建议，首版 `modelg1` 推荐：

- optimizer：`AdamW`
- 初始学习率：`3e-4`
- weight decay：`1e-5`
- gradient clip：`1.0`

学习率调度采用：

- `CosineAnnealingWarmRestarts`

建议首版参数：

- `T_0 = 20`
- `T_mult = 2`
- `eta_min = 1e-5`

这样做是为了避免固定衰减过早进入平坦期。

### 16.9 训练细节

- dropout：`0.03`
- batch size：
  - `N<=4`：`16`
  - `N=5,6`：`8` 或 `12`
- eval batch size：
  - `16~32`
- 训练总轮数上限：
  - `120`
- 早停：
  - 以 `val_id_exact_rate` 为第一排序指标
  - patience 建议 `25`

### 16.10 推理规则

首版 `modelg1` 不引入额外后处理，保持标准口径：

- 先把归一化预测值还原为 `ΔR`
- 按 `|\hat{\Delta R}|` 排序
- 取前 `K` 条边作为变化边预测

这样可以与现有所有主线结果直接对比。

### 16.11 为什么这是首版最合适的方案

因为它同时满足四个条件：

1. 比 `modelo2_gnn` 轻很多，适合中小规模扫表
2. 比 `modelo1_gnn_mse` 更系统地吸收了有效经验
3. 不重新引入复杂多任务 loss
4. 仍然保留最关键的图结构先验与多层信息

### 16.12 实现后的第一批实验

`modelg1` 实现后，第一批实验固定为：

- `N = 3, 4, 5, 6`
- 按现有数据逐个扫描 `K`
- 先只做子项目 1
- 先判断 `6x6` 是否仍有继续展开子项目 2 和 3 的必要
