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
