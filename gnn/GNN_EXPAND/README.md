# GNN_EXPAND 说明

说明：
- 本目录用于在不修改原有 GNN 主线代码的前提下，扩展到不同拓扑和节点规模。
- `CLS / REG / joint_inference` 继续沿用当前 clean 主线的最佳方法：
  - 分类：`GNN_CLS/modelo3`
  - 回归：`GNN_REG/o4a2`
- 所有扩展代码均复制并收敛到 `gnn/GNN_EXPAND/common` 与四个阶段子目录中，不回写 `gnn/GNN_CLS`、`gnn/GNN_REG`、`gnn/GNN_CMEI_INFERENCE` 原程序。

## 一、四阶段目录
- `stage1_square_10x10`
  - `10x10` 正方形，`100` 节点，`180` 电阻边
- `stage2_rect_6x10`
  - `6x10` 矩形，`60` 节点，`104` 电阻边
- `stage3_honeycomb_63`
  - `7x9` 蜂窝状邻接，`63` 节点，`158` 电阻边
- `stage4_transfer_circlecut_69`
  - `9x9` 角点裁切近似圆形，`69` 节点，`120` 电阻边

每个阶段都固定包含：
- `cls/train.py`
- `cls/inference.py`
- `reg/train.py`
- `reg/inference.py`
- `joint_inference/inference.py`
- `README.md`
- `Log.md`

## 二、数据
- `GNN_EXPAND` 现已新增原生数据目录：
  - `gnn/GNN_EXPAND/data`
- 四套默认数据：
  - `square_10x10.csv`
  - `rect_6x10.csv`
  - `honeycomb_63.csv`
  - `circlecut_69.csv`
- 每套数据都带：
  - `*_meta.json`
- 这些数据由：
  - `gnn/GNN_EXPAND/generate_expand_datasets.py`
  直接在目标拓扑上做基尔霍夫正演生成，而不是继续复用根目录 `8x8` CSV。

### 激励与测量约束
- 激励只使用外部节点。
- 测量只输出外部节点电压。
- 当前四套数据都固定：
  - `28` 个外部节点
  - `32` 组激励

## 三、与旧 clean 数据的兼容适配
- `GNN_EXPAND/common` 仍保留对旧 `8x8` clean 数据的兼容逻辑。
- 如果手动把数据路径切回根目录旧 CSV，仍会启用两层适配：

### 1) 边界电极映射
- 原 CSV 的 28 个 `v_node*` 通道仍按顺时针电极顺序读取。
- 扩展拓扑会为每个阶段生成自己的 28 个顺时针边界节点列表。
- 电压观测、`src_node`、`gnd_node` 都按电极顺序映射到目标拓扑，不再直接复用 8x8 的节点 id。

### 2) 电阻标签映射
- 原始 `r*_id` 仍来自 8x8 clean 数据。
- 扩展回归入口会读取对应 `_meta.json`，恢复源拓扑电阻边。
- 然后按归一化边中点位置与方向相近性，把源 `112` 条边映射到目标拓扑电阻边索引。

## 四、默认 warm start
- `stage1 / stage2 / stage3`
  - `CLS` 默认从 `gnn/GNN_CLS/modelo3/outputs/training_data64Nodes_2/model_last.pt`
  - `REG` 默认从 `gnn/GNN_REG/o4a2/outputs/training_data64Nodes_2/model_last.pt`
- `stage4_transfer_circlecut_69`
  - `CLS` 默认尝试从 `gnn/GNN_EXPAND/stage1_square_10x10/cls/outputs/training_data64Nodes_2/model_last.pt`
  - `REG` 默认尝试从 `gnn/GNN_EXPAND/stage1_square_10x10/reg/outputs/training_data64Nodes_2/model_last.pt`
  - 若 stage1 权重尚不存在，会自动跳过 warm start

## 五、输出约定
- `cache` 与 `outputs` 全部留在各阶段自己的 `cls/reg/joint_inference` 子目录内部。
- 当前定位是“拓扑/规模扩展容器”，重点先保证：
  - 目录结构完整
  - 训练/推理入口可独立运行
  - 输出口径不与原主线混写
  - 文档与日志可持续追加

## 六、2026-04-02 数据口径更正
- 先前文档中把四套 `EXPAND` 原生数据统一写成 `28` 个外部节点、`32` 组激励，这个说法不正确。
- `GNN_EXPAND` 的原生数据生成遵守的真实规则是：
  - 激励只使用外部节点
  - 测量只输出外部节点电压
  - 外部节点数和激励组数由目标拓扑自身的边界规模决定
- 当前四套数据的真实规模：
  - `square_10x10`: `36` 个外部节点，`40` 组激励
  - `rect_6x10`: `28` 个外部节点，`32` 组激励
  - `honeycomb_63`: `28` 个外部节点，`32` 组激励
  - `circlecut_69`: `24` 个外部节点，`28` 组激励
- 对 `square_10x10` 而言，`10x10` 网格的外边界节点数为 `2*(10+10)-4 = 36`，因此激励不是 `32`，而是 `36` 组相邻边界激励加 `4` 组额外跨边界激励，共 `40` 组。

## 七、2026-04-03 训练结果与汇总图
- 已吸收根目录 `拓展训练日志.txt`，并重新以各阶段真实输出文件为准核对结果。
- 当前四阶段结果如下：
  - `stage1_square_10x10`
    - `CLS macro_f1=0.8581`
    - `REG mae_changed=37.0220`
    - `joint CMEI=89.30`
  - `stage2_rect_6x10`
    - `CLS macro_f1=0.9018`
    - `REG mae_changed=16.9012`
    - `joint CMEI=94.38`
  - `stage3_honeycomb_63`
    - `CLS macro_f1=0.8671`
    - `REG mae_changed=31.3267`
    - `joint CMEI=91.05`
  - `stage4_transfer_circlecut_69`
    - `CLS macro_f1=0.8818`
    - `REG mae_changed=43.2324`
    - `joint CMEI=88.42`

### 当前评价
- `stage2_rect_6x10` 是当前扩展结果最好的阶段，说明现有方法对规则但非正方形网格的推广性较强。
- `stage1_square_10x10` 与 `stage3_honeycomb_63` 仍保持可用，但回归误差相比 `stage2` 明显增大。
- `stage4_circlecut_69` 的分类仍有竞争力，但回归与联合指标最弱，说明不规则拓扑仍是当前最主要的扩展难点。

### 解释口径补充
- 本轮 `stage1/2/3` 的 `CLS` 输出中，`warm_start.loaded=0`，说明当前记录下来的分类结果并没有成功加载 clean `modelo3` 权重。
- 本轮 `stage4 transfer` 还存在一处真实路径错误：
  - 默认 warm start 原先写成了 `stage1_square_10x10/.../outputs/training_data64Nodes_2/model_last.pt`
  - 但 `stage1` 实际输出目录是 `.../outputs/square_10x10/model_last.pt`
- 因此当前 `stage4` 结果不能直接当作最终 transfer 结论，而应先解释为当前不规则拓扑基线。
- 该默认路径现已修正，后续可按正确口径重跑 `stage1 -> stage4` 迁移实验。

### 汇总图
- 已新增简洁科研风格汇总图：
  - `gnn/GNN_EXPAND/expand_summary.svg`
- 配套脚本与数据：
  - `gnn/GNN_EXPAND/plot_expand_summary.py`
  - `gnn/GNN_EXPAND/expand_summary_metrics.json`
- 该图采用三行小面板：
  - `CLS Macro-F1`
  - `REG MAE_changed`
  - `Joint CMEI`
  用于统一比较四阶段表现，并在图下注明本轮 warm start 约束。
## 2026-04-03 可视化正式版更新
- 正式汇总图从手写 `svg` 版切换为 `matplotlib` 版。
- 当前正式输出：
  - `gnn/GNN_EXPAND/expand_summary.png`
  - `gnn/GNN_EXPAND/expand_summary.pdf`
- 配套文件：
  - `gnn/GNN_EXPAND/plot_expand_summary.py`
  - `gnn/GNN_EXPAND/expand_summary_metrics.json`
- 绘图口径：
  - 左图统一比较 `S_num / S_F1 / S_id / S_mse`
  - 右图展示各阶段 `CMEI`
- 旧 `expand_summary.svg` 已删除，不再作为正式输出。
## 2026-04-03 Figure 目录与拓扑图
- `GNN_EXPAND` 下新增图像目录：
  - `gnn/GNN_EXPAND/Figure`
- 当前正式汇总图位置：
  - `gnn/GNN_EXPAND/Figure/expand_summary.png`
  - `gnn/GNN_EXPAND/Figure/expand_summary.pdf`
- 新增拓扑结构图生成脚本：
  - `gnn/GNN_EXPAND/plot_expand_topologies.py`
- 当前四张拓扑结构图：
  - `gnn/GNN_EXPAND/Figure/topology_square_10x10.png`
  - `gnn/GNN_EXPAND/Figure/topology_rect_6x10.png`
  - `gnn/GNN_EXPAND/Figure/topology_honeycomb_63.png`
  - `gnn/GNN_EXPAND/Figure/topology_circlecut_69.png`
- 绘图原则：
  - 白底、细线、结构优先
  - 外部边界节点单独着色，便于与内部节点区分

## 2026-04-04 真正 transfer 结果与图像重做
- 已吸收根目录 `0404训练日志.txt`，并以各阶段真实 `metrics.json / cmei_metrics.json` 重新核对结果。
- `stage4_transfer_circlecut_69` 现已确认不是“路径修正后的待重跑状态”，而是真正成功加载 `stage1` 权重后的 transfer 结果：
  - `CLS warm_start.loaded=36`
  - `REG warm_start.loaded=36`
  - `CLS macro_f1=0.8928`
  - `REG mae_changed=36.1173`
  - `joint CMEI=91.14`

### 相比 2026-04-03 基线的变化
- `macro_f1: 0.8818 -> 0.8928`
- `mae_changed: 43.2324 -> 36.1173`
- `CMEI: 88.42 -> 91.14`

### 当前判断修正
- `stage2_rect_6x10` 仍是当前四阶段中的最佳阶段。
- 但 `stage4_circlecut_69` 在真实 transfer 下已经明显回升，说明：
  - `stage1 -> stage4` 的迁移方向本身是有效的
  - 当前剩余难点主要是如何继续压低不规则拓扑下的回归误差，而不是 warm start 根本无效

### 可视化
- `plot_expand_summary.py` 已按最新真实数据重画汇总图。
- `plot_expand_topologies.py` 已重生成四张拓扑结构图。
- 当前正式输出仍统一保留在：
  - `gnn/GNN_EXPAND/Figure/expand_summary.png`
  - `gnn/GNN_EXPAND/Figure/expand_summary.pdf`
  - `gnn/GNN_EXPAND/Figure/topology_square_10x10.png`
  - `gnn/GNN_EXPAND/Figure/topology_rect_6x10.png`
  - `gnn/GNN_EXPAND/Figure/topology_honeycomb_63.png`
  - `gnn/GNN_EXPAND/Figure/topology_circlecut_69.png`
- 旧版同名图已直接由最新数据覆盖，不再保留旧指标版本。
