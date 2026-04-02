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
