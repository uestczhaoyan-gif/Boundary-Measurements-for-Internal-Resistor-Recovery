# Stage4 Transfer Circle-Cut 69

## 目标
- 对应第四阶段：Transfer / Zero-Shot 亮点。
- 使用近似圆形的 `9x9` 角点裁切拓扑，承接 stage1 的放大拓扑权重。

## 默认配置
- 拓扑：`circlecut_69`
- 节点数：`69`
- 电阻边数：`120`
- 默认数据：`gnn/GNN_EXPAND/data/circlecut_69.csv`
- 默认 warm start：
  - `CLS <- gnn/GNN_EXPAND/stage1_square_10x10/cls`
  - `REG <- gnn/GNN_EXPAND/stage1_square_10x10/reg`

## 说明
- 当前数据由 `gnn/GNN_EXPAND/generate_expand_datasets.py` 直接在裁角圆形近似拓扑上正演生成。
- 激励和测量都只使用外部节点。
- 本阶段不再默认回到原 8x8 主线权重，而是优先尝试接 stage1 的 `10x10` 权重，体现 transfer 设定。
- 推理入口支持 partial load，因此即便拓扑 buffer 形状不同，也能加载同构权重做迁移试验。

## 2026-04-02 数据说明补充
- 本阶段原生数据严格遵守：
  - 激励只使用外部节点
  - 测量只输出外部节点电压
- `circlecut_69` 的真实数据规模为：
  - `24` 个外部节点
  - `28` 组激励

## 2026-04-03 当前结果与 transfer 口径
- 当前结果：
  - `CLS macro_f1=0.8818`
  - `REG mae_all=0.8202`
  - `REG mae_changed=43.2324`
  - `joint CMEI=88.42`
- 当前解释：
  - 不规则裁角圆形拓扑下，分类仍能保持在较高水平
  - 但回归误差和联合指标是四阶段中最弱，说明当前主要难点仍集中在不规则拓扑的内部重建
- 这一阶段必须特别说明：
  - 本轮记录时，默认 transfer warm start 路径写错，实际没有成功加载 `stage1_square_10x10` 权重
  - 因此当前结果不能直接当作最终 transfer 结论，而应先视作当前不规则拓扑结果基线
- 本轮已修正默认路径为：
  - `gnn/GNN_EXPAND/stage1_square_10x10/cls/outputs/square_10x10/model_last.pt`
  - `gnn/GNN_EXPAND/stage1_square_10x10/reg/outputs/square_10x10/model_last.pt`
- 后续重跑时，才适合正式比较 `stage1 -> stage4` transfer 效果。
