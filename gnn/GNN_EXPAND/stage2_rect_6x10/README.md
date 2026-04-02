# Stage2 Rect 6x10

## 目标
- 对应第二阶段：打破对称性。
- 用 `6x10 / 60` 节点长方形网格验证现有最佳 GNN 是否能承接非正方形拓扑。

## 默认配置
- 拓扑：`rect_6x10`
- 节点数：`60`
- 电阻边数：`104`
- 默认数据：`gnn/GNN_EXPAND/data/rect_6x10.csv`
- 默认 warm start：
  - `CLS <- gnn/GNN_CLS/modelo3`
  - `REG <- gnn/GNN_REG/o4a2`

## 说明
- 当前数据由 `gnn/GNN_EXPAND/generate_expand_datasets.py` 直接在 `6x10` 拓扑上正演生成。
- 激励和测量都只使用外部节点。
- 原主线目录与原脚本不做修改。

## 2026-04-02 数据说明补充
- 本阶段原生数据严格遵守：
  - 激励只使用外部节点
  - 测量只输出外部节点电压
- `rect_6x10` 的真实数据规模为：
  - `28` 个外部节点
  - `32` 组激励
