# Stage3 Honeycomb 63

## 目标
- 对应第三阶段：复杂物理结构。
- 按当前要求固定选择蜂窝状拓扑，在 `63` 节点规模上验证当前最佳方法。

## 默认配置
- 拓扑：`honeycomb_63`
- 节点数：`63`
- 电阻边数：`158`
- 默认数据：`gnn/GNN_EXPAND/data/honeycomb_63.csv`
- 默认 warm start：
  - `CLS <- gnn/GNN_CLS/modelo3`
  - `REG <- gnn/GNN_REG/o4a2`

## 说明
- 当前数据由 `gnn/GNN_EXPAND/generate_expand_datasets.py` 直接在蜂窝拓扑上正演生成。
- 激励和测量都只使用外部节点。
- 蜂窝状邻接只在 `GNN_EXPAND` 内实现，不回写主线模型目录。

## 2026-04-02 数据说明补充
- 本阶段原生数据严格遵守：
  - 激励只使用外部节点
  - 测量只输出外部节点电压
- `honeycomb_63` 的真实数据规模为：
  - `28` 个外部节点
  - `32` 组激励
