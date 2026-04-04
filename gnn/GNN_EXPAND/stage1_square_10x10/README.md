# Stage1 Square 10x10

## 目标
- 对应第一阶段：纯规模膨胀。
- 在不改原主线程序的前提下，把当前最佳 `CLS / REG / joint` 方法迁移到 `10x10 / 100` 节点正方形网格。

## 默认配置
- 拓扑：`square_10x10`
- 节点数：`100`
- 电阻边数：`180`
- 默认数据：`gnn/GNN_EXPAND/data/square_10x10.csv`
- 默认 warm start：
  - `CLS <- gnn/GNN_CLS/modelo3`
  - `REG <- gnn/GNN_REG/o4a2`

## 说明
- 当前数据由 `gnn/GNN_EXPAND/generate_expand_datasets.py` 直接在 `10x10` 拓扑上正演生成。
- 激励和测量都只使用外部节点。
- `cache` 与 `outputs` 都留在本阶段目录内部。
- 原 `modelo3 / o4a2` 代码不做任何修改。

## 2026-04-02 数据说明补充
- 本阶段原生数据严格遵守：
  - 激励只使用外部节点
  - 测量只输出外部节点电压
- `square_10x10` 的真实数据规模为：
  - `36` 个外部节点
  - `40` 组激励
- 原因是 `10x10` 方格外边界节点数为 `2*(10+10)-4 = 36`，生成器会使用：
  - `36` 组相邻边界激励
  - `4` 组额外跨边界激励

## 2026-04-03 训练结果
- 当前结果：
  - `CLS macro_f1=0.8581`
  - `REG mae_all=0.4694`
  - `REG mae_changed=37.0220`
  - `joint CMEI=89.30`
- 当前解释：
  - `10x10 / 100` 节点规模扩张后，分类仍然可用
  - 主要压力集中在回归侧，`mae_changed` 明显高于当前 clean 主线
- 本轮记录还需要注意：
  - `CLS warm_start.loaded=0`
  - 因而当前分类结果不是“成功承接 clean modelo3 权重”后的结果，而是本次云端条件下的实际记录值
