# Stage2 Rect 6x10 Log

## 2026-04-02 初始化
- 已建立矩形拓扑阶段目录：
  - `cls`
  - `reg`
  - `joint_inference`
- 当前状态：
  - 入口已就绪，等待正式训练/推理结果补充

## 2026-04-02 数据生成
- 已生成原生 stage2 数据：
  - `gnn/GNN_EXPAND/data/rect_6x10.csv`
  - `gnn/GNN_EXPAND/data/rect_6x10_meta.json`
- 已确认：
  - 激励只使用外部节点
  - 测量只输出外部节点电压

## 2026-04-02 数据口径补充
- 已按 `rect_6x10_meta.json` 核对：
  - 外部节点数为 `28`
  - 激励组数为 `32`
- 约束保持不变：
  - 激励只使用外部节点
  - 测量只输出外部节点电压

## 2026-04-03 训练结果
- `stage2_rect_6x10` 当前结果：
  - `CLS macro_f1=0.9018`
  - `REG mae_all=0.4110`
  - `REG mae_changed=16.9012`
  - `joint CMEI=94.38`
- 当前判断：
  - 这是本轮最佳扩展阶段
  - 非正方形规则网格并未成为当前方法的主要障碍
- 口径补充：
  - `CLS warm_start.loaded=0`
  - `REG warm_start.loaded=36`
