# Stage4 Transfer Circle-Cut 69 Log

## 2026-04-02 初始化
- 已建立 transfer 阶段目录：
  - `cls`
  - `reg`
  - `joint_inference`
- 默认迁移来源：
  - `stage1_square_10x10`
- 当前状态：
  - 入口与 partial-load 机制已到位
  - 等待后续记录 stage1 -> stage4 的零样本或微调结果

## 2026-04-02 数据生成
- 已生成原生 stage4 数据：
  - `gnn/GNN_EXPAND/data/circlecut_69.csv`
  - `gnn/GNN_EXPAND/data/circlecut_69_meta.json`
- 已确认：
  - 激励只使用外部节点
  - 测量只输出外部节点电压

## 2026-04-02 数据口径补充
- 已按 `circlecut_69_meta.json` 核对：
  - 外部节点数为 `24`
  - 激励组数为 `28`
- 约束保持不变：
  - 激励只使用外部节点
  - 测量只输出外部节点电压
