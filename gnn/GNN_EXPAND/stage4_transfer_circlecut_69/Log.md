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

## 2026-04-03 当前结果与路径修正
- `stage4_transfer_circlecut_69` 当前结果：
  - `CLS macro_f1=0.8818`
  - `REG mae_all=0.8202`
  - `REG mae_changed=43.2324`
  - `joint CMEI=88.42`
- 当前判断：
  - 不规则拓扑下分类仍可用
  - 回归与联合推理表现为四阶段最弱
- 本轮同时确认：
  - 默认 transfer warm start 路径原先写错
  - 当前这次记录并未真正加载 `stage1_square_10x10` 权重
- 因此当前结果暂不作为最终 transfer 结论，而只作为当前不规则拓扑基线。
- 已修正代码中的默认路径到：
  - `stage1_square_10x10/.../outputs/square_10x10/model_last.pt`
