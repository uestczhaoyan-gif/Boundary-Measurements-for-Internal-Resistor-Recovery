# Stage3 Honeycomb 63 Log

## 2026-04-02 初始化
- 已建立蜂窝状阶段目录：
  - `cls`
  - `reg`
  - `joint_inference`
- 当前状态：
  - 目录、脚本与拓扑注册已落地
  - 等待后续补充正式训练和联合推理结果

## 2026-04-02 数据生成
- 已生成原生 stage3 数据：
  - `gnn/GNN_EXPAND/data/honeycomb_63.csv`
  - `gnn/GNN_EXPAND/data/honeycomb_63_meta.json`
- 已确认：
  - 激励只使用外部节点
  - 测量只输出外部节点电压

## 2026-04-02 数据口径补充
- 已按 `honeycomb_63_meta.json` 核对：
  - 外部节点数为 `28`
  - 激励组数为 `32`
- 约束保持不变：
  - 激励只使用外部节点
  - 测量只输出外部节点电压

## 2026-04-03 训练结果
- `stage3_honeycomb_63` 当前结果：
  - `CLS macro_f1=0.8671`
  - `REG mae_all=0.4831`
  - `REG mae_changed=31.3267`
  - `joint CMEI=91.05`
- 当前判断：
  - 蜂窝状拓扑下当前方法仍保持可用
  - 但回归误差明显高于 `stage2_rect_6x10`
- 口径补充：
  - `CLS warm_start.loaded=0`
  - `REG warm_start.loaded=36`
