# GNN_EXPAND 日志

## 2026-04-02 初始化记录
- 已正式建立 `gnn/GNN_EXPAND` 作为拓扑与规模扩展专用目录。
- 原则：
  - 不修改原有 clean 主线程序
  - 所有扩展代码只落在 `GNN_EXPAND` 内部
  - 每个阶段都在自己的 `cls/reg/joint_inference` 下闭环保存 `cache/outputs`

### 本次落地内容
- 已新增共享层：
  - `common/topologies.py`
  - `common/models.py`
  - `common/expand_common.py`
  - `common/train_cls_expand.py`
  - `common/train_reg_expand.py`
  - `common/inference_cls_expand.py`
  - `common/inference_reg_expand.py`
  - `common/inference_joint_expand.py`
- 已建立四个阶段：
  - `stage1_square_10x10`
  - `stage2_rect_6x10`
  - `stage3_honeycomb_63`
  - `stage4_transfer_circlecut_69`

### 关键工程修正
- 修正了运行目录机制：
  - `cache/outputs` 不再落到 `common/`
  - 现在会落到每个阶段自己的 `cls/reg/joint_inference` 目录
- 新增边界节点顺序映射：
  - 原始 clean 数据的 28 通道边界电压会按电极顺序映射到目标拓扑边界节点
- 新增电阻 id 重映射：
  - 原始 8x8 `r*_id` 会根据 `_meta.json` 恢复源拓扑，再映射到目标拓扑电阻边
- 新增 partial load：
  - 推理入口可加载不同拓扑下的同构权重，只跳过 shape 不匹配的 buffer/参数
  - 为 stage4 的 transfer / zero-shot 试验保留了入口

### 当前状态
- 各阶段目录与入口脚本已落地。
- 正式训练结果、推理结果与后续日志继续累计在本文件和各阶段 `Log.md` 中。

## 2026-04-02 原生拓扑数据生成完成
- 已新增通用拓扑数据生成器：
  - `gnn/GNN_EXPAND/generate_expand_datasets.py`
- 该脚本不再做“8x8 数据重映射生成”，而是：
  - 直接在目标拓扑电阻网络上做基尔霍夫正演
  - 输出目标拓扑自己的 `csv + meta`
- 已生成四套数据到：
  - `gnn/GNN_EXPAND/data/square_10x10.csv`
  - `gnn/GNN_EXPAND/data/rect_6x10.csv`
  - `gnn/GNN_EXPAND/data/honeycomb_63.csv`
  - `gnn/GNN_EXPAND/data/circlecut_69.csv`
- 同时生成：
  - `square_10x10_meta.json`
  - `rect_6x10_meta.json`
  - `honeycomb_63_meta.json`
  - `circlecut_69_meta.json`
- 统一生成口径：
  - `10000` 个 combo
  - `0.01A` 电流激励
  - `0/1/2/3` 变化比例仍为 `0.07 / 0.31 / 0.31 / 0.31`
- 已确认：
  - 四套数据的激励都只使用外部节点
  - 四套数据的测量都只输出外部节点电压
  - 每套数据均为 `28` 个外部节点、`32` 组激励
- 各阶段默认数据入口已切换为读取本目录内对应 CSV。

## 2026-04-02 数据口径更正
- 更正上一条“原生拓扑数据生成完成”中的数量说明：
  - “四套数据均为 `28` 个外部节点、`32` 组激励”不正确
- `GNN_EXPAND` 原生数据的真实规则是：
  - 激励只使用外部节点
  - 测量只输出外部节点电压
  - 外部节点数与激励组数跟随目标拓扑边界规模变化
- 已按 `gnn/GNN_EXPAND/data/*_meta.json` 重新核对：
  - `square_10x10`: `36` 个外部节点，`40` 组激励
  - `rect_6x10`: `28` 个外部节点，`32` 组激励
  - `honeycomb_63`: `28` 个外部节点，`32` 组激励
  - `circlecut_69`: `24` 个外部节点，`28` 组激励
