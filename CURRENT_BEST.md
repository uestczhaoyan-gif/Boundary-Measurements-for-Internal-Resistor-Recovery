# 64Nodes 当前最佳清单

## 1. 文件定位
- 本文件用于记录当前正式路线和当前最佳本地产物锚点。
- 在 Git 方案 A 下，大量 `outputs / cache / 权重 / 可重建 csv` 不直接进仓库。
- 因此本文件负责帮助后续窗口快速知道：
  - 当前正式该用什么
  - 当前最佳结果本地在哪
  - 如果要恢复当前最佳链路，应从哪里开始

## 2. clean 主线
- 默认 clean 数据：
  - `data/training_data64Nodes_2.csv`
- 当前正式 `CLS`：
  - 目录：`gnn/GNN_CLS/modelo3`
  - 本地锚点：`gnn/GNN_CLS/modelo3/outputs/training_data64Nodes_2/`
  - 参考指标：`test_macro_f1=0.9027`
- 当前正式 `REG`：
  - 目录：`gnn/GNN_REG/o4a2`
  - 本地锚点：`gnn/GNN_REG/o4a2/outputs/training_data64Nodes_2/`
  - 参考指标：`mae_all=0.4679`
  - 参考指标：`mae_changed=23.5724`
- 当前正式 joint：
  - 目录：`gnn/GNN_CMEI_INFERENCE`
  - 脚本：`gnn/GNN_CMEI_INFERENCE/inference_gnn_cmei.py`
  - 参考组合：`modelo3 + o4a2`
  - 参考指标：`CMEI=93.53`

## 3. noisy 主线
- 当前正式 noisy 路线：
  - `rand_boundary`
- 当前正式 `CLS`：
  - 目录：`gnn/GNN_NOISE/CLS_modelo3_ft`
  - 本地锚点：`gnn/GNN_NOISE/CLS_modelo3_ft/outputs/training_data64Nodes_2_noiseft_rand_boundary_20260401/`
- 当前正式 `REG`：
  - 目录：`gnn/GNN_NOISE/REG_o4a2_ft`
  - 本地锚点：`gnn/GNN_NOISE/REG_o4a2_ft/outputs/training_data64Nodes_2_noiseft_rand_boundary_20260401/`
- 当前正式 joint：
  - 脚本：`gnn/GNN_CMEI_INFERENCE/inference_gnn_cmei.py`
  - 本地输出锚点：
    - `gnn/GNN_CMEI_INFERENCE/outputs/gnn_cmei_noiseft_rand_boundary_clean_20260401/`
    - `gnn/GNN_CMEI_INFERENCE/outputs/gnn_cmei_noiseft_rand_boundary_20db_20260401/`
- 当前参考指标：
  - clean: `CMEI=91.01`
  - `40dB`: `CMEI=90.83`
  - `30dB`: `CMEI=89.62`
  - `20dB`: `CMEI=82.56`

## 4. joint 正式入口
- 正式联合推理脚本：
  - `gnn/GNN_CMEI_INFERENCE/inference_gnn_cmei.py`
- 实验版脚本：
  - `gnn/GNN_CMEI_INFERENCE/inference_gnn_cmei_v2.py`
- 当前判断：
  - `v2` 暂未超过 `v1`
  - 正式路线继续用 `v1`

## 5. expand 主线
- 当前默认推广体系：
  - `CLS = modelo3`
  - `REG = o4a2`
  - `joint = v1 inference_gnn_cmei`
- 当前四阶段目录：
  - `gnn/GNN_EXPAND/stage1_square_10x10`
  - `gnn/GNN_EXPAND/stage2_rect_6x10`
  - `gnn/GNN_EXPAND/stage3_honeycomb_63`
  - `gnn/GNN_EXPAND/stage4_transfer_circlecut_69`
- 当前原生数据锚点：
  - `gnn/GNN_EXPAND/data/square_10x10_meta.json`
  - `gnn/GNN_EXPAND/data/rect_6x10_meta.json`
  - `gnn/GNN_EXPAND/data/honeycomb_63_meta.json`
  - `gnn/GNN_EXPAND/data/circlecut_69_meta.json`
- 当前边界口径：
  - `square_10x10`: `36` 个外部节点，`40` 组激励
  - `rect_6x10`: `28` 个外部节点，`32` 组激励
  - `honeycomb_63`: `28` 个外部节点，`32` 组激励
  - `circlecut_69`: `24` 个外部节点，`28` 组激励

## 6. 当前重点工作
- `GNN_NOISE v2`
  - `gnn/GNN_NOISE/CLS_modelo3_ft_v2`
  - `gnn/GNN_NOISE/REG_o4a2_ft_v2`
- `GNN_EXPAND`
  - 在四阶段拓扑与规模上做正式训练和推广测试

## 7. 恢复顺序建议
- 新窗口接手时，建议按以下顺序恢复当前最佳状态：
  1. 读 `RULES.md`
  2. 读本文件 `CURRENT_BEST.md`
  3. 读根目录 `README.md / Log.md` 最新部分
  4. 读 `gnn/README.md / gnn/Log.md` 最新部分
  5. 再进入当前具体目录

## 8. 备注
- 本文件记录的是“当前正式与当前最佳的锚点说明”，不是完整实验历史。
- 如果后续最佳路线切换，必须同步更新本文件。
- 如果未来需要把最佳权重本身纳入版本管理，应升级到更重的 Git 方案，而不是继续扩大方案 A 的跟踪范围。
