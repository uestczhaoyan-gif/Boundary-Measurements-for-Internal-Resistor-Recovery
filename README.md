# 64Nodes: Boundary Measurements for Internal Resistor Recovery

研究一个固定拓扑的逆问题：只观测电阻网络边界节点的电压响应，能否判断内部哪些电阻发生了变化、变化了多少，以及这种判断在噪声和拓扑变化下还能保持多久。

项目对应的本科毕业设计题目为“面向拓扑光电融合网络的边缘端口测量与内部参数提取”。当前仓库中的数据和结果全部来自**基于基尔霍夫方程的合成仿真**，尚不包含真实器件或真实硬件测量验证。仓库公开的是代码、脱敏数据元数据、实验图表和研究记录；论文终稿、答辩材料和含个人信息的文件位于本地 `private_materials/`，并被 Git 忽略。

## 研究问题

主实验使用 `8 x 8` 网格：64 个节点、112 条电阻边、28 个边界测量节点和 32 组激励。每个样本由一组内部电阻变化和对应的边界电压响应组成。模型需要解决三个相关任务：

1. **变化数量分类**：判断变化数量 `K ∈ {0, 1, 2, 3}`。
2. **变化量回归**：输出 112 条电阻边的电阻变化量 `ΔR`。
3. **联合推理**：用分类得到的 `K` 截断回归候选边，形成最终的位置和幅值预测。

问题的困难来自信息压缩和非唯一性：内部节点不可直接测量，不同内部变化组合可能产生相近的边界响应。因此，较低的平均误差并不等于可靠的位置恢复，仓库同时记录数量、幅值、候选覆盖率和联合指标。

## 方法路线

```text
拓扑与电阻参数
        │
        ▼
基尔霍夫方程正演 ──► 多激励边界电压数据
        │                         │
        │                         ├── MLP baseline
        │                         └── GNN 主线
        │                               ├── CLS: 预测 K
        │                               ├── REG: 预测 112 维 ΔR
        │                               └── CMEI: 联合推理与评估
        │
        ├── inverse_identifiability: 可检测性与非唯一性
        └── square_scale_study: 规模、端口数与 K_max
```

当前公开主线是 GNN。MLP 用于固定拓扑下的结构化向量基线，`history/` 保存更早的 attention、CNN、CNN2D-MLP 和 UNet-MLP 路线。所有实验输出、缓存和权重均默认不进入 Git，具体规则见 [`.gitignore`](.gitignore)。

## 当前推荐结果

这些数字是仓库中已记录的阶段性结果，不是经过独立复核的统一排行榜；比较时必须同时读取数据集标签、随机种子和指标定义。

| 路线 | 角色 | 记录值 | 入口 |
| --- | --- | ---: | --- |
| GNN CLS `modelo3` | 变化数量 | `test_macro_f1 ≈ 0.90` | [`gnn/GNN_CLS/modelo3`](gnn/GNN_CLS/modelo3) |
| GNN REG `o4a2` | 变化量回归 | `mae_all = 0.4679`；`mae_changed = 23.5724` | [`gnn/GNN_REG/o4a2`](gnn/GNN_REG/o4a2) |
| GNN CMEI v1 | 联合推理 | `CMEI = 93.53` | [`gnn/GNN_CMEI_INFERENCE`](gnn/GNN_CMEI_INFERENCE) |
| GNN noise v2 | clean 到中等噪声 | clean `CMEI = 93.49`；20 dB `CMEI = 80.42` | [`gnn/GNN_NOISE`](gnn/GNN_NOISE) |
| GNN expand | 跨拓扑迁移 | stage2 记录 `CMEI = 94.38` | [`gnn/GNN_EXPAND`](gnn/GNN_EXPAND) |

“当前最好”与“历史曾达到”并不总是同一个概念。完整口径见 [`CURRENT_BEST.md`](CURRENT_BEST.md)，实验过程见各目录的 `Log.md`。

## 仓库地图

| 目录 | 职责 | 读者入口 |
| --- | --- | --- |
| `data/` | 8x8 主数据元数据、数据生成器和可选本地 CSV | [`data/README.md`](data/README.md) |
| `scripts/` | 跨目录的公开工具和数据生成脚本 | [`scripts/README.md`](scripts/README.md) |
| `gnn/` | 当前主线：分类、回归、联合推理、噪声和拓扑迁移 | [`gnn/README.md`](gnn/README.md) |
| `mlp/` | 固定拓扑 MLP baseline 与 fixed-change 诊断线 | [`mlp/README.md`](mlp/README.md) |
| `inverse_identifiability/` | 响应差异、相似性、可检测性和非唯一性分析 | [`inverse_identifiability/README.md`](inverse_identifiability/README.md) |
| `square_scale_study/` | 正方形规模、端口数和最大可识别变化数研究 | [`square_scale_study/README.md`](square_scale_study/README.md) |
| `history/` | 已停止推进的模型家族归档 | [`history/README.md`](history/README.md) |
| `Figure/` | 可公开的论文/PPT 图表源文件和汇总图 | [`Figure/README.md`](Figure/README.md) |
| `docs/` | 复现约定、指标定义和项目导航 | [`docs/PROJECT_MAP.md`](docs/PROJECT_MAP.md) |
| `private_materials/` | 仅本地保存的论文、答辩和个人材料 | 不上传 |

## 快速开始

### 1. 查看主数据口径

```powershell
Get-Content data\training_data64Nodes_2_meta.json
```

默认主线是未筛选的 10 mA 数据：`data/training_data64Nodes_2.csv`。CSV 文件通常被 `.gitignore` 忽略；没有本地 CSV 时，应先用生成脚本按元数据重建。

### 2. 生成一个小数据集做冒烟测试

```powershell
python scripts\generate_training_data64.py --help
python scripts\generate_training_data64.py `
  --total-combos 100 `
  --current-a 0.01 `
  --output data\_smoke\training_data64_smoke.csv `
  --meta-output data\_smoke\training_data64_smoke_meta.json
```

这条命令显式指定了仿真电流、CSV 路径和 metadata 路径，不会覆盖正式数据的 metadata。要生成当前默认的 10 mA 正式数据，使用 `--current-a 0.01`、`--output data\training_data64Nodes_2.csv` 和 `--meta-output data\training_data64Nodes_2_meta.json`。

具体参数以脚本的 `--help` 为准。正式训练前先用小数据集验证路径、依赖和输出目录。

### 3. 运行 GNN 主线

```powershell
python gnn\GNN_CLS\modelo3\train.py --help
python gnn\GNN_REG\o4a2\train.py --help
python gnn\GNN_CMEI_INFERENCE\inference_gnn_cmei.py --help
```

训练命令应显式指定 `--data-path`、`--dataset-tag` 和随机种子。输出会写入对应模型目录的 `cache/<dataset_tag>/` 与 `outputs/<dataset_tag>/`，这些目录默认不提交。想验证完整链路，可运行 [`scripts/reproduce_smoke.ps1`](scripts/reproduce_smoke.ps1)。

## 复现约定

- 使用项目根目录作为工作目录，避免把个人电脑绝对路径写进脚本或日志。
- 记录数据文件名、数据集标签、随机种子、训练/验证/测试划分和关键超参数。
- 只用训练集统计量做标准化。
- `metrics.json` 是机器可读的结果源，`Log.md` 用于解释实验选择和失败原因。
- 论文中的数字必须能回指到一个具体输出目录；若只能从终端日志得到，应标为待复核。

更完整的环境与复现说明见 [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)。

## 项目贡献

本项目的工作重点不是只训练一个网络，而是建立一条可追踪的研究链路：实现基尔霍夫方程正演数据生成；把网络拓扑编码为 GNN 输入；拆分数量分类、变化量回归和联合推理；用可检测性分析解释难例；并在噪声、规模和拓扑变化下检查方法边界。

## 许可证与材料边界

原创代码按 [`LICENSE`](LICENSE) 发布。第三方论文、学校模板、图标、字体和论文附件不因本仓库的许可证而自动获得再分发授权。含个人信息的材料保存在 `private_materials/`，不加入公开仓库。
