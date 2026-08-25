# GNN_EXPAND data

本目录保存四套目标拓扑的原生 clean 数据元数据。CSV 文件很大，默认由 `.gitignore` 排除；`*_meta.json` 应随生成脚本和实验记录保留。

| 数据集 | 边界节点 | 激励组 | 生成入口 |
| --- | ---: | ---: | --- |
| `square_10x10` | 36 | 40 | `generate_expand_datasets.py` |
| `rect_6x10` | 28 | 32 | `generate_expand_datasets.py` |
| `honeycomb_63` | 28 | 32 | `generate_expand_datasets.py` |
| `circlecut_69` | 24 | 28 | `generate_expand_datasets.py` |

边界节点数和激励组数由目标拓扑决定，不能套用主 8x8 数据的 28/32 口径。所有数据都只在外部节点施加激励并测量外部节点电压，内部节点通过基尔霍夫方程求解。

```powershell
python gnn\GNN_EXPAND\generate_expand_datasets.py --help
python gnn\GNN_EXPAND\generate_expand_datasets.py
```

生成后，每套数据都应有对应的 `*_meta.json`。训练命令通过目标拓扑的数据标签读取正确的边数、边界顺序和激励列表。
