# Data layer

`data/` 保存主项目的 8x8 数据说明、元数据和数据生成入口。大型 CSV 通常留在本地并由 `.gitignore` 排除；元数据 JSON 和生成脚本是公开复现所需的最小资产。

## 主数据集

| 文件 | 激励 | 用途 |
| --- | ---: | --- |
| `training_data64.csv` | 5 mA | 早期主线对照 |
| `training_data64Nodes_2.csv` | 10 mA | 当前默认主线 |
| `training_data64Nodes_3.csv` | 20 mA | 激励强度对照 |
| `training_data64Nodes_2_screened.csv` | 10 mA | 分布筛选实验，不是默认数据 |
| `training_data64_smoke.csv` | 小规模 | 本地冒烟测试 |

每个正式 CSV 对应一个 `*_meta.json`。元数据记录网格大小、边数、边界节点顺序、激励列表、变化比例范围、类别比例和生成时间。训练脚本应读取元数据或通过数据本身推断同一口径，不要手工复制这些常数。

## 生成

```powershell
python scripts\generate_training_data64.py --help
python scripts\generate_training_data64_screened.py --help
```

正演过程基于电阻网络的基尔霍夫线性方程。默认 8x8 网络包含 64 个节点和 112 条边，边界顺时针排列为 28 个节点，另加 4 组跨边界激励。

## 数据治理

- 数据文件名应包含激励版本或实验标签，避免 `cache` 复用。
- 新数据必须同时提交元数据 JSON 和生成命令/参数。
- 经过筛选的数据必须在 README、Log 和结果表中明确标注，不能与原始分布混称。
- 含个人信息的论文附件不属于本目录。
