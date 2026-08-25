# Reproducibility contract

## 运行环境

建议使用 Python 3.10 或更高版本，并单独创建虚拟环境。依赖大致列在根目录 [`requirements.txt`](../requirements.txt)；PyTorch/PyG 在不同操作系统和 CUDA 环境下安装方式可能不同，必要时应按官方 wheel 选择版本。仓库中的 `.vendor_torchpy311/` 是本地依赖缓存，不是公开依赖清单。

运行前先确认：

```powershell
python --version
python gnn\GNN_CLS\modelo3\train.py --help
```

如果 import 失败，先按当前机器的 PyTorch/PyG 官方安装说明配置环境，再运行项目脚本。不要把 vendor 目录提交到 GitHub。

仓库提供 [`scripts/reproduce_smoke.ps1`](../scripts/reproduce_smoke.ps1) 作为端到端冒烟入口。它会生成独立的 10 mA 小数据集、各训练一轮 CLS/REG，然后运行 CMEI；需要已安装 NumPy、SciPy、PyTorch 和 PyG。

## 实验记录最小字段

每次可比较实验至少记录：

- 数据文件和 `dataset-tag`；
- 数据生成参数与元数据文件；
- 随机种子和 train/val/test 划分；
- 模型目录、关键超参数和 warm-start 来源；
- 指标定义、输出目录和运行日期；
- 是否使用噪声、筛选数据或拓扑迁移。

## 输出约定

模型目录下的 `cache/<dataset-tag>/` 存预处理数据，`outputs/<dataset-tag>/` 存权重、标准化参数、`metrics.json` 和样例推理。联合推理输出使用 `cmei_metrics.json`。这些文件默认被忽略，但应在本地保留以支持结果复核。

## 复核原则

同一表格中的数字必须来自相同数据口径和测试划分。历史日志中的“最佳”可能是单次运行的可达上限，不一定是稳定现役模型；引用前应检查 `CURRENT_BEST.md` 与实际 JSON 输出。
