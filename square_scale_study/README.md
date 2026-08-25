# Square Scale Study

这个子项目把问题从“某个固定网络能否恢复”推进到“端口数量和拓扑规模如何限制可恢复性”。研究对象是正方形网格 `N x N`，规模从 `3x3` 到 `10x10`，每次固定真实变化数量 `K`，只做纯回归和可识别性判定。

## 核心问题

定义边界端口数 `P = 4N - 4`。对每个 `(N, K)` 组合训练和测试模型，寻找满足以下条件的最大 `K_max`：

- 样本级精确 support 恢复率 `>= 0.98`；
- 变化量数值准确率 `>= 0.90`。

主曲线是 `P` 与 `K_max` 的关系。它不是主项目的替代品，而是用来研究端口资源、网络规模、响应秩和可辨识性之间的关系。

## 目录

| 目录 | 内容 |
| --- | --- |
| `data/` | 按 `N`、`K` 组织的数据和元数据；CSV 默认不提交 |
| `models/` | `modelv1`、`modelv2`、`modelo1/2` 的回归模型 |
| `scripts/` | 数据生成、批量训练、汇总、误差和图表脚本 |
| `analysis/` | 激励信息通道、端口负载和灵敏度分析 |
| `combo_identifiability/` | 固定候选池下的多电阻组合可辨识性 |
| `theory_validation/` | 响应矩阵秩/SVD 与白箱重构验证 |
| `Figure/` | 汇报图、曲线和表格图 |
| `PLAN.md` | 实验设计与判定规则 |
| `Log.md` | 运行记录、结果和阶段结论 |

## 最小工作流

以下命令从仓库根目录执行，参数以对应脚本的 `--help` 为准：

```powershell
python square_scale_study\scripts\generate_square_fixedk_data.py --help
python square_scale_study\scripts\generate_square_fixedk_data.py --n 3 --k 1

python square_scale_study\models\modelv1\train.py --help
python square_scale_study\models\modelv1\inference.py --help

python square_scale_study\scripts\generate_square_fixedk_range.py `
  --n-values 3,4,5 `
  --k-values 1,2,3,4,5,6

python square_scale_study\scripts\summarize_scale_sweep.py --help
```

先在 `3x3` 上完成一轮数据生成、训练、推理和指标汇总，再扩大规模。`N=6~10` 的训练数据可能很大，必须确认磁盘空间和 `.gitignore` 状态。

## 两条分析线

1. **学习型回归**：比较不同模型在固定 `N,K` 下的 support 与数值恢复。
2. **白箱/理论验证**：在已知候选池或已知变化位置时直接解线性系统，检查响应矩阵的秩、奇异值和条件数是否能预测难度。

只有当学习型结果和白箱结果使用相同拓扑、端口、激励和变化分布时，二者才可以放在同一结论中讨论。详细假设见 [`PLAN.md`](PLAN.md)，阶段记录见 [`Log.md`](Log.md)。
