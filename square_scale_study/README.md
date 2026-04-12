# 正方形规模研究子项目

`square_scale_study` 是从已冻结 `gnn` 主线中拆分出来的独立子项目，用来研究一个更聚焦的问题：

**在固定正方形拓扑下，随着端口数/规模增加，能够稳定识别的最大同时变化电阻数量 `K_max` 是多少？**

主线研究口径固定为：

- 只使用正方形网格拓扑
- 规模从 `3x3` 扫到 `10x10`
- 每个实验只使用固定变化数量 `K`
- 只研究回归，不做数量分类
- 主线阶段不加噪声

项目的核心曲线是：

- 横轴：端口数 `P = 4N - 4`
- 纵轴：最大可识别变化数量 `K_max`

其中一次运行只有同时满足以下两项时，才算“可识别”：

- 样本级精确 support 恢复率 `>= 0.98`
- 数值精度 `value_accuracy >= 0.90`

## 目录说明

- `PLAN.md`：问题定义、实验规则和阶段规划
- `Log.md`：实际实验记录、阶段性结论和交接信息
- `data/`：按规模划分的数据目录，如 `N3x3/`、`N4x4/`
- `scripts/`：数据生成、训练扫表、主线汇总、误差分析和表格导出
- `models/modelv1/`：第一版最简纯回归基线
- `models/modelv1_1/`：在 `modelv1` 基础上只修改损失函数的版本
- `models/modelv2/`：带 score head 的 support-aware 回归版本
- `models/modelv3/`：面向 `3x3` 优先验证的轻量双头版本，使用浅层图编码和简单跨激励融合
- `analysis/`：后续机制分析，例如激励信息通道、秩和条件数研究
- `outputs/`：`modelv1` 的训练与推理输出
- `outputs_modelv2/`：`modelv2` 的训练与推理输出
- `Figure/`：汇报图、曲线图和指标表图

## 基本工作流

1. 生成固定 `K` 数据集：

```powershell
python square_scale_study\scripts\generate_square_fixedk_data.py --n 3 --k 1
```

2. 训练 `modelv1`：

```powershell
python square_scale_study\models\modelv1\train.py --meta-path square_scale_study\data\N3x3\square_N3x3_K1_meta.json
```

3. 推理并导出指标：

```powershell
python square_scale_study\models\modelv1\inference.py --meta-path square_scale_study\data\N3x3\square_N3x3_K1_meta.json
```

4. 其他模型只需要更换入口路径：

```powershell
python square_scale_study\models\modelv1_1\train.py --meta-path square_scale_study\data\N3x3\square_N3x3_K2_meta.json
python square_scale_study\models\modelv1_1\inference.py --meta-path square_scale_study\data\N3x3\square_N3x3_K2_meta.json

python square_scale_study\models\modelv2\train.py --meta-path square_scale_study\data\N3x3\square_N3x3_K2_meta.json
python square_scale_study\models\modelv2\inference.py --meta-path square_scale_study\data\N3x3\square_N3x3_K2_meta.json

python square_scale_study\models\modelv3\train.py --meta-path square_scale_study\data\N3x3\square_N3x3_K2_meta.json
python square_scale_study\models\modelv3\inference.py --meta-path square_scale_study\data\N3x3\square_N3x3_K2_meta.json
```

5. 汇总主线结果：

```powershell
python square_scale_study\scripts\summarize_scale_sweep.py --outputs-root square_scale_study\outputs
```

## 当前执行策略

- 主线结果、图表和记录都只沉淀在 `square_scale_study/` 内，不再继续扩写根目录长文档。
- 当前已经完成 `N=3~5, K=1~6` 的 `modelv1` 与 `modelv2` 基线。
- 下一阶段的模型改进，优先在 `3x3` 网络上快速验证；只有在 `3x3` 上出现明确提升后，才扩展到更大规模，避免单次迭代周期过长、资源消耗过大。
- `scripts/summarize_scale_sweep.py` 用于生成主曲线图和总表。
- `scripts/analyze_support_errors.py` 用于单点 support 恢复误差分析。
- `scripts/export_test_metric_table.py` 用于导出可直接放进 PPT 的测试集指标表。
## 2026-04-12 模型设计更新

- 新增模型设计记录文档：`MODEL_DESIGN.md`
- 本轮新增三条模型线：
  - `models/modelo1_gnn/`
  - `models/modelo1_mlp1/`
  - `models/modelo1_mlp2/`
- `MODEL_DESIGN.md` 统一记录以下内容：
  - 三个模型各自回答的问题和对照作用
  - 三种结果组合的分析口径
  - 每个模型的正式架构记录
  - 各自 loss 的精确定义
  - `N=3, K=2~4` 的推荐训练命令
