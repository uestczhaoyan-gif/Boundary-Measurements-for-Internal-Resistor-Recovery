# MLP 基线与诊断线

`mlp/` 用固定 8x8 拓扑上的展平响应向量建立非图结构 baseline。它的作用有两个：给 GNN 提供一个强度可控的参照；验证在固定拓扑、固定激励顺序下，单纯增加网络容量或损失约束能否解决数量和位置恢复问题。

## 数据表示

每个样本聚合 32 组激励和 28 个边界节点电压。主线使用相对基准响应 `dV = V_meas - V_base`，输入形状为 `32 x 28`，训练时展平为 896 维向量。训练、验证、测试按 `combo_id` 划分为 `8:1:1`，标准化统计量只从训练集计算。

默认主数据是 `data/training_data64Nodes_2.csv`（10 mA），其他电流数据通过 `--data-path` 和 `--dataset-tag` 切换。模型输出放在各版本自己的 `cache/` 和 `outputs/` 下。

## 目录职责

```text
MLP_CLS/                 变化数量分类
MLP_REG/                 112 维电阻变化量回归
MLP_FULL/                分类与回归共享主干的多任务尝试
fixed_change_recon/      固定 K=2/3 的纯回归诊断
```

每个版本通常包含 `model/model.py`、`train.py` 和 `inference.py`。`modelo1`/`modelo2` 是早期 baseline，`modelo3` 以后主要记录损失、阈值和训练流程的局部改动。不要把整个版本目录当作同时运行的主线。

## 推荐使用方式

MLP 当前不承担项目默认联合推理。若需要建立基线，优先比较：

```powershell
python mlp\MLP_CLS\modelo5\train.py --help
python mlp\MLP_REG\modelo5\train.py --help
python mlp\MLP_CLS\modelo5\inference.py --help
python mlp\MLP_REG\modelo5\inference.py --help
```

固定变化数量诊断线：

```powershell
python mlp\fixed_change_recon\modelv3_new\train.py --help
python mlp\fixed_change_recon\modelv3_new\inference.py --help
```

该诊断线只回答“已知固定 `K` 时能否重构变化量”，不应被误读为完整的数量分类器。

## 评价重点

- 分类：`macro_f1`、各类别混淆矩阵，特别是 `2/3` 边界。
- 回归：`mae_all`、`mae_changed`、阈值化后的假阳性数量。
- 固定 K：support 是否恢复、变化位与未变化位的幅值分离、不同物理约束之间是否冲突。

MLP 的详细演进和失败原因保留在 [`Log.md`](Log.md)。项目总体结论不应只引用 MLP 的单项指标，因为固定拓扑上的展平输入没有显式利用网络邻接关系。

## 与 GNN 的关系

MLP 是控制变量：输入、数据、划分和评价口径尽量保持一致，只改变归纳偏置。若 MLP 与 GNN 的结果差异很小，问题可能受边界信息本身限制；若 GNN 明显更强，则说明拓扑结构对内部参数恢复提供了有效先验。

相关入口：[`../gnn/README.md`](../gnn/README.md)、[`../README.md`](../README.md)。
