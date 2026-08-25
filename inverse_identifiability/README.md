# Inverse Identifiability

这是主模型之外的机制分析子项目。它不训练最终的分类器或回归器，而是直接比较不同电阻变化对边界电压的响应，回答一个更基础的问题：**某个变化在当前激励与端口配置下是否可检测，不同变化组合是否会产生近似相同的观测。**

## 研究对象

- 拓扑：8x8 网格，64 个节点、112 条电阻边。
- 观测：28 个边界节点电压。
- 激励：与主数据一致的 32 组激励；也可切换为 4 组典型长程激励。
- 输出：响应幅值、归一化相似度、近重复案例、检测率和汇总图表。

主脚本：`scripts/run_identifiability_study.py`。

```powershell
python inverse_identifiability\scripts\run_identifiability_study.py --help
python inverse_identifiability\scripts\run_identifiability_study.py
```

默认输出写入 `inverse_identifiability/outputs/`，包括响应示例图、范数与幅值关系、余弦相似度热图、相似案例 CSV 和 `analysis_summary.md`。这些输出默认被 Git 忽略。

## 如何解释结果

- 响应范数小，不代表变化不存在，只表示它在当前测量配置下较难分辨。
- 高相似度只说明两个观测模式接近，不自动证明它们在所有激励下不可区分。
- 可检测性分析用于解释模型难例和设计激励，不替代真实的训练/测试评估。

这个子项目与主线的关系是“机制解释层”：GNN/MLP 负责预测，identifiability 负责解释为什么某些位置或组合天然更难。

## 目录

- `scripts/`：响应计算、激励/基准电阻对照和汇总脚本。
- `data/`：可选的中间数据或元数据。
- `outputs/`：本地生成结果，不提交。
- `README.md`：本页。

运行环境依赖 NumPy、SciPy 和 Matplotlib；仓库内 `.vendor` 仅用于本机实验，不应上传。
