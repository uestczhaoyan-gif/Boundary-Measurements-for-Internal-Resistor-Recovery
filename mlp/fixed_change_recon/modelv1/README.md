# modelv1

- 任务：固定3变化的纯重构（输出112维ΔR）
- 主要损失：
  - 变化位置回归（SmoothL1）
  - 未变化抑制（L1 + hinge）
  - Top3排序约束
  - 后期可选Kirchhoff一致性约束（每batch随机4或8个激励）

## 运行
```bash
python 64Nodes/mlp/fixed_change_recon/modelv1/train.py
python 64Nodes/mlp/fixed_change_recon/modelv1/inference.py
```

