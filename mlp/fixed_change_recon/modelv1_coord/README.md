# modelv1_coord

- 任务：固定3变化的纯重构（输出112维 `dR`）
- 定位：`modelv1` 的坐标约束增强版（原 `modelv2_coord` 重编号）
- 核心：在 `modelv1` 损失上增加坐标矩约束（基于 `id -> (x,y)`）

## 运行
```bash
python 64Nodes/mlp/fixed_change_recon/modelv1_coord/train.py
python 64Nodes/mlp/fixed_change_recon/modelv1_coord/inference.py
```

