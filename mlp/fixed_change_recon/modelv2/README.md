# modelv2

- 任务：固定3变化的纯重构（输出112维 `dR`）
- 定位：`modelv1` 的“重构约束增强版（无坐标默认）”
- 主要新增：
  - 固定计数软约束 `L_count3`
  - hardest 正负分离约束 `L_sep`
  - 评分加入 `top3_id_precision`
- 默认 `lambda_coord=0.0`，作为非坐标对照版本

## 运行
```bash
python 64Nodes/mlp/fixed_change_recon/modelv2/train.py
python 64Nodes/mlp/fixed_change_recon/modelv2/inference.py
```

