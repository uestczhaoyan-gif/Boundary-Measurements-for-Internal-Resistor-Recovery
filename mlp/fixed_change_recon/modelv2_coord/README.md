# modelv2_coord

- 任务：固定3变化的纯重构（输出112维 `dR`）
- 定位：`modelv2` 的坐标约束版
- 主要新增：
  - 固定计数软约束 `L_count3`
  - hardest 正负分离约束 `L_sep`
  - 坐标矩约束 `L_coord`
  - 评分加入 `top3_id_precision`

## 运行
```bash
python 64Nodes/mlp/fixed_change_recon/modelv2_coord/train.py
python 64Nodes/mlp/fixed_change_recon/modelv2_coord/inference.py
```

