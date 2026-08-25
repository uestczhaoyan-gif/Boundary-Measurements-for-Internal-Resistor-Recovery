# GNN noise robustness

本目录研究边界电压测量噪声对数量分类、变化量回归和联合 CMEI 的影响。噪声只注入外部测量节点，训练分支通过 fine-tune 或 curriculum 学习恢复性能。

## 分支

- `CLS_modelo3_ft` / `REG_o4a2_ft`：早期 `rand_boundary` 方案。
- `CLS_modelo3_ft_v2` / `REG_o4a2_ft_v2`：结构化边界噪声与 clean mix 的 v2 主线。
- `robustness_archive/`：已停止推进的噪声方案。

## 推荐评估

```powershell
python gnn\GNN_NOISE\run_noise_eval_suite.py --help
python gnn\GNN_NOISE\run_noise_eval_suite.py `
  --run-tag training_data64Nodes_2_noiseft_struct_boundary_v2_20260402 `
  --dry-run
```

去掉 `--dry-run` 后才会真正执行 clean、40 dB、30 dB 和 20 dB 评估。请确认 `--run-tag` 是实际字符串，不要把未展开的 shell 占位符传给 Python。

## 阶段性曲线

v2 记录：clean `CMEI=93.49`、40 dB `92.81`、30 dB `90.44`、20 dB `80.42`。旧 `rand_boundary` 在 20 dB 记录过 `82.56`。因此 v2 适合作为 clean 到中噪声区间的主线，不能宣称在所有噪声强度下都更优。

单模型输出留在各训练分支，联合输出统一写入 `gnn/GNN_CMEI_INFERENCE/outputs/`。详细噪声口径见 `Log.md`。
