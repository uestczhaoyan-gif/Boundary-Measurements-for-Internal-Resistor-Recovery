# modelv1_new

- 任务：固定 3 变化场景下的纯重建（输出 112 维 `dR`）
- 架构：`896 -> 1024 -> 896 -> 512 -> 256 -> 112`
- 每层：`BN + ReLU + Dropout`
- 残差：输入 `896` 到隐藏层 `896` 做跳连
- 输出约束：`tanh(y) * max_abs`，默认 `max_abs=310`

## 指标输出
- `mae_all`
- `mae_changed`
- 位置预测准确率（对 0 个 / 对 1 个 / 对 2 个 / 全对）

## 损失构成（权重优先级）
- `L_mse`（最高）
- `L_id`（坐标法预测 ID 误差）
- `L_phys`（基尔霍夫约束，每批随机 4 个激励）
- `L_sparse`（L1 稀疏约束，最低）

## 运行
```bash
python 64Nodes/mlp/fixed_change_recon/modelv1_new/train.py
python 64Nodes/mlp/fixed_change_recon/modelv1_new/inference.py
```

