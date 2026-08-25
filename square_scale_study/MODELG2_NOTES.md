# modelg2 Notes

## 定位

`modelg2` 是 `modelg1` 的训练协议修正版，不改变主干结构，只调整训练过程。

它的目标不是“换一套更复杂的模型”，而是先回答一个更朴素的问题：

- 如果保持图结构主干不变
- 只修学习率策略、数据吞吐和评估节奏
- exact-support 恢复是否会明显改善

## 相比 modelg1 的三项变化

1. 学习率调度

- 从 `CosineAnnealingWarmRestarts`
- 改为 `ReduceLROnPlateau`

原因：

- `modelg1` 已经看到明显的 warm restart 扰动
- 在当前任务里，`val_id_exact_rate` 对学习率重启较敏感
- 单调下降更有利于 support 排序逐步收敛

2. DataLoader

- 增加 `num_workers`
- 增加 `pin_memory`
- 增加 `persistent_workers`
- 增加 `prefetch_factor`

原因：

- 5090 上的小图训练不是纯算力瓶颈
- 训练存在明显锯齿状利用率
- 需要尽量减少 batch 准备与主机到设备搬运的空转时间

3. 评估频率

- 不再每个 epoch 都对完整 `train/val` 做全量评估
- 改为：
  - `val` 低频评估
  - `train` 更低频评估
  - 训练结束后再对 best checkpoint 重新做完整 `train/val` 评估

原因：

- `modelg1` 中后期存在“评估比训练更重”的问题
- 这会拉低吞吐，也会让训练日志出现很多离散跳动

## 保持不变的内容

为保证结果可归因，`modelg2` 保持下列内容不变：

- `GINE + 4` 层 residual graph block
- `JK concat`
- `mean/max` 跨激励融合
- 纯回归边输出头
- `weighted normalized MSE + L1`
- `top-K(|pred|)` 的 `id` 识别规则

## 为什么当前更建议子项目 2 / 3 先用 4x4

当前 `modelg1` 下：

- `6x6` 的严格阈值 `K_max` 太低
- 如果直接在 `6x6` 上做 `M_var-K_max` 或 `P_active-K_max`
- 曲线很可能只有 `0/1` 或很低区间的变化

这会带来两个问题：

- 即使控制变量真的有效，也可能看不出清晰趋势
- 即使没效果，也无法判断是“变量没作用”还是“基线已经贴地”

因此第一轮更合理的策略是：

- 子项目 1：仍保留 `3x3~6x6`
- 子项目 2：先在 `4x4` 上做 pilot
- 子项目 3：先在 `4x4` 上做 pilot
- 若趋势明确，再把协议迁回 `6x6` 做第二轮验证

## 当前建议的实际执行顺序

1. 用 `modelg2` 重跑子项目 1
2. 观察 `P-K_max` 主曲线是否优于 `modelg1`
3. 若要尽快和师兄讨论，再用 `4x4` 先跑子项目 2
4. 用同一版 `modelg2` 跑 `4x4` 的子项目 3
5. 讨论后再决定是否值得把子项目 2 / 3 扩展回 `6x6`
