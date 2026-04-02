# GNN_NOISE 说明（64Nodes）

说明：
- 本目录用于执行 `Noise_test` 的策略 B：数据增强 + 带噪微调。
- 当前实现不是从头训练，而是在 clean 最优权重上做 low-lr fine-tune。

## 一、目录结构
- `GNN_NOISE/CLS_modelo3_ft`
  - 基于 `GNN_CLS/modelo3`
  - 用于分类模型带噪微调
- `GNN_NOISE/REG_o4a2_ft`
  - 基于 `GNN_REG/o4a2`
  - 用于回归模型带噪微调

## 二、当前策略
- warm start：
  - `CLS_modelo3_ft/train.py` 默认加载 `GNN_CLS/modelo3` 的 clean `model_last.pt`
  - `REG_o4a2_ft/train.py` 默认加载 `GNN_REG/o4a2` 的 clean `model_last.pt`
- 学习率：
  - 默认 `lr=5e-5`
- 微调轮数：
  - 默认 `epochs=30`
- 噪声注入：
  - 仅训练集开启
  - 每个样本动态随机采样噪声强度
  - 默认 `noise_schedule=random`
  - 默认 `noise_mode=gaussian`
  - 默认 `noise_std_max=0.1`
  - 默认 `fixed_noise_std=0.1`
  - 默认 `noise_scope=boundary`
  - 每次取样时实际噪声强度为 `noise_std = noise_std_max * rand()`

## 三、实现细节
- 输入仍保持原始主线相同的图表示：
  - `(Batch, 32, 64, 4)`
  - 四个通道分别为 `src_mask / gnd_mask / voltage_delta / boundary_mask`
- 噪声只注入到：
  - 外部边界节点的 `voltage_delta` 通道
- 验证集与测试集保持 clean：
  - 这样可以分离“训练增强”与“评估阶段显式加噪”的作用

## 四、推荐训练命令

### CLS
```bash
python gnn/GNN_NOISE/CLS_modelo3_ft/train.py \
  --data-path data/training_data64Nodes_2.csv \
  --dataset-tag training_data64Nodes_2 \
  --epochs 30 \
  --lr 5e-5 \
  --noise-schedule random \
  --noise-mode gaussian \
  --noise-scope boundary \
  --noise-std-max 0.1
```

### REG
```bash
python gnn/GNN_NOISE/REG_o4a2_ft/train.py \
  --data-path data/training_data64Nodes_2.csv \
  --dataset-tag training_data64Nodes_2 \
  --epochs 30 \
  --lr 5e-5 \
  --noise-schedule random \
  --noise-mode gaussian \
  --noise-scope boundary \
  --noise-std-max 0.1
```

## 五、原始 `Noise_test` 步骤 B 保留版

- 根目录原 `Noise_test.txt` 的步骤 B 并不和当前默认增强版完全相同。
- 为避免根目录继续散落说明文件，原始版本已经迁移到：
  - `gnn/GNN_NOISE/原始步骤B_fixed20dB.md`
- 它对应的固定 `20dB` 口径为：
  - `noise_schedule=fixed`
  - `fixed_noise_std=0.1`
  - `noise_mode=gaussian`
  - `noise_scope=all`

## 六、推荐评估命令

### 单模型带噪评估
```bash
python gnn/GNN_NOISE/CLS_modelo3_ft/inference.py \
  --dataset-tag training_data64Nodes_2 \
  --noise-std 0.1 \
  --noise-seed 20260331
```

```bash
python gnn/GNN_NOISE/REG_o4a2_ft/inference.py \
  --dataset-tag training_data64Nodes_2 \
  --noise-std 0.1 \
  --noise-seed 20260331
```

### 联合带噪评估
```bash
python gnn/inference_gnn_cmei.py \
  --dataset-tag training_data64Nodes_2 \
  --cls-dir GNN_NOISE/CLS_modelo3_ft \
  --reg-dir GNN_NOISE/REG_o4a2_ft \
  --noise-std 0.1 \
  --noise-seed 20260331 \
  --out-dir outputs/gnn_cmei_noise_ft_20db
```

## 七、当前判断
- 这条线的目标不是在 clean 测试集上继续追求更高分
- 而是显式修复 zero-shot 20dB 噪声下的流形塌陷
- 如果这条线有效，后续更值得比较的是：
  - 带噪微调前后的 `20dB` 指标恢复程度
  - `CMEI / macro_f1 / id_recall / mse_all_edges`

## 八、附加记录
- `首轮20dB噪声诊断记录.md` 已从根目录迁移到本目录：
  - `gnn/GNN_NOISE/首轮20dB噪声诊断记录.md`
- 这样当前 `Noise_test` 相关材料已经集中为：
  - `首轮20dB噪声诊断记录.md`
  - `原始步骤B_fixed20dB.md`
  - `CLS_modelo3_ft / REG_o4a2_ft` 训练与评估脚本

## 九、`0401训练记录` 结论
- `rand_boundary` 推荐增强版在本次云端记录中没有完整闭环：
  - `REG` 训练完成
  - `CLS` 对应 tag 下没有真正落出 `model_last.pt`
- `fixed20db_all` 原始步骤 B 保留版已完整跑通：
  - `CLS` 最终保留结果：`test_macro_f1=0.7275`
  - `REG` 最终保留结果：`mae_all=0.9201`，`mae_changed=48.8259`
  - clean 联合：`CMEI=83.39`
  - noisy 单模型：`CLS macro_f1=0.7121`，`REG count_macro_f1=0.5829`
- 当前建议：
  - 不把 `fixed20db_all` 直接作为默认带噪方案
  - 先补齐 `rand_boundary` 的 `CLS + REG` 成对训练，再做正式 noisy CMEI 比较

## 十、2026-04-01 可视化与当前补训项
- 按新的中期汇报需求，旧的图文一体式可视化材料已经删除：
  - `midterm_assets/20260401_data_figures`
  - `中期汇报_数据可视化说明.md`
  - `tools/generate_midterm_figures.py`
- 新的最终图集只保留图像结果，位于：
  - `midterm_assets/20260401_visuals/01_topology_boundary_nodes.svg`
  - `midterm_assets/20260401_visuals/02_dataset_composition.svg`
  - `midterm_assets/20260401_visuals/03_changed_edge_frequency.svg`
  - `midterm_assets/20260401_visuals/04_boundary_response_heatmaps.svg`
- 本轮再次确认的工程状态：
  - `rand_boundary` 分支里真正缺的是 `CLS_modelo3_ft` 对应 tag 的完整训练落盘
  - `fixed20db_all` 分支已经具备完整权重与评估文件
  - 若云端要继续做 noisy 联合推理，需要确保 `gnn/inference_gnn_cmei.py` 已同步为本地最新版

## 十一、2026-04-01 `0401补充训练` 结果
- 本次补充训练把推荐增强版 `rand_boundary` 的分类侧正式补齐：
  - `CLS clean test_macro_f1=0.8750`
  - `CLS noisy(20dB) test_macro_f1=0.7780`
- noisy 联合结果：
  - `CMEI=82.56`
  - `num_accuracy=0.7360`
  - `macro_f1=0.7780`
  - `id_recall=0.7579`
  - `mse_all_edges=154.4499`
- 与当前本地目录核对后确认：
  - `CLS rand_boundary` 的输出目录已完整
  - `REG rand_boundary` 当前缺 `noise_eval.json`
  - `gnn/outputs` 当前还缺本次 `rand_boundary` noisy 联合输出目录
- 因此现阶段更准确的总结是：
  - 推荐增强版已经在 noisy 联合链路上跑通
  - 但本地落档还不完整，后续应把对应 joint 目录再从云端同步一次

## 十二、2026-04-01 鲁棒性曲线脚本
- 新增：
  - `gnn/GNN_NOISE/plot_noise_robustness.py`
- 功能：
  - 读取 `20/30/40dB` 不同方法的 `json` 指标文件
  - 生成同一张 `svg` 对比曲线图
- 附带清理：
  - 根目录 `0401补充训练.txt` 已删除，因为内容已经吸收完成

## 十三、2026-04-01 `rand_boundary` 主线阈值细化结论
- 已新增 `gnn/GNN_CLS/modelo3/two_stage_threshold_search.py`，用于对 `CLS` 兼容权重做独立阈值细化。
- 已对正式主线 `training_data64Nodes_2_noiseft_rand_boundary_20260401` 实跑：
  - 原阈值：`[0.05, 0.17, 0.37]`
  - 细化后：`[0.05, 0.164, 0.368]`
  - `val_macro_f1=0.8976 -> 0.8976`
  - `test_macro_f1=0.8750 -> 0.8749`
- 结论：
  - 当前阈值粗搜已经足够接近最优
  - 正式主线后续不应把主要时间继续投在阈值调参上

## 十四、2026-04-02 `0402补充日志` 与主线状态
- `rand_boundary` 的 `20dB` 补推理现已齐全：
  - `REG noisy`: `mae_all=1.2692`，`mae_changed=54.1729`，`count_macro_f1=0.5844`
  - clean joint: `CMEI=91.01`
  - 20dB joint: `CMEI=82.56`
- `fixed20db_all` 现已齐全：
  - clean joint: `CMEI=83.39`
  - 20dB joint: `CMEI=81.79`
- 当前正式结论：
  - `rand_boundary` 仍优于 `fixed20db_all`
  - 因此后续只建议继续做 `30/40dB` 扩展，不需要再补新的 20dB 命令

## 十五、2026-04-02 最终鲁棒性曲线
- 已吸收 `0402大范围噪声训练.txt`
- 已生成最终图：
  - `rand_boundary_robustness_curve.svg`
- 作图脚本：
  - `plot_rand_boundary_robustness.py`
- `rand_boundary` 最终 joint 曲线：
  - clean `CMEI=91.01`
  - `40dB CMEI=90.83`
  - `30dB CMEI=89.62`
  - `20dB CMEI=82.56`
- 最终工程结论：
  - `40/30dB` 区间已接近 clean 水平
  - `20dB` 虽有衰减，但相较 zero-shot 已是质变式恢复
  - 这条线已经足以作为当前“可汇报、可继续扩展”的正式最佳方案
- 附带清理：
  - `0402补充日志.txt` 已删除
  - `0402大范围噪声训练.txt` 已删除
## 八、后续目录约定（2026-04-02）
- `GNN_NOISE` 只负责：
  - 带噪训练
  - 单模型推理
  - 各自分支内的 `outputs/`
- joint `CMEI` 输出不再继续写到 `gnn/outputs`，而是统一迁到：
  - `gnn/GNN_CMEI_INFERENCE/outputs`
- 因此下一版带噪实验如继续做，建议目录统一新建为：
  - `GNN_NOISE/CLS_modelo3_ft_v2`
  - `GNN_NOISE/REG_o4a2_ft_v2`
- 每一版都保持：
  - 自己的 `cache/`
  - 自己的 `outputs/`
  - 不与上一版共用输出目录

## 九、`v2` 带噪训练分支（2026-04-02）
- 已新建：
  - `GNN_NOISE/CLS_modelo3_ft_v2`
  - `GNN_NOISE/REG_o4a2_ft_v2`
- 设计目标：
  - 保留 `rand_boundary` 的 clean / noisy 平衡
  - 在其基础上进一步引入更贴近真实电极测量的结构化边界噪声
- `v2` 默认策略：
  - warm start 从 `v1 rand_boundary` 权重继续 fine-tune
  - `noise_schedule=curriculum`
  - `noise_mode=structured`
  - `noise_scope=boundary`
  - 同时混入少量 clean 样本 `clean_mix_prob=0.15`
- 结构化噪声由 4 部分组成：
  - 局部独立白噪声 `iid`
  - 电极级慢漂移 `drift`
  - 每次激励的共模偏移 `common`
  - 少量异常电极 `bad electrode`
- 说明：
  - 单模型训练与推理结果仍保留在各自 `v2/outputs/`
  - 不会迁入 `GNN_CMEI_INFERENCE`
