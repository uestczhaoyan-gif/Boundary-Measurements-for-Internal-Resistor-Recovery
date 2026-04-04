# GNN_NOISE 日志（64Nodes）

说明：
- 本文件记录 `Noise_test` 策略 B 的实现、训练与评估结果。
- 更新规则：只追加，不覆盖历史记录。

## 2026-03-31 - GNN_NOISE 建立
- 新建目录：
  - `gnn/GNN_NOISE/CLS_modelo3_ft`
  - `gnn/GNN_NOISE/REG_o4a2_ft`
- 设计目标：
  - 不从头训练
  - 直接继承 clean 最优权重
  - 用训练阶段的动态随机噪声增强做低学习率微调

## 2026-03-31 - 带噪微调脚本首版落地
### `CLS_modelo3_ft`
- 来源：
  - 复制自 `gnn/GNN_CLS/modelo3`
- 关键改动：
  - 默认 `pretrained_model_path=../../GNN_CLS/modelo3/outputs/training_data64Nodes_2/model_last.pt`
  - 默认 `epochs=30`
  - 默认 `lr=5e-5`
  - 默认开启 `add_noise=True`
  - 新增：
    - `--noise-schedule {random,fixed}`
    - `--noise-mode {gaussian,uniform}`
    - `--noise-std-max`
    - `--fixed-noise-std`
    - `--noise-scope {boundary,all}`
    - `--pretrained-model-path`
- 数据增强逻辑：
  - 仅训练集注入噪声
  - 每次 `__getitem__` 都重新采样噪声强度
  - 默认 `noise_std = 0.1 * rand()`
  - 噪声仅注入边界节点的 `voltage_delta` 通道

### `REG_o4a2_ft`
- 来源：
  - 复制自 `gnn/GNN_REG/o4a2`
- 关键改动：
  - 默认 `pretrained_model_path=../../GNN_REG/o4a2/outputs/training_data64Nodes_2/model_last.pt`
  - 默认 `epochs=30`
  - 默认 `lr=5e-5`
  - 默认开启 `add_noise=True`
  - 新增：
    - `--noise-schedule {random,fixed}`
    - `--noise-mode {gaussian,uniform}`
    - `--noise-std-max`
    - `--fixed-noise-std`
    - `--noise-scope {boundary,all}`
    - `--pretrained-model-path`
- 数据增强逻辑：
  - 仅训练集注入噪声
  - 每次 `__getitem__` 动态采样噪声强度
  - 仍保持 `o4a2` 原本的 gated regression、`mask BCE` 与 `SmoothL1` 训练逻辑不变

## 2026-03-31 - 当前状态
- 已完成：
  - 目录建立
  - 脚本改写
  - 语法校验与 `--help` 验证待执行
- 尚未完成：
  - 云端实际 fine-tune
  - 带噪微调后在 `20dB` 下的恢复效果评估

## 2026-04-01 - 原始步骤 B 收编
- 已确认当前默认 `GNN_NOISE/*_ft` 不是根目录原 `Noise_test` 步骤 B 的完全等价实现：
  - 当前默认版是 `warm start + random noise + boundary-only`
  - 原始步骤 B 更接近 `fixed 20dB gaussian + all-voltage-channel`
- 因此新增：
  - `gnn/GNN_NOISE/原始步骤B_fixed20dB.md`
- 同时在 `CLS_modelo3_ft/train.py` 与 `REG_o4a2_ft/train.py` 补齐了可复现原始步骤 B 的参数：
  - `--noise-schedule fixed`
  - `--fixed-noise-std 0.1`
  - `--noise-scope all`

## 2026-04-01 - 首轮诊断文件迁入
- 已将根目录 `首轮20dB噪声诊断记录.md` 迁移到本目录：
  - `gnn/GNN_NOISE/首轮20dB噪声诊断记录.md`
- 现在 `Noise_test` 相关的三类材料已集中：
  - 首轮 20dB zero-shot 崩塌诊断
  - 原始步骤 B 保留版
  - 当前推荐的 noisy fine-tuning 实现

## 2026-04-01 - `0401训练记录` 吸收
- clean 主线复训结果：
  - `modelo3`: `test_macro_f1=0.9027`
  - `o4a2`: `mae_all=0.4679`，`mae_changed=23.5724`
  - 联合：`CMEI=93.53`
- `noiseft_rand_boundary_20260401`：
  - `REG_o4a2_ft` 已完成：`mae_all=0.5900`，`mae_changed=27.5311`，`count_macro_f1=0.8140`
  - `CLS_modelo3_ft` 对应输出目录仅有 `standardization.npz`
  - 说明该 tag 下没有完整训练好的分类权重
- `noiseft_fixed20db_all_20260401`：
  - `CLS_modelo3_ft` 同 tag 被连续训练两次，最终保留 `test_macro_f1=0.7275`
  - `REG_o4a2_ft` 最终为 `mae_all=0.9201`，`mae_changed=48.8259`，`count_macro_f1=0.6178`
  - clean 联合为 `CMEI=83.39`
  - noisy 单模型为：
    - `CLS macro_f1=0.7121`
    - `REG count_macro_f1=0.5829`
- 当前判断：
  - fixed-all 的鲁棒性恢复是真的；
  - 但代价过大，不适合作为默认带噪线；
  - 下一步应补齐 `rand_boundary` 的分类侧训练，形成完整联合评估。

## 2026-04-01 - 可视化结果替换与补训确认
- 已删除旧可视化目录与说明材料：
  - `midterm_assets/20260401_data_figures`
  - `中期汇报_数据可视化说明.md`
  - `tools/generate_midterm_figures.py`
- 已生成新的最终图集：
  - `midterm_assets/20260401_visuals/01_topology_boundary_nodes.svg`
  - `midterm_assets/20260401_visuals/02_dataset_composition.svg`
  - `midterm_assets/20260401_visuals/03_changed_edge_frequency.svg`
  - `midterm_assets/20260401_visuals/04_boundary_response_heatmaps.svg`
- 当前仍待补齐的训练闭环：
  - `CLS_modelo3_ft` with `dataset_tag=training_data64Nodes_2_noiseft_rand_boundary_20260401`
- 当前云端推理前置检查：
  - `gnn/inference_gnn_cmei.py` 需与本地最新版一致，才能识别 `--noise-std / --noise-seed`

## 2026-04-01 - `0401补充训练` 记录吸收
- 已吸收根目录 `0401补充训练.txt`
- 推荐增强版 `rand_boundary` 新增结果：
  - `CLS clean test_macro_f1=0.8750`
  - `CLS noisy test_macro_f1=0.7780`
  - noisy joint `CMEI=82.56`
  - `num_accuracy=0.7360`
  - `id_recall=0.7579`
  - `mse_all_edges=154.4499`
- 本地目录核对：
  - `CLS rand_boundary` 输出完整
  - `REG rand_boundary` 当前未见 `noise_eval.json`
  - `gnn/outputs` 当前未见本次 `rand_boundary` 的 joint 输出目录
- 当前判断：
  - 训练是成功补齐了
  - 但本地下载结果并不完整，后续应补同步 joint 输出与缺失评估文件

## 2026-04-01 - 鲁棒性曲线脚本与草稿清理
- 已删除根目录 `0401补充训练.txt`
- 已新增：
  - `gnn/GNN_NOISE/plot_noise_robustness.py`
- 作用：
  - 面向 `20/30/40dB` 噪声评估结果绘制统一对比曲线

## 2026-04-01 - `rand_boundary` 阈值细化复核
- 已使用 `gnn/GNN_CLS/modelo3/two_stage_threshold_search.py` 对 `rand_boundary` 分类器做两阶段细阈值搜索
- 结果：
  - 原阈值：`[0.05, 0.17, 0.37]`
  - 细化后：`[0.05, 0.164, 0.368]`
  - `val_macro_f1` 持平
  - `test_macro_f1` 基本不变且略回落
- 当前判断：
  - `rand_boundary` 的核心改进空间不在阈值离散步长
  - 更应继续增强 `boundary-only` 带噪训练的鲁棒性

## 2026-04-02 - `0402补充日志` 结果吸收
- 已吸收根目录 `0402补充日志.txt`
- 当前 `20dB` 结果已补齐：
  - `rand_boundary` clean joint `CMEI=91.01`
  - `rand_boundary` 20dB joint `CMEI=82.56`
  - `fixed20db_all` clean joint `CMEI=83.39`
  - `fixed20db_all` 20dB joint `CMEI=81.79`
  - `REG rand_boundary` 20dB 单模型 `mae_all=1.2692`，`mae_changed=54.1729`
- 当前判断：
  - 不再需要补新的 20dB 推理
  - 后续只建议继续跑 `30/40dB`

## 2026-04-02 - 最终鲁棒性曲线与主线收敛
- 已吸收 `0402大范围噪声训练.txt`
- 已生成最终图：
  - `gnn/GNN_NOISE/rand_boundary_robustness_curve.svg`
- joint 曲线结果：
  - clean `CMEI=91.01`
  - `40dB CMEI=90.83`
  - `30dB CMEI=89.62`
  - `20dB CMEI=82.56`
- 当前判断：
  - `rand_boundary` 在 `40/30dB` 区间基本保持 clean 性能
  - `20dB` 下依然显著优于 zero-shot baseline
  - 正式主线已收敛到 `rand_boundary`
- 已删除：
  - `0402补充日志.txt`
  - `0402大范围噪声训练.txt`
## 2026-04-02 - 带噪目录职责收敛
- `GNN_NOISE` 后续只保留：
  - 带噪训练
  - 单模型推理
  - 分支内 `outputs/`
- joint `CMEI` 结果目录已统一迁往：
  - `gnn/GNN_CMEI_INFERENCE/outputs`
- 下一版命名建议：
  - `CLS_modelo3_ft_v2`
  - `REG_o4a2_ft_v2`

## 2026-04-02 - `v2` 带噪训练分支建立
- 已新建：
  - `gnn/GNN_NOISE/CLS_modelo3_ft_v2`
  - `gnn/GNN_NOISE/REG_o4a2_ft_v2`
- 已清理复制时带入的旧目录：
  - `cache/`
  - `outputs/`
  - `__pycache__/`
- `v2` 当前默认口径：
  - warm start from `v1 rand_boundary`
  - `noise_schedule=curriculum`
  - `noise_mode=structured`
  - `noise_scope=boundary`
- 噪声组成：
  - `iid`
  - `drift`
  - `common`
  - `bad electrode`
- 已完成：
  - `train.py` 语法校验
  - 参数入口校验

## 2026-04-03 - `0402噪声v2训练日志` 吸收与验证脚本补强
- 已吸收根目录 `0402噪声v2训练日志.txt`，但未直接转抄原始终端流水。
- 本轮 `v2` 云端已确认：
  - clean `CLS test_macro_f1=0.9149`
  - clean `REG mae_all=0.4664`，`mae_changed=24.2457`，`count_macro_f1=0.8349`
  - clean joint `CMEI=93.49`
  - `40dB CLS test_macro_f1=0.9078`
  - `40dB REG mae_all=0.5317`，`mae_changed=25.3754`，`count_macro_f1=0.8342`
- `30dB / 20dB` 本轮未得到有效结果，已定位为：
  - 验证命令中的 `--dataset-tag ${TAG}` 未展开
  - 参数解析在推理开始前直接退出
- 已新增：
  - `gnn/GNN_NOISE/run_noise_eval_suite.py`
- 该脚本用于：
  - 统一串行执行 clean / `40dB` / `30dB` / `20dB` 的 `CLS / REG / joint` 评估
  - 自动把单模型结果按噪声等级保存成独立 `json`
  - 支持 `--dry-run` 先打印完整命令
- 同时已在以下入口补充 `dataset-tag` 空值 / 占位符未展开的明确报错：
  - `gnn/GNN_NOISE/CLS_modelo3_ft_v2/train.py`
  - `gnn/GNN_NOISE/CLS_modelo3_ft_v2/inference.py`
  - `gnn/GNN_NOISE/REG_o4a2_ft_v2/train.py`
  - `gnn/GNN_NOISE/REG_o4a2_ft_v2/inference.py`
  - `gnn/GNN_CMEI_INFERENCE/inference_gnn_cmei.py`

## 2026-04-04 - `0404训练日志` 吸收与 `v2` 完整曲线
- 已吸收根目录 `0404训练日志.txt`，但最终记录继续以本地真实 `outputs/*.json` 为准。
- 本轮已真正补齐 `clean / 40dB / 30dB / 20dB` 四档 `CLS / REG / joint` 结果：
  - clean：`CLS macro_f1=0.9149`，`REG mae_changed=24.2457`，`joint CMEI=93.49`
  - `40dB`：`CLS macro_f1=0.9078`，`REG mae_changed=25.3754`，`joint CMEI=92.81`
  - `30dB`：`CLS macro_f1=0.8903`，`REG mae_changed=34.0454`，`joint CMEI=90.44`
  - `20dB`：`CLS macro_f1=0.7582`，`REG mae_changed=58.1169`，`joint CMEI=80.42`
- 与 `rand_boundary` 对比：
  - clean：`91.01 -> 93.49`
  - `40dB`：`90.83 -> 92.81`
  - `30dB`：`89.62 -> 90.44`
  - `20dB`：`82.56 -> 80.42`
- 当前判断修正为：
  - `v2` 的优势已经明确落在 clean 到中噪声区间
  - 但 `20dB` 端点仍由 `rand_boundary` 保持领先
  - 后续如果继续迭代 `v2`，应优先针对最重噪声下的数量与定位退化做增强，而不是继续只抠 clean 指标
- 本轮已新增正式可视化脚本与底表：
  - `plot_noise_v2_summary.py`
  - `noise_v2_summary_metrics.json`
- 当前正式图像输出：
  - `Figure/noise_v2_summary.png`
  - `Figure/noise_v2_summary.pdf`
- 旧图：
  - `rand_boundary_robustness_curve.svg`
  已删除，不再保留。
