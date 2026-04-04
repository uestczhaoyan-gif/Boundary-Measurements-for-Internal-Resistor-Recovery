# GNN 运行日志（64Nodes）

说明：
- 记录 `64Nodes/gnn` 各版本关键指标、问题分析与改进计划。
- 更新规则：仅追加，不覆盖历史。

记录模板：
- 版本与训练命令
- 关键指标（分类混淆矩阵；回归 mae_all/mae_changed/avg(|dR|>50)）
- 现象分析
- 下一轮修改动作

## 2026-03-20 - v1 工程骨架初始化
- 新建 `GNN_CLS/modelo1`
- 新建 `GNN_REG/modelo1`
- 新建 `GNN_FULL/modelo1_h_multitask`
- 架构：GraphSAGE 风格图主干 + 边解码器基准
- 目标：验证图结构归纳偏置在 64Nodes 上的有效性

## 2026-03-23 - 主线训练脚本支持按数据集切换与分目录输出
- 目的：
- 方便在云端并行训练 `5mA` 与 `10mA` 两套主数据，不再手动改代码路径。
- 本轮改动入口：
- `GNN_CLS/modelo1/train.py`
- `GNN_REG/modelo1/train.py`
- `GNN_FULL/modelo1_h_multitask/train.py`
- 新增命令行参数：
- `--data-path`
- `--dataset-tag`
- `--dataset-subdir / --no-dataset-subdir`
- 行为调整：
- 默认 cache 改为写入 `cache/<dataset_tag>/...`
- 默认输出改为写入 `outputs/<dataset_tag>/...`
- 直接收益：
- 避免不同数据集复用同一个 cache
- 避免 `5mA` / `10mA` 模型结果覆盖

## 2026-03-23 - 主线 inference 脚本支持按数据集标签自动找模型
- 本轮补齐入口：
- `GNN_CLS/modelo1/inference.py`
- `GNN_REG/modelo1/inference.py`
- `GNN_FULL/modelo1_h_multitask/inference.py`
- 新增参数：
- `--data-path`
- `--dataset-tag`
- `--dataset-subdir / --no-dataset-subdir`
- 默认行为：
- 自动从 `cache/<dataset_tag>/...` 读取缓存
- 自动从 `outputs/<dataset_tag>/...` 读取 `model_last.pt / metrics.json / standardization.npz`
- `inference_samples.json` 也会写回对应数据集子目录

## 2026-03-24 - GNN_CLS/modelo1 三电流首轮对比（用户回传）
- 测试结果摘要：
- `5mA`：最佳验证 `val_macro_f1=0.7791`
- `10mA`：最佳验证 `val_macro_f1=0.7825`
- `20mA`：最佳验证 `val_macro_f1=0.7776`
- 现象分析：
- 三套数据差距不大，说明单纯增大激励并没有明显改变当前 GNN 分类瓶颈。
- `10mA` 略优，但 `2/3` 类双向混淆依然明显，分类线仍不适合作为当前主推进方向。

## 2026-03-24 - GNN_REG/modelo1 三电流首轮对比（用户回传）
- `5mA`：
- `mae_all=3.2819`
- `mae_changed=44.4122`
- `best_count_threshold(val)=64.0`
- `avg(|dR|>64)=2.34`
- `10mA`：
- `mae_all=3.5430`
- `mae_changed=41.7682`
- `best_count_threshold(val)=64.0`
- `avg(|dR|>64)=2.33`
- `20mA`：
- `mae_all=3.0921`
- `mae_changed=43.2367`
- `best_count_threshold(val)=61.0`
- `avg(|dR|>61)=2.27`
- 现象分析：
- `10mA` 当前给出了最优 `mae_changed`；
- `20mA` 在 `mae_all` 和稀疏性上更好，但没有继续改善 `mae_changed`。
- 结论：
- 若 GNN 只保留一套主数据继续迭代，当前建议优先用 `10mA`。

## 2026-03-24 - GNN_FULL/modelo1_h_multitask 报错与修正
- 0324 首轮训练命令在启动 Stage1 前报错：
- `AttributeError: 'CNN2DHMultiTask' object has no attribute 'reg_head'`
- 原因：
- 训练脚本冻结/解冻回归分支时访问了错误属性名；
- 模型中实际的回归头名称是 `edge_mlp`。
- 当前处理：
- 已修正训练脚本；
- `GNN_FULL` 结果需要重新补跑后再评价，不应把这次报错当成模型结论。

## 2026-03-24 - GNN 主线优先级与默认数据调整
- 当前 GNN 主推进顺序改为：
- 第一优先级：`GNN_REG/modelo1`
- 第二优先级：fixed-change 重构对照
- 第三优先级：`GNN_CLS/modelo1`
- `GNN_FULL/modelo1_h_multitask` 暂不作为每轮同步推进对象。
- 非 fixed-change GNN 主线默认数据统一切到 `10mA`：
- `data/training_data64Nodes_2.csv`
- 保留 `--data-path` 与 `--dataset-tag`，因此后续仍能按需回切 `5mA / 20mA` 做对照。

## 2026-03-24 - GNN 主线下一版优化（按当前优先级）
- `GNN_REG/modelo1`：
  - 阈值搜索从纯 `macro-F1` 改为 `macroF1 + class2_F1`，默认 `count_cls2_weight=0.35`
  - 新增 `L_fp_next` 抑制真实变化数之后的第一个伪峰：
    - `lambda_fp_next=0.12`
    - `fp_next_weight_k2=1.25`
    - `fp_next_weight_k3=1.45`
  - early stopping 改为更重建导向的 `val_score`
- `GNN_CLS/modelo1`：
  - 阈值搜索改为 weighted score
- 新增：
  - `penalty_32=0.12`
  - `bonus_r3=0.06`
  - `bonus_r2=0.05`

## 2026-03-25 - GNN_CLS/modelo1（10mA，权重调整后首轮回传）
- 训练结果：
- 最佳验证 `val_macro_f1=0.7855`
- 测试混淆矩阵：
  - `[[66,0,0,0],[0,263,54,0],[0,45,175,101],[0,8,81,207]]`
- 现象分析：
- 相比 0324 首轮没有形成明确增益；
- `2/3` 类仍大面积互相吞并，说明 weighted score 对主瓶颈帮助有限。
- 阶段结论：
- `GNN_CLS` 继续保持低优先级，只建议做小范围校准，不建议投入新的大版本。

## 2026-03-25 - GNN_REG/modelo1（10mA，权重调整后首轮回传）
- 测试结果：
- `mae_all=4.6721`
- `mae_changed=70.9448`
- `best_count_threshold(val)=45.0`
- `val_macro_f1=0.5930`
- `val_score=0.7400`
- `avg(|dR|>45)=2.22`
- 计数混淆矩阵：
  - `[[66,0,0,0],[24,179,52,62],[3,100,107,111],[1,41,89,165]]`
- 与 0324 的 `10mA` 基线对比：
- `mae_changed: 41.7682 -> 70.9448`，显著退化；
- 阈值也从 `64` 降到 `45`，表明当前新增约束把模型整体压到了更保守但失真的区域。
- 阶段判断：
- 本轮 `L_fp_next + class2 加权评分 + val_score` 的组合不适合作为继续主线；
- 下一轮应优先回退到上一个有效 baseline，再做单因素小步验证，而不是继续在当前组合上叠加改动。

## 2026-03-25 - GNN_REG/modelo1（默认值回退到 0324 更稳口径）
- 针对本轮显著退化，当前已把默认训练口径回退为更稳版本：
  - `lambda_fp_next: 0.12 -> 0.0`
  - `count_cls2_weight: 0.35 -> 0.0`
  - early stopping：`val_score -> val_loss`
- 保留项：
  - `val_overpredict` 仍可作为诊断日志保留
  - `false_positive_next_loss` 代码保留，但默认关闭，便于后续单因素消融
- 当前意图：
- 先把 `GNN_REG/modelo1` 恢复到 0324 的有效水平，再重新逐项测试新增约束是否真的有益。

## 2026-03-25 - GNN_REG/modelo1（10mA，回退后首轮回传）
- 测试结果：
- `mae_all=3.2785`
- `mae_changed=43.0275`
- `best_count_threshold(val)=67.0`
- `val_macro_f1=0.6871`
- `val_score=0.6871`
- `avg(|dR|>67)=2.17`
- 与回退前对比：
- `mae_changed: 70.9448 -> 43.0275`，大幅恢复；
- 阈值也回到更合理的 `67.0`。
- 与 0324 的最佳 `10mA` 基线对比：
- `mae_changed: 41.7682 -> 43.0275`，仍略差，但已回到可继续迭代的有效区间。
- 阶段判断：
- 当前回退方向有效，后续可以重新做单因素消融，但不应再把多项新增约束一次性叠加。

## 2026-03-25 - GNN_REG/modelo1 inference 修正
- 修正内容：
- 默认推理路径现在会按以下顺序自动兼容：
  - `outputs/<dataset_tag>/...`
  - `outputs/<data_path.stem>/...`
  - 旧版 `outputs/` 根目录文件
- 推理输出新增：
  - `pred_change_ids`
  - `pred_change_deltas`
  - `true_change_ids`
  - `true_change_deltas`
  - `true_deltas / pred_deltas`
  - `true_resistances / pred_resistances / abs_error_resistance`
- 目的：
- 解决旧输出目录命名和新 `dataset-tag` 并存时的查找问题；
- 让推理样例能直接核对预测位置与变化值。

## 2026-03-25 - GNN 新版实验目录建立（modelo2）
- 本轮新增：
  - `GNN_REG/modelo2`
  - `GNN_CLS/modelo2`
- `GNN_REG/modelo2` 结构要点：
  - 输入从旧版 97 通道网格改为 `(Batch, 32, 64, 4)` 的物理图表示
  - 单次激励先独立过 3 层残差式 GATv2 风格注意力层
  - 再对 32 次激励做 cross-excitation attention pooling
  - 边解码器输入为 `cat([H_u, H_v, |H_u-H_v|])`
  - 回归头改为 `mask(sigmoid) * value(tanh)` 门控输出
  - 损失改为 `MSE + lambda_mask_l1 * mean(mask_prob)`，默认 `lambda_mask_l1=0.05`
- `GNN_CLS/modelo2` 结构要点：
  - 图输入与注意力主干同 `REG`
  - 图级特征由节点池化后的 mean/max 聚合得到
  - 分类头保持 CORAL
  - 在分类头前增加 supervised contrastive 特征投影，重点拉开真实 `2/3` 类
- 当前状态：
  - `modelo2` 仅完成代码实现，尚未训练
  - `modelo1` 继续作为当前 GNN 稳定基线
- 本地验证说明：
  - 已通过 `python -m py_compile`
  - 由于当前本地环境缺少 `torch`，未完成真实前向冒烟测试
## 2026-03-25 - GNN modelo2 切换为 PyG 原生 GATv2Conv，并完成简单验证
- 按用户要求，已通过清华源在项目内建立本地依赖目录：`64Nodes/.vendor_torchpy311`
- 当前确认版本：
  - `torch 2.11.0+cpu`
  - `torch-geometric 2.7.0`
- 代码层改动：
  - `GNN_REG/modelo2/model/model.py`
  - `GNN_CLS/modelo2/model/model.py`
  - `GNN_REG/modelo2/train.py`
  - `GNN_REG/modelo2/inference.py`
  - `GNN_CLS/modelo2/train.py`
  - `GNN_CLS/modelo2/inference.py`
- 结构修正：
  - 3 层图主干现已改为原生 `GATv2Conv + 残差 + LayerNorm`
  - 跨激励 pooling、图级聚合、门控边回归头保持不变
- 本地兼容性处理：
  - `modelo2` 的训练、推理与模型文件已支持自动探测 `.vendor_torchpy311`
  - 因此当前机器上可直接运行 `python gnn/GNN_REG/modelo2/train.py ...`
- 简单验证：
  - `from torch_geometric.nn import GATv2Conv` 成功
  - `PhysicsInformedGNNRegressor` 最小前向通过，输出 `(2, 112)`
  - `PhysicsInformedGNNClassifier` 最小前向通过，输出 `(2, 3)`
  - `python gnn/GNN_REG/modelo2/train.py --help` 启动正常
## 2026-03-25 - GNN modelo2 OOM 修复（按 excitation 分块过 GAT）
- 云端报错现象：
  - `GNN_CLS/modelo2/train.py` 在 `GATv2Conv.edge_update` 阶段触发 CUDA OOM
  - 当时默认 batch 为 48，32 次激励被一次性展开成 `48*32` 张图并行计算
- 根因判断：
  - 问题不在数据或维度错误，而在原生 `GATv2Conv` 对大批量多图并行时的峰值显存占用过高
- 修复内容：
  - `PhysicsInformedGNNRegressor` / `PhysicsInformedGNNClassifier` 新增 `excitation_chunk_size`
  - 前向过程改为：
    - 先按 excitation 维分块
    - 每块单独构图并过 3 层 `GATv2Conv`
    - 最后沿 excitation 维拼回完整 32 次激励表示
- 同步调整：
  - `GNN_REG/modelo2/train.py` 默认 `batch_size=8`
  - `GNN_CLS/modelo2/train.py` 默认 `batch_size=8`
  - 训练/推理都新增 `--excitation-chunk-size`，默认 `4`
- 一致性验证：
  - 在本地随机输入上比较“整批编码”和“分块编码”结果
  - REG 输出最大差异约 `3.8e-06`
  - CLS 输出最大差异约 `1.5e-08`
  - 说明该修改只影响显存路径，不改变模型定义
- 当前推荐云端启动命令：
  - `python gnn/GNN_CLS/modelo2/train.py --dataset-tag 10mA --batch-size 8 --excitation-chunk-size 4`
  - `python gnn/GNN_REG/modelo2/train.py --dataset-tag 10mA --batch-size 8 --excitation-chunk-size 4`
- 若 10GB 左右显存仍紧张：
  - 进一步用 `--batch-size 4 --excitation-chunk-size 2`
  - 或临时减小 `--gat-heads 2`
## 2026-03-26 - GNN 新版本首轮结果（modelo2）
- `GNN_CLS/modelo2`：
  - 训练已可稳定运行，说明 `OOM` 修复方向有效；
  - `test_macro_f1=0.8871`
  - 验证最优阈值：`[0.05, 0.05, 0.27]`
  - 测试混淆矩阵：
    - `[[71,0,0,0],[0,297,3,0],[0,21,245,46],[0,1,69,247]]`
- 与旧主线 `GNN_CLS/modelo1@10mA` 对比：
  - `0.7855 -> 0.8871`
  - 提升约 `+0.1016`
- 结论：
  - `modelo2` 在分类线上是明确成功的；
  - `2/3` 混淆仍在，但已经比旧主线显著缓解；
  - 当前 `GNN_CLS` 新主线可以直接切到 `modelo2`。

- `GNN_REG/modelo2`：
  - `mae_all=0.8814`
  - `mae_changed=50.9660`
  - `best_count_threshold(val)=40.0`
  - `val_macro_f1=0.5816`
  - `avg(|dR|>40)=1.40`
  - `avg(mask_prob)=0.0073`
  - 派生数量混淆矩阵：
    - `[[71,0,0,0],[70,230,0,0],[16,124,169,3],[3,53,149,112]]`
- 与旧主线 `GNN_REG/modelo1` 对比：
  - 优势：
    - `mae_all` 极大改善；
    - 输出更稀疏；
    - `0/1/2` 变化样例已有较干净定位表现。
  - 劣势：
    - `mae_changed` 仍明显高于旧主线最好结果；
    - 阈值被压到 `40`，且对 `2/3` 样本普遍偏低估。
- 结论：
  - `modelo2` 的问题不是“没学到”，而是“学得太保守”；
  - 当前门控输出与 `lambda_mask_l1=0.05` 组合大概率压低了真实变化位的幅值和数量召回；
  - 下一步不建议回退结构，而应优先放松稀疏压力，先把 `mae_changed` 往 `43` 甚至 `41.x` 拉回。

- 本轮附带观察：
  - `GNN_CLS/modelo2` 验证/测试阶段出现 `np.exp` overflow warning；
  - 这是数值稳定性问题，不影响当前结果趋势判断，但建议后续改为更稳定的 sigmoid 计算方式。
## 2026-03-26 - GNN 下一版建立（modelo3）
- 基于 `modelo2` 首轮结果补充分析：
  - `GNN_CLS/modelo2` 已实现实质提升，残余误差同样主要集中在 `2/3` 边界；
  - `overflow encountered in exp` 暴露出 logits 过大、模型过度自信的问题，需要先做数值稳定性修正。
  - `GNN_REG/modelo2` 的主要问题不是“不会预测”，而是“太保守”：
    - 漏报明显；
    - 但一旦预测非零，数值幅值通常更稳。
- 本轮新建：
  - `GNN_CLS/modelo3`
  - `GNN_REG/modelo3`
- 目录说明：
  - 复制自 `modelo2`
  - 复制带来的旧 `outputs/cache/__pycache__` 已清理
- `GNN_CLS/modelo3` 本轮改动：
  - 验证/测试阶段的概率计算改为 `scipy.special.expit`
  - 默认 `weight_decay: 1e-4 -> 1e-3`
  - `inference.py` 现在默认优先抽样真实 `2/3` 变化样本
- `GNN_REG/modelo3` 本轮改动：
  - `lambda_mask_l1: 0.05 -> 0.01`
  - 目标是缓解过度保守的漏报倾向
  - `inference.py` 现在默认优先抽样真实 `2/3` 变化样本
  - 新增 `inference_full.py`
- `GNN_REG/modelo3/inference_full.py` 逻辑：
  - 先用 `GNN_CLS/modelo3` 预测 `K`
  - 再取 `GNN_REG/modelo3` 输出中绝对值最大的前 `K` 个电阻作为最终变化边
  - 其余边强制置 0
- 兼容性修复：
  - 若运行环境缺少 `scipy`，`modelo3` 的 `train.py / inference.py / inference_full.py` 现在会自动回退到数值稳定的本地 `expit` 实现，不再因 `ModuleNotFoundError` 中断。
- 当前意图：
  - 不回退 `Physics-Informed GNN` 结构
  - 优先验证“放松稀疏惩罚后，`mae_changed` 能否明显回升，同时保住低 `mae_all` 与低假阳性”

## 2026-03-26 - GNN 物理迭代新分支（model_tp1）
- 根据根目录 `新架构.md` 新建：
  - `gnn/GNN_REG/model_tp1/model/model.py`
  - `gnn/GNN_REG/model_tp1/train.py`
  - `gnn/GNN_REG/model_tp1/inference.py`
- 设计目标：
  - 更贴合电阻网络的物理结构，不再让 `GATv2` 主干承担全部逆问题学习；
  - 用共享边电导的 KCL 迭代先把边界电压往内部传播，再做边级回归。
- 与现有项目的贴合方式：
  - 不改主数据来源，仍直接使用 `data/training_data64Nodes_2.csv` 等现有 CSV；
  - 仍保留 `--data-path`、`--dataset-tag`、`cache/<tag>/...`、`outputs/<tag>/...` 口径；
  - 仍输出 `112` 维 `dR`，评价指标保持与其他 REG 主线一致。
- `model_tp1` 结构摘要：
  - 每个样本构造成一个 `PyG Data` 图，`x=(64, 32)`，边界 28 节点填电压差，内部节点为 0；
  - 112 条电阻边各有一个可学习电导，所有物理迭代层共享；
  - 更新公式采用稳定写法 `V <- V - alpha * L_g(V)`，并在每步后重新固定边界节点；
  - 解码器对每条边统计跨 32 次激励的 `Vu/Vv/|Vu-Vv|/avg` 的 `mean/max/std`，再拼接共享电导做门控回归。
- 本轮训练损失：
  - `MSE(pred, true_delta)`
  - `+ lambda_mask_l1 * mean(mask_prob)`
  - `+ lambda_kcl * mean(interior_residual^2)`
- 当前状态：
  - 已完成代码落地；
  - 本地已通过 `py_compile`、`--help` 与最小前向验证：
    - 前向输出 shape 为 `(2, 112)`
    - 默认 `alpha` 初值为 `0.1`
  - 尚未形成正式训练结果，后续需和 `modelo3` 做直接对比。

## 2026-03-26 - GNN_CLS/modelo3 首轮正式结果 + GNN_REG/modelo3 再放松
- `GNN_CLS/modelo3` 首轮结果：
  - `test_macro_f1=0.9075`
  - `val_best_thresholds=[0.05, 0.05, 0.26]`
  - 测试混淆矩阵：
    - `[[71,0,0,0],[0,299,1,0],[0,13,253,46],[0,1,54,262]]`
  - 判断：
    - 分类线继续有效提升；
    - `0/1` 基本稳定，剩余误差仍主要集中在 `2/3` 边界。
- `GNN_REG/modelo3` 本轮中途观察：
  - 到 epoch 45 时，`val_avg(|dR|>50)` 仍只有 `1.18` 左右，离主数据真实平均变化数 `1.86` 明显偏低；
  - `val_mask_mean≈0.006`，说明门控仍偏保守。
- 针对性修正：
  - `lambda_mask_l1: 0.01 -> 0.002`
  - `val_sparse_alpha: 0.20 -> 0.05`
  - 目的：
    - 继续放松对非零输出的压制；
    - 减少选模阶段对“激活数增加”的过强惩罚。

## 2026-03-27 - GNN 0326 训练结果记录
- `GNN_REG/modelo3`：
  - `mae_all=0.5573`
  - `mae_changed=25.0888`
  - `best_count_threshold(val)=45.0`
  - `val_macro_f1=0.8314`
  - `avg(|dR|>45)=1.75`
  - 测试派生数量混淆矩阵：
    - `[[71,0,0,0],[7,290,3,0],[0,48,249,15],[0,7,112,198]]`
- 结论：
  - `modelo3` 这次结果记录为当前 `GNN_REG` 线最佳 baseline；
  - 虽然它当前难以稳定复现，但它仍然给出了这条路线曾经达到过的精度目标。
- `GNN_REG/model_tp1`：
  - `mae_all=2.1173`
  - `mae_changed=106.4545`
  - `best_count_threshold(val)=40.0`
  - `val_macro_f1=0.2847`
  - `avg(|dR|>40)=0.93`
  - 测试派生数量混淆矩阵：
    - `[[73,0,0,0],[119,187,16,1],[70,175,44,8],[30,168,84,25]]`
- 失败判断：
  - 当前 `model_tp1` 的 `KCL residual` 约束过早、过强；
  - 模型更容易学到“整体趋近 0 扰动”这种保守捷径，而不是学真实变化位。
- 因此本轮已重写 `model_tp1`：
  - 从“共享静态电导 + 软 KCL”改为 `node-edge-global` GN 更新块；
  - 边状态支持根据 `vi/vj/eij/u` 做动态更新；
  - 节点更新显式使用物理电流聚合；
  - 全局状态显式编码总电流与全局统计；
  - 同时把 `lambda_mask_l1` 与 `lambda_kcl` 都进一步放轻，并加入 `kcl warmup`。

## 2026-03-27 - GNN model_tp1 修正版结果与当前取舍
- `GNN_REG/model_tp1` 修正版结果：
  - `mae_all=2.0972`
  - `mae_changed=101.9642`
  - `best_count_threshold(val)=40.0`
  - `val_macro_f1=0.3097`
  - `avg(|dR|>40)=1.06`
  - `avg(mask_prob)=0.0068`
  - `avg(kcl_residual)=0.021538`
  - 派生数量混淆矩阵：
    - `[[73,0,0,0],[58,203,46,16],[51,189,48,9],[20,201,68,18]]`
- 判断：
  - 修正版相比上一轮 `model_tp1` 仅有轻微改善，但总体仍显著落后于 `modelo3`
  - `avg(|dR|>40)` 仍只有 `1.06`，说明模型依旧被压在偏保守区间
  - 当前不再继续扩大 `model_tp1` 的实验投入，先暂停这条线
- 当前 GNN 主线取舍：
  - 回归主线仍是 `GNN_REG/modelo3`
  - 分类主线仍是 `GNN_CLS/modelo3`
  - 后续更优先推进与 `MLP_CLS/modelo7`、`MLP_REG/modelo7` 的异构整合推理
## 2026-03-27 - GNN_REG / modelo4a 与 modelo4b 新版本建立
- 新建目录：
  - `gnn/GNN_REG/modelo4a`
  - `gnn/GNN_REG/modelo4b`
- `modelo4a` 主要改动：
  - 引入 `resistor_embedding`
  - 新增 `top-K` 位置值损失
  - 新增小权重电压重投影物理损失 `lambda_physics=0.01`
  - 训练加入 `CosineAnnealingWarmRestarts(T_0=10, T_mult=2)`
- `modelo4b` 主要改动：
  - 与 `modelo4a` 共用同一主干与损失
  - 额外输出 `top3 / top4 / top5` 候选覆盖率
- 当前定位：
  - `modelo4a` 是下一轮 GNN 回归增强主线
  - `modelo4b` 是候选集口径的并行诊断分支
  - 二者都继续兼容旧 `10mA` 数据和新筛选版 `10mA` 数据

## 2026-03-27 - GNN_REG/modelo4a 首轮部分结果与路线撤回
- `modelo4a` 已观察到的关键指标：
  - 到 `epoch 25` 时，`val_mae_changed` 仍约 `71.0671`
  - `val_avg(|dR|>45)` 长时间停在约 `1.15`
  - `val_phys` 长时间维持在 `4600+`
- 当前判断：
  - KCL 约束本质上是极其严苛的非线性等式约束；
  - 将其作为 soft loss 与正在拟合 `MSE` 的网络同时优化，会导致明显梯度冲突；
  - 模型为了降低整体风险，会退回“少预测变化”的保守安全区。
- 相邻变化边现象补充：
  - 在强门控 / 强稀疏压力下，若相邻电阻 `A` 和 `B` 同时变化，优化器更容易选择“只保留一条边，把另一条边的幅值挤压到它身上”；
  - 这会表现为相邻变化被强行孤立，候选中只剩单边。
- 路线取舍修正：
  - `modelo4a / modelo4b` 当前均撤回，不再继续作为 GNN 回归主线
  - GNN 回归主线继续保持为 `GNN_REG/modelo3`
  - 若后续需要候选集口径，直接基于 `modelo3` 追加 `top3 / top4 / top5` 候选覆盖率即可，不再另起 `modelo4b` 主干
- 当前默认数据：
  - 未筛选 `10mA`：`data/training_data64Nodes_2.csv`
- `gnn/GNN_REG/modelo3b` 已建立：
  - 作为 `modelo3` 的独立推理候选集版本
  - 复用 `modelo3` 的已训练权重、cache、standardization 与 metrics
  - 在自身目录下单独输出 `candidate_metrics.json` 与 `candidate_samples.json`

## 2026-03-28 - GNN_REG/modelo3b 首轮候选集推理记录
- 运行命令：
  - `python gnn/GNN_REG/modelo3b/inference.py --source-model-dir gnn/GNN_REG/modelo3 --data-path data/training_data64Nodes_2.csv --dataset-tag training_data64Nodes_2`
- 结果：
  - `top3_candidate_cover=0.8300`
  - `top3_candidate_cover_changed_only=0.8170`
  - `top4_candidate_cover=0.8540`
  - `top4_candidate_cover_changed_only=0.8428`
  - `top5_candidate_cover=0.8610`
  - `top5_candidate_cover_changed_only=0.8504`

### 样例含义
- 样例 `9994`：
  - `Pred ids=[14, 19, 33]`
  - `True ids=[14, 19, 26]`
  - `Top3 hit=False`
  - `Top4 hit=True`
  - 说明这类错误更像“最后一条真实边只差一个名次”，属于候选集可补救型错误
- 样例 `9517`：
  - `Top3 hit=True`
  - 说明在一部分 3-change 样本中，`modelo3` 已经能把三条真实边都排进最前面，排序质量足够直接支持最终定位
- 样例 `7378` 与 `8916`：
  - `Top5 hit=False`
  - 说明仍有一批难样本里，漏掉的真实边根本没有进入前五，这不是简单扩候选集就能解决的问题

### 阶段分析
- `modelo3b` 证明了一个重要事实：
  - 当前 `GNN_REG/modelo3` 已经不只是“幅值回归强”，它的候选排序质量本身也相当可用
- `top3 -> top4` 的提升较明显：
  - 说明很多失败样本属于“差一名”的轻度排序误差
  - 若未来要做两阶段筛选或人工复核，`top4` 是当前较合理的候选规模
- `top4 -> top5` 的提升已经明显变小：
  - 说明剩余难样本中，问题不再只是阈值或 `K` 截断
  - 有一部分真实边在主模型排序里被压到了更后面
- 当前结论：
  - `modelo3b` 适合被视为 `modelo3` 的候选生成版与能力上限诊断版
  - 它支持后续探索“候选集 + 二阶段重排/物理筛选”的路线
  - 但它不意味着主回归任务已经靠扩展 topK 被解决

## 2026-03-28 - GNN_REG `o4` 系列建立与首轮结果
- 新建目录：
  - `gnn/GNN_REG/o4a`
  - `gnn/GNN_REG/o4a2`
  - `gnn/GNN_REG/o4b`
  - `gnn/GNN_REG/o4b2`
  - `gnn/GNN_REG/o4b3`

### 版本定位
- `o4a`：
  - 保留耦合输出 `pred = mask_prob * value`
  - 通过 `mask` 正偏置初始化和 `lambda_mask_l1 warmup` 抵抗早期门控塌陷
- `o4a2`：
  - 保留 `o4a` 的耦合输出结构
  - 在训练中新增显式 `BCE(mask_logits)` 监督门控
  - 同时把回归主损失改为 `SmoothL1`
- `o4b`：
  - 把 `mask` 和 `value` 解耦训练
  - 用 `BCEWithLogitsLoss(mask_logits, y_change)` 监督位置
  - 用 `masked MSE(value_pred, y_delta)` 监督真实变化边的幅值
- `o4b2`：
  - 针对 `o4b` 过报问题，降低 `mask/value` 激进度，并增加背景幅值抑制
- `o4b3`：
  - 基于 Loss 量级分析，把总损失重平衡为高权重 `BCE` + 低权重 `masked MSE`

### `o4b` 首轮不完整训练记录
- 用户运行：
  - `python gnn/GNN_REG/o4b/train.py --data-path data/training_data64Nodes_2.csv --dataset-tag o4b_0328_try1 --cache-path gnn/GNN_REG/o4b/cache/o4b_0328_try1/cache_dataset_reg_graphattn.npz --out-dir gnn/GNN_REG/o4b/outputs`
- 关键日志：
  - `epoch 1`：
    - `val_mae_all=17.3121`
    - `val_mae_changed=85.6182`
    - `val_avg(|dR|>50)=11.18`
    - `val_mask_mean=0.3417`
  - `epoch 10`：
    - `val_mae_changed=39.9807`
    - `val_avg(|dR|>50)=14.91`
  - `epoch 20`：
    - `val_mae_changed=36.1730`
    - `val_avg(|dR|>50)=11.93`
  - `epoch 35`：
    - `val_mae_changed=29.6492`
    - `val_avg(|dR|>50)=8.48`
    - `val_mask_mean=0.1140`
  - `epoch 45`：
    - `val_mae_changed=30.3320`
    - `val_avg(|dR|>50)=9.40`
    - `val_mask_mean=0.1170`

### 对 `o4b` 的阶段分析
- 正向结论：
  - 解耦训练确实有效打破了最近 `modelo3` 复现中的门控塌陷；
  - `val_mae_changed` 已经从 `68~72` 档降到了 `29~33` 档。
- 负向结论：
  - `val_avg(|dR|>50)` 高得离谱，说明模型严重过报；
  - `val_mask_mean` 维持在 `0.10+`，意味着 112 条边上平均有十几条边处在较高激活状态。
- 根因判断：
  - 虽然前向传播已经解耦，但原始 `o4b` 的总损失里 `masked MSE` 量级仍远大于 `BCE`
  - 优化器仍主要服务于回归头，导致 `mask` 头没有足够梯度去压制假阳性

### 当前记录口径
- `val_mask_mean`：
  - 表示验证集上 112 条边平均 `mask_prob` 的均值
  - 不是越大越好，也不是越小越好
  - 太小对应塌陷，太大对应过报
- 当前判断：
  - `o4a / o4a2 / o4b2 / o4b3` 都属于围绕这一问题继续排障的实验版本；
  - 截至目前，尚未出现新的稳定可复现主线，`o4` 系列仍处于诊断与修正阶段。

## 2026-03-28 - `o4a` 部分结果与 `o4a2` 建立
- 用户回传 `o4a` 的不完整日志：
  - `epoch 1`：
    - `val_mae_changed=86.2339`
    - `val_avg(|dR|>50)=1.32`
    - `val_mask_mean=0.0340`
    - `mask_l1=0.000000`
  - `epoch 5`：
    - `val_mae_changed=73.5238`
    - `val_avg(|dR|>50)=1.16`
    - `val_mask_mean=0.0069`
    - `mask_l1=0.000571`
  - `epoch 10`：
    - `val_mae_changed=71.5638`
    - `val_avg(|dR|>50)=1.17`
    - `val_mask_mean=0.0064`
    - `mask_l1=0.001286`
- 分析：
  - `o4a` 说明单靠 `mask` 偏置初始化和 `lambda_mask_l1 warmup`，不足以扭转当前 `modelo3` 的坏优化轨迹；
  - 门控虽然起步没有完全关闭，但很快又被耦合回归损失拖回了低激活区；
  - 换句话说，问题已经不只是“稀疏惩罚过早”，而是“门控头缺少足够直接的监督”。
- 因此新建 `o4a2`：
  - 架构继续保持耦合输出 `pred = mask_prob * value`
  - 训练改为：
    - `SmoothL1(pred, y_delta)`
    - `+ mask_bce_weight * BCEWithLogits(mask_logits, y_change)`
    - `+ warmup(lambda_mask_l1) * mean(mask_prob)`
  - 设计目标：
    - 比 `o4a` 更抗塌陷；
    - 比 `o4b` 更不容易进入极端过报。
## 2026-03-28 - 统一切换为纯 GNN 联合推理
- 新建统一入口：
  - `gnn/inference_gnn_cmei.py`
- 当前默认组合：
  - `gnn/GNN_CLS/modelo3`
  - `gnn/GNN_REG/modelo3`
- 方案选择依据：
  - 旧 `joint_inference` 中固定逻辑版整体优于动态融合版，动态权重未体现稳定增益；
  - `GNN_CLS` 与当前最佳 `MLP_CLS` 的分类差距已经很小；
  - `CLS + REG` 同时使用 GNN，更有利于统一图结构表示、拓扑归纳偏置和后续综合建模。
- 当前正式放弃：
  - `MLP_CLS + GNN_REG (+ MLP_REG)` 异构联合推理路线；
  - `joint_inference/` 目录及其两版旧推理入口。
- 新入口保留：
  - `Near-Miss` 轻量后处理；
  - 完整测试集 `CMEI` 评分；
  - 混淆矩阵、detail samples 和输出文件落盘。

## 2026-03-29 - `o4a2 / o4b2 / o4b3` 完整结果与阶段结论
- 已根据根目录 `GNN_REG训练记录.txt` 补充完整记录。

### `o4a2`
- 关键验证曲线：
  - `epoch 15`：`val_mae_changed=40.3115`，`val_avg(|dR|>50)=1.59`，`val_mask_mean=0.0325`
  - `epoch 45`：`val_mae_changed=27.9950`，`val_avg(|dR|>50)=1.69`，`val_mask_mean=0.0233`
  - `epoch 85`：`val_mae_changed=26.6961`，`val_avg(|dR|>50)=1.66`，`val_mask_mean=0.0205`
  - `epoch 120`：`val_mae_changed=24.8569`，`val_avg(|dR|>50)=1.71`，`val_mask_mean=0.0191`
- 最终测试：
  - `mae_all=0.4854`
  - `mae_changed=24.2925`
  - `best_count_threshold(val)=40.0`
  - `val_macro_f1=0.8683`
  - `avg(|dR|>40)=1.77`
  - `avg(mask_prob)=0.0192`
- 分析：
  - `o4a2` 是当前第一条真正落在合理稀疏区的 `o4` 新支线；
  - 它没有像 `o4a` 那样塌回 `1.17` 左右，也没有像 `o4b` 家族那样冲到 `2.5+` 甚至更高的过报区；
  - 说明“保留耦合输出 + 显式 `BCE(mask)` 监督 + `SmoothL1` 主回归”这组改动，成功把训练轨迹拉回了中间带；
  - 从单次结果看，它已经重新达到并略优于 `modelo3` 的历史最好上限。

### `o4b2`
- 最终测试：
  - `mae_all=2.5001`
  - `mae_changed=22.4532`
  - `best_count_threshold(val)=56.0`
  - `val_macro_f1=0.6904`
  - `avg(|dR|>56)=2.82`
  - `avg(mask_prob)=0.0377`
- 分析：
  - `mae_changed` 很好，说明解耦后的 value 头确实更会拟合真实变化边的幅值；
  - 但 `mae_all`、`val_macro_f1` 和平均预测变化数都明显偏差，说明它仍在大量误报；
  - 因此它更像“高召回候选生成器”，不适合直接作为最终主回归器。

### `o4b3`
- 最终测试：
  - `mae_all=2.2490`
  - `mae_changed=21.2190`
  - `best_count_threshold(val)=67.0`
  - `val_macro_f1=0.6990`
  - `avg(|dR|>67)=2.27`
  - `avg(mask_prob)=0.0307`
- 分析：
  - 相比 `o4b2`，`o4b3` 通过提高 `BCE` 权重，确实把过报压下去了一截；
  - 但它仍明显高于理想稀疏区，阈值也被抬到 `67.0`，说明底层分布还是过活跃；
  - 这条线继续证明“完全解耦”能救 `mae_changed`，但目前还没学会稳定地把背景压回 0。

### 当前总判断
- `o4a2`：
  - 当前最像正式主线；
  - 优点是稀疏度、`mae_all`、`mae_changed`、派生数量四项同时平衡。
- `o4b2 / o4b3`：
  - 当前更适合作为候选生成/高召回诊断支线；
  - 它们对“真实变化边幅值”学得很强，但对“不要乱报”还不够强。
- 下一步建议：
  - 先围绕 `o4a2` 做 fresh cache / fresh outdir / 多 seed 复验；
  - 在 `o4a2` 稳住之前，不再继续平行扩展过多新 `REG` 结构。

## 2026-03-29 - GNN 通用可视化脚本落地
- 新建：
  - `gnn/visualize_gnn_results.py`
- 设计目标：
  - 用统一视觉风格展示 `CLS / REG / 候选集 / 联合推理`
  - 兼顾“模型演进对比”和“代表样例展示”
- 已验证的输入：
  - `gnn/outputs/gnn_cmei/training_data64Nodes_2`
  - `gnn/GNN_CLS/modelo2/outputs/training_data64Nodes_2`
  - `gnn/GNN_CLS/modelo3/outputs/training_data64Nodes_2`
- 当前输出：
  - `overview.png`
  - `samples.png`
  - `comparison.png`
  - `manifest.json`
- 当前默认输出目录：
  - `gnn/outputs/visualizations`
- 工具定位：
  - 以后每次版本迭代，不再只贴纯文本分数；
  - 可以直接用同一套图板展示 `score`、混淆矩阵、样例拓扑和版本演进。
## 2026-03-30 - `o4a2` 四 seed 复验完成，正式升级为当前最佳 `GNN_REG`
- 根目录 `0330训练日志.txt` 已整理进正式记录。
- 4 个 seed 结果如下：
  - `o4a2_seed20260325`
    - `mae_all=0.4806`
    - `mae_changed=25.9977`
    - `val_count_macro_f1=0.8508`
    - `avg_abs_gt_threshold=1.713`
    - `best_epoch=100`
  - `o4a2_seed20260326`
    - `mae_all=0.5013`
    - `mae_changed=25.3102`
    - `val_count_macro_f1=0.8655`
    - `avg_abs_gt_threshold=1.722`
    - `best_epoch=110`
  - `o4a2_seed20260327`
    - `mae_all=0.5387`
    - `mae_changed=27.5191`
    - `val_count_macro_f1=0.8535`
    - `avg_abs_gt_threshold=1.704`
    - `best_epoch=105`
  - `o4a2_seed20260328`
    - `mae_all=0.5654`
    - `mae_changed=26.3538`
    - `val_count_macro_f1=0.8558`
    - `avg_abs_gt_threshold=1.768`
    - `best_epoch=110`
- 汇总统计：
  - `mae_all mean=0.5215, std=0.0328`
  - `mae_changed mean=26.2952, std=0.8000`
  - `val_count_macro_f1 mean=0.8564, std=0.0056`
  - `avg_abs_gt_threshold mean=1.7268, std=0.0247`
- 当前阶段结论：
  - `o4a2` 已经不再是“偶然跑出一次好结果”的状态；
  - 它是第一条同时满足“结果强”和“多 seed 可复验”的 `GNN_REG` 新主线；
  - 因此从今天开始，`GNN_REG/o4a2` 正式取代 `GNN_REG/modelo3`，成为当前最佳模型与后续优化母版。

### 当前最好单 checkpoint 与当前最好复验锚点
- 当前最好单 checkpoint 仍是：
  - `gnn/GNN_REG/o4a2/outputs/training_data64Nodes_2/`
  - 指标：
    - `mae_all=0.4854`
    - `mae_changed=24.2925`
    - `val_count_macro_f1=0.8683`
    - `avg_abs_gt_threshold=1.771`
- 当前最好复验锚点建议使用：
  - `gnn/GNN_REG/o4a2/outputs/o4a2_seed20260326/`
  - 原因：
    - 四个 seed 里它在 `mae_changed`、`count_f1` 与整体平衡上最稳；
    - 适合作为以后做增量修改时的对照参考。

### 对当前问题的判断
- 轻微过拟合：
  - 训练末段确实存在 train loss 继续下降、验证指标轻微震荡的现象；
  - 但幅度不大，而且训练脚本保存的是验证最优权重，因此当前不是阻碍上线使用的问题。
- 保守预测倾向：
  - 仍然存在；
  - 主要体现为高变化数样本中偏向少报，说明 `o4a2` 虽已进入正确区间，但还没有彻底摆脱“稳妥少报”的偏置。
- `mask_l1=0.002000` 后期不再变化：
  - 这是当前 `o4a2` 设计使然；
  - warmup 结束后，`mask_l1` 固定为常数，不会继续增长；
  - 这意味着它后期更像一个恒定弱稀疏先验，而不是持续增强的调度器；
  - 从这次训练现象看，它没有完全失效，但也不是后期主要驱动因素，后续优化应更多瞄准“保守偏置修正”和“高变化数召回”。

### Outputs 是否已可直接复用
- 用户下载回本地的 `o4a2` outputs 已检查通过。
- 当前这些目录都具备完整复用条件：
  - `model_last.pt`
  - `metrics.json`
  - `standardization.npz`
  - `confusion_matrix_count_test.txt`
- 结论：
  - 它们已经可以直接拿来做：
    - 推理复现
  - 联合推理接入
  - 可视化生成
  - 结果归档/备份
  - 不需要再重新训练才能“使用”。

### 统一 GNN 推理链路冒烟结果
- 已使用当前默认组合完成一次真实推理验证：
  - `GNN_CLS/modelo3 + GNN_REG/o4a2`
- 结果：
  - `CMEI=93.73`
  - `num_accuracy=0.8850`
  - `macro_f1=0.9075`
  - `id_recall=0.9248`
  - `mse_all_edges=49.7686`
- 说明：
  - `o4a2` 现在不只是“训练阶段表现好”的候选模型；
  - 它已经能直接接入当前 `gnn/inference_gnn_cmei.py` 的最终联合推理流程，并稳定产出完整 `CMEI` 结果。

## 2026-03-30 - `o5a / o5b / GNN_FULL/Mv1` 建立
### 指标口径确认
- 已核对 `GNN_REG/modelo3` 与 `GNN_REG/o4a2` 的实现：
  - `val_mae_all` 与 `val_mae_changed` 的定义没有被改小；
  - 两者都还是基于 `abs(pred - true)` 的 MAE；
  - 变化的是训练损失和训练轨迹，不是这两个 epoch 指标的定义。

### `o5a`
- 路径：
  - `gnn/GNN_REG/o5a`
- 设计定位：
  - 只针对 `o4a2` 现有的保守预测倾向做小修；
  - 不改主骨架，不改输入结构。
- 具体做法：
  - 在原 `o4a2` loss 上新增真实变化边的幅值下限约束；
  - 形式上等价于：当真实变化边的 `|pred_delta|` 低于给定 margin 时，额外施加 hinge 惩罚。
- 预期作用：
  - 把“找到了但幅值太小，最后在 count threshold 上被算成没报”的样本往上拉；
  - 尤其面向 `3 -> 2` 这类保守少报场景。

### `o5b`
- 路径：
  - `gnn/GNN_REG/o5b`
- 设计来源：
  - 根目录 `GNN_REG优化.txt`
- 具体做法：
  - 在 regressor 的边解码阶段引入 `edge embedding`
  - 每条边在 `edge_feat = [hu, hv, |hu-hv|, edge_emb]` 中携带绝对位置先验
  - 稀疏项改成 relaxed top-k penalty：
    - 对最可疑的前 k 条边放松惩罚
    - 背景边继续承受稀疏压力
- 预期作用：
  - 补足原模型缺少绝对位置先验的问题；
  - 降低全局 L1 对相邻真实变化边的过度抑制。

### `GNN_FULL/Mv1`
- 路径：
  - `gnn/GNN_FULL/Mv1`
- 当前结构：
  - `model/model_cls.py`
  - `model/model_reg.py`
  - `train_cls.py`
  - `train_reg.py`
  - `inference.py`
  - `outputs/`
- 设计定位：
  - 这是第一版把当前 `GNN_CLS` 与 `GNN_REG` 最优主线收拢到同一个实验容器中的融合目录；
  - 它不是端到端单模型，而是统一管理“两个训练入口 + 一个完整联合推理入口”。
- 已完成本地冒烟：
  - `train_cls.py --help` 通过
  - `train_reg.py --help` 通过
  - `inference.py --help` 通过
  - 借用现有权重做了一次完整推理：
    - `GNN_CLS/modelo3 + GNN_REG/o4a2`
    - `CMEI=93.73`
    - `num_accuracy=0.8850`
    - `macro_f1=0.9075`
    - `id_recall=0.9248`
    - `mse_all_edges=49.7686`
- 结论：
  - `Mv1` 的壳层和默认路径已经可运行；
  - 后续只需把 `Mv1` 自己训练出来的 `cls/reg` 权重接入即可形成正式版联合目录。

## 2026-03-31 - `o5a / o5b` 首轮完整结果与联合推理验证
- 根目录 `0331训练记录.txt` 已完成吸收，原始训练文本可删除。

### `o5a`
- 最终测试：
  - `mae_all=0.7770`
  - `mae_changed=38.8351`
  - `best_count_threshold(val)=40.0`
  - `val_macro_f1=0.8524`
  - `avg(|dR|>40)=1.77`
  - `avg(mask_prob)=0.0200`
  - `best_epoch=115`
- 阶段判断：
  - 这条线没有修正 `o4a2` 的保守少报核心问题；
  - 新增的 true-edge margin 更像是在硬拉真实边幅值，导致回归整体校准变差；
  - 因此 `o5a` 作为失败尝试记录保留，不再继续投入。

### `o5b`
- 最终测试：
  - `mae_all=0.5119`
  - `mae_changed=25.4532`
  - `best_count_threshold(val)=40.0`
  - `val_macro_f1=0.8564`
  - `avg(|dR|>40)=1.74`
  - `avg(mask_prob)=0.0207`
  - `best_epoch=120`
- 阶段判断：
  - `o5b` 明显强于 `o5a`；
  - 它在不破坏合理稀疏度的前提下，把 `mae_all / mae_changed` 拉回到了接近 `o4a2` 主线的水平；
  - 说明 `edge embedding + relaxed top-k sparsity` 确实值得继续做多 seed 验证。
- 与 `o4a2` 的关系：
  - 对比当前最佳单 checkpoint `o4a2`，`o5b` 仍略弱；
  - 对比 `o4a2` 的 4-seed 均值，`o5b` 已经接近甚至略优；
  - 因此它目前的正确定位不是“立刻替代主线”，而是“首要复验候选”。

### 统一 GNN 联合推理验证
- 已完成：
  - `GNN_CLS/modelo3 + GNN_REG/o5b`
- 输出目录：
  - `gnn/outputs/gnn_cmei_o5b_eval/o5b_10mA/`
- 结果：
  - `CMEI=93.20`
  - `num_accuracy=0.8850`
  - `macro_f1=0.9075`
  - `id_recall=0.9120`
  - `mse_all_edges=56.4333`
- 对比当前默认组合 `modelo3 + o4a2`：
  - 分类侧几乎持平；
  - 但 `id_recall` 更低，`mse_all_edges` 也更差；
  - 说明 `o5b` 虽然是个有希望的回归分支，但还没有在最终联合推理层面赢过 `o4a2`。

### 当前结论
- 正式主线不变：
  - `GNN_CLS/modelo3`
  - `GNN_REG/o4a2`
- 下一步优先级：
  - 停止沿 `o5a` 方向继续试错；
  - 对 `o5b` 做 fresh-cache / fresh-outdir / 多 seed 复验；
  - 只有当 `o5b` 的多 seed 均值稳定优于 `o4a2`，才考虑升级默认回归器。

## 2026-03-31 - `GNN_FULL/Mv1/inference_v2` 规则实验
- 新增：
  - `gnn/GNN_FULL/Mv1/inference_v2.py`
- 规则改动：
  - near-miss 新增保护墙：若相邻簇中的较弱边 `reg_prob > 0.85` 或 `|ΔR| > 45Ω`，则不允许替换；
  - top-k 先做 `35Ω` deadband 截断；
  - 当 `k < 3` 且下一条边 `|ΔR| >= 45Ω` 时，允许 REG 抢救回一条高置信边。
- 目标原本是：
  - 减少 `o5b` 对相邻真实损坏的误杀；
  - 用物理先验修正 CLS 对少量样本的数量误判。
- 实测组合：
  - `GNN_CLS/modelo3 + GNN_REG/o5b`
- 实测结果：
  - `CMEI=91.38`
  - `raw_cls_num_accuracy=0.8850`
  - `num_accuracy=0.8400`
  - `macro_f1=0.8587`
  - `id_recall=0.9019`
  - `mse_all_edges=56.7826`
- 结果解读：
  - 这版并没有改善 `o5b` 的最终联合表现；
  - `raw_cls_num_accuracy` 仍是 `0.8850`，说明 CLS 本身没坏；
  - 真正变差的是“后处理后的最终计数”，这说明当前物理死区 + rescue 规则太强，已经开始过度篡改原本合理的 top-k 结果。
- 结论：
  - 保留 `inference_v2.py` 作为实验分支；
  - 当前默认联合推理仍保持旧入口；
  - 若后续继续优化联合推理，应优先保留“near-miss 保护墙”，而不要直接保留当前这版整套 deadband/rescue 组合。

## 2026-03-31 - `o5b1` 与简化版 `inference_v2`
### `o5b1`
- 新建：
  - `gnn/GNN_REG/o5b1`
- 与 `o5b` 的区别：
  - 唯一改动是 `mask_bce_weight=20.0`
  - 其余结构、位置编码、relaxed top-k 稀疏约束全部保持不变
- 当前定位：
  - 这是针对 `o5b` 的最小超参试探版；
  - 先不动结构，只测试更温和的 `mask` 监督是否更有利于减少保守偏置。

### 简化版 `inference_v2`
- 已修改：
  - 删除 `35Ω deadband`
  - 删除 `45Ω rescue`
  - 只保留“高置信相邻边保护”逻辑
- 实测组合：
  - `GNN_CLS/modelo3 + GNN_REG/o5b`
- 输出目录：
  - `gnn/GNN_FULL/Mv1/outputs/inference_v2_guard_only_o5b_eval/o5b_10mA/`
- 结果：
  - `CMEI=93.17`
  - `raw_cls_num_accuracy=0.8850`
  - `num_accuracy=0.8850`
  - `macro_f1=0.9075`
  - `id_recall=0.9115`
  - `mse_all_edges=57.6391`
- 阶段判断：
  - 这说明“只保留高置信相邻边保护”后，性能已经从 `91.38` 恢复到了接近原始水平；
  - 但它仍然没有优于旧版统一推理下对同一组模型得到的 `93.20`
  - 因此当前默认联合推理入口不切换，`inference_v2.py` 继续只作为实验版。

## 2026-03-31 - `GNN_FULL/Mv1` 路径兼容修复与本地复跑
- 暴露问题：
  - 用户已按 `Mv1训练记录.txt` 训练完成 `Mv1` 的 `cls/reg` 两个模型；
  - 但从根目录执行
    - `python gnn/GNN_FULL/Mv1/inference.py --cls-out-dir gnn/GNN_FULL/Mv1/outputs/cls --reg-out-dir gnn/GNN_FULL/Mv1/outputs/reg ...`
    时，脚本会把相对路径再次拼到脚本目录下，形成重复的 `gnn/GNN_FULL/Mv1/gnn/GNN_FULL/Mv1/...`
  - 此外，本地只下载了 `outputs/` 时，`Mv1/cache/...` 缺失也会直接导致推理失败
- 本次修复：
  - `gnn/GNN_FULL/Mv1/inference.py`
  - `gnn/GNN_FULL/Mv1/inference_v2.py`
  - 统一采用“绝对路径 / 当前工作目录相对路径 / 项目根相对路径”三层候选解析
  - 新增 `ensure_cache_npz(...)`：
    - 若 `cache_dataset_cls_graphattn.npz` 不存在，则调用 `train_cls.py` 中的 `build_dataset(...)`
    - 若 `cache_dataset_reg_graphattn.npz` 不存在，则调用 `train_reg.py` 中的 `build_dataset(...)`
    - 重建后再继续读取 `standardization.npz` 与模型权重
- 本地验证结果：
  - 首次执行已自动补建：
    - `gnn/GNN_FULL/Mv1/cache/training_data64Nodes_2/cache_dataset_cls_graphattn.npz`
    - `gnn/GNN_FULL/Mv1/cache/training_data64Nodes_2/cache_dataset_reg_graphattn.npz`
  - 再次执行原命令后已正常输出到：
    - `gnn/GNN_FULL/Mv1/outputs/inference/training_data64Nodes_2/`
  - 联合结果：
    - `CMEI=93.11`
    - `num_accuracy=0.8740`
    - `macro_f1=0.8975`
    - `id_recall=0.9173`
    - `mse_all_edges=53.3761`
- 对 `Mv1` 当前状态的判断：
  - 路径与 cache 依赖问题已经不再阻塞本地复现；
  - `Mv1` 现在是可独立复跑的正式联合实验目录；
  - 但其当前分数仍弱于默认 `modelo3 + o4a2` 的 `CMEI=93.73`，因此不替换当前默认联合主线。

## 2026-03-31 - `Noise_test` 步骤 A（20dB zero-shot noise）
- 执行配置：
  - 数据：`training_data64Nodes_2`（当前默认 `10mA`）
  - 噪声：standardized voltage 通道高斯白噪声，`noise_std=0.1`
  - 作用位置：仅测试集
  - 随机种子：`20260331`
- `GNN_CLS/modelo3`
  - clean：`test_macro_f1=0.9075`
  - noise：`test_macro_f1=0.1203`
  - 混淆现象：
    - `noise_eval.json` 显示 1000 个测试样本几乎全部被判成类别 `3`
- `GNN_REG/o4a2`
  - clean：
    - `mae_all=0.4854`
    - `mae_changed=24.2925`
    - `count_macro_f1=0.8683`
    - `avg(|dR|>40)=1.771`
    - `avg(mask_prob)=0.0192`
  - noise：
    - `mae_all=23.2065`
    - `mae_changed=70.3696`
    - `count_macro_f1=0.1203`
    - `avg(|dR|>40)=17.668`
    - `avg(mask_prob)=0.1927`
  - 现象解释：
    - 噪声下 `mask_prob` 整体抬升近一个数量级；
    - 派生计数同样几乎完全塌到 `3`
- 统一联合推理 `GNN_CLS/modelo3 + GNN_REG/o4a2`
  - clean：
    - `CMEI=93.73`
    - `num_accuracy=0.8850`
    - `macro_f1=0.9075`
    - `id_recall=0.9248`
    - `mse_all_edges=49.7686`
  - noise：
    - 输出目录：`gnn/outputs/gnn_cmei_noise20db/training_data64Nodes_2/`
    - `CMEI=41.79`
    - `num_accuracy=0.3170`
    - `macro_f1=0.1203`
    - `id_recall=0.2608`
    - `mse_all_edges=1735.1177`
- 阶段结论：
  - 步骤 A 已经给出明确结论：当前最佳 clean GNN 链条在这组 `20dB` 噪声设定下没有鲁棒性余量；
  - 当前最值得推进的是 `Noise_test` 的步骤 B，即训练期噪声增强，而不是继续微调 near-miss 或 count 后处理；
  - 这批结果可直接作为后续鲁棒性分析的 zero-shot baseline。

## 2026-03-31 - `GNN_NOISE` 首版落地
- 已新建：
  - `gnn/GNN_NOISE/CLS_modelo3_ft`
  - `gnn/GNN_NOISE/REG_o4a2_ft`
  - `gnn/GNN_NOISE/README.md`
  - `gnn/GNN_NOISE/Log.md`
- 设计目标：
  - 正式执行 `Noise_test` 的策略 B
  - 不再围绕 clean-only 模型做 zero-shot 猜测
  - 而是直接通过数据增强让模型见到不同强度的边界噪声

### `CLS_modelo3_ft`
- 来源：
  - 复制自 `gnn/GNN_CLS/modelo3`
- 当前新增入口：
  - `--pretrained-model-path`
  - `--add-noise / --no-add-noise`
  - `--noise-schedule {random,fixed}`
  - `--noise-mode {gaussian,uniform}`
  - `--noise-std-max`
  - `--fixed-noise-std`
  - `--noise-scope {boundary,all}`
- 默认微调配置：
  - `pretrained-model-path=../../GNN_CLS/modelo3/outputs/training_data64Nodes_2/model_last.pt`
  - `epochs=30`
  - `lr=5e-5`
  - `patience=10`
- 噪声增强逻辑：
  - 仅训练集 `Dataset.__getitem__` 注入
  - 每次采样重新生成 `noise_std = 0.1 * rand()`
  - 噪声作用在边界节点 `voltage_delta` 通道

### `REG_o4a2_ft`
- 来源：
  - 复制自 `gnn/GNN_REG/o4a2`
- 当前新增入口：
  - `--pretrained-model-path`
  - `--add-noise / --no-add-noise`
  - `--noise-schedule {random,fixed}`
  - `--noise-mode {gaussian,uniform}`
  - `--noise-std-max`
  - `--fixed-noise-std`
  - `--noise-scope {boundary,all}`
- 默认微调配置：
  - `pretrained-model-path=../../GNN_REG/o4a2/outputs/training_data64Nodes_2/model_last.pt`
  - `epochs=30`
  - `lr=5e-5`
  - `patience=10`
- 保留不变的核心逻辑：
  - `o4a2` 的 gated regression 主干
  - `SmoothL1(pred, y_delta, beta=25)`
  - `mask BCE`
  - `mask L1 warmup`

### 当前判断
- 这次改动的重点不是“发明一个新结构”
- 而是把当前已经验证过有效的 clean 主线，转成可执行的 noisy fine-tuning 容器
- 下一步真正关键的实验，不是再看 clean 测试集是否微升，而是：
  - 带噪微调后，`20dB` 下的 `macro_f1 / id_recall / mse / CMEI` 能恢复多少

## 2026-04-01 - `Noise_test` 原始步骤 B 收编
- 已确认当前默认 `GNN_NOISE` 与根目录原 `Noise_test` 步骤 B 不完全等价。
- 为保留原定义，又不继续在根目录散放说明文件，现已新增：
  - `gnn/GNN_NOISE/原始步骤B_fixed20dB.md`
- 代码层面已补齐可表达原始步骤 B 的参数组合：
  - `noise_schedule=fixed`
  - `fixed_noise_std=0.1`
  - `noise_scope=all`

## 2026-04-01 - `o5b1` 训练记录复盘
- 已读取根目录 `o5b1训练记录.txt` 并核对 `metrics.json`。
- `o5b1` 结果：
  - `mae_all=0.5097`
  - `mae_changed=25.5355`
  - `val_count_macro_f1=0.8422`
  - `avg(|dR|>40)=1.73`
  - `avg(mask_prob)=0.0213`
- 对比 `o5b`：
  - `mae_all` 几乎持平
  - `mae_changed` 轻微变差
  - `val_count_macro_f1` 有所下降
  - 说明 `mask_bce_weight: 25 -> 20` 并没有释放出额外性能
- 但同时也说明：
  - `o5b` 系列并不是靠更大的 `mask BCE` 硬压着才稳定
  - 放松后没有出现明显假阳性膨胀，这支持“结构与归纳偏置本身已较稳”的判断
- 当前结论：
  - `o5b1` 不足以替代 `o5b`
  - `o5b1` 更不足以替代当前默认 `o4a2`
  - 现阶段更合理的战略不是继续微调 clean-only loss，而是把主力转向带噪训练与鲁棒性验证

## 2026-04-01 - 根目录草稿文件清理
- 已迁移：
  - `首轮20dB噪声诊断记录.md -> gnn/GNN_NOISE/首轮20dB噪声诊断记录.md`
- 已删除：
  - `o5b1训练记录.txt`
  - `Mv1训练记录.txt`
  - `Noise_test.txt`
- 依据：
  - `o5b1` 训练记录已被正式吸收；
  - `Mv1` 训练与路径修复结论已被正式吸收；
  - `Noise_test` 步骤 A/B 已被拆分并沉淀到 `GNN_NOISE` 目录下。

## 2026-04-01 - `0401训练记录` 复盘
- clean 主线复训：
  - `modelo3`: `test_macro_f1=0.9027`
  - `o4a2`: `mae_all=0.4679`，`mae_changed=23.5724`，`val_count_macro_f1=0.8628`
  - `modelo3 + o4a2`: `CMEI=93.53`
- 结论：
  - clean 表现继续稳定
  - 但未超过历史最好联合结果 `93.73`
- `noiseft_rand_boundary_20260401`：
  - `REG_o4a2_ft` 已完成，结果为 `mae_all=0.5900`，`mae_changed=27.5311`，`count_macro_f1=0.8140`
  - `CLS_modelo3_ft` 对应输出目录仅有 `standardization.npz`
  - 因此这条推荐增强版分支在本次云端记录中并未完整闭环
- `noiseft_fixed20db_all_20260401`：
  - `CLS_modelo3_ft` 同 tag 连续训练两次，最终保留结果为 `test_macro_f1=0.7275`
  - `REG_o4a2_ft` 最终结果为 `mae_all=0.9201`，`mae_changed=48.8259`，`count_macro_f1=0.6178`
  - clean 联合 `CMEI=83.39`
  - noisy 单模型结果为：
    - `CLS macro_f1=0.7121`
    - `REG mae_all=1.1800`
    - `REG mae_changed=54.2884`
    - `REG count_macro_f1=0.5829`
- 综合判断：
  - fixed-all 强增强对噪声恢复有效；
  - 但 clean 性能牺牲过大；
  - 后续默认带噪路线仍应优先补完整 `rand_boundary` 成对实验

## 2026-04-01 - 可视化图集重做与运行口径确认
- 已删除旧可视化材料：
  - `midterm_assets/20260401_data_figures`
  - `中期汇报_数据可视化说明.md`
  - `tools/generate_midterm_figures.py`
- 已生成新的最终图集：
  - `midterm_assets/20260401_visuals/01_topology_boundary_nodes.svg`
  - `midterm_assets/20260401_visuals/02_dataset_composition.svg`
  - `midterm_assets/20260401_visuals/03_changed_edge_frequency.svg`
  - `midterm_assets/20260401_visuals/04_boundary_response_heatmaps.svg`
- 已再次确认：
  - 当前本地联合推理脚本是 `gnn/inference_gnn_cmei.py`
  - 若云端版本仍报 `--noise-std/--noise-seed` 参数错误，则应优先同步该文件
  - 当前唯一必须补训的是 `CLS_modelo3_ft` 的 `training_data64Nodes_2_noiseft_rand_boundary_20260401`

## 2026-04-01 - `0401补充训练` 吸收与 outputs 完整性检查
- 已吸收根目录 `0401补充训练.txt`
- 推荐增强版 `rand_boundary` 新增结果：
  - `CLS clean test_macro_f1=0.8750`
  - `CLS noisy test_macro_f1=0.7780`
  - noisy joint `CMEI=82.56`
  - `num_accuracy=0.7360`
  - `id_recall=0.7579`
  - `mse_all_edges=154.4499`
- 本地输出目录检查：
  - `CLS rand_boundary` 目录内容齐全
  - `REG rand_boundary` 目录缺 `noise_eval.json`
  - `gnn/outputs` 尚未看到 `gnn_cmei_noiseft_rand_boundary_20db_20260401` 与 `gnn_cmei_noiseft_fixed20db_all_*`
- 判断：
  - 训练本身已完成
  - 但本地下载并非全量镜像，联合结果目录仍需补同步

## 2026-04-01 - 记录清理与鲁棒性画图脚本
- 已删除 `0401补充训练.txt`
- 已新增 `gnn/GNN_NOISE/plot_noise_robustness.py`
- 用途：
  - 将多组 `json` 指标文件合并到一张 `svg` 鲁棒性曲线图中
  - 方便直接比较 `20/30/40dB` 下的不同训练策略

## 2026-04-01 - `modelo3` 两阶段阈值搜索脚本落地
- 已新增：
  - `gnn/GNN_CLS/modelo3/two_stage_threshold_search.py`
- 已本地验证 `rand_boundary` 分类器：
  - 原阈值：`[0.05, 0.17, 0.37]`
  - 细化后：`[0.05, 0.164, 0.368]`
  - `val_macro_f1` 持平
  - `test_macro_f1` 基本不变且略有回落
- 判断：
  - 当前粗搜步长不是主瓶颈
  - 后续优化优先级仍应放在鲁棒特征学习与噪声建模

## 2026-04-02 - `0402补充日志` 吸收
- 已确认本地当前 joint 输出目录齐全：
  - `gnn_cmei_noiseft_rand_boundary_clean_20260401`
  - `gnn_cmei_noiseft_rand_boundary_20db_20260401`
  - `gnn_cmei_noiseft_fixed20db_all_clean_20260401`
  - `gnn_cmei_noiseft_fixed20db_all_20db_20260401`
- 新增结果：
  - `rand_boundary` clean joint `CMEI=91.01`
  - `rand_boundary` 20dB joint `CMEI=82.56`
  - `fixed20db_all` clean joint `CMEI=83.39`
  - `fixed20db_all` 20dB joint `CMEI=81.79`
  - `REG rand_boundary` 20dB 单模型 `mae_all=1.2692`，`mae_changed=54.1729`
- 当前判断：
  - 20dB 的补推理工作已完成
  - 剩余仅建议继续扩展 `30/40dB`

## 2026-04-02 - `rand_boundary` 最终鲁棒性图完成
- 已吸收 `0402大范围噪声训练.txt`
- 已生成：
  - `gnn/GNN_NOISE/rand_boundary_robustness_curve.svg`
- 已新增：
  - `gnn/GNN_NOISE/plot_rand_boundary_robustness.py`
- joint 结果总结：
  - clean `CMEI=91.01`
  - `40dB CMEI=90.83`
  - `30dB CMEI=89.62`
  - `20dB CMEI=82.56`
- 最终判断：
  - `rand_boundary` 已经成为当前唯一应继续推进的主线
  - `fixed20db_all` 不再作为正式候选
  - `0402补充日志.txt` 与 `0402大范围噪声训练.txt` 已删除
## 2026-04-02 - 联合推理目录收敛到 `GNN_CMEI_INFERENCE`
- 已新建：
  - `gnn/GNN_CMEI_INFERENCE`
- 已迁移：
  - `gnn/inference_gnn_cmei.py -> gnn/GNN_CMEI_INFERENCE/inference_gnn_cmei.py`
  - `gnn/outputs -> gnn/GNN_CMEI_INFERENCE/outputs`
- 已保留兼容入口：
  - `gnn/inference_gnn_cmei.py`
- 已新增：
  - `gnn/GNN_CMEI_INFERENCE/inference_gnn_cmei_v2.py`
- `v2` 当前仅吸收推理层可直接落地项：
  - `near-miss` 高置信保护
  - `REG` 动态 `K` / 数量仲裁
- 已删除：
  - `GNN_联合优化.txt`
- 已补跑当前正式主线 `rand_boundary` 的 `v2` 验证：
  - `v2(full arbitration)` clean `CMEI=90.08`
  - `v2(full arbitration)` 20dB `CMEI=79.40`
  - `v2(guard_only)` clean `CMEI=90.85`
  - `v2(guard_only)` 20dB `CMEI=82.40`
  - 对比 `v1`：
    - clean `91.01`
    - 20dB `82.56`
- 结论：
  - `v2` 尚未带来收益
  - `REG` 数量仲裁在当前主线上偏激进
  - 正式入口仍保持 `gnn/GNN_CMEI_INFERENCE/inference_gnn_cmei.py`

## 2026-04-02 - `GNN_NOISE v2` 分支建立
- 已新建：
  - `gnn/GNN_NOISE/CLS_modelo3_ft_v2`
  - `gnn/GNN_NOISE/REG_o4a2_ft_v2`
- 设计口径：
  - 单模型 outputs 继续保留在 `GNN_NOISE/*_v2/outputs`
  - joint outputs 保留在 `GNN_CMEI_INFERENCE/outputs`
- `v2` 训练改动：
  - `noise_schedule=curriculum`
  - `noise_mode=structured`
  - `noise_scope=boundary`
  - warm start from `v1 rand_boundary`
- 已通过：
  - `py_compile`
  - `--help`

## 2026-04-02 v80
- 已正式新建 `gnn/GNN_EXPAND`。
- 本轮不是在原主支上继续改模，而是：
  - 复制并抽象现有 clean 最优 `CLS / REG / joint` 逻辑
  - 在 `GNN_EXPAND` 内独立承接不同拓扑和节点规模
- 当前阶段目录：
  - `stage1_square_10x10`
  - `stage2_rect_6x10`
  - `stage3_honeycomb_63`
  - `stage4_transfer_circlecut_69`
- 已完成的关键工程项：
  - 共享 topology registry
  - 共享 `CLS/REG/joint` expand 训练与推理脚本
  - 阶段级包装入口
  - 阶段级 `README.md / Log.md`
- 已修正两个关键问题：
  - `cache/outputs` 现在会落到各阶段自己的子目录，而不是写进 `common/`
  - `6x10 / 蜂窝` 阶段使用原始 clean 数据时的节点与电阻 id 越界问题，现已通过边界映射和电阻几何重映射解决
- `stage4` 默认 warm start 已切到 `stage1_square_10x10`，用于后续 transfer 试验。
- `模型扩展路径.txt` 内容已完成吸收入正式文档，待校验后删除。

## 2026-04-02 v81
- 已新增 `GNN_EXPAND` 通用拓扑数据生成器：
  - `gnn/GNN_EXPAND/generate_expand_datasets.py`
- 已正式生成四套原生拓扑 clean 数据：
  - `square_10x10`
  - `rect_6x10`
  - `honeycomb_63`
  - `circlecut_69`
- 当前四套数据都已满足：
  - 激励只使用外部节点
  - 测量只输出外部节点电压
  - `28` 个外部节点
  - `32` 组边界激励
- 各阶段默认 `train.py / inference.py` 数据路径已切换到：
  - `gnn/GNN_EXPAND/data/*.csv`
- 已验证 `EXPAND` 的数据构建入口可以直接读取这些新 `csv + meta` 并生成训练 cache。

## 2026-04-02 v82
- 更正 `GNN_EXPAND` 原生数据规模描述：
  - 先前“四套数据均为 `28` 个外部节点、`32` 组激励”的记录不成立
  - `generate_expand_datasets.py` 的实际规则是：激励和测量都严格限制在外部节点，但数量随拓扑边界变化
- 已按 `*_meta.json` 核对当前四套数据：
  - `square_10x10`: `36` 个外部节点，`40` 组激励
  - `rect_6x10`: `28` 个外部节点，`32` 组激励
  - `honeycomb_63`: `28` 个外部节点，`32` 组激励
  - `circlecut_69`: `24` 个外部节点，`28` 组激励
- `square_10x10` 对应的 `10x10` 方格边界节点数为 `36`，因此其激励规模也应随之扩展到 `40`。

## 2026-04-02 v83
- 已配合根目录升级 `DOC_RULES.md`，将其作为后续新窗口的优先接手入口。
- 对当前 `gnn` 主线，`DOC_RULES.md` 已固化以下正式口径：
  - clean：`GNN_CLS/modelo3 + GNN_REG/o4a2 + GNN_CMEI_INFERENCE/inference_gnn_cmei.py`
  - noisy：`rand_boundary` 为当前正式带噪主线
  - joint：`inference_gnn_cmei_v2.py` 仍是实验版，不转正
  - expand：所有推广测试只在 `GNN_EXPAND` 内进行，不回写原主线
- 后续若 `gnn` 正式路线切换，应同时更新：
  - `gnn/README.md`
  - `gnn/Log.md`
  - `DOC_RULES.md`

## 2026-04-02 v84
- 已配合根目录将长期规则主文件从 `DOC_RULES.md` 切换为 `RULES.md`。
- 当前 `gnn` 线后续执行时，新增固定规则：
  - 修改代码前，先向用户呈现修改思路
  - 新版本优先复制原模型目录后再改，不直接改旧模型目录
  - `README.md / Log.md` 继续遵守只追加、不删旧记录
- `DOC_RULES.md` 现仅作为历史兼容入口保留；后续正式规则维护统一看 `RULES.md`。

## 2026-04-02 v85
- 已为 `gnn` 主线补充 Git 分支与“当前最佳版本” tag 操作规范。
- 当前固定口径为：
  - `main` 只保留稳定、可回退、可作为正式路线的 `gnn` 状态
  - 日常实验默认不直接在 `main` 上推进
  - 新实验默认新建 `codex/` 前缀分支
  - 推荐分支命名格式为 `codex/<模块>-<版本>-<目的>`
- 当 `gnn` 某轮结果准备升级为当前最佳时，应同步完成：
  - 稳定代码进入 `main`
  - 更新根目录 `CURRENT_BEST.md`
  - 追加更新根目录与 `gnn` 对应 `README.md / Log.md`
  - 新增 `best-*` Git tag 作为恢复锚点
- 同时明确：
  - tag 只新增，不覆盖旧 tag
  - 未形成正式结论的实验版本只停留在实验分支

## 2026-04-03 v86
- 已吸收 `0402噪声v2训练日志.txt` 中与 `gnn` 主线相关的有效结果。
- `GNN_NOISE v2` 当前已确认：
  - clean `CLS test_macro_f1=0.9149`
  - clean `REG mae_all=0.4664`，`mae_changed=24.2457`，`count_macro_f1=0.8349`
  - clean joint `CMEI=93.49`
  - `40dB CLS test_macro_f1=0.9078`
  - `40dB REG mae_all=0.5317`，`mae_changed=25.3754`，`count_macro_f1=0.8342`
- `30dB / 20dB` 本轮未得到有效值，已定位为云端命令里的 `--dataset-tag ${TAG}` 未展开，属于验证入口问题。
- 已新增：
  - `gnn/GNN_NOISE/run_noise_eval_suite.py`
- 作用：
  - 统一串行执行 clean / `40dB` / `30dB` / `20dB` 的 `CLS / REG / joint` 评估
  - 自动把单模型评估结果按噪声等级复制为独立 `json`
  - 支持 `--dry-run`
- 并已在 `GNN_NOISE v2` 与正式 `GNN_CMEI_INFERENCE` 入口补充 `dataset-tag` 空值 / 占位符未展开的明确报错。

## 2026-04-03 v87
- 已吸收 `拓展训练日志.txt`，并以 `GNN_EXPAND` 四阶段实际输出文件作为最终记录依据。
- 当前结果：
  - `stage1_square_10x10`: `macro_f1=0.8581`，`mae_changed=37.0220`，`CMEI=89.30`
  - `stage2_rect_6x10`: `macro_f1=0.9018`，`mae_changed=16.9012`，`CMEI=94.38`
  - `stage3_honeycomb_63`: `macro_f1=0.8671`，`mae_changed=31.3267`，`CMEI=91.05`
  - `stage4_transfer_circlecut_69`: `macro_f1=0.8818`，`mae_changed=43.2324`，`CMEI=88.42`
- 当前判断：
  - `stage2_rect_6x10` 为本轮最佳扩展阶段
  - `stage4_circlecut_69` 的回归与联合推理最弱，不规则拓扑仍是主难点
- 已修正 `stage4 transfer` 的默认 warm start 路径：
  - 从错误的 `.../outputs/training_data64Nodes_2/model_last.pt`
  - 改为正确的 `.../outputs/square_10x10/model_last.pt`
- 因此当前已记录的 `stage4` 结果暂不作为最终 transfer 结论，只作为当前不规则拓扑基线。
- 已新增扩展结果汇总图与配套脚本：
  - `gnn/GNN_EXPAND/plot_expand_summary.py`
  - `gnn/GNN_EXPAND/expand_summary_metrics.json`
  - `gnn/GNN_EXPAND/expand_summary.svg`
- v88
  - `GNN_EXPAND` 汇总图切换为 `matplotlib` 双面板版本。
  - 正式输出更新为 `expand_summary.png` 与 `expand_summary.pdf`，旧 `expand_summary.svg` 停用并删除。
- v89
  - `GNN_EXPAND` 新增 `Figure` 目录，统一存放 `expand_summary` 与四张拓扑结构图。
  - 新增脚本：
    - `gnn/GNN_EXPAND/plot_expand_topologies.py`

## 2026-04-04 v90
- 已吸收 `0404训练日志.txt`，但 `gnn` 线正式记录继续以本地真实输出文件为准。

### `GNN_NOISE v2` 完整补评估
- 本轮终于补齐 `clean / 40dB / 30dB / 20dB` 四档完整 `CLS / REG / joint` 结果：
  - clean：`macro_f1=0.9149`，`mae_changed=24.2457`，`CMEI=93.49`
  - `40dB`：`macro_f1=0.9078`，`mae_changed=25.3754`，`CMEI=92.81`
  - `30dB`：`macro_f1=0.8903`，`mae_changed=34.0454`，`CMEI=90.44`
  - `20dB`：`macro_f1=0.7582`，`mae_changed=58.1169`，`CMEI=80.42`
- 与 `rand_boundary` 对比后的判断修正为：
  - `v2` 已经成为 clean 到中噪声区间更强的曲线
  - 但 `20dB` 端点仍由 `rand_boundary` 保持领先
  - 因此 noisy 最佳口径应改成“双锚点”而不是单条线绝对统治
- 已新增：
  - `gnn/GNN_NOISE/plot_noise_v2_summary.py`
  - `gnn/GNN_NOISE/noise_v2_summary_metrics.json`
  - `gnn/GNN_NOISE/Figure/noise_v2_summary.png`
  - `gnn/GNN_NOISE/Figure/noise_v2_summary.pdf`
- 已删除旧图：
  - `gnn/GNN_NOISE/rand_boundary_robustness_curve.svg`

### `GNN_EXPAND` 真正 transfer 结果补齐
- `stage4_transfer_circlecut_69` 当前记录已更新为真正成功加载 `stage1` 权重后的结果：
  - `CLS warm_start.loaded=36`
  - `REG warm_start.loaded=36`
  - `CLS macro_f1=0.8928`
  - `REG mae_changed=36.1173`
  - `joint CMEI=91.14`
- 相比上一轮未成功 warm start 的基线：
  - `macro_f1: 0.8818 -> 0.8928`
  - `mae_changed: 43.2324 -> 36.1173`
  - `CMEI: 88.42 -> 91.14`
- 当前阶段判断更新为：
  - `stage2_rect_6x10` 仍是最强阶段
  - `stage4_circlecut_69` 已从“路径错误导致的弱基线”修正为“真实 transfer 后可用但仍待继续优化”的阶段
- `EXPAND` 图像已按最新数据重画：
  - `gnn/GNN_EXPAND/Figure/expand_summary.png`
  - `gnn/GNN_EXPAND/Figure/expand_summary.pdf`
  - `gnn/GNN_EXPAND/Figure/topology_square_10x10.png`
  - `gnn/GNN_EXPAND/Figure/topology_rect_6x10.png`
  - `gnn/GNN_EXPAND/Figure/topology_honeycomb_63.png`
  - `gnn/GNN_EXPAND/Figure/topology_circlecut_69.png`

### 清理
- `0404训练日志.txt` 的内容已吸收入正式文档，现已删除。
