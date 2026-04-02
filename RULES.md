# 64Nodes RULES

## 0. 文件定位
- 本文件是 `64Nodes` 项目的长期规则文件，也是新对话窗口接手项目时的第一入口。
- 目标不是记录所有历史细节，而是固定三类高优先级信息：
  - 文档与日志管理规则
  - 必须遵守的项目原则与禁忌
  - 当前正式主线、当前最佳路线、当前重点工作
- 任何新窗口在开始工作前，必须先读本文件，再按需要继续读根目录与对应子目录的 `README.md / Log.md`。
- 如果旧文档与本文件存在冲突：
  - 以本文件和最新日期的更正记录为准
  - 不能静默忽略冲突，必须在后续日志里补一条“更正说明”

## 1. 项目简介
- 问题本质：
  - 在某种固定拓扑下的电阻网络中，当内部电阻组织发生变化，只能通过外部节点的激励与测量去重建内部变化状态。
- 当前仓库的标准 clean 主线实现是：
  - `8x8` 网格
  - `64` 节点
  - `112` 条电阻边
- 当前仓库围绕三个任务组织：
  - `CLS`：变化数量分类，类别为 `0/1/2/3`
  - `REG`：电阻变化量回归
  - `joint inference`：用 `CLS + REG` 组合完成最终联合推理
- 当前可用方法主要有两条：
  - `mlp`：可用，但不是当前主线
  - `gnn`：效果和可解释性都优于 `mlp`，所以当前所有主线推进都在 `gnn` 内进行
- 当前 clean 默认数据口径：
  - 默认使用未筛选 `10mA` 数据
  - 数据路径为 `data/training_data64Nodes_2.csv`
  - `screened` 数据不再作为当前正式主线默认数据

## 2. 不可违反的硬规则

### 2.1 边界节点规则
- 数据生成时，激励节点只能使用外部节点。
- 数据生成时，测量节点只能使用外部节点。
- 这个规则对 clean、noisy、expand 三类数据都成立。
- 不允许为了和旧数据形状对齐，强行把新拓扑裁成固定数量的外部节点。
- 只要拓扑变了，外部节点数和激励组数就应按该拓扑自己的真实边界规模确定。

### 2.2 沟通与执行规则
- 在修改代码之前，必须先向用户呈现本次修改思路，不能直接先动手改代码。
- 这个“修改思路”不需要很长，但必须先说明：
  - 改什么
  - 为什么这么改
  - 是否会新建版本目录或复制现有模型
- 用户如果继续推进，再开始实际修改。

### 2.3 主线保护规则
- 不要在已经稳定的主线目录上直接做结构性试验。
- 一切模型改动和优化，原则上都不应破坏现有模型。
- 每一版更新都应优先在“复制原模型后的新目录”上进行，而不是直接改原模型。
- 新分支、新版本、新实验应优先放在独立目录中，避免污染正式主线。
- `GNN_EXPAND` 内的扩展工作必须在 `gnn/GNN_EXPAND` 内独立完成，不回写原始 `GNN_CLS / GNN_REG / GNN_CMEI_INFERENCE`。
- `GNN_NOISE` 的职责仅限：
  - 带噪训练
  - 单模型推理
  - 各自分支内的 `outputs`
- 联合推理与 `CMEI` 输出统一放在：
  - `gnn/GNN_CMEI_INFERENCE`

### 2.4 文档纪律
- 任何模型修改、路线调整、用户回传训练日志、关键实验结论吸收后，都必须更新文档。
- `Log.md` 和 `README.md` 原则上不允许删除内容，只允许增添新内容。
- 如果旧记录口径错误，不删旧记录，而是在新条目里明确写“更正”。
- `README.md` 负责说明“是什么、为什么、当前怎么用”。
- `Log.md` 负责记录“做了什么、结果如何、如何判断、下一步是什么”。
- `RULES.md` 负责保存长期稳定的规则与当前正式主线，不写逐轮细碎训练过程。

### 2.5 路线纪律
- `history` 目录中的方法默认视为已尝试后放弃，不应重新当作当前主线推进。
- `mlp` 仍可作为可用对照或备用基线，但默认不再占用主要开发资源。
- 没有明确新证据时，不要把实验版脚本、实验版模型或一次性好结果升级为正式主线。

## 3. 文件职责

### 3.1 根目录
- `README.md`
  - 项目总览
  - 总体任务定义
  - 方法目录导航
  - 当前正式路线摘要
- `Log.md`
  - 全局版本时间线
  - 阶段性判断
  - 当前主线变化摘要
- `RULES.md`
  - 新窗口第一入口
  - 长期规则
  - 当前正式路线
  - 当前重点工作
- `CURRENT_BEST.md`
  - 当前最佳路线与本地最佳产物锚点清单
  - 用于在 Git 轻量方案下快速重塑当前最佳状态

### 3.2 `gnn`
- `gnn/README.md`
  - GNN 方法演进说明
  - 当前正式 clean/noisy/joint/expand 路线说明
- `gnn/Log.md`
  - GNN 线的详细实验时间线
  - 关键结果、回退、转正与路线收敛记录

### 3.3 `gnn` 子目录职责
- `gnn/GNN_CLS`
  - clean 分类训练与单模型推理
- `gnn/GNN_REG`
  - clean 回归训练与单模型推理
- `gnn/GNN_NOISE`
  - 带噪训练与单模型推理
- `gnn/GNN_CMEI_INFERENCE`
  - 联合推理与 `CMEI` 输出
- `gnn/GNN_EXPAND`
  - 不同拓扑、不同规模下的模型推广测试

### 3.4 其他目录
- `mlp`
  - 保留为实际可用的历史主方法与对照基线
- `history`
  - 已尝试后放弃的方法归档，不作为当前开发入口

## 4. 文档更新触发条件
- 满足以下任一条件时，必须更新对应 `README.md / Log.md`，必要时同步更新本文件：
  - 新建模型版本目录
  - 训练脚本、推理脚本、数据生成脚本有实质改动
  - 用户提供了新的训练日志、评估结果或云端回传
  - 当前最佳模型或正式路线发生变化
  - 当前重点工作发生变化
  - 发现了旧记录中的原则性错误

## 5. 新窗口接手顺序
- 任何新窗口默认按以下顺序建立上下文：
  1. `RULES.md`
  2. `CURRENT_BEST.md`
  3. 根目录 `README.md`
  4. 根目录 `Log.md` 末尾最新条目
  5. `gnn/README.md`
  6. `gnn/Log.md` 末尾最新条目
  7. 当前相关子目录的 `README.md / Log.md`
- 如果任务属于：
  - clean 主线：重点读 `GNN_CLS / GNN_REG / GNN_CMEI_INFERENCE`
  - 带噪主线：重点读 `GNN_NOISE`
  - 拓扑推广：重点读 `GNN_EXPAND`

## 6. 当前正式路线

### 6.1 clean 主线
- 当前 clean 默认数据：
  - `data/training_data64Nodes_2.csv`
- 当前正式 `CLS`：
  - `gnn/GNN_CLS/modelo3`
- 当前正式 `REG`：
  - `gnn/GNN_REG/o4a2`
- 当前正式 joint inference：
  - `gnn/GNN_CMEI_INFERENCE/inference_gnn_cmei.py`
- 说明：
  - `modelo3` 是当前正式分类主线
  - `o4a2` 是当前正式回归主线
  - `inference_gnn_cmei.py` 是当前正式联合推理入口

### 6.2 当前最佳 clean 参考口径
- 当前稳定 clean 参考组合：
  - `GNN_CLS/modelo3 + GNN_REG/o4a2 + inference_gnn_cmei.py`
- 当前文档中的稳定参考结果为：
  - `CLS modelo3`: `test_macro_f1=0.9027`
  - `REG o4a2`: `mae_all=0.4679`
  - `REG o4a2`: `mae_changed=23.5724`
  - clean joint: `CMEI=93.53`
- 说明：
  - `GNN_REG/modelo3` 的历史单次最好结果可以作为上限参考
  - 但当前正式、稳定、可复验的主线已经切换到 `o4a2`

### 6.3 noisy 主线
- 当前正式带噪主线是：
  - `gnn/GNN_NOISE/CLS_modelo3_ft`
  - `gnn/GNN_NOISE/REG_o4a2_ft`
- 当前正式 noisy 策略是：
  - `rand_boundary`
  - 核心思想是“边界可测节点上的带噪训练”
- 当前 noisy 正式路线的原则是：
  - 噪声优先加在可测边界节点上
  - 不再把 `fixed20db_all` 当作默认增强方案
- 当前 noisy joint 正式入口仍是：
  - `gnn/GNN_CMEI_INFERENCE/inference_gnn_cmei.py`
- 当前正式 noisy 结论：
  - `rand_boundary` 优于 `fixed20db_all`
  - `fixed20db_all` 保留为对照和历史记录，不再是正式主线

### 6.4 noisy 当前最佳参考口径
- 当前 `rand_boundary` joint 参考结果：
  - clean: `CMEI=91.01`
  - `40dB`: `CMEI=90.83`
  - `30dB`: `CMEI=89.62`
  - `20dB`: `CMEI=82.56`
- 工程判断：
  - `40/30dB` 已接近 clean 水平
  - `20dB` 仍有明显衰减
  - 但相对 zero-shot 已实现决定性恢复

### 6.5 joint inference 正式口径
- 联合推理正式目录：
  - `gnn/GNN_CMEI_INFERENCE`
- 联合推理正式脚本：
  - `gnn/GNN_CMEI_INFERENCE/inference_gnn_cmei.py`
- `gnn/inference_gnn_cmei.py` 当前只保留兼容转发角色，不应再当作正式逻辑主文件来维护。
- `gnn/GNN_CMEI_INFERENCE/inference_gnn_cmei_v2.py` 目前仍是实验版：
  - `v2` 没有超过 `v1`
  - 不能转正

### 6.6 expand 主线
- `GNN_EXPAND` 的定位不是重写主线，而是把当前最佳 GNN 方法推广到不同拓扑与规模进行测试。
- 当前 `GNN_EXPAND` 默认沿用：
  - `CLS = modelo3`
  - `REG = o4a2`
  - `joint = inference_gnn_cmei` 体系
- 当前四阶段目录：
  - `stage1_square_10x10`
  - `stage2_rect_6x10`
  - `stage3_honeycomb_63`
  - `stage4_transfer_circlecut_69`
- `stage3` 固定为蜂窝状，这是当前项目约束之一。
- `stage4` 默认优先承接 `stage1_square_10x10` 权重，体现 transfer 设定。

## 7. GNN_EXPAND 数据硬口径
- `GNN_EXPAND` 的原生数据必须直接在目标拓扑上正演生成。
- 生成器路径：
  - `gnn/GNN_EXPAND/generate_expand_datasets.py`
- 不允许把“旧 `8x8` 固定 28 外部节点 / 32 激励”的口径硬套到新拓扑。
- 当前四套原生数据真实规模为：
  - `square_10x10`: `36` 个外部节点，`40` 组激励
  - `rect_6x10`: `28` 个外部节点，`32` 组激励
  - `honeycomb_63`: `28` 个外部节点，`32` 组激励
  - `circlecut_69`: `24` 个外部节点，`28` 组激励
- 这四套数据的共同规则仍然是：
  - 激励只使用外部节点
  - 测量只输出外部节点电压

## 8. 明确不再作为正式主线的路线
- `history/*`
  - 统一视为归档
- `MLP` 主线继续保留，但不是当前主推进方向
- 异构联合推理：
  - `MLP_CLS + GNN_REG (+ MLP_REG)`
  - 当前不再作为正式主线
- `joint_method`
  - 已经判定失败并撤回
- `GNN_REG/model_tp1`
  - 当前暂停
- `GNN_REG/o5a`
  - 当前不如 `o4a2`
- `GNN_REG/o5b / o5b1`
  - 有价值，但尚未超过当前正式主线 `o4a2`
- `GNN_FULL/Mv1/inference_v2.py`
  - 保留实验意义，不是正式联合推理入口
- `GNN_CMEI_INFERENCE/inference_gnn_cmei_v2.py`
  - 当前不转正
- `fixed20db_all`
  - 保留为历史对照，不再作为 noisy 正式路线

## 9. 当前重点工作
- 当前项目的主线任务只有两项：
  - `GNN_NOISE v2` 的第二版鲁棒性测试
  - `GNN_EXPAND` 的模型推广测试

### 9.1 `GNN_NOISE v2`
- 当前目录：
  - `gnn/GNN_NOISE/CLS_modelo3_ft_v2`
  - `gnn/GNN_NOISE/REG_o4a2_ft_v2`
- 当前目标：
  - 保留 `rand_boundary` 的 clean/noisy 平衡
  - 引入更贴近真实测量场景的结构化边界噪声
- 当前默认策略：
  - warm start from `v1 rand_boundary`
  - `noise_schedule=curriculum`
  - `noise_mode=structured`
  - `noise_scope=boundary`
  - `clean_mix_prob=0.15`
- 当前状态：
  - 代码与入口已建立
  - 仍需要正式训练与结果吸收

### 9.2 `GNN_EXPAND`
- 当前目标：
  - 用当前最佳 `modelo3 + o4a2 + v1 joint` 体系去测试不同拓扑、不同规模的可推广性
- 当前要求：
  - 不改原主线
  - 所有扩展代码仅在 `gnn/GNN_EXPAND` 内维护
  - 每个阶段都要保留 `cls / reg / joint_inference`
  - 每个阶段的输出都留在自己子目录内
- 当前状态：
  - 四阶段目录已建立
  - 原生拓扑数据生成器已建立
  - 四套 `csv + meta` 已生成
  - 后续重点应放在正式训练、推理和结果吸收

## 10. 新窗口必须遵守的执行流程
- 接到任务后先判断属于哪条主线：
  - clean
  - noisy
  - expand
- 先读本文件，再读相关目录的 `README.md / Log.md` 最新部分。
- 修改代码前，必须先向用户呈现本次修改思路，再开始动手。
- 做版本更新时，优先复制原模型目录并在新目录上改，不直接改老版本目录。
- 完成代码或吸收训练日志后，必须同步更新：
  - 根目录 `README.md`
  - 根目录 `Log.md`
  - 对应子目录 `README.md`
  - 对应子目录 `Log.md`
- 如果影响“当前正式路线 / 当前最佳模型 / 当前重点工作”，必须同步更新 `RULES.md`。

## 11. 后续维护原则
- 本文件属于“长期规则 + 当前路线”文件，更新频率应低于普通日志。
- 只有以下内容变化时，才建议修改本文件：
  - 正式主线切换
  - 当前最佳模型切换
  - 当前重点工作切换
  - 长期规则新增或修正
  - 项目原则性错误被发现并需要永久防呆
- 普通训练细节、单次试验结果、临时分析，不应堆积到本文件中，应写入对应 `Log.md`。

## 12. Git 规则
- 当前项目已采用 Git 轻量方案（方案 A）。
- 方案 A 的目标是：
  - 用 Git 保护代码、脚本、文档、规则文件、元数据和当前最佳路线说明
  - 不把大量训练产物、缓存、权重和可重建数据直接纳入版本库
- 当前 Git 默认应跟踪：
  - `.py / .md / .json` 等源码、文档、规则、元数据
  - 目录结构与数据生成脚本
  - `CURRENT_BEST.md`
- 当前 Git 默认不跟踪：
  - `outputs/`
  - `cache/`
  - `*.pt / *.pth / *.npz`
  - 可由脚本重建的 `csv` 数据
  - 本地依赖目录与解释器缓存
- 这样做的原因是：
  - 保持仓库轻量
  - 避免训练产物把版本历史淹没
  - 让“当前最佳路线”通过文档和锚点文件保持可恢复
- `CURRENT_BEST.md` 必须记录：
  - 当前正式 clean/noisy/joint/expand 路线
  - 当前最佳本地目录锚点
  - 当前推荐恢复顺序
- 每次出现以下情况时，建议提交一次 Git 版本：
  - 正式主线切换
  - 新模型目录建立并完成稳定验证
  - 重要规则文件更新
  - 数据生成脚本、训练脚本、联合推理脚本有关键改动
