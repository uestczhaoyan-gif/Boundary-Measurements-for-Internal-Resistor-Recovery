# `Noise_test` 原始步骤 B 保留版

本文件用于把根目录原 `Noise_test.txt` 里的步骤 B 收编到 `gnn/GNN_NOISE` 中，避免根目录继续保留散落的实验说明文件。

## 1. 与当前默认 `*_ft` 方案的关系

- 当前默认 `GNN_NOISE/*_ft`：
  - warm start 到 clean 最优权重
  - 训练集动态随机噪声
  - 默认只在边界节点 `voltage_delta` 通道加噪
- 原 `Noise_test` 步骤 B：
  - 核心关注点是“固定 `20dB` 高斯噪声增强训练”
  - 原始伪代码是对整个 `voltage` 通道执行 `x_sample[:, :, 2] += noise`
  - 没有强调动态随机 SNR，也没有强调边界节点限定

## 2. 现在如何在 `GNN_NOISE` 中复现它

两条训练脚本都已新增以下入口：

- `--noise-schedule {random,fixed}`
- `--fixed-noise-std`
- `--noise-scope {boundary,all}`

因此，原 `Noise_test` 的步骤 B 现在可由以下参数组合严格表达：

- `--noise-schedule fixed`
- `--fixed-noise-std 0.1`
- `--noise-mode gaussian`
- `--noise-scope all`

## 3. 推荐命令

### CLS

```bash
python gnn/GNN_NOISE/CLS_modelo3_ft/train.py \
  --data-path data/training_data64Nodes_2.csv \
  --dataset-tag training_data64Nodes_2 \
  --epochs 30 \
  --lr 5e-5 \
  --noise-schedule fixed \
  --fixed-noise-std 0.1 \
  --noise-mode gaussian \
  --noise-scope all
```

### REG

```bash
python gnn/GNN_NOISE/REG_o4a2_ft/train.py \
  --data-path data/training_data64Nodes_2.csv \
  --dataset-tag training_data64Nodes_2 \
  --epochs 30 \
  --lr 5e-5 \
  --noise-schedule fixed \
  --fixed-noise-std 0.1 \
  --noise-mode gaussian \
  --noise-scope all
```

## 4. 备注

- 这份“保留版”主要是为了完整吸收根目录原始步骤 B 的定义。
- 当前真正优先建议跑的，仍然是默认增强版：
  - `noise-schedule=random`
  - `noise-scope=boundary`
  - warm start fine-tune
