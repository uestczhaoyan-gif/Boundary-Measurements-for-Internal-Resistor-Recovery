# MLP classification family

`MLP_CLS/` 将 `32 x 28` 边界响应展平为 896 维向量，预测 `K ∈ {0,1,2,3}`。`modelo5` 是当前可比较的稳定 baseline；更高版本应结合各自输出和日志判断，不自动视为更优。

```powershell
python mlp\MLP_CLS\modelo5\train.py --help
python mlp\MLP_CLS\modelo5\inference.py --help
```

主指标是 `macro_f1` 和 `2/3` 混淆。该目录用于和 GNN 做同数据口径的归纳偏置对照。
