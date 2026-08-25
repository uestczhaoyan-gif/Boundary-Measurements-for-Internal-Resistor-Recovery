# MLP multi-task family

`MLP_FULL/` 保存共享主干、多头分类/回归以及由回归推断数量的实验。它主要用于研究任务耦合是否帮助或损害单任务性能；当前不是默认生产链路。

优先阅读各版本 `README.md`、`Log.md` 和输出 `metrics.json`，再决定是否复跑。当前主项目的联合推理使用 GNN 的 CLS+REG 拆分路线。
