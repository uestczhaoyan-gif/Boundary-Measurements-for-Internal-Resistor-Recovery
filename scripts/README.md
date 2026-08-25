# Public scripts

这里放跨子项目复用的公开脚本，重点是数据生成和不会依赖某个模型权重的工具。模型训练/推理脚本保留在对应模型目录，论文排版脚本和私有材料处理脚本位于 `private_materials/working_notes/`。

| 脚本 | 作用 |
| --- | --- |
| `generate_training_data64.py` | 生成 8x8 主数据 |
| `generate_training_data64_screened.py` | 生成筛选版对照数据 |

查看参数：

```powershell
python scripts\generate_training_data64.py --help
python scripts\generate_training_data64_screened.py --help
```

脚本应从仓库根目录运行，并使用相对路径。新工具请补充 `--help`、输入输出说明和元数据落盘逻辑；不要把模型权重、缓存或个人电脑绝对路径写入仓库。
