# GNN_REG / modelo3b

- Purpose:
  - Candidate-set inference wrapper around `gnn/GNN_REG/modelo3`
  - Reuses `modelo3` training outputs and adds `top3 / top4 / top5` coverage evaluation
- Scope:
  - Inference only
  - Does not change global `joint_inference`
- Default source model:
  - `../modelo3`
- Default data:
  - `../../../data/training_data64Nodes_2.csv`

Typical run:

```powershell
python gnn/GNN_REG/modelo3b/inference.py
```

Useful overrides:

```powershell
python gnn/GNN_REG/modelo3b/inference.py --source-model-dir ../modelo3 --dataset-tag training_data64Nodes_2
python gnn/GNN_REG/modelo3b/inference.py --candidate-sizes 3,4,5 --num-samples 6
```
