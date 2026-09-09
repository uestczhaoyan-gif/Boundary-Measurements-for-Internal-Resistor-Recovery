# 64Nodes: Boundary Measurements for Internal Resistor Recovery

[中文](README.md) | **English**

This project studies an inverse problem on a fixed resistor-network topology: using only voltage responses measured at boundary nodes, can we identify which internal resistors have changed, estimate the changes, and determine how reliable that recovery remains under noise and changes in topology?

The project supports an undergraduate thesis on boundary-port measurements and internal-parameter extraction for a topological optoelectronic network. All data and results currently in the repository come from **synthetic simulations based on Kirchhoff's laws**. They have not yet been validated on physical devices or measured hardware. The public repository contains code, sanitized dataset metadata, experimental figures, and research records. The final thesis, defense materials, and files containing personal information remain in the local, Git-ignored `private_materials/` directory.

## Research questions

The main experiment uses an `8 × 8` grid with 64 nodes, 112 resistor edges, 28 boundary measurement nodes, and 32 excitation configurations. Each sample pairs a set of internal resistance changes with the resulting boundary voltage responses. The models address three related tasks:

1. **Change-count classification:** predict the number of changed resistors, `K ∈ {0, 1, 2, 3}`.
2. **Change-magnitude regression:** predict a 112-dimensional vector of resistance changes, `ΔR`.
3. **Joint inference:** use the predicted `K` to retain the corresponding number of candidate edges from the regression output, producing final location and magnitude predictions.

The difficulty comes from compressed observations and non-uniqueness: internal nodes cannot be measured directly, and different internal changes may yield similar boundary responses. A low average error therefore does not by itself establish reliable localization. The repository records count, magnitude, candidate-coverage, and joint metrics.

## Method overview

```text
Topology and resistor parameters
        │
        ▼
Kirchhoff-law forward simulation ──► Boundary voltages under multiple excitations
        │                                      │
        │                                      ├── MLP baseline
        │                                      └── Main GNN approach
        │                                            ├── CLS: predict K
        │                                            ├── REG: predict 112-dimensional ΔR
        │                                            └── CMEI: joint inference and evaluation
        │
        ├── inverse_identifiability: detectability and non-uniqueness
        └── square_scale_study: network size, port count, and K_max
```

GNNs are the current main approach. MLPs provide structured-vector baselines on a fixed topology. Earlier attention, CNN, CNN2D–MLP, and UNet–MLP approaches are archived in `history/`. Experiment outputs, caches, and trained weights are excluded from Git by default; see [`.gitignore`](.gitignore).

## Currently recommended results

These are intermediate results recorded in the repository, not an independently verified leaderboard under a single evaluation protocol. Read the dataset tags, random seeds, and metric definitions when making comparisons.

| Approach | Role | Recorded result | Entry point |
| --- | --- | ---: | --- |
| GNN CLS `modelo3` | Change count | `test_macro_f1 ≈ 0.90` | [`gnn/GNN_CLS/modelo3`](gnn/GNN_CLS/modelo3) |
| GNN REG `o4a2` | Change-magnitude regression | `mae_all = 0.4679`; `mae_changed = 23.5724` | [`gnn/GNN_REG/o4a2`](gnn/GNN_REG/o4a2) |
| GNN CMEI v1 | Joint inference | `CMEI = 93.53` | [`gnn/GNN_CMEI_INFERENCE`](gnn/GNN_CMEI_INFERENCE) |
| GNN noise v2 | Clean to moderate noise | Clean `CMEI = 93.49`; 20 dB `CMEI = 80.42` | [`gnn/GNN_NOISE`](gnn/GNN_NOISE) |
| GNN expand | Cross-topology transfer | Stage 2 recorded `CMEI = 94.38` | [`gnn/GNN_EXPAND`](gnn/GNN_EXPAND) |

The currently recommended configuration and the best result ever observed are not always the same. See [`CURRENT_BEST.md`](CURRENT_BEST.md) for the complete conventions and each directory's `Log.md` for the experimental history. These records and the detailed documentation remain in Chinese.

## Repository map

| Directory | Purpose | Start here |
| --- | --- | --- |
| `data/` | Main 8×8 dataset metadata, generators, and optional local CSV files | [`data/README.md`](data/README.md) |
| `scripts/` | Shared public utilities and data-generation scripts | [`scripts/README.md`](scripts/README.md) |
| `gnn/` | Main approach: classification, regression, joint inference, noise, and topology transfer | [`gnn/README.md`](gnn/README.md) |
| `mlp/` | Fixed-topology MLP baselines and fixed-change diagnostics | [`mlp/README.md`](mlp/README.md) |
| `inverse_identifiability/` | Response differences, similarity, detectability, and non-uniqueness | [`inverse_identifiability/README.md`](inverse_identifiability/README.md) |
| `square_scale_study/` | Square-network size, port count, and maximum identifiable change count | [`square_scale_study/README.md`](square_scale_study/README.md) |
| `history/` | Archived model families no longer being developed | [`history/README.md`](history/README.md) |
| `Figure/` | Public thesis/presentation figure sources and summary figures | [`Figure/README.md`](Figure/README.md) |
| `docs/` | Reproducibility conventions, metric definitions, and navigation | [`docs/PROJECT_MAP.md`](docs/PROJECT_MAP.md) |
| `private_materials/` | Local-only thesis, defense, and personal materials | Not uploaded |

## Quick start

### 1. Inspect the main dataset specification

```powershell
Get-Content data\training_data64Nodes_2_meta.json
```

The main approach uses the **unscreened 10 mA dataset**, `data/training_data64Nodes_2.csv`. CSV files are normally ignored by Git. If the CSV is not available locally, regenerate it using the generator and the recorded metadata.

### 2. Generate a small dataset for a smoke test

```powershell
python scripts\generate_training_data64.py --help
python scripts\generate_training_data64.py `
  --total-combos 100 `
  --current-a 0.01 `
  --output data\_smoke\training_data64_smoke.csv `
  --meta-output data\_smoke\training_data64_smoke_meta.json
```

This command explicitly sets the simulation current, CSV path, and metadata path, so it does not overwrite the main dataset's metadata. To generate the standard 10 mA dataset, use `--current-a 0.01`, `--output data\training_data64Nodes_2.csv`, and `--meta-output data\training_data64Nodes_2_meta.json`.

Refer to each script's `--help` for its supported arguments. Before a full training run, use a small dataset to check paths, dependencies, and output directories.

### 3. Run the main GNN approach

```powershell
python gnn\GNN_CLS\modelo3\train.py --help
python gnn\GNN_REG\o4a2\train.py --help
python gnn\GNN_CMEI_INFERENCE\inference_gnn_cmei.py --help
```

Specify `--data-path`, `--dataset-tag`, and the random seed explicitly in training commands. Outputs are written to `cache/<dataset_tag>/` and `outputs/<dataset_tag>/` within the corresponding model directory; these directories are not committed by default. For an end-to-end smoke test, see [`scripts/reproduce_smoke.ps1`](scripts/reproduce_smoke.ps1).

## Reproducibility conventions

- Run commands from the project root. Avoid embedding personal absolute paths in scripts or logs.
- Record the data filename, dataset tag, random seed, train/validation/test split, and key hyperparameters.
- Use only training-set statistics for normalization.
- Treat `metrics.json` as the machine-readable result source. Use `Log.md` to explain experimental choices and failures.
- Every number reported in the thesis should point to a specific output directory. Numbers recoverable only from terminal logs should be marked as awaiting verification.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for environment setup and further reproducibility guidance.

## Project contributions

The project develops a traceable research workflow: Kirchhoff-law forward data generation; encoding network topology for GNN input; separating change-count classification, change-magnitude regression, and joint inference; using detectability analysis to interpret difficult cases; and examining limitations under noise, changes in network size, and changes in topology.

## License and material boundaries

Original code is released under [`LICENSE`](LICENSE). The repository license does not automatically grant redistribution rights for third-party papers, university templates, icons, fonts, or thesis attachments. Files containing personal information belong in `private_materials/` and are not part of the public repository.
