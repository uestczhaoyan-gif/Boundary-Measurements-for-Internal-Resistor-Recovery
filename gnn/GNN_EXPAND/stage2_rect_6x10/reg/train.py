from __future__ import annotations

import sys
from pathlib import Path

COMMON_DIR = Path(__file__).resolve().parents[2] / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from train_reg_expand import main


if __name__ == "__main__":
    main(
        stage_name="stage2_rect_6x10",
        topology_key="rect_6x10",
        default_data_path="gnn/GNN_EXPAND/data/rect_6x10.csv",
        default_pretrained_model_path="gnn/GNN_REG/o4a2/outputs/training_data64Nodes_2/model_last.pt",
        runtime_dir=Path(__file__).resolve().parent,
    )
