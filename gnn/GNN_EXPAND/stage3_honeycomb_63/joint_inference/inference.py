from __future__ import annotations

import sys
from pathlib import Path

COMMON_DIR = Path(__file__).resolve().parents[2] / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from inference_joint_expand import main


if __name__ == "__main__":
    main(
        stage_name="stage3_honeycomb_63",
        topology_key="honeycomb_63",
        default_data_path="gnn/GNN_EXPAND/data/honeycomb_63.csv",
        runtime_dir=Path(__file__).resolve().parent,
    )
