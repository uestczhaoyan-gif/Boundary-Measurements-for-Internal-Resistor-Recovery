from pathlib import Path
import runpy


TARGET = Path(__file__).resolve().parent / "GNN_CMEI_INFERENCE" / "inference_gnn_cmei.py"


if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
