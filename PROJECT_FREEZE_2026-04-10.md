## Project Freeze Record

Date: 2026-04-10

### Freeze reason

After discussion with the advisor, the current GNN project has produced a complete stage snapshot, but the research direction no longer fully matches the advisor's latest expectations. The project is therefore frozen at the current state before replanning the next stage.

This freeze means:

- The current mainline results are kept as a reproducible baseline.
- No further model expansion should be done on top of this snapshot before the new research plan is clarified.
- Follow-up work should start from a new clearly defined direction, while this snapshot remains available for rollback, comparison, and presentation.

### Frozen scope

The current snapshot includes the following completed lines of work:

- Clean mainline:
  - `gnn/GNN_CLS/modelo3`
  - `gnn/GNN_REG/o4a2`
  - `gnn/GNN_CMEI_INFERENCE/inference_gnn_cmei.py`
- Noisy training:
  - `rand_boundary` baseline
  - `structured_boundary_v2` balanced noisy line
- Topology / scale expansion:
  - `stage1_square_10x10`
  - `stage2_rect_6x10`
  - `stage3_honeycomb_63`
  - `stage4_transfer_circlecut_69`
- Physical analysis:
  - `inverse_identifiability`

### Snapshot notes

- The clean GNN pipeline has already formed a stable staged baseline.
- The noisy line has completed a first-round comparison between v1 and v2.
- The expand line has completed regular-topology and irregular-topology transfer verification.
- `inverse_identifiability` has completed the first round of detectability / ambiguity analysis and now serves as an explanatory reference for the inverse problem difficulty.

### Asset updates included in this freeze

This freeze commit also preserves the current presentation and reporting assets prepared during the latest reporting cycle, including:

- comparison figures under:
  - `gnn/GNN_CLS/Figure/`
  - `gnn/GNN_REG/Figure/`
  - `gnn/GNN_CMEI_INFERENCE/Figure/`
  - `gnn/GNN_NOISE/Figure/`
  - `gnn/GNN_EXPAND/Figure/`
- root-level reporting support notes and materials:
  - `下一阶段重点.txt`
  - files under `Figure/`

### Recommended rule after freeze

- Treat this commit as the final closure point of the current GNN stage.
- If a new direction is started later, do not silently continue editing under the old narrative.
- Prefer to create a new clearly labeled branch, plan file, or project line for the next-stage work.

### Suggested recovery order

If this frozen project needs to be reopened in the future, review in the following order:

1. `RULES.md`
2. `CURRENT_BEST.md`
3. `README.md`
4. `Log.md`
5. `gnn/README.md`
6. `gnn/GNN_CMEI_INFERENCE/inference_gnn_cmei.py`

### Final note

This file records a deliberate research freeze, not a failure state. The current project has already reached a meaningful and reusable stage boundary; the freeze is intended to preserve that boundary clearly before the next plan is defined.
