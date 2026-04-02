# Inverse Identifiability

This mini-project quantifies how resistor changes in the `8x8 / 64-node / 112-resistor`
grid affect boundary voltages, and how often different changes produce nearly identical
voltage patterns.

Why this name:
- The main project is an inverse problem.
- This study focuses on detectability and non-uniqueness.
- "Identifiability" captures both.

Important alignment with the existing data-generation spec:
- The grid boundary has `28` unique boundary nodes, not `32`.
- The project data-generation script uses `32` excitation pairs.
- This study keeps the same topology and boundary-node order, and by default uses the
  four canonical long-range excitations already present in the project:
  `0->63`, `7->56`, `3->60`, `31->32`.
- The script can also switch to the full project `32` excitations.

Main script:
- `scripts/run_identifiability_study.py`

Typical run:

```powershell
& 'C:\Program Files\Python311\python.exe' inverse_identifiability\scripts\run_identifiability_study.py
```

Outputs:
- `outputs/fig1_delta_v_examples.png`
- `outputs/fig2_norm_vs_amplitude.png`
- `outputs/fig3_cosine_similarity_heatmap.png`
- `outputs/case_metrics.csv`
- `outputs/high_similarity_pairs.csv`
- `outputs/special_case_triplets.csv`
- `outputs/detection_summary.json`
- `outputs/analysis_summary.md`

Notes:
- The plotting backend is headless (`Agg`) so the script can run in the terminal.
- Dependencies are expected in `inverse_identifiability/.vendor`.
