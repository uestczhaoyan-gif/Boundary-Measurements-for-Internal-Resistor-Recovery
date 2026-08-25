#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

N_VALUES=(3 4 5 6 7 8 9 10)
K_VALUES=(1 2 3 4 5 6)

for n in "${N_VALUES[@]}"; do
  scale_dir="N${n}x${n}"
  for k in "${K_VALUES[@]}"; do
    meta_path="square_scale_study/data/${scale_dir}/square_N${n}x${n}_K${k}_meta.json"
    if [[ ! -f "$meta_path" ]]; then
      echo "[skip] missing meta: $meta_path"
      continue
    fi

    echo "==== modelo2_gnn | ${scale_dir} | K=${k} ===="
    python square_scale_study/models/modelo2_gnn/train.py --meta-path "$meta_path"
    python square_scale_study/models/modelo2_gnn/inference.py --meta-path "$meta_path" --split test
  done
done

