$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$smokeDir = Join-Path $root "data\_smoke"
$csvPath = Join-Path $smokeDir "training_data64_smoke.csv"
$metaPath = Join-Path $smokeDir "training_data64_smoke_meta.json"
$tag = "training_data64_smoke"

New-Item -ItemType Directory -Path $smokeDir -Force | Out-Null

python (Join-Path $root "scripts\generate_training_data64.py") `
  --total-combos 100 `
  --current-a 0.01 `
  --output $csvPath `
  --meta-output $metaPath

python (Join-Path $root "gnn\GNN_CLS\modelo3\train.py") `
  --data-path $csvPath `
  --dataset-tag $tag `
  --epochs 1 `
  --patience 1

python (Join-Path $root "gnn\GNN_REG\o4a2\train.py") `
  --data-path $csvPath `
  --dataset-tag $tag `
  --epochs 1 `
  --patience 1

python (Join-Path $root "gnn\GNN_CMEI_INFERENCE\inference_gnn_cmei.py") `
  --data-path $csvPath `
  --dataset-tag $tag `
  --no-near-miss

Write-Host "Smoke run completed for $tag"
