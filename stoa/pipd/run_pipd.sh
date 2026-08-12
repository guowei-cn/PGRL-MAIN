#!/usr/bin/env bash
set -euo pipefail

datasets=(imagenette_adaptivecifar10)
classes=10
device=cuda:0
poison=poison
log_dir=logs   # already exists

# Ensure the absolute log path exists (safe even if it already does)
mkdir -p "/storageA/david_projects/PGRL-main/${log_dir}"

ts() { date +"%Y%m%d_%H%M%S"; }

run() {
  local pr="$1"
  local log="/storageA/david_projects/PGRL-main/${log_dir}/stoa_pipd_${dataset}_class${classes}_${poison}_pr${pr//./p}_$(ts).log"
  echo "===> $(date) running dataset=${dataset} pr=${pr} | logging to: $log"

  PYTHONUNBUFFERED=1 stdbuf -oL -eL \
    python -u stoa_pipd.py \
      -t "$dataset" -class "$classes" -pb "$poison" -pr "$pr" -d "$device" \
    2>&1 | tee "$log"
}

for dataset in "${datasets[@]}"; do
  run 0.05
  run 0.003
done
