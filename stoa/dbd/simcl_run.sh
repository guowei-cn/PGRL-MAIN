#!/usr/bin/env bash
set -euo pipefail

# --- Config ---
dataset=imagenette_freq_meg_500
classes=10
poison=poison
device=cuda:1
log_dir="logs"

mkdir -p "$log_dir"

ts() { date +"%Y%m%d_%H%M%S"; }


# pr = 0.003
log="${log_dir}/simclr_${dataset}_class${classes}_${poison}_pr0p003_$(ts).log"
python simclr.py -t "$dataset" -class "$classes" -pb "$poison" -pr 0.003 -d "$device" \
  | tee "$log"

# pr = 0.05
log="${log_dir}/simclr_${dataset}_class${classes}_${poison}_pr0p05_$(ts).log"
python simclr.py -t "$dataset" -class "$classes" -pb "$poison" -pr 0.05 -d "$device" \
  | tee "$log"
