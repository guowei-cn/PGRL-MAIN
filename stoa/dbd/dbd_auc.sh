#!/usr/bin/env bash
set -euo pipefail

# --- Config ---
dataset=imagenette_freq_meg_500
classes=10
poison=poison
device=cuda:0
log_dir="logs"

mkdir -p "$log_dir"

ts() { date +"%Y%m%d_%H%M%S"; }
master_log="../../$log_dir/dbd_run_$(ts).log"

run_and_log() {
  local script="$1"
  local pr="$2"
  local pr_tag="${pr//./p}"
  local base="${script%.py}_${dataset}_class${classes}_${poison}_pr${pr_tag}_$(ts)"
  local log="$log_dir/${base}.log"

  echo "[$(date +'%F %T')] START: $script (pr=$pr)" | tee -a "$master_log"
  echo "CMD: python $script -t $dataset -class $classes -pb $poison -pr $pr -d $device" | tee -a "$master_log"

  # stdbuf ensures line-buffered output so logs update in real time
  stdbuf -oL -eL python "$script" \
    -t "$dataset" -class "$classes" -pb "$poison" -pr "$pr" -d "$device" 2>&1 \
    | tee -a "$master_log" | tee "$log"

  echo "[$(date +'%F %T')] DONE : $script (pr=$pr) -> $log" | tee -a "$master_log"
  echo
}

#run_and_log "simclr.py" 0.003
run_and_log "dbd_auc.py"    0.003
#run_and_log "simclr.py" 0.05
run_and_log "dbd_auc.py"    0.05
