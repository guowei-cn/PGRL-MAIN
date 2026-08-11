#!/usr/bin/env bash
set -euo pipefail

script="our_method_with_OT_emb_as_act.py"

dataset="imagenette"
classes=10
poison="poison"
sample=10
aug_n=6
device="cuda:1"

log_dir="logs"
mkdir -p "$log_dir"

# two pr values to run
prs=(0.05 0.003)
xloss_l=("lcv" "fde")

for pr in "${prs[@]}"; do
  for xloss in "${xloss_l[@]}"; do
    ts="$(date +"%Y%m%d_%H%M%S")"
    # replace '.' with 'p' for a filesystem-friendly tag: 0.003 -> 0p003
    pr_tag="${pr//./p}"
    dev_tag="$(echo "$device" | tr ':' '-')"

    log_file="${log_dir}/$(basename "$script" .py)_t${dataset}_class${classes}_pb${poison}_pr${pr_tag}_sample${sample}_aug${aug_n}_xloss${xloss}_d${dev_tag}_${ts}.log"

    echo "===> Running: python $script -t $dataset -class $classes -pb $poison -pr $pr -sample $sample -aug_n $aug_n -xloss $xloss -d $device"
    echo "===> Logging to: $log_file"

    # run and capture both stdout and stderr
    PYTHONUNBUFFERED=1 python "$script" \
      -t "$dataset" \
      -class "$classes" \
      -pb "$poison" \
      -pr "$pr" \
      -sample "$sample" \
      -aug_n "$aug_n" \
      -xloss "$xloss" \
      -d "$device" \
      2>&1 | tee "$log_file"
  done
done