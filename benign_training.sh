#!/usr/bin/env bash
set -euo pipefail

dataset=imagenette
classes=10
model=cnn
log_dir=logs
mkdir -p "$log_dir"

for pr in 0.003 0.05 0; do
  ts=$(date +"%Y%m%d_%H%M%S")

  # set -pb to 'benign' when pr == 0, else 'poison'
  if [[ "$pr" == "0" || "$pr" == "0.0" || "$pr" == "0.000" ]]; then
    poison='benign'
  else
    poison='poison'
  fi

  pr_tag=${pr//./p}
  log="$log_dir/train_${dataset}_class${classes}_${poison}_pr${pr_tag}_${model}_$ts.log"

  # everything in the braces (status line + python output) goes to the log
  {
    echo "===> $(date -Is) Running: -t $dataset -class $classes -pb $poison -pr $pr -mn $model"
    # -u makes Python unbuffered so logs stream line-by-line
    python -u train.py -t "$dataset" -class "$classes" -pb "$poison" -pr "$pr" -mn "$model"
  } 2>&1 | tee "$log"

  # optional: maintain a 'latest' symlink for quick access
  ln -sfn "$log" "$log_dir/latest_${poison}_pr${pr_tag}.log"
done
