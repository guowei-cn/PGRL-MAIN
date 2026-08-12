#!/usr/bin/env bash
set -euo pipefail

# Array of datasets (bash syntax, no '=' and no brackets)
datasets=(imagenette_freq_meg_500 imagenette_adaptivecifar10 )

classes=10
device="cuda:0"
poison="poison"
log_dir="logs"   # already exists

ts() { date +"%Y%m%d_%H%M%S"; }

for dataset in "${datasets[@]}"; do

    pr=0.05
    log="/storageA/david_projects/PGRL-main/${log_dir}/stoa_abl_${dataset}_class${classes}_${poison}_pr${pr//./p}_$(ts).log"
    python stoa_abl.py -t "$dataset" -class "$classes" -pb "$poison" -pr "$pr" -d "$device" 2>&1 | tee "$log"

    pr=0.003
    log="/storageA/david_projects/PGRL-main/${log_dir}/stoa_abl_${dataset}_class${classes}_${poison}_pr${pr//./p}_$(ts).log"
    python stoa_abl.py -t "$dataset" -class "$classes" -pb "$poison" -pr "$pr" -d "$device" 2>&1 | tee "$log"

done
