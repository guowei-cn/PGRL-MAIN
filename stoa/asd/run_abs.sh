#!/usr/bin/env bash
set -euo pipefail

datasets=(imagenette_freq_meg_500 imagenette_adaptivecifar10 )
classes=10
device=cuda:0
poison=poison
log_dir=logs   # already exists

ts() { date +"%Y%m%d_%H%M%S"; }

dataset=${datasets[0]}
pr=0.05
log="/storageA/david_projects/PGRL-main/${log_dir}/stoa_asd_${dataset}_class${classes}_${poison}_pr${pr//./p}_$(ts).log"
python ASD.py --config "./config/${dataset}_pr_0.05_seed_num_10.yaml" 2>&1 | tee "$log"

dataset=${datasets[1]}
pr=0.003
log="/storageA/david_projects/PGRL-main/${log_dir}/stoa_asd_${dataset}_class${classes}_${poison}_pr${pr//./p}_$(ts).log"
python ASD.py --config "./config/${dataset}_pr_0.003_seed_num_10.yaml" 2>&1 | tee "$log"

pr=0.05
log="/storageA/david_projects/PGRL-main/${log_dir}/stoa_asd_${dataset}_class${classes}_${poison}_pr${pr//./p}_$(ts).log"
python ASD.py --config "./config/${dataset}_pr_0.05_seed_num_10.yaml" 2>&1 | tee "$log"
