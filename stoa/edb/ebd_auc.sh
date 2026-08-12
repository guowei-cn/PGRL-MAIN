#!/bin/bash

python separate_samples_auc.py --dataset pattern --poison_rate 0.05 --cover_rate 0.0 --target_label 2 --model resnet18 --trigger_type gridTrigger --batch_size 1 --clean_ratio 0.20 --poison_ratio 0.05 --gamma_low 0.0 --gamma_high 0.11064413189888 --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans_ftsimi/pattern/resnet18/gridTrigger/9.tar


#python separate_samples_auc.py --dataset adaptivecifar10 --poison_rate 0.003 --target_label 2 --model resnet18 --trigger_type gridTrigger --batch_size 1 --clean_ratio 0.20 --poison_ratio 0.05 --gamma_low 0.0 --gamma_high 0.07910242676734924 --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans_ftsimi/adaptivecifar10/resnet18/gridTrigger/9.tar

python separate_samples_auc.py --dataset adaptivecifar10 --poison_rate 0.05 --cover_rate 0.05 --target_label 2 --model resnet18 --trigger_type gridTrigger --batch_size 1 --clean_ratio 0.20 --poison_ratio 0.05 --gamma_low 0.0 --gamma_high 0.11064413189888 --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans_ftsimi/adaptivecifar10/resnet18/gridTrigger/9.tar

#python separate_samples_auc.py --dataset ultrasonic --poison_rate 0.003 --target_label down --model resnet18 --trigger_type gridTrigger --batch_size 1 --clean_ratio 0.20 --poison_ratio 0.05 --gamma_low 0.0 --gamma_high 0.0 --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans_ftsimi/ultrasonic/resnet18/gridTrigger/9.tar

#python separate_samples_auc.py --dataset ultrasonic --poison_rate 0.05 --target_label down --model resnet18 --trigger_type gridTrigger --batch_size 1 --clean_ratio 0.20 --poison_ratio 0.05 --gamma_low 0.0 --gamma_high 0.01 --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans_ftsimi/ultrasonic/resnet18/gridTrigger/9.tar

#python separate_samples_auc.py --dataset freq --poison_rate 0.003 --target_label 0 --model resnet18 --trigger_type gridTrigger --batch_size 1 --clean_ratio 0.20 --poison_ratio 0.05 --gamma_low 5.771383371211414e-12 --gamma_high 0.008333366394042969 --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans_ftsimi/freq/resnet18/gridTrigger/9.tar

python separate_samples_auc.py --dataset freq --poison_rate 0.05 --target_label 0 --model resnet18 --trigger_type gridTrigger --batch_size 1 --clean_ratio 0.20 --poison_ratio 0.05 --gamma_low 0 --gamma_high 0.002101900987327099 --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans_ftsimi/freq/resnet18/gridTrigger/9.tar

#python separate_samples_auc.py --dataset freq --poison_rate 0.01 --target_label 0 --model resnet18 --trigger_type gridTrigger --batch_size 1 --clean_ratio 0.20 --poison_ratio 0.05 --gamma_low 4.847548916586675e-05 --gamma_high 5.244356155395508 --checkpoint_load ./saved/backdoored_model/poison_rate_0.01/noTrans_ftsimi/freq/resnet18/gridTrigger/9.tar
