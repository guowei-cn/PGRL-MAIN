python train_attack_noTrans.py --dataset adaptivecifar10 --poison_rate 0.003 --target_label 2 --model resnet18 --epochs 2

python finetune_attack_noTrans.py --dataset adaptivecifar10 --poison_rate 0.003 --target_label 2 --model resnet18 --trigger_type gridTrigger --epochs 10 --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans/adaptivecifar10/resnet18/gridTrigger/1.tar

python calculate_consistency.py --dataset adaptivecifar10 --poison_rate 0.003 --target_label 2 --model resnet18 --trigger_type gridTrigger --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans_ftsimi/adaptivecifar10/resnet18/gridTrigger/9.tar

python visualize_consistency.py --dataset adaptivecifar10 --poison_rate 0.003 --target_label 2 --model resnet18 --trigger_type gridTrigger --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans_ftsimi/adaptivecifar10/resnet18/gridTrigger/9.tar

python calculate_gamma.py --clean_ratio 0.20 --poison_ratio 0.05 --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans_ftsimi/adaptivecifar10/resnet18/gridTrigger/9.tar
# output of calculate_gamma.py is used in the command of separate_samples.py
python separate_samples.py --dataset adaptivecifar10 --poison_rate 0.003 --target_label 2 --model resnet18 --trigger_type gridTrigger --batch_size 1 --clean_ratio 0.20 --poison_ratio 0.05 --gamma_low 0.0 --gamma_high 0.07910242676734924 --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans_ftsimi/adaptivecifar10/resnet18/gridTrigger/9.tar

cd ST

python train_extractor.py --dataset adaptivecifar10 --poison_rate 0.003 --target_label 2 --model resnet18 --trigger_type gridTrigger --epochs 200 --learning_rate 0.5 --temp 0.1 --cosine --save_freq 20 --batch_size 512

python train_classifier.py --dataset adaptivecifar10 --poison_rate 0.003 --target_label 2 --model resnet18 --trigger_type gridTrigger --epochs 10 --learning_rate 5 --batch_size 512 --ckpt ./save/poison_rate_0.003/SupCon_models/adaptivecifar10/resnet18/gridTrigger_0.2_0.05/SupCon_adaptivecifar10_resnet18_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm/last.pth

# adaptivecifar10 0.05
python train_attack_noTrans.py --dataset adaptivecifar10 --poison_rate 0.05 --target_label 2 --model resnet18 --epochs 2

python finetune_attack_noTrans.py --dataset adaptivecifar10 --poison_rate 0.05 --target_label 2 --model resnet18 --trigger_type gridTrigger --epochs 10 --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans/adaptivecifar10/resnet18/gridTrigger/1.tar
python calculate_consistency.py --dataset adaptivecifar10 --poison_rate 0.05 --target_label 2 --model resnet18 --trigger_type gridTrigger --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans_ftsimi/adaptivecifar10/resnet18/gridTrigger/9.tar

python calculate_gamma.py --clean_ratio 0.20 --poison_ratio 0.05 --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans_ftsimi/adaptivecifar10/resnet18/gridTrigger/9.tar
# output of calculate_gamma.py is used in the command of separate_samples.py
python separate_samples.py --dataset adaptivecifar10 --poison_rate 0.05 --target_label 2 --model resnet18 --trigger_type gridTrigger --batch_size 1 --clean_ratio 0.20 --poison_ratio 0.05 --gamma_low 0.0 --gamma_high 0.11064413189888 --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans_ftsimi/adaptivecifar10/resnet18/gridTrigger/9.tar

cd ST

python train_extractor.py --dataset adaptivecifar10 --poison_rate 0.05 --target_label 2 --model resnet18 --trigger_type gridTrigger --epochs 200 --learning_rate 0.5 --temp 0.1 --cosine --save_freq 20 --batch_size 512

python train_classifier.py --dataset adaptivecifar10 --poison_rate 0.05 --target_label 2 --model resnet18 --trigger_type gridTrigger --epochs 10 --learning_rate 5 --batch_size 512 --ckpt ./save/poison_rate_0.05/SupCon_models/adaptivecifar10/resnet18/gridTrigger_0.2_0.05/SupCon_adaptivecifar10_resnet18_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm/last.pth


# blto 0.003
python train_attack_noTrans.py --dataset blto --poison_rate 0.003 --target_label 9 --model resnet18 --epochs 2

python finetune_attack_noTrans.py --dataset blto --poison_rate 0.003 --target_label 9 --model resnet18 --trigger_type gridTrigger --epochs 10 --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans/blto/resnet18/gridTrigger/1.tar

python calculate_consistency.py --dataset blto --poison_rate 0.003 --target_label 9 --model resnet18 --trigger_type gridTrigger --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans_ftsimi/blto/resnet18/gridTrigger/9.tar

python calculate_gamma.py --clean_ratio 0.20 --poison_ratio 0.05 --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans_ftsimi/blto/resnet18/gridTrigger/9.tar
#gamma_low:  0.0
#gamma_high:  7.660406112670898

python separate_samples.py --dataset blto --poison_rate 0.003 --target_label 9 --model resnet18 --trigger_type gridTrigger --batch_size 1 --clean_ratio 0.20 --poison_ratio 0.05 --gamma_low 0.0 --gamma_high 7.660406112670898 --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans_ftsimi/blto/resnet18/gridTrigger/9.tar

cd ST

python train_extractor.py --dataset blto --poison_rate 0.003 --target_label 9 --model resnet18 --trigger_type gridTrigger --epochs 200 --learning_rate 0.5 --temp 0.1 --cosine --save_freq 20 --batch_size 512

python train_classifier.py --dataset blto --poison_rate 0.003 --target_label 9 --model resnet18 --trigger_type gridTrigger --epochs 10 --learning_rate 5 --batch_size 512 --ckpt ./save/poison_rate_0.003/SupCon_models/blto/resnet18/gridTrigger_0.2_0.05/SupCon_blto_resnet18_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm/last.pth

# blto 0.05
python train_attack_noTrans.py --dataset blto --poison_rate 0.05 --target_label 9 --model resnet18 --epochs 2

python finetune_attack_noTrans.py --dataset blto --poison_rate 0.05 --target_label 9 --model resnet18 --trigger_type gridTrigger --epochs 10 --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans/blto/resnet18/gridTrigger/1.tar

python calculate_consistency.py --dataset blto --poison_rate 0.05 --target_label 9 --model resnet18 --trigger_type gridTrigger --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans_ftsimi/blto/resnet18/gridTrigger/9.tar

python calculate_gamma.py --clean_ratio 0.20 --poison_ratio 0.05 --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans_ftsimi/blto/resnet18/gridTrigger/9.tar
#gamma_low:  4.464893208933063e-05
#gamma_high:  10.564499855041504
python separate_samples.py --dataset blto --poison_rate 0.05 --target_label 9 --model resnet18 --trigger_type gridTrigger --batch_size 1 --clean_ratio 0.20 --poison_ratio 0.05 --gamma_low 4.464893208933063e-05 --gamma_high 10.564499855041504 --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans_ftsimi/blto/resnet18/gridTrigger/9.tar

cd ST

python train_extractor.py --dataset blto --poison_rate 0.05 --target_label 9 --model resnet18 --trigger_type gridTrigger --epochs 200 --learning_rate 0.5 --temp 0.1 --cosine --save_freq 20 --batch_size 512

python train_classifier.py --dataset blto --poison_rate 0.05 --target_label 9 --model resnet18 --trigger_type gridTrigger --epochs 10 --learning_rate 5 --batch_size 512 --ckpt ./save/poison_rate_0.05/SupCon_models/blto/resnet18/gridTrigger_0.2_0.05/SupCon_blto_resnet18_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm/last.pth

# ultrasonic 0.003
python train_attack_noTrans.py --dataset ultrasonic --poison_rate 0.003 --target_label down --model resnet18 --epochs 2

python finetune_attack_noTrans.py --dataset ultrasonic --poison_rate 0.003 --target_label down --model resnet18 --trigger_type gridTrigger --epochs 10 --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans/ultrasonic/resnet18/gridTrigger/1.tar

python calculate_consistency.py --dataset ultrasonic --poison_rate 0.003 --target_label down --model resnet18 --trigger_type gridTrigger --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans_ftsimi/ultrasonic/resnet18/gridTrigger/9.tar

python calculate_gamma.py --clean_ratio 0.20 --poison_ratio 0.05 --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans_ftsimi/ultrasonic/resnet18/gridTrigger/9.tar
#gamma_low:  0.0
#gamma_high:  0.0
python separate_samples.py --dataset ultrasonic --poison_rate 0.003 --target_label down --model resnet18 --trigger_type gridTrigger --batch_size 1 --clean_ratio 0.20 --poison_ratio 0.05 --gamma_low 0.0 --gamma_high 0.0 --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans_ftsimi/ultrasonic/resnet18/gridTrigger/9.tar

cd ST

python train_extractor.py --dataset ultrasonic --poison_rate 0.003 --target_label down --model resnet18 --trigger_type gridTrigger --epochs 200 --learning_rate 0.5 --temp 0.1 --cosine --save_freq 20 --batch_size 512

python train_classifier.py --dataset ultrasonic --poison_rate 0.003 --target_label down --model resnet18 --trigger_type gridTrigger --epochs 10 --learning_rate 5 --batch_size 512 --ckpt ./save/poison_rate_0.003/SupCon_models/ultrasonic/resnet18/gridTrigger_0.2_0.05/SupCon_ultrasonic_resnet18_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm/last.pth


# ultrasonic 0.05
python train_attack_noTrans.py --dataset ultrasonic --poison_rate 0.05 --target_label down --model resnet18 --epochs 2

python finetune_attack_noTrans.py --dataset ultrasonic --poison_rate 0.05 --target_label down --model resnet18 --trigger_type gridTrigger --epochs 10 --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans/ultrasonic/resnet18/gridTrigger/1.tar

python calculate_consistency.py --dataset ultrasonic --poison_rate 0.05 --target_label down --model resnet18 --trigger_type gridTrigger --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans_ftsimi/ultrasonic/resnet18/gridTrigger/9.tar

python calculate_gamma.py --clean_ratio 0.20 --poison_ratio 0.05 --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans_ftsimi/ultrasonic/resnet18/gridTrigger/9.tar

python separate_samples.py --dataset ultrasonic --poison_rate 0.05 --target_label down --model resnet18 --trigger_type gridTrigger --batch_size 1 --clean_ratio 0.20 --poison_ratio 0.05 --gamma_low 0.0 --gamma_high 0.01 --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans_ftsimi/ultrasonic/resnet18/gridTrigger/9.tar

# freq 0.003
python train_attack_noTrans.py --dataset freq --poison_rate 0.003 --target_label 0 --model resnet18 --epochs 2

python finetune_attack_noTrans.py --dataset freq --poison_rate 0.003 --target_label 0 --model resnet18 --trigger_type gridTrigger --epochs 10 --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans/freq/resnet18/gridTrigger/1.tar

python calculate_consistency.py --dataset freq --poison_rate 0.003 --target_label 0 --model resnet18 --trigger_type gridTrigger --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans_ftsimi/freq/resnet18/gridTrigger/9.tar

python calculate_gamma.py --clean_ratio 0.20 --poison_ratio 0.05 --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans_ftsimi/freq/resnet18/gridTrigger/9.tar
gamma_low:  5.771383371211414e-12
gamma_high:  0.008333366394042969
python separate_samples.py --dataset freq --poison_rate 0.003 --target_label 0 --model resnet18 --trigger_type gridTrigger --batch_size 1 --clean_ratio 0.20 --poison_ratio 0.05 --gamma_low 5.771383371211414e-12 --gamma_high 0.008333366394042969 --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans_ftsimi/freq/resnet18/gridTrigger/9.tar

cd ST

python train_extractor.py --dataset freq --poison_rate 0.003 --target_label 0 --model resnet18 --trigger_type gridTrigger --epochs 200 --learning_rate 0.5 --temp 0.1 --cosine --save_freq 20 --batch_size 512

python train_classifier.py --dataset freq --poison_rate 0.003 --target_label 0 --model resnet18 --trigger_type gridTrigger --epochs 10 --learning_rate 5 --batch_size 512 --ckpt ./save/poison_rate_0.003/SupCon_models/freq/resnet18/gridTrigger_0.2_0.05/SupCon_freq_resnet18_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm/last.pth

# freq 0.05
python train_attack_noTrans.py --dataset freq --poison_rate 0.05 --target_label 0 --model resnet18 --epochs 2

python finetune_attack_noTrans.py --dataset freq --poison_rate 0.05 --target_label 0 --model resnet18 --trigger_type gridTrigger --epochs 10 --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans/freq/resnet18/gridTrigger/1.tar

python calculate_consistency.py --dataset freq --poison_rate 0.05 --target_label 0 --model resnet18 --trigger_type gridTrigger --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans_ftsimi/freq/resnet18/gridTrigger/9.tar

python calculate_gamma.py --clean_ratio 0.20 --poison_ratio 0.05 --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans_ftsimi/freq/resnet18/gridTrigger/9.tar
gamma_low:  0.0
gamma_high:  0.002101900987327099
python separate_samples.py --dataset freq --poison_rate 0.05 --target_label 0 --model resnet18 --trigger_type gridTrigger --batch_size 1 --clean_ratio 0.20 --poison_ratio 0.05 --gamma_low 0 --gamma_high 0.002101900987327099 --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans_ftsimi/freq/resnet18/gridTrigger/9.tar

cd ST

python train_extractor.py --dataset freq --poison_rate 0.05 --target_label 0 --model resnet18 --trigger_type gridTrigger --epochs 200 --learning_rate 0.5 --temp 0.1 --cosine --save_freq 20 --batch_size 512

python train_classifier.py --dataset freq --poison_rate 0.05 --target_label 0 --model resnet18 --trigger_type gridTrigger --epochs 10 --learning_rate 5 --batch_size 512 --ckpt ./save/poison_rate_0.05/SupCon_models/freq/resnet18/gridTrigger_0.2_0.05/SupCon_freq_resnet18_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm/last.pth






# freq 0.01
python train_attack_noTrans.py --dataset freq --poison_rate 0.01 --target_label 0 --model resnet18 --epochs 2

python finetune_attack_noTrans.py --dataset freq --poison_rate 0.01 --target_label 0 --model resnet18 --trigger_type gridTrigger --epochs 10 --checkpoint_load ./saved/backdoored_model/poison_rate_0.01/noTrans/freq/resnet18/gridTrigger/1.tar

python calculate_consistency.py --dataset freq --poison_rate 0.01 --target_label 0 --model resnet18 --trigger_type gridTrigger --checkpoint_load ./saved/backdoored_model/poison_rate_0.01/noTrans_ftsimi/freq/resnet18/gridTrigger/9.tar

python calculate_gamma.py --clean_ratio 0.20 --poison_ratio 0.05 --checkpoint_load ./saved/backdoored_model/poison_rate_0.01/noTrans_ftsimi/freq/resnet18/gridTrigger/9.tar
gamma_low:  4.847548916586675e-05
gamma_high:  5.244356155395508
python separate_samples.py --dataset freq --poison_rate 0.01 --target_label 0 --model resnet18 --trigger_type gridTrigger --batch_size 1 --clean_ratio 0.20 --poison_ratio 0.05 --gamma_low 4.847548916586675e-05 --gamma_high 5.244356155395508 --checkpoint_load ./saved/backdoored_model/poison_rate_0.01/noTrans_ftsimi/freq/resnet18/gridTrigger/9.tar

cd ST

python train_extractor.py --dataset freq --poison_rate 0.01 --target_label 0 --model resnet18 --trigger_type gridTrigger --epochs 200 --learning_rate 0.5 --temp 0.1 --cosine --save_freq 20 --batch_size 512

python train_classifier.py --dataset freq --poison_rate 0.01 --target_label 0 --model resnet18 --trigger_type gridTrigger --epochs 10 --learning_rate 5 --batch_size 512 --ckpt ./save/poison_rate_0.01/SupCon_models/freq/resnet18/gridTrigger_0.2_0.05/SupCon_freq_resnet18_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm/ckpt_epoch_139_knn_acc_83.980_back_acc_65.530.pth


# freq 0.04
python train_attack_noTrans.py --dataset freq --poison_rate 0.04 --target_label 0 --model resnet18 --epochs 2

python finetune_attack_noTrans.py --dataset freq --poison_rate 0.04 --target_label 0 --model resnet18 --trigger_type gridTrigger --epochs 10 --checkpoint_load ./saved/backdoored_model/poison_rate_0.04/noTrans/freq/resnet18/gridTrigger/1.tar

python calculate_consistency.py --dataset freq --poison_rate 0.04 --target_label 0 --model resnet18 --trigger_type gridTrigger --checkpoint_load ./saved/backdoored_model/poison_rate_0.04/noTrans_ftsimi/freq/resnet18/gridTrigger/9.tar

python calculate_gamma.py --clean_ratio 0.20 --poison_ratio 0.05 --checkpoint_load ./saved/backdoored_model/poison_rate_0.04/noTrans_ftsimi/freq/resnet18/gridTrigger/9.tar
gamma_low:  0.0
gamma_high:  2.904647350311279
python separate_samples.py --dataset freq --poison_rate 0.04 --target_label 0 --model resnet18 --trigger_type gridTrigger --batch_size 1 --clean_ratio 0.20 --poison_ratio 0.05 --gamma_low 0.0 --gamma_high 2.904647350311279 --checkpoint_load ./saved/backdoored_model/poison_rate_0.04/noTrans_ftsimi/freq/resnet18/gridTrigger/9.tar

cd ST

python train_extractor.py --dataset freq --poison_rate 0.04 --target_label 0 --model resnet18 --trigger_type gridTrigger --epochs 200 --learning_rate 0.5 --temp 0.1 --cosine --save_freq 20 --batch_size 512

python train_classifier.py --dataset freq --poison_rate 0.04 --target_label 0 --model resnet18 --trigger_type gridTrigger --epochs 10 --learning_rate 5 --batch_size 512 --ckpt ./save/poison_rate_0.01/SupCon_models/freq/resnet18/gridTrigger_0.2_0.05/SupCon_freq_resnet18_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm/ckpt_epoch_139_knn_acc_83.980_back_acc_65.530.pth


















# freq_meg_2000 0.003
python train_attack_noTrans.py --dataset freq_meg_2000 --poison_rate 0.003 --target_label 0 --model resnet18 --epochs 2

python finetune_attack_noTrans.py --dataset freq_meg_2000 --poison_rate 0.003 --target_label 0 --model resnet18 --trigger_type gridTrigger --epochs 10 --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans/freq_meg_2000/resnet18/gridTrigger/1.tar

python calculate_consistency.py --dataset freq_meg_2000 --poison_rate 0.003 --target_label 0 --model resnet18 --trigger_type gridTrigger --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans_ftsimi/freq_meg_2000/resnet18/gridTrigger/9.tar

python calculate_gamma.py --clean_ratio 0.20 --poison_ratio 0.05 --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans_ftsimi/freq_meg_2000/resnet18/gridTrigger/9.tar
gamma_low:  1.1913854223166709e-06
gamma_high:  0.9868118762969971
python separate_samples.py --dataset freq_meg_2000 --poison_rate 0.003 --target_label 0 --model resnet18 --trigger_type gridTrigger --batch_size 1 --clean_ratio 0.20 --poison_ratio 0.05 --gamma_low 1.1913854223166709e-06 --gamma_high 0.9868118762969971 --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans_ftsimi/freq_meg_2000/resnet18/gridTrigger/9.tar

cd ST

python train_extractor.py --dataset freq_meg_2000 --poison_rate 0.003 --target_label 0 --model resnet18 --trigger_type gridTrigger --epochs 200 --learning_rate 0.5 --temp 0.1 --cosine --save_freq 20 --batch_size 512

python train_classifier.py --dataset freq_meg_2000 --poison_rate 0.003 --target_label 0 --model resnet18 --trigger_type gridTrigger --epochs 10 --learning_rate 5 --batch_size 512 --ckpt ./save/poison_rate_0.003/SupCon_models/freq_meg_2000/resnet18/gridTrigger_0.2_0.05/SupCon_freq_resnet18_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm/last.pth

# freq_meg_2000 0.05
python train_attack_noTrans.py --dataset freq_meg_2000 --poison_rate 0.05 --target_label 0 --model resnet18 --epochs 2

python finetune_attack_noTrans.py --dataset freq_meg_2000 --poison_rate 0.05 --target_label 0 --model resnet18 --trigger_type gridTrigger --epochs 10 --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans/freq_meg_2000/resnet18/gridTrigger/1.tar

python calculate_consistency.py --dataset freq_meg_2000 --poison_rate 0.05 --target_label 0 --model resnet18 --trigger_type gridTrigger --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans_ftsimi/freq_meg_2000/resnet18/gridTrigger/9.tar

python calculate_gamma.py --clean_ratio 0.20 --poison_ratio 0.05 --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans_ftsimi/freq_meg_2000/resnet18/gridTrigger/9.tar
gamma_low:  4.0645309340447966e-10
gamma_high:  13.528178215026855
python separate_samples.py --dataset freq_meg_2000 --poison_rate 0.05 --target_label 0 --model resnet18 --trigger_type gridTrigger --batch_size 1 --clean_ratio 0.20 --poison_ratio 0.05 --gamma_low 4.0645309340447966e-10 --gamma_high 13.528178215026855 --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans_ftsimi/freq_meg_2000/resnet18/gridTrigger/9.tar

cd ST

python train_extractor.py --dataset freq_meg_2000 --poison_rate 0.05 --target_label 0 --model resnet18 --trigger_type gridTrigger --epochs 200 --learning_rate 0.5 --temp 0.1 --cosine --save_freq 20 --batch_size 512

python train_classifier.py --dataset freq_meg_2000 --poison_rate 0.05 --target_label 0 --model resnet18 --trigger_type gridTrigger --epochs 10 --learning_rate 5 --batch_size 512 --ckpt ./save/poison_rate_0.05/SupCon_models/freq_meg_2000/resnet18/gridTrigger_0.2_0.05/SupCon_freq_resnet18_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm/last.pth

# pattern 0.003
python train_attack_noTrans.py --dataset pattern --poison_rate 0.003 --target_label 2 --model resnet18 --epochs 2
python finetune_attack_noTrans.py --dataset pattern --poison_rate 0.003 --target_label 2 --model resnet18 --trigger_type gridTrigger --epochs 10 --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans/pattern/resnet18/gridTrigger/1.tar
python calculate_consistency.py --dataset pattern --poison_rate 0.003 --target_label 2 --model resnet18 --trigger_type gridTrigger --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans_ftsimi/pattern/resnet18/gridTrigger/9.tar
python calculate_gamma.py --clean_ratio 0.20 --poison_ratio 0.05 --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans_ftsimi/pattern/resnet18/gridTrigger/9.tar
gamma_low:  1.6782316379249096e-05
gamma_high:  15.90577507019043
python separate_samples.py --dataset pattern --poison_rate 0.003 --target_label 2 --model resnet18 --trigger_type gridTrigger --batch_size 1 --clean_ratio 0.20 --poison_ratio 0.05 --gamma_low 1.6782316379249096e-05 --gamma_high 15.90577507019043 --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans_ftsimi/pattern/resnet18/gridTrigger/9.tar
cd ST
python train_extractor.py --dataset pattern --poison_rate 0.003 --target_label 2 --model resnet18 --trigger_type gridTrigger --epochs 200 --learning_rate 0.5 --temp 0.1 --cosine --save_freq 20 --batch_size 512
python train_classifier.py --dataset pattern --poison_rate 0.003 --target_label 2 --model resnet18 --trigger_type gridTrigger --epochs 10 --learning_rate 5 --batch_size 512 --ckpt ./save/poison_rate_0.003/SupCon_models/pattern/resnet18/gridTrigger_0.2_0.05/SupCon_pattern_resnet18_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm/last.pth

# pattern 0.05
python train_attack_noTrans.py --dataset pattern --poison_rate 0.05 --target_label 2 --model resnet18 --epochs 2
python finetune_attack_noTrans.py --dataset pattern --poison_rate 0.05 --target_label 2 --model resnet18 --trigger_type gridTrigger --epochs 10 --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans/pattern/resnet18/gridTrigger/1.tar
python calculate_consistency.py --dataset pattern --poison_rate 0.05 --target_label 2 --model resnet18 --trigger_type gridTrigger --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans_ftsimi/pattern/resnet18/gridTrigger/9.tar
python calculate_gamma.py --clean_ratio 0.20 --poison_ratio 0.05 --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans_ftsimi/pattern/resnet18/gridTrigger/9.tar
gamma_low:  0.0
gamma_high:  0.30285775661468506
python separate_samples.py --dataset pattern --poison_rate 0.05 --target_label 2 --model resnet18 --trigger_type gridTrigger --batch_size 1 --clean_ratio 0.20 --poison_ratio 0.05 --gamma_low 0.0 --gamma_high 0.30285775661468506 --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans_ftsimi/pattern/resnet18/gridTrigger/9.tar
cd ST
python train_extractor.py --dataset pattern --poison_rate 0.05 --target_label 2 --model resnet18 --trigger_type gridTrigger --epochs 200 --learning_rate 0.5 --temp 0.1 --cosine --save_freq 20 --batch_size 512
python train_classifier.py --dataset pattern --poison_rate 0.05 --target_label 2 --model resnet18 --trigger_type gridTrigger --epochs 10 --learning_rate 5 --batch_size 512 --ckpt ./save/poison_rate_0.05/SupCon_models/pattern/resnet18/gridTrigger_0.2_0.05/SupCon_pattern_resnet18_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm/last.pth

# wanet 0.003
python train_attack_noTrans.py --dataset wanet --poison_rate 0.003 --target_label 2 --model resnet18 --epochs 2
python finetune_attack_noTrans.py --dataset wanet --poison_rate 0.003 --target_label 2 --model resnet18 --trigger_type gridTrigger --epochs 10 --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans/wanet/resnet18/gridTrigger/1.tar
python calculate_consistency.py --dataset wanet --poison_rate 0.003 --target_label 2 --model resnet18 --trigger_type gridTrigger --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans_ftsimi/wanet/resnet18/gridTrigger/9.tar
python calculate_gamma.py --clean_ratio 0.20 --poison_ratio 0.05 --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans_ftsimi/wanet/resnet18/gridTrigger/9.tar
gamma_low:  4.8783902457216755e-05
gamma_high:  9.361274719238281
python separate_samples.py --dataset wanet --poison_rate 0.003 --target_label 2 --model resnet18 --trigger_type gridTrigger --batch_size 1 --clean_ratio 0.20 --poison_ratio 0.05 --gamma_low 4.8783902457216755e-05 --gamma_high 9.361274719238281 --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans_ftsimi/wanet/resnet18/gridTrigger/9.tar
cd ST
python train_extractor.py --dataset wanet --poison_rate 0.003 --target_label 2 --model resnet18 --trigger_type gridTrigger --epochs 200 --learning_rate 0.5 --temp 0.1 --cosine --save_freq 20 --batch_size 512
python train_classifier.py --dataset pattern --poison_rate 0.003 --target_label 2 --model resnet18 --trigger_type gridTrigger --epochs 10 --learning_rate 5 --batch_size 512 --ckpt ./save/poison_rate_0.003/SupCon_models/wanet/resnet18/gridTrigger_0.2_0.05/SupCon_wanet_resnet18_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm/last.pth

# wanet 0.05
python train_attack_noTrans.py --dataset wanet --poison_rate 0.05 --target_label 2 --model resnet18 --epochs 2
python finetune_attack_noTrans.py --dataset wanet --poison_rate 0.05 --target_label 2 --model resnet18 --trigger_type gridTrigger --epochs 10 --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans/wanet/resnet18/gridTrigger/1.tar
python calculate_consistency.py --dataset wanet --poison_rate 0.05 --target_label 2 --model resnet18 --trigger_type gridTrigger --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans_ftsimi/wanet/resnet18/gridTrigger/9.tar
python calculate_gamma.py --clean_ratio 0.20 --poison_ratio 0.05 --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans_ftsimi/wanet/resnet18/gridTrigger/9.tar
gamma_low:  0.0
gamma_high:  0.004730707500129938
python separate_samples.py --dataset wanet --poison_rate 0.05 --target_label 2 --model resnet18 --trigger_type gridTrigger --batch_size 1 --clean_ratio 0.20 --poison_ratio 0.05 --gamma_low 0.0 --gamma_high 0.004730707500129938 --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans_ftsimi/wanet/resnet18/gridTrigger/9.tar
cd ST
python train_extractor.py --dataset wanet --poison_rate 0.05 --target_label 2 --model resnet18 --trigger_type gridTrigger --epochs 200 --learning_rate 0.5 --temp 0.1 --cosine --save_freq 20 --batch_size 512
python train_classifier.py --dataset pattern --poison_rate 0.05 --target_label 2 --model resnet18 --trigger_type gridTrigger --epochs 10 --learning_rate 5 --batch_size 512 --ckpt ./save/poison_rate_0.05/SupCon_models/wanet/resnet18/gridTrigger_0.2_0.05/SupCon_wanet_resnet18_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm/last.pth


# freq_meg_500 0.003
python train_attack_noTrans.py --dataset freq_meg_500 --poison_rate 0.003 --target_label 0 --model resnet18 --epochs 2
python finetune_attack_noTrans.py --dataset freq_meg_500 --poison_rate 0.003 --target_label 0 --model resnet18 --trigger_type gridTrigger --epochs 10 --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans/freq_meg_500/resnet18/gridTrigger/1.tar
python calculate_consistency.py --dataset freq_meg_500 --poison_rate 0.003 --target_label 0 --model resnet18 --trigger_type gridTrigger --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans_ftsimi/freq_meg_500/resnet18/gridTrigger/9.tar
python calculate_gamma.py --clean_ratio 0.20 --poison_ratio 0.05 --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans_ftsimi/freq_meg_500/resnet18/gridTrigger/9.tar
gamma_low:  0.0008487682789564133
gamma_high:  3.1480712890625
python separate_samples.py --dataset freq_meg_500 --poison_rate 0.05 --target_label 0 --model resnet18 --trigger_type gridTrigger --batch_size 1 --clean_ratio 0.20 --poison_ratio 0.05 --gamma_low 0.0008487682789564133 --gamma_high 3.1480712890625 --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans_ftsimi/freq_meg_500/resnet18/gridTrigger/9.tar

# freq_meg_500 0.05
python train_attack_noTrans.py --dataset freq_meg_500 --poison_rate 0.05 --target_label 0 --model resnet18 --epochs 2
python finetune_attack_noTrans.py --dataset freq_meg_500 --poison_rate 0.05 --target_label 0 --model resnet18 --trigger_type gridTrigger --epochs 10 --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans/freq_meg_500/resnet18/gridTrigger/1.tar
python calculate_consistency.py --dataset freq_meg_500 --poison_rate 0.05 --target_label 0 --model resnet18 --trigger_type gridTrigger --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans_ftsimi/freq_meg_500/resnet18/gridTrigger/9.tar
python calculate_gamma.py --clean_ratio 0.20 --poison_ratio 0.05 --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans_ftsimi/freq_meg_500/resnet18/gridTrigger/9.tar
gamma_low:  0.0
gamma_high:  0.18553170561790466
python separate_samples.py --dataset freq_meg_500 --poison_rate 0.05 --target_label 0 --model resnet18 --trigger_type gridTrigger --batch_size 1 --clean_ratio 0.20 --poison_ratio 0.05 --gamma_low 0.0 --gamma_high 0.18553170561790466 --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans_ftsimi/freq_meg_500/resnet18/gridTrigger/9.tar

cd ST
python train_extractor.py --dataset freq_meg_500 --poison_rate 0.05 --target_label 0 --model resnet18 --trigger_type gridTrigger --epochs 200 --learning_rate 0.5 --temp 0.1 --cosine --save_freq 20 --batch_size 512
python train_classifier.py --dataset freq_meg_500 --poison_rate 0.05 --target_label 0 --model resnet18 --trigger_type gridTrigger --epochs 10 --learning_rate 5 --batch_size 512 --ckpt ./save/poison_rate_0.05/SupCon_models/freq_meg_500/resnet18/gridTrigger_0.2_0.05/SupCon_freq_meg_500_resnet18_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm/last.pth


# imagenette 0.003
python train_attack_noTrans.py --dataset imagenette --poison_rate 0.003 --target_label 0 --model resnet18 --epochs 2
python finetune_attack_noTrans.py --dataset imagenette --poison_rate 0.003 --target_label 0 --model resnet18 --trigger_type gridTrigger --epochs 10 --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans/imagenette/resnet18/gridTrigger/1.tar
python calculate_consistency.py --dataset imagenette --poison_rate 0.003 --target_label 0 --model resnet18 --trigger_type gridTrigger --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans_ftsimi/imagenette/resnet18/gridTrigger/9.tar
python calculate_gamma.py --clean_ratio 0.20 --poison_ratio 0.05 --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans_ftsimi/imagenette/resnet18/gridTrigger/9.tar
gamma_low:  0.003022143617272377
gamma_high:  4.798514366149902
python separate_samples.py --dataset imagenette --poison_rate 0.003 --target_label 0 --model resnet18 --trigger_type gridTrigger --batch_size 1 --clean_ratio 0.20 --poison_ratio 0.05 --gamma_low 0.003022143617272377 --gamma_high 4.798514366149902 --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans_ftsimi/imagenette/resnet18/gridTrigger/9.tar
cd ST
python train_extractor.py --dataset imagenette --poison_rate 0.003 --target_label 0 --model resnet18 --trigger_type gridTrigger --epochs 200 --learning_rate 0.5 --temp 0.1 --cosine --save_freq 20 --batch_size 512
python train_classifier.py --dataset imagenette --poison_rate 0.003 --target_label 0 --model resnet18 --trigger_type gridTrigger --epochs 10 --learning_rate 5 --batch_size 512 --ckpt ./save/poison_rate_0.003/SupCon_models/imagenette/resnet18/gridTrigger_0.2_0.05/SupCon_imagenette_resnet18_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm/last.pth

# imagenette 0.05
python train_attack_noTrans.py --dataset imagenette --poison_rate 0.05 --target_label 0 --model resnet18 --epochs 2
python finetune_attack_noTrans.py --dataset imagenette --poison_rate 0.05 --target_label 0 --model resnet18 --trigger_type gridTrigger --epochs 10 --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans/imagenette/resnet18/gridTrigger/1.tar
python calculate_consistency.py --dataset imagenette --poison_rate 0.05 --target_label 0 --model resnet18 --trigger_type gridTrigger --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans_ftsimi/imagenette/resnet18/gridTrigger/9.tar
python calculate_gamma.py --clean_ratio 0.20 --poison_ratio 0.05 --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans_ftsimi/imagenette/resnet18/gridTrigger/9.tar
gamma_low:  0.004198372829705477
gamma_high:  5.340563774108887
python separate_samples.py --dataset imagenette --poison_rate 0.05 --target_label 0 --model resnet18 --trigger_type gridTrigger --batch_size 1 --clean_ratio 0.20 --poison_ratio 0.05 --gamma_low 0.004198372829705477 --gamma_high 5.340563774108887 --checkpoint_load ./saved/backdoored_model/poison_rate_0.05/noTrans_ftsimi/imagenette/resnet18/gridTrigger/9.tar
cd ST
python train_extractor.py --dataset imagenette --poison_rate 0.05 --target_label 0 --model resnet18 --trigger_type gridTrigger --epochs 200 --learning_rate 0.5 --temp 0.1 --cosine --save_freq 20 --batch_size 512
python train_classifier.py --dataset imagenette --poison_rate 0.05 --target_label 0 --model resnet18 --trigger_type gridTrigger --epochs 10 --learning_rate 5 --batch_size 512 --ckpt ./save/poison_rate_0.05/SupCon_models/imagenette/resnet18/gridTrigger_0.2_0.05/SupCon_imagenette_resnet18_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm/last.pth


# imagenattee_freq_meg_500
python train_attack_noTrans.py --dataset imagenette_freq_meg_500 --poison_rate 0.003 --target_label 0 --model resnet18 --epochs 2
python finetune_attack_noTrans.py --dataset imagenette_freq_meg_500 --poison_rate 0.003 --target_label 0 --model resnet18 --trigger_type gridTrigger --epochs 10 --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans/imagenette_freq_meg_500/resnet18/gridTrigger/1.tar
python calculate_consistency.py --dataset imagenette_freq_meg_500 --poison_rate 0.003 --target_label 0 --model resnet18 --trigger_type gridTrigger --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans_ftsimi/imagenette_freq_meg_500/resnet18/gridTrigger/9.tar
python calculate_gamma.py --clean_ratio 0.20 --poison_ratio 0.003 --checkpoint_load ./saved/backdoored_model/poison_rate_0.003/noTrans_ftsimi/imagenette_freq_meg_500/resnet18/gridTrigger/9.tar
