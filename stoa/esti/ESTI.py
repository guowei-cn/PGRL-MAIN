import os, sys

home_folder = os.path.join(os.getcwd().split('PGRL-main')[0], 'PGRL-main')
sys.path.append(home_folder)

import torch
import csv


import dataset
import predict

import split
import train
from model_config import model_config
from torch.utils.data import ConcatDataset
import argparse
import utils
import os
import shutil

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='./config/baseline_esti.yaml', help='Path to the config file.')
opts = parser.parse_args()

config = utils.get_config(opts.config)
model_name = config['model_name']
num_workers = config['num_workers']
seed_sample_ratio = config['seed_sample_ratio']
dual_model_iter = config['dual_model_iter']
trap_model_iter = config['trap_model_iter']
dataset_root = config['dataset_root']
save_root = config['save_root']
dataset_name = config['dataset_name']
random_seed = config['random_seed']
attack_type_list = config['attack_type_list']
log_dir = config['log_dir']
attack_type = attack_type_list[0]

if not os.path.exists(save_root):
    os.makedirs(save_root)
config_filename = os.path.basename(opts.config)
shutil.copy(opts.config, os.path.join(save_root, config_filename))
logger = utils.setup_logger(log_dir, attack_type)

for key, value in config.items():
    logger.info(f"{key}: {value}")

try:
    assert dual_model_iter >= 2, f"dual_model_iter ({dual_model_iter}) must be greater than 2"
except AssertionError as e:
    logger.error(e)
    raise e
try:
    assert trap_model_iter >= 1, f"trap_model_iter ({trap_model_iter}) must be greater than 1"
except AssertionError as e:
    logger.error(e)
    raise e


if random_seed is not None:
    utils.fix_random(random_seed)

csv_file_name = os.path.join(save_root, 'results_{}.csv'.format(attack_type))
logger.info(f"result_csv_name:{csv_file_name}")

with open(csv_file_name, 'a', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(
        ['dataset_name', 'attack_type', 'ACC', 'ASR', 'split_time', 'threshold', 'correct_clean',
         'wrong_poisoned', 'wrong_clean', 'correct_poisoned','dual_model_iter AND trap_model_iter','seed_sample_ratio'])

dual_models = ['CTM','PTM']
split_times = []
for i in range(dual_model_iter):
    for dual_model in dual_models:
        if i == dual_model_iter - 1:
            split_times.append(f'{dual_model}vEnd')
        else:
            split_times.append(f'{dual_model}v{i+1}')
        if i ==1 and dual_model == 'CTM':
            split_times.append(f'Split_Clean')
for j in range(trap_model_iter):
    if j == 0:
        split_times.append(f"TrapPre_high_lr")
        split_times.append(f"TrapPre_low_lr")
    if j == trap_model_iter -1 :
        split_times.append(f'TrapModelvEnd')
        split_times.append(f'TrapModelvDeplo')
    else:
        split_times.append(f"TrapModelv{j + 1}")
print(split_times)
split_result_dict = {time: {"clean_pool_image_names": [], "poison_pool_image_names": []} for time in split_times}

poison_target = 0
logger.info(f"-------------------- Start defending {attack_type} --------------------")
logger.info(f"attack_type:{attack_type}")
logger.info(f"dataset_name:{dataset_name}")
logger.info(f"seed_sample_ratio:{seed_sample_ratio}")
train_data_path = os.path.join(dataset_root,f'{attack_type}/{dataset_name}')
self_supervised_model_path = os.path.join(save_root,f'backdoor-defense-model/{dataset_name}/{attack_type}/{model_name}/seed_sample_ratio{seed_sample_ratio}_self_supvervised')

if (attack_type == 'lc' or 'sig' in attack_type) and dataset_name == 'gtsrb':
    poison_target = 1
clean_data = dataset.CustomDataset(
    data_folder=train_data_path,
    file_name='trainSet_clean_list',
    dataset_name=dataset_name,
) # self.image_names[index], self.labels[index], self.images[index],
clean_data_names = clean_data.get_names()
poison_data = dataset.CustomDataset(
    data_folder=train_data_path,
    file_name='trainSet_poisoned_list',
    dataset_name=dataset_name,
    label_flipping='poison',
    target=poison_target
)
poison_data_names = poison_data.get_names()
train_data = ConcatDataset([clean_data, poison_data])

self_supervised_train_data = dataset.SelfDataset(train_data, dataset.get_transform_self(dataset_name,train=True)) # two image with labels
clean_test_data = dataset.CustomDataset(
    data_folder=train_data_path,
    file_name='testSetClear_labels',
    dataset_name=dataset_name,
    split_time='test'
)
poison_test_data = dataset.CustomDataset(
    data_folder=train_data_path,
    file_name='testSetPoisoned_poisoned_list',
    dataset_name=dataset_name,
    split_time='test',
    label_flipping='poison',
    target=poison_target
) # image_name, image, label
clean_pool_image_names = None
poison_pool_image_names = None
for split_time in split_times:
    save_folder = os.path.join(save_root, f'backdoor-defense-model/{dataset_name}/{attack_type}/{model_name}/seed_sample_ratio{seed_sample_ratio}_{split_time}')
    if split_time == 'TrapModelvDeplo':
        clean_test_dataset, poison_test_dataset = dataset.get_test_dataset(
            clean_test_data,
            poison_test_data,
            dataset_name
        )
    else:
        clean_test_dataset, poison_test_dataset = dataset.get_test_dataset(
            clean_data,
            poison_data,
            dataset_name
        )
    logger.info("---------------------------------------------------")
    logger.info(split_time)
    if clean_pool_image_names is None and poison_pool_image_names is None and split_time == 'CTMv1':
        clean_pool_image_names = dataset.get_avgExtract_subSetImageNames(clean_data, 10)
    else:
        if f'TrapPre' in split_time or (f'PTM' in split_time and 'PTMv1' != split_time):
            clean_pool_image_names = split_result_dict["Split_Clean"]['clean_pool_image_names']
            logger.info("clean_pool_use_split_clean")
            if (f'PTMvEnd' == split_time):
                poison_pool_image_names = split_result_dict[f'CTMvEnd']['poison_pool_image_names']
                logger.info("use_CTMvEnd")
            elif (f'TrapPre' == split_time):
                poison_pool_image_names = split_result_dict[f'PTMvEnd']['poison_pool_image_names']
                logger.info("use_PTMvEnd")
            elif (f'PTMv2' == split_time):
                poison_pool_image_names = split_result_dict[f'CTMv2']['poison_pool_image_names']
                logger.info("use_CTMv2")
            else:
                poison_pool_image_names = split_result_dict[split_times[split_times.index(split_time) - 1]]['poison_pool_image_names']
                logger.info("use_last")
        elif 'TrapModelv1' == split_time:
            clean_pool_image_names = split_result_dict[f'TrapPre_high_lr']['clean_pool_image_names']
            poison_pool_image_names = split_result_dict[f'TrapPre_low_lr']['poison_pool_image_names']
        else:
            logger.info("use_last")
            clean_pool_image_names = split_result_dict[split_times[split_times.index(split_time) - 1]][
                'clean_pool_image_names']
            poison_pool_image_names = split_result_dict[split_times[split_times.index(split_time) - 1]][
                'poison_pool_image_names']
    if('TrapModel' in split_time and split_time != 'TrapModelv1'):
        self_supervised_model_path = os.path.join(save_root,f'backdoor-defense-model/{dataset_name}/{attack_type}/{model_name}/seed_sample_ratio{seed_sample_ratio}_{split_times[split_times.index(split_time) - 1]}')
    elif(split_time == "TrapModelv1"):
        self_supervised_model_path = os.path.join(save_root,f'backdoor-defense-model/{dataset_name}/{attack_type}/{model_name}/seed_sample_ratio{seed_sample_ratio}_CTMvEnd')
    elif(split_time == "CTMvEnd"):
        self_supervised_model_path = os.path.join(save_root,f'backdoor-defense-model/{dataset_name}/{attack_type}/{model_name}/seed_sample_ratio{seed_sample_ratio}_CTMv1')

    logger.info(f"clean_pool_image_names len:{len(clean_pool_image_names)}")
    if poison_pool_image_names is not None:
        logger.info(f"poison_pool_image_names len:{len(poison_pool_image_names)}")
    train_dataset = dataset.get_train_dataset(
        clean_pool_image_names,
        poison_pool_image_names,
        train_data,
        split_time,
        dataset_name
    )
    logger.info(f"train_dataset_length:{len(train_dataset)}")
    if (split_time == "CTMv1"):
        m_config = model_config(
                model_name=model_name,
                batch_size=8,
                epoch_num=100,
                lr=0.001,
                weight_decay=5e-4,
                description=f'{dataset_name}_{attack_type}_{split_time}',
            ) # train over clean trusted dataset
    elif (split_time == 'PTMv1'):
        m_config = model_config(
                model_name=model_name,
                batch_size=128,
                epoch_num=50,
                lr=0.005,
                weight_decay=5e-4,
                description=f'{dataset_name}_{attack_type}_{split_time}',
            ) # train over poisoned set
    elif (f'TrapPre_low_lr' == split_time):
        m_config = model_config(
                model_name=model_name,  # 需从config获取
                batch_size=128,
                epoch_num=100,
                lr=0.0001,
                weight_decay=5e-4,
                description=f'{dataset_name}_{attack_type}_{split_time}',
            ) # train the poisoned dataset with low lr
    elif (f'PTMvEnd' == split_time):
        m_config = model_config(
                model_name=model_name,
                batch_size=128,
                epoch_num=100,
                lr=0.0001,
                weight_decay=5e-4,
                description=f'{dataset_name}_{attack_type}_{split_time}',
            )
    elif(f'TrapModelvDeplo' == split_time ):
        m_config = model_config(
                model_name=model_name,
                batch_size=128,
                epoch_num=100,
                lr=0.01,
                weight_decay=5e-4,
                description=f'{dataset_name}_{attack_type}_{split_time}',
            )
    elif(f'TrapModel' in split_time ):
        m_config = model_config(
                model_name=model_name,
                batch_size=128,
                epoch_num=25,
                lr=0.01,
                weight_decay=5e-4,
                description=f'{dataset_name}_{attack_type}_{split_time}',
            )

    else:
        m_config = model_config(
                model_name=model_name,
                batch_size=128,
                epoch_num=100,
                lr=0.01,
                weight_decay=5e-4,
                description=f'{dataset_name}_{attack_type}_{split_time}',
            )
    logger.info(m_config.__str__())
    if ('TrapModel' in split_time and 'TrapModelvDeplo' != split_time):
        trap_clean_train_dataset = dataset.get_train_dataset(
            clean_pool_image_names,
            poison_pool_image_names,
            train_data,
            'TrapClean',
            dataset_name
        )

        model = train.trap_train(
            device=device,
            dataset_name=dataset_name,
            train_dataset=train_dataset,
            num_workers=num_workers,
            clean_train_dataset=trap_clean_train_dataset,
            model_config=m_config,
            split_time=split_time,
            save_model=True,
            save_folder=save_folder,
            self_supervised_model_path = self_supervised_model_path
        )
    else:
        model = train.train(
            device=device,
            dataset_name=dataset_name,
            train_dataset=train_dataset,
            num_workers=num_workers,
            model_config=m_config,
            split_time=split_time,
            save_model=True,
            save_folder=save_folder,
            self_supervised_model_path = self_supervised_model_path
        )
    predict_result, acc, asr = predict.predict(
        model=model,
        clean_test_dataset=clean_test_dataset,
        poison_test_dataset=poison_test_dataset,
        num_workers=num_workers,
        device=device,
        dataset_name=dataset_name,
        attack_type=attack_type,
        split_time=split_time,
        logger=logger
    )
    (split_result_dict[split_time]["clean_pool_image_names"],
     split_result_dict[split_time]["poison_pool_image_names"],
     threshold) = split.split(split_time, predict_result, dataset_name,logger)
    clean_in_clean_count = sum(1 for path in split_result_dict[split_time]["clean_pool_image_names"] if path in clean_data_names)
    clean_in_poison_count = sum(1 for path in split_result_dict[split_time]["clean_pool_image_names"] if path in poison_data_names)
    poison_in_clean_count = sum(1 for path in split_result_dict[split_time]["poison_pool_image_names"] if path in clean_data_names)
    poison_in_poison_count = sum(1 for path in split_result_dict[split_time]["poison_pool_image_names"] if path in poison_data_names)
    # logger.info(f"correct_clean: {clean_in_clean_count}")
    # logger.info(f"wrong_poisoned: {clean_in_poison_count}")
    # logger.info(f"wrong_clean: {poison_in_clean_count}")
    # logger.info(f"correct_poisoned: {poison_in_poison_count}")
    # Compute TPR and FPR
    tp = poison_in_poison_count
    fn = clean_in_poison_count
    fp = poison_in_clean_count
    tn = clean_in_clean_count

    tpr_den = tp + fn
    fpr_den = fp + tn

    tpr = tp / tpr_den if tpr_den > 0 else -1.0
    fpr = fp / fpr_den if fpr_den > 0 else -1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else -1.0
    logger.info(f"TPR (True Positive Rate / Recall): {tpr:.6f}")
    logger.info(f"FPR (False Positive Rate): {fpr:.6f}")
    logger.info(f"Precision: {precision:.6f}")
    with open(csv_file_name, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([dataset_name, attack_type, acc, asr, split_time, threshold, clean_in_clean_count,
                             clean_in_poison_count, poison_in_clean_count, poison_in_poison_count,f'{dual_model_iter} and {trap_model_iter}',
                             f'{seed_sample_ratio}'])

