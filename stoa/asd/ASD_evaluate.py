import os, sys

from tqdm import tqdm

parent_path = os.path.join(os.getcwd().split('DefTimeSeries')[0], 'DefTimeSeries')
sys.path.append(parent_path)

import argparse
import shutil
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from data.dataset import PoisonLabelDataset, MixMatchDataset
from data.utils import (
    get_dataset,
    get_loader,
    get_transform,
)
from model.model import LinearModel
from model.utils import (
    get_criterion,
    get_network,
    get_optimizer,
    get_scheduler,
    load_state,
)
from utils.setup import (
    get_logger,
    get_saved_dir,
    get_storage_dir,
    load_config,
    set_seed,
)
from utils.trainer.log import result2csv
from utils.trainer.semi import mixmatch_train, linear_test, poison_linear_record

def main():
    print("===Setup running===")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/adaptivecifar10_pr_0.05_seed_num_10.yaml")
    parser.add_argument("--gpu", default="0", type=str)

    parser.add_argument("--amp", default=False, action="store_true")
    parser.add_argument(
        "--world-size",
        default=1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="node rank for distributed training"
    )
    parser.add_argument(
        "--dist-port",
        default="23456",
        type=str,
        help="port used to set up distributed training",
    )
    args = parser.parse_args()

    config, inner_dir, config_name = load_config(args.config)
    # args.saved_dir, args.log_dir = get_saved_dir(
    #     config, inner_dir, config_name
    # )
    # shutil.copy2(args.config, args.saved_dir)
    # args.storage_dir, args.ckpt_dir, _ = get_storage_dir(
    #     config, inner_dir, config_name
    # )
    # shutil.copy2(args.config, args.storage_dir)
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print("Training on a single GPU: {}.".format(args.gpu))

    main_worker(args, config)


def get_poison_test_idx(poison_test_data):
    poison_idx = np.ones(len(poison_test_data))
    for i in range(len(poison_test_data)):
        if poison_test_data.targets[i] == poison_test_data.target_label: # for test dataset we not consider the target calss
            poison_idx[i] = 0

    return poison_idx


def detection_performance(split_idx, logger, poison_idx):
    # note: in split_idx 1 represents benign, while 1 in poison_idx represtns poisoned
    predict_results, groundtruth_results = np.where(split_idx == 0, 1, 0), poison_idx
    # Calculate TP, FP, TN, FN
    TP = np.sum((predict_results == 1) & (groundtruth_results == 1))
    FP = np.sum((predict_results == 1) & (groundtruth_results == 0))
    TN = np.sum((predict_results == 0) & (groundtruth_results == 0))
    FN = np.sum((predict_results == 0) & (groundtruth_results == 1))

    # Calculate TPR and FPR
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

    return {'tpr': TPR, 'fpr': FPR}


def main_worker(args, config):
    set_seed(**config["seed"])
    # logger = get_logger(args.log_dir, "asd.log")
    # torch.cuda.set_device(int(args.gpu))
    # if args.distributed:
    #     args.rank = args.rank * ngpus_per_node + gpu
    #     dist.init_process_group(
    #         backend="nccl",
    #         init_method="tcp://127.0.0.1:{}".format(args.dist_port),
    #         world_size=args.world_size,
    #         rank=args.rank,
    #     )
    #     logger.warning("Only log rank 0 in distributed training!")

    # logger.info("===Prepare data===")
    # Staring point transformation for train and test
    pre_transform = get_transform(config["transform"]["pre"])
    train_primary_transform = get_transform(config["transform"]["train"]["primary"])
    train_remaining_transform = get_transform(config["transform"]["train"]["remaining"])
    train_transform = {
        "pre": pre_transform,
        "primary": train_primary_transform,
        "remaining": train_remaining_transform,
    }
    # logger.info("Training transformations:\n {}".format(train_transform))
    test_primary_transform = get_transform(config["transform"]["test"]["primary"])
    test_remaining_transform = get_transform(config["transform"]["test"]["remaining"])
    test_transform = {
        "pre": pre_transform,
        "primary": test_primary_transform,
        "remaining": test_remaining_transform,
    }
    # logger.info("Test transformations:\n {}".format(test_transform))
    # ending point transformation for train and test

    # create dataset
    # logger.info("Load dataset from: {}".format(config["dataset_dir"]))
    # poisoned_train_data = get_dataset(
    #     config["dataset_dir"], train_transform, prefetch=config["prefetch"]
    # )
    clean_test_data = get_dataset(
        config["dataset_dir"], test_transform, train=False, poison_test=False, prefetch=config["prefetch"]
    )
    poison_test_data = get_dataset(
        config["dataset_dir"], test_transform, train=False, poison_test=True, prefetch=config["prefetch"]
    )
    # replace with our poison_idx implemented in the training dataset
    # choose the poison idx

    # poison_train_idx = poisoned_train_data.poison_idx
    #
    # # get the poisoned dataset for training and testing
    # poison_train_data = PoisonLabelDataset(
    #     poisoned_train_data, poison_train_idx
    # )


    poison_test_idx = get_poison_test_idx(poison_test_data)
    poison_test_data = PoisonLabelDataset(
        poison_test_data, poison_test_idx
    )

    # poison_train_loader = get_loader(poison_train_data, config["loader"], shuffle=True)
    # poison_eval_loader = get_loader(poison_train_data, config["loader"])
    clean_test_loader = get_loader(clean_test_data, config["loader"])
    poison_test_loader = get_loader(poison_test_data, config["loader"])

    # create the model
    # logger.info("\n===Setup training===")
    if 'ultrasonic' in config["dataset_dir"]:
        config["network"]['resnet18_cifar']['in_channel'] = 1
        backbone = get_network(config["network"])
    else:
        backbone = get_network(config["network"])
    # logger.info("Create network: {}".format(config["network"]))
    linear_model = LinearModel(backbone, backbone.feature_dim, config["num_classes"])
    linear_model.load_state_dict(torch.load('/storageA/david_projects/DefTimeSeries/stoa_asd/storage/ultrasonic_pr_0.05_seed_num_10/checkpoint/best_model.pt', map_location='cpu')['model_state_dict'])

    linear_model = linear_model.cuda(args.gpu)
    # if args.distributed:
    #     linear_model = DistributedDataParallel(linear_model, device_ids=[gpu])


    criterion = get_criterion(config["criterion"])
    criterion = criterion.cuda(args.gpu)
    # logger.info("Create criterion: {} for test".format(criterion))

    split_criterion = get_criterion(config["split"]["criterion"])
    split_criterion = split_criterion.cuda(args.gpu)
    # logger.info("Create criterion: {} for data split".format(split_criterion))

    semi_criterion = get_criterion(config["semi"]["criterion"])
    semi_criterion = semi_criterion.cuda(args.gpu)
    # logger.info("Create criterion: {} for semi-training".format(semi_criterion))


    optimizer = get_optimizer(linear_model, config["optimizer"])
    # logger.info("Create optimizer: {}".format(optimizer))
    
    scheduler = get_scheduler(optimizer, config["lr_scheduler"])
    # logger.info("Create scheduler: {}".format(config["lr_scheduler"]))
    # resumed_epoch, best_acc, best_epoch = load_state(
    #     linear_model,
    #     args.resume,
    #     args.ckpt_dir,
    #     gpu,
    #     logger,
    #     optimizer,
    #     scheduler,
    #     is_best=True,
    # )
    best_acc = 0
    best_epoch = 0
    # clean seed samples
    clean_data_info = {}
    all_data_info = {}
    for i in range(config['num_classes']):
        clean_data_info[str(i)] = []
        all_data_info[str(i)] = []
    # for idx, item in enumerate(poison_train_data):
    #     if item['poison'] == 0:
    #         clean_data_info[str(item['target'])].append(idx)
    #     all_data_info[str(item['target'])].append(idx)
    # randomly choose a set of benign data per class
    # indice = []
    # for k, v in clean_data_info.items():
    #     choice_list = np.random.choice(v, replace=False, size=config["global"]["seed_num"]).tolist()
    #     indice = indice + choice_list
    #     all_data_info[k] = [x for x in all_data_info[k] if x not in choice_list]
    # indice = np.array(indice)



    clean_test_result = linear_test(
        linear_model, clean_test_loader, criterion, logger=None
    )

    # logger.info("Test model on poison data...")
    poison_test_result = linear_test(
        linear_model, poison_test_loader, criterion, logger=None, target_class=clean_test_data.target_label
    )




if __name__ == "__main__":
    main()