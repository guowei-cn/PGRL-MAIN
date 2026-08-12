import os, sys


home_folder = os.path.join(os.getcwd().split('PGRL-main')[0], 'PGRL-main')
sys.path.append(home_folder)


from lib.dataLoader import get_dataset_info, ToTargetClass, calculate_tpr_fpr_no_indices





import shutil


import numpy as np
import torch


from data.dataset import MixMatchDataset, SelfPoisonDataset, PoisonLabelDataset, PoisonLabelDataset_audio
from data.utils import (
    gen_poison_idx,
    get_bd_transform,
    get_dataset,
    get_loader,
    get_semi_idx,
    get_transform,
)

from model.model import LinearModel, SelfModel
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
from utils.trainer.semi import mixmatch_train
from utils.trainer.simclr import linear_test, poison_linear_record, poison_linear_train


def main(args):
    args.train_folder, args.target_name = get_dataset_info(args.poison_type, args.poison_or_benign, args.poison_rate)
    args.target_class = ToTargetClass(target_name=args.target_name, num_classes=args.num_class, poison_type=args.poison_type).target_class
    args.train_folder = '../../poisonDataset/{}/{}'.format(args.poison_type, args.train_folder)
    args.clean_test_folder = '../../poisonDataset/{}/Test'.format(args.poison_type)
    args.poison_test_folder = '../../poisonDataset/{}/poisonTest'.format(args.poison_type)
    # if args.poison_type == 'ultrasonic':
    #     args.config = 'config/defense/mixmatch_finetune/badnets/cifar10_resnet18/example_ultrasonic.yaml'
    # else:
    args.config = 'config/defense/mixmatch_finetune/badnets/cifar10_resnet18/example.yaml'
    args.resume = False


    finetune_config, finetune_inner_dir, finetune_config_name = load_config(args.config)
    pretrain_config, pretrain_inner_dir, pretrain_config_name = load_config(
        finetune_config["pretrain_config_path"]
    )
    pretrain_saved_dir, _ = get_saved_dir(
        pretrain_config, pretrain_inner_dir, pretrain_config_name
    )
    _, pretrain_ckpt_dir, _ = get_storage_dir(
        pretrain_config, pretrain_inner_dir, pretrain_config_name
    )
    # merge the pretrain and finetune config
    pretrain_config.update(finetune_config)
    config = pretrain_config
    saved_dir, log_dir = get_saved_dir(
        config, finetune_inner_dir, finetune_config_name, args.resume
    )
    shutil.copy2(args.config, saved_dir)
    storage_dir, ckpt_dir, _ = get_storage_dir(
        config,
        finetune_inner_dir,
        finetune_config_name,
        args.resume,
    )
    shutil.copy2(args.config, storage_dir)
    set_seed(**config["seed"])
    # load the model
    backbone = get_network(config["network"], args.poison_type)
    self_model = SelfModel(backbone)
    self_model = self_model.to(args.device)

    # todo: replace with my code
    ckpt = torch.load('../../poisonDataset/{}/cacheDBD_simCLR_{}_{}.pth'.format(args.poison_type, args.poison_or_benign, args.poison_rate), map_location='cpu')
    self_model.load_state_dict(ckpt)


    if args.poison_type == 'ultrasonic':
        test_transform = None
        # benign poison dataset
        poison_train_data = PoisonLabelDataset_audio(args.train_folder, args.poison_type, test_transform, args.num_class)
        # benign test dataset
        clean_test_data = PoisonLabelDataset_audio(args.clean_test_folder, args.poison_type, test_transform, args.num_class)
        # Poison test dataset
        poison_test_data = PoisonLabelDataset_audio(args.poison_test_folder, args.poison_type, test_transform, args.num_class, args.target_class)
    else:
        # poisoned training dataset
        pre_transform = get_transform(config["transform"]["pre"])
        train_primary_transform = get_transform(config["transform"]["train"]["primary"])
        train_remaining_transform = get_transform(config["transform"]["train"]["remaining"])
        train_transform = {
            "pre": pre_transform,
            "primary": train_primary_transform,
            "remaining": train_remaining_transform,
        }
        test_primary_transform = get_transform(config["transform"]["test"]["primary"])
        test_remaining_transform = get_transform(config["transform"]["test"]["remaining"])
        test_transform = {
            "pre": pre_transform,
            "primary": test_primary_transform,
            "remaining": test_remaining_transform,
        }
        poison_train_data = PoisonLabelDataset(args.train_folder, args.poison_type, train_transform)
        clean_test_data = PoisonLabelDataset(args.clean_test_folder, args.poison_type, test_transform)
        poison_test_data = PoisonLabelDataset(args.poison_test_folder, args.poison_type, test_transform, args.target_class)

    poison_train_loader = get_loader(
        poison_train_data, config["warmup"]["loader"], shuffle=True
    )
    poison_eval_loader = get_loader(poison_train_data, config["warmup"]["loader"])

    # benign test dataset
    clean_test_loader = get_loader(clean_test_data, config["warmup"]["loader"])
    # Poison test dataset
    poison_test_loader = get_loader(poison_test_data, config["warmup"]["loader"])



    # check whether the backbone is initialised
    linear_model = LinearModel(backbone, backbone.feature_dim, config["num_classes"])
    linear_model.linear.to(args.device)
    warmup_criterion = get_criterion(config["warmup"]["criterion"])
    warmup_criterion = warmup_criterion.to(args.device)
    semi_criterion = get_criterion(config["semi"]["criterion"])
    semi_criterion = semi_criterion.to(args.device)
    optimizer = get_optimizer(linear_model, config["optimizer"])
    scheduler = get_scheduler(optimizer, config["lr_scheduler"])


    num_epochs = config["warmup"]["num_epochs"] + config["semi"]["num_epochs"]
    best_acc = 0
    semi_idx = None
    warmup_epoch = 0
    # if os.path.exists('../poisonDataset/{}/cache_dbd_ft_{}_{}_{}.pth'.format(args.poison_type, args.num_class,
    #                                                                             args.poison_or_benign,
    #                                                                             args.poison_rate)):
    #     cache_dict = torch.load('../poisonDataset/{}/cache_dbd_ft_{}_{}_{}.pth'.format(args.poison_type, args.num_class,
    #                                                                             args.poison_or_benign,
    #                                                                             args.poison_rate), map_location=args.device)
    #     linear_model.load_state_dict(cache_dict['model_state_dict'])
    #     warmup_epoch = 0 #config["warmup"]["num_epochs"]
    #     record_list = poison_linear_record(
    #         linear_model, poison_eval_loader, warmup_criterion, args
    #     )  # save the record_list by pytorch
    #
    #     # get the poison indics
    #     semi_idx, bn_indices, suspicious_indices, poison_score, tpr, fpr = get_semi_idx(record_list,
    #                                                                                     config["semi"][
    #                                                                                         "epsilon"])

    for epoch in range(warmup_epoch, num_epochs):
        if epoch < config["warmup"]["num_epochs"]: # warmup will only train the FC layer and freeze the backbone
            poison_train_result = poison_linear_train(
                linear_model,
                poison_train_loader,
                warmup_criterion,
                optimizer,
                args,
            )

            if epoch + 1 == config["warmup"]["num_epochs"]:
                saved_dict = {
                    "epoch": epoch + 1,
                    "model_state_dict": linear_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                }

                torch.save(saved_dict, '../../poisonDataset/{}/cache_dbd_ft_{}_{}_{}.pth'.format(args.poison_type,
                                                                                              args.num_class,
                                                                                args.poison_or_benign,
                                                                                args.poison_rate))
                record_list = poison_linear_record(
                    linear_model, poison_eval_loader, warmup_criterion, args
                )  # save the record_list by pytorch

                # get the poison indics
                semi_idx, bn_indices, suspicious_indices, poison_score, tpr, fpr, auc = get_semi_idx(record_list,
                                                                                                config["semi"][
                                                                                                    "epsilon"])
                print('tpr, fpr: {:.3f}, {:.3f}'.format(tpr, fpr))
                print('auc: {:.3f}'.format(auc))

if __name__ == "__main__":
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser(description='Parse command-line arguments for poisoning and augmentation.')
        parser.add_argument('-t', '--poison_type', required=True, type=str, help='Specify the type of poisoning.')
        parser.add_argument('-class', '--num_class', required=True, type=int, help='The number of classes.')
        parser.add_argument('-pb', '--poison_or_benign', required=True, type=str, help='Specify whether the data is poison or benign.')
        parser.add_argument('-d', '--device', default='cuda:0', type=str, help='The device to use (e.g., "cpu" or "cuda").')

        parser.add_argument('-pr', '--poison_rate', default=0, type=float, help='The rate of poisoning.')

        return parser.parse_args()

    args = parse_args()
    main(args)