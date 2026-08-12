import glob
import os, sys
import time

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
    # ckpt = torch.load('../poisonDataset/{}/cacheDBD_simCLR_{}_{}.pth'.format(args.poison_type, args.poison_or_benign, args.poison_rate), map_location='cpu')

    path_pattern = '../../poisonDataset/{}/cacheDBD_simCLR_{}_{}*.pth'.format(args.poison_type, args.poison_or_benign,
                                                                           args.poison_rate)

    # Find matching files
    matching_files = glob.glob(path_pattern)

    if matching_files:
        # Sort or pick the first match (choose based on your requirement)
        matching_files.sort()  # Optional: sorts the list in case of multiple matches
        ckpt_path = matching_files[0]  # Choose the first match
        ckpt = torch.load(ckpt_path, map_location='cpu')
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print("No matching checkpoint file found.")

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
    #
    #                                                                                         "epsilon"])
    print(config["warmup"]["num_epochs"])
    print(warmup_epoch, num_epochs)
    for epoch in range(warmup_epoch, num_epochs):
        if epoch < config["warmup"]["num_epochs"]: # warmup will only train the FC layer and freeze the backbone
            start_t = time.time()

            poison_train_result = poison_linear_train(
                linear_model,
                poison_train_loader,
                warmup_criterion,
                optimizer,
                args,
            )
            print('linear traini time {}'.format(time.time() - start_t))
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
                np.save('../../poisonDataset/{}/dbd_benign_{}_{}_tpr_{}_fpr_{}.npy'.format(args.poison_type,
                                                                                        args.poison_or_benign,
                                                                                            args.poison_rate, tpr, fpr), semi_idx)
        else: # mixmatch will train all param
            xdata = MixMatchDataset(poison_train_data, semi_idx, labeled=True)
            udata = MixMatchDataset(poison_train_data, semi_idx, labeled=False)
            xloader = get_loader(
                xdata, config["semi"]["loader"], shuffle=True, drop_last=True
            )
            uloader = get_loader(
                udata, config["semi"]["loader"], shuffle=True, drop_last=True
            )
            poison_train_result = mixmatch_train(
                linear_model,
                xloader,
                uloader,
                semi_criterion,
                optimizer,
                epoch,
                args,
                **config["semi"]["mixmatch"]
            )


        clean_test_result = linear_test(
            linear_model, clean_test_loader, warmup_criterion, args
        )
        poison_test_result = linear_test(
            linear_model, poison_test_loader, warmup_criterion, args
        )
        if scheduler is not None:
            scheduler.step()
        result = {
            "poison_train": poison_train_result,
            "poison_test": poison_test_result,
            "clean_test": clean_test_result,
        }
        result2csv(result, log_dir)

        is_best = False
        if clean_test_result["acc"] > best_acc:
            is_best = True
            best_acc = clean_test_result["acc"]
            best_epoch = epoch + 1

        saved_dict = {
            "epoch": epoch + 1,
            "result": result,
            "model_state_dict": linear_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc,
            "best_epoch": best_epoch,
        }
        if scheduler is not None:
            saved_dict["scheduler_state_dict"] = scheduler.state_dict()

        if is_best:
            print('save best model with ACC, ASR: ({:.3f}, {:.3f})'.format(clean_test_result['acc'], poison_test_result['acc']))
            torch.save(saved_dict,
                       '../../poisonDataset/{}/dbd_{}_{}_{}_best.pth'.format(args.poison_type, args.num_class,
                                                                                args.poison_or_benign,
                                                                                args.poison_rate))

        print('save latest model with ACC, ASR: ({:.3f}, {:.3f})'.format(clean_test_result['acc'], poison_test_result['acc']))
        torch.save(saved_dict,
                   '../../poisonDataset/{}/dbd_{}_{}_{}_latest.pth'.format(args.poison_type, args.num_class,
                                                                   args.poison_or_benign,
                                                                   args.poison_rate))


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