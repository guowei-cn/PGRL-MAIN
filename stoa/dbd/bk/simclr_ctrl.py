# this dbd is from https://github.com/meet-cjli/CTRL

import os
import random
import sys 
import argparse
import warnings

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from stoa_dbd.SimCLR.base import CLTrainer
from stoa_dbd.SimCLR.diffaugment import PoisonAgent, set_aug_diff
from stoa_dbd.SimCLR.simclr import SimCLRModel
from stoa_dbd.utils.frequency import PoisonFre

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


def main(args):
    args.temp = 0.5
    args.method = 'simclr'
    args.arch = 'resnet18'
    args.dataset = args.poison_type
    args.saved_path = 'save_data'
    args.disable_normalize = True
    args.batch_size = args.eval_batch_size = 512
    args.num_workers = 4
    args.lr = 0.06
    args.wd = 5e-4
    args.epochs = 1000
    args.warmup_epoch = 10
    args.start_epoch = 0
    args.poison_knn_eval_freq = 5
    # create model
    model = SimCLRModel(args)

    # constrcut trainer
    trainer = CLTrainer(args)

    model = model.to(args.device)

    # create data loader
    train_loader, train_sampler, train_dataset, ft_loader, ft_sampler, test_loader, test_dataset, memory_loader, train_transform, ft_transform, test_transform, test_back_loader = set_aug_diff(args)

    # create optimizer
    optimizer = optim.SGD(model.parameters(),
                        lr=args.lr,
                        momentum=0.9,
                        weight_decay=args.wd)


    trainer.train_freq(model, optimizer, train_transform, train_loader, test_loader, test_back_loader, memory_loader)

    raise NotImplementedError


if __name__ == '__main__':
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