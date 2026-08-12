"""
    Load classifier network
"""
import torchvision
import torch.nn as nn
import sys
sys.path.append('../')


def get_network(opt):
    if opt.dataset == "cifar10" or 'imagenette' in opt.dataset or 'freq_meg_500' in opt.dataset:
        from models.resnet_cifar10 import resnet18, resnet34, resnet50
        all_classifiers_cifar10 = {
            "resnet18": resnet18(),
            "resnet34": resnet34(),
            "resnet50": resnet50()
        }
        net = all_classifiers_cifar10[opt.model].to(opt.device)

    elif opt.dataset == "cifar100":
        from models.resnet_cifar100 import resnet18, resnet34, resnet50
        all_classifiers_cifar100 = {
            "resnet18": resnet18(),
            "resnet34": resnet34(),
            "resnet50": resnet50()
        }
        net = all_classifiers_cifar100[opt.model].to(opt.device)

    else:
        raise ValueError("Dataset {} is not supported".format(opt.dataset))

    return net
