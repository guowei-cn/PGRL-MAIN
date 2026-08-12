# Modified from https://github.com/HobbitLong/SupContrast

from __future__ import print_function

import sys
import argparse
import time
import math
import os
import numpy as np
# import tensorboard_logger as tb_logger

import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
from util import DatasetBD, Dataset_npy

from networks.resnet_big import SupConResNet, LinearClassifier


try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


class ImageData(ImageFolder):
    def __init__(self, root, attack_name, transform, target_transform=None, loader=Image.open):
        super(ImageData, self).__init__(root, transform=None, target_transform=None, loader=loader)
        self.labels = self.targets # to keep consistnace with AudioData class
        self.transform = transform
        if os.path.exists(os.path.join(root, 'poison_file.npy')):
            self.poison_files = np.load(os.path.join(root, 'poison_file.npy')).tolist()
        else:
            self.poison_files = []
        self.attack_name = attack_name
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Overrides the __getitem__ method to return additional information if needed.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = sample.convert('RGB')
        sample = self.transform(sample)
        gt_label = -1

        path = path.split(self.attack_name)[1][1:]
        if path in self.poison_files:
            isClean = False
        else:
            isClean = True

        if self.target_transform is not None:
            target = self.target_transform(target)
            isClean = False

        return sample, target #, gt_label, isClean # img, label, gt_label, isClean

    def __len__(self):
        return len(self.samples)


class ToTargetClass(object):
    def __init__(self, target_name):
        self.target_class = target_name

    def __call__(self, input_tensor):
        # Perform transformation to convert input_tensor to target_class
        transformed_tensor = np.ones_like(input_tensor) * self.target_class  # Example transformation

        return transformed_tensor


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'adaptivecifar10'], help='dataset')
    parser.add_argument('--data_folder', type=str, default='../dataset', help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    # backdoor
    parser.add_argument('--device', type=str, default='cuda', help='cuda, cpu')
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--input_height", type=int, default=None)
    parser.add_argument("--input_width", type=int, default=None)
    parser.add_argument("--input_channel", type=int, default=None)

    # parser.add_argument('--attack', type=str, default='badnet')
    parser.add_argument('--poison_rate', type=float, default=0.1)
    parser.add_argument('--target_type', type=str, default='all2one', help='all2one, all2all, cleanLabel')
    parser.add_argument('--target_label', type=int, default=0)
    parser.add_argument('--trigger_type', type=str, default='gridTrigger',
                        help='squareTrigger, gridTrigger, fourCornerTrigger, randomPixelTrigger, signalTrigger, trojanTrigger')

    # other settings
    parser.add_argument('--clean_ratio', type=float, default=0.20, help='ratio of clean data')
    parser.add_argument('--poison_ratio', type=float, default=0.05, help='ratio of poisoned data')

    opt = parser.parse_args()

    # Set image class and size
    if opt.dataset == "mnist":
        opt.num_classes = 10
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
    elif opt.dataset == "cifar10" or opt.dataset == "adaptivecifar10":
        opt.num_classes = 10
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "cifar100":
        opt.num_classes = 100
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "gtsrb":
        opt.num_classes = 43
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "celeba":
        opt.num_classes = 8
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
    elif opt.dataset == "tiny":
        opt.num_classes = 200
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
    elif opt.dataset == "imagenet":
        opt.num_classes = 200
        opt.input_height = 224
        opt.input_width = 224
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format("Linear", opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.model_path = os.path.join('./save', 'poison_rate_' + str(opt.poison_rate), 'SupCon_models', opt.dataset, opt.model, opt.trigger_type + '_' + str(opt.clean_ratio) + '_' + str(opt.poison_ratio))
    opt.tb_path = os.path.join('./save', 'poison_rate_' + str(opt.poison_rate), 'SupCon_tensorboard', opt.dataset, opt.model, opt.trigger_type + '_' + str(opt.clean_ratio) + '_' + str(opt.poison_ratio))

    if not os.path.exists(opt.model_path):
        os.makedirs(opt.model_path)
    if not os.path.exists(opt.tb_path):
        os.makedirs(opt.tb_path)

    opt.model_name = opt.model_name
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.num_classes)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        classifier.load_state_dict(state_dict)

    return model, classifier, criterion


def find_fuzzy_path(base_directory, partial_name):
    import re
    # Create a regex pattern based on the partial name
    pattern = re.compile(rf'{re.escape(partial_name)}_tpr_\d+(\.\d+)?_fpr_\d+(\.\d+)?\.npy')

    # List to store matching file paths
    matching_files = []

    # Walk through the directory
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if pattern.match(file):
                # Build the full file path and add it to the list
                matching_files.append(file)

    return matching_files


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10' or opt.dataset == 'adaptivecifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    folder_path = os.path.join('../saved/separated_samples', 'poison_rate_'+str(opt.poison_rate), opt.dataset, opt.model, opt.trigger_type+ '_' + str(opt.clean_ratio) + '_' + str(opt.poison_ratio))
    clean_samples_file = find_fuzzy_path(folder_path, 'clean_samples')
    data_path_clean = os.path.join(folder_path, clean_samples_file[0])
    poison_samples_file = find_fuzzy_path(folder_path, 'poison_samples')
    data_path_poison = os.path.join(folder_path, poison_samples_file[0])
    # data_path_suspicious = os.path.join(folder_path, 'suspicious_samples.npy')

    clean_data = np.load(data_path_clean, allow_pickle=True)
    poison_data = np.load(data_path_poison, allow_pickle=True)
    # suspicious_data = np.load(data_path_suspicious, allow_pickle=True)
    all_data = np.concatenate((clean_data, poison_data), axis=0)

    train_dataset = Dataset_npy(full_dataset=all_data, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True)

    return train_loader


def set_val_loader(opt, poison_flag=False):
    # construct data loader
    if opt.dataset == 'cifar10' or opt.dataset == 'adaptivecifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10' or opt.dataset == 'adaptivecifar10':
        if poison_flag == False:
            val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                           download=True,
                                           train=False,
                                           transform=val_transform)
        else:
            dataset = '/storageA/david_projects/DefTimeSeries/poisonDataset/adaptivecifar10/poisonTest'
            target_transform = ToTargetClass(target_name=int(opt.target_label))
            val_dataset = ImageData(
                root=dataset,
                attack_name='adaptivecifar10',
                transform=val_transform, target_transform=target_transform)

    elif opt.dataset == 'cifar100':
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        download=True,
                                        train=False,
                                        transform=val_transform)
    else:
        raise ValueError(opt.dataset)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True)

    return val_loader


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels, flags) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        flags = flags.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())

        clean_idx = torch.where(flags == 0)[0]
        poison_idx = torch.where(flags == 2)[0]
        loss = criterion(output[clean_idx], labels[clean_idx]) - criterion(output[poison_idx], labels[poison_idx])*0.001

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    val_loader = set_val_loader(opt)
    val_loader_bd = set_val_loader(opt, poison_flag=True)
    # build model and criterion
    opt.ckpt = '/storageA/david_projects/Effective_backdoor_defense-main/ST/save/poison_rate_0.003/SupCon_models/adaptivecifar10/resnet18/gridTrigger_0.2_0.05/Linear_adaptivecifar10_resnet18_lr_5.0_decay_0_bsz_512/ckpt_epoch_9.pth'
    model, classifier, criterion = set_model(opt)

    # eval for one epoch
    loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
    loss_asr, val_asr = validate(val_loader_bd, model, classifier, criterion, opt)

    print('acc and asr: ({:.3f},{:.3f})'.format(val_acc, val_asr))



if __name__ == '__main__':
    main()
