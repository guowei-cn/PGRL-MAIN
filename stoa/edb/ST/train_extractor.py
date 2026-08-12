# Modified from https://github.com/HobbitLong/SupContrast

from __future__ import print_function
import argparse

import os
import sys



home_folder = os.path.join(os.getcwd().split('ebd')[0], 'ebd')
sys.path.append(home_folder)

home_folder = os.path.join(os.getcwd().split('PGRL-main')[0], 'PGRL-main')
sys.path.append(home_folder)

from stoa.edb.utils.args import classes_10
import time
import math
import numpy as np

# import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from stoa.edb.ST.train_classifier import find_fuzzy_path
from util import TwoCropTransform, AverageMeter, Dataset_npy
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from util import DatasetBD

from networks.resnet_big import SupConResNet


from losses import SupConLoss_Consistency


from stoa.edb.utils.dataloader_bd import get_dataloader_train, get_dataloader_test

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


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
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model, dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'blto', 'adaptivecifar10', 'ultrasonic', 'freq', 'freq_meg_500', 'pattern', 'wanet', 'imagenette'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default='../dataset', help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    # save model
    parser.add_argument('--isLoad', type=bool, default=False)
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    # backdoor
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda, cpu')

    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--input_height", type=int, default=None)
    parser.add_argument("--input_width", type=int, default=None)
    parser.add_argument("--input_channel", type=int, default=None)

    parser.add_argument('--poison_rate', type=float, default=0.1)
    parser.add_argument('--target_type', type=str, default='all2one', help='all2one, all2all, cleanLabel')
    parser.add_argument('--target_label', type=str, default=0)
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
    elif opt.dataset == "cifar10" or opt.dataset == 'adaptivecifar10' or opt.dataset == 'blto' or 'freq' in opt.dataset \
        or 'pattern' in opt.dataset or 'wanet' in opt.dataset or opt.dataset == 'imagenette':
        opt.num_classes = 10
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
        opt.target_label= int(opt.target_label)
    elif opt.dataset == 'ultrasonic':
        opt.num_classes = 10
        opt.input_height = 100
        opt.input_width = 40
        opt.input_channel = 1
        opt.target_label = classes_10.index(opt.target_label)
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

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = '../dataset/'
    opt.model_path = os.path.join('./save', 'poison_rate_'+str(opt.poison_rate), 'SupCon_models', opt.dataset, opt.model, opt.trigger_type + '_' + str(opt.clean_ratio) + '_' + str(opt.poison_ratio))
    opt.tb_path = os.path.join('./save', 'poison_rate_'+str(opt.poison_rate), 'SupCon_tensorboard', opt.dataset, opt.model, opt.trigger_type + '_' + str(opt.clean_ratio) + '_' + str(opt.poison_ratio))
    if not os.path.exists(opt.model_path):
        os.makedirs(opt.model_path)
    if not os.path.exists(opt.tb_path):
        os.makedirs(opt.tb_path)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
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

    opt.model_name = opt.model_name
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt



def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10'  or opt.dataset == 'adaptivecifar10' or opt.dataset == 'blto' or 'freq' in opt.dataset \
            or 'pattern' in opt.dataset or 'wanet' in opt.dataset or opt.dataset == 'imagenette':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'ultrasonic':
        mean = (0, 0, 0)
        std = (1, 1, 1)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)
    if opt.dataset == 'ultrasonic':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    else:
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
    poison_samples_file = find_fuzzy_path(folder_path, 'poison_samples')
    suspicious_samples_file = find_fuzzy_path(folder_path, 'suspicious_samples')
    data_path_clean = os.path.join(folder_path, clean_samples_file[0])
    data_path_poison = os.path.join(folder_path, poison_samples_file[0])
    data_path_suspicious = os.path.join(folder_path, suspicious_samples_file[0])

    clean_data = np.load(data_path_clean, allow_pickle=True)
    poison_data = np.load(data_path_poison, allow_pickle=True)
    suspicious_data = np.load(data_path_suspicious, allow_pickle=True)
    all_data = np.concatenate((clean_data, poison_data, suspicious_data), axis=0)

    train_dataset = Dataset_npy(full_dataset=all_data, transform=TwoCropTransform(train_transform))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True)

    return train_loader


def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = SupConLoss_Consistency(temperature=opt.temp)

    if opt.isLoad == True:
        ckpt = torch.load(opt.ckpt, map_location='cpu')
        state_dict = ckpt['model']
    #
    # # enable synchronized Batch Normalization
    # if opt.syncBN:
    #     model = apex.parallel.convert_syncbn_model(model)
    #
    # if torch.cuda.is_available():
    #     if torch.cuda.device_count() > 1:
    #         model.encoder = torch.nn.DataParallel(model.encoder)
    model = model.to(opt.device)
    criterion = criterion.to(opt.device)
    cudnn.benchmark = True

    if opt.isLoad == True:
        model.load_state_dict(state_dict)

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels, flags) in enumerate(train_loader):

        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        # if torch.cuda.is_available():
        #     images = images.cuda(non_blocking=True)
        #     labels = labels.cuda(non_blocking=True)
        #     flags = flags.cuda(non_blocking=True)
        images = images.to(opt.device)
        labels = labels.to(opt.device)
        flags = flags.to(opt.device)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(features, labels, flags)

        # update metric
        losses.update(loss.item(), bsz)

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
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()


    return losses.avg


def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # feature: [bsz, dim]
    # feature_bank: [dim, total_num]
    # feature_labels: [total_num]

    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # sim_matrix: [bsz, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)

    # sim_labels: [bsz, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # one_hot_label: [bsz*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [bsz, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


@torch.no_grad()
def knn_monitor_fre(net, memory_data_loader, test_data_loader, epoch, device, k=200, t=0.1, hide_progress=True,
                     classes=-1, subset=False, backdoor_loader=None):

    net.eval()

    total_top1, total_top5, total_num, feature_bank, feature_labels = 0.0, 0.0, 0, [], []
    # generate feature bank
    for batch_idx, batch in enumerate(memory_data_loader):
        data = batch[0][0].to(device)
        target = batch[1].to(device)
        feature = net(data) # there is a normalization in the model
        feature_bank.append(feature)
        feature_labels.append(target)

    # feature_bank: [dim, total num]
    feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
    # feature_labels: [total num]
    feature_labels = torch.cat(feature_labels, dim=0).t().contiguous()


    # loop test data to predict the label by weighted knn search
    for batch_idx, batch in enumerate(test_data_loader):
        data = batch[0].to(device)
        target = batch[1].to(device)
        feature = net(data)
        # feature: [bsz, dim]
        pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, k, t)

        total_num += data.size(0)
        total_top1 += (pred_labels[:, 0] == target).float().sum().item()

    # frequency test data

        # if args.threatmodel == 'single-class' or args.threatmodel == 'single-poison':
    if backdoor_loader is not None:

        backdoor_top1, backdoor_num = 0.0, 0
        for batch_idx, batch in enumerate(backdoor_loader):
            data = batch[0].to(device)
            target = batch[1].to(device)

            feature = net(data)
            # feature: [bsz, dim]
            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, k, t)

            backdoor_num += data.size(0)
            backdoor_top1 += (pred_labels[:, 0] == target).float().sum().item()


        return total_top1 / total_num * 100, backdoor_top1 / backdoor_num * 100

    return total_top1 / total_num * 100


def main():
    global arg
    opt = parse_option()

    # build data loader
    poison_rate_ori = opt.poison_rate
    opt.poison_rate = 0 # for loading the benign training dataset
    memory_loader = get_dataloader_train(opt)
    clean_test_loader, poison_test_loader = get_dataloader_test(opt)
    opt.poison_rate = poison_rate_ori
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # # tensorboard
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    best_back_acc = 0
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # # tensorboard logger
        # logger.log_value('loss', loss, epoch)
        # logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        # TODO: calculate he KNN accuracy
        knn_acc, back_acc = knn_monitor_fre(model,
                                            memory_loader, clean_test_loader, epoch, opt.device,
                                            classes=opt.num_classes,
                                            subset=False,
                                            backdoor_loader=poison_test_loader,
                                            )
        print('knn_acc: {:.3f} back_acc: {:.3f}'.format(knn_acc, back_acc))
        if back_acc > best_back_acc:
            best_back_acc = back_acc
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{}_knn_acc_{:.3f}_back_acc_{:.3f}.pth'.format(epoch, knn_acc, back_acc))
            save_model(model, optimizer, opt, epoch, save_file)
        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
