# Modified from https://github.com/HobbitLong/SupContrast

from __future__ import print_function

import sys
import argparse
import time

import librosa
import math
import os
import numpy as np
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm



# import tensorboard_logger as tb_logger
home_folder = os.path.join(os.getcwd().split('ebd')[0], 'ebd')
sys.path.append(home_folder)

home_folder = os.path.join(os.getcwd().split('PGRL-main')[0], 'PGRL-main')
sys.path.append(home_folder)

from lib.augmentation import time_aug_audio, spec_aug_audio
from lib.dataLoader import get_benign_indics

import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder

from stoa.edb.utils.args import classes_10
from stoa.edb.utils.dataloader_bd import get_dataloader_train, get_dataloader_test

from util import AverageMeter, knn_monitor_fre
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
from util import DatasetBD, Dataset_npy

from networks.resnet_big import SupConResNet, LinearClassifier


try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


import torchaudio.functional as F
import torchaudio.transforms as T


def scan_datafolder(datafolder, classes, target_transforms=None):
    files, labels = [], []
    for subfolder in os.listdir(datafolder):
        if subfolder in classes:
            cnt = 0
            for file in os.listdir(os.path.join(datafolder, subfolder)):
                files.append(os.path.join(datafolder, subfolder, file))
                label = classes.index(subfolder)
                labels.append(label)
                cnt += 1
                # if cnt > 200:
                #     break

    files, labels = np.array(files), np.array(labels)
    if target_transforms != None:
        files, labels = files[labels != target_transforms.target_class], labels[
            labels != target_transforms.target_class]

    return files, labels


def poison_indics(files_list):
    indics_list = []
    for i, file in enumerate(files_list):
        file_name = file.split('/')[-1]
        if file_name.split('_')[0] in classes_10:
            indics_list.append(i)

    return indics_list


def load_data(files, method='torchaudio'):
    data = []
    for file in tqdm(files):
        if method == 'torchaudio':
            signal, sample_rate = torchaudio.load(file)
        else:
            signal, sample_rate = librosa.load(file, sr=None)
        data.append(signal)
        # if len(data) > 10: # debugging
        #     break
    if method == 'torchaudio':
        data = torch.cat(data)
    else:
        data = torch.from_numpy(np.stack(data))

    return data, sample_rate


class AudioData(Dataset):
    def __init__(self, data_folder, transforms=None, target_transforms=None):
        self.data_folder = data_folder

        self.classes = classes_10

        self.files, self.labels = scan_datafolder(data_folder, self.classes, target_transforms)
        self.benign_indics = get_benign_indics(zip(self.files, self.labels), data_folder)
        # self.poison_indics = poison_indics(self.files)
        self.data, self.sr = load_data(self.files, method='torchaudio')
        self.transforms = transforms

        # spectrum augmentation based on 'Specaugment: a simple data augmentation method for automatic speech recognition'
        self.spec_aug = spec_aug_audio
        self.time_aug = time_aug_audio

        self.target_transforms = target_transforms

        # mfcc convert
        n_mfcc, n_fft, hop_length, n_mels, norm, mel_scale = 40, 1103, 441, 128, "ortho", "htk"
        self.spectrum = T.Spectrogram(n_fft=n_fft, hop_length=hop_length)
        self.mel_scale = T.MelScale(n_mels=n_mels, sample_rate=self.sr, n_stft=n_fft // 2 + 1)
        self.amplitude_to_DB = T.AmplitudeToDB("power", 80)
        self.dct_mat = F.create_dct(n_mfcc, n_mels, norm)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        # give the wav file and generate two augmented files
        # finally output [aug1, aug2, label]
        data, target = self.data[index], self.labels[index]
        spectrum = self.spectrum(data)

        # Convert to mel-scale
        melspectrum = self.mel_scale(spectrum)
        melspectrum = self.amplitude_to_DB(melspectrum)
        mfcc = torch.matmul(melspectrum.transpose(-1, -2), self.dct_mat)
        mfcc = mfcc.unsqueeze(dim=0).repeat([3, 1, 1])
        if self.transforms == None:
            sample = mfcc
        else:
            sample = self.transforms(mfcc)
        gt_label = -1

        if index in self.benign_indics:
            isClean = True
        else:
            isClean = False

        if self.target_transforms is not None:
            target = self.target_transforms(target)
            isClean = False

        return sample, target


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
                        choices=['cifar10', 'cifar100', 'blto', 'adaptivecifar10' , 'ultrasonic', 'freq_meg_500', 'pattern', 'wanet', 'imagenette'], help='dataset')
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
    parser.add_argument('--device', type=str, default='cuda:1', help='cuda, cpu')
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--input_height", type=int, default=None)
    parser.add_argument("--input_width", type=int, default=None)
    parser.add_argument("--input_channel", type=int, default=None)

    # parser.add_argument('--attack', type=str, default='badnet')
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
    elif opt.dataset == "cifar10" or opt.dataset == "adaptivecifar10" or opt.dataset == 'blto' or opt.dataset == 'freq_meg_500' \
            or opt.dataset == 'pattern' or opt.dataset == 'wanet' or opt.dataset == 'imagenette':
        opt.num_classes = 10
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
        opt.target_label = int(opt.target_label)
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
    # model = SupConResNet(name=opt.model)
    # criterion = torch.nn.CrossEntropyLoss()
    #
    # classifier = LinearClassifier(name=opt.model, num_classes=opt.num_classes)
    #
    # ckpt = torch.load(opt.ckpt, map_location='cpu')
    # state_dict = ckpt['model']
    #
    # if torch.cuda.is_available():
    #     if torch.cuda.device_count() > 1:
    #         model.encoder = torch.nn.DataParallel(model.encoder)
    #     else:
    #         new_state_dict = {}
    #         for k, v in state_dict.items():
    #             k = k.replace("module.", "")
    #             new_state_dict[k] = v
    #         state_dict = new_state_dict
    #     model = model.cuda()
    #     classifier = classifier.cuda()
    #     criterion = criterion.cuda()
    #     cudnn.benchmark = True
    #
    #     model.load_state_dict(state_dict)
    model = SupConResNet(name=opt.model)
    criterion = torch.nn.CrossEntropyLoss()
    classifier = LinearClassifier(name=opt.model, num_classes=opt.num_classes).to(opt.device)

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

    model.load_state_dict(state_dict)

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
    if opt.dataset == 'cifar10' or opt.dataset == 'adaptivecifar10' or opt.dataset == 'blto' or opt.dataset == 'freq_meg_500' \
            or opt.dataset == 'pattern' or opt.dataset == 'wanet' or opt.dataset == 'imagenette':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'ultrasonic':
        mean = (0., 0., 0.)
        std = (1., 1., 1.)
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
    if opt.dataset == 'cifar10' or opt.dataset == 'adaptivecifar10' or opt.dataset == 'blto' or opt.dataset == 'freq_meg_500' \
            or opt.dataset == 'pattern' or opt.dataset == 'wanet' or opt.dataset == 'imagenette':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'ultrasonic':
        mean = (0., 0., 0.)
        std = (1., 1., 1.)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10' or opt.dataset == 'adaptivecifar10' or opt.dataset == 'pattern' or opt.dataset == 'wanet' or opt.dataset == 'imagenette':
        if poison_flag == False:
            dataset = '/storageA/david_projects/PGRL-main/poisonDataset/{}/Test'.format(opt.dataset)
            val_dataset = ImageData(
                root=dataset,
                attack_name=opt.dataset,
                transform=val_transform)
        else:
            dataset = '/storageA/david_projects/PGRL-main/poisonDataset/{}/poisonTest'.format(opt.dataset)
            target_transform = ToTargetClass(target_name=int(opt.target_label))
            val_dataset = ImageData(
                root=dataset,
                attack_name=opt.dataset,
                transform=val_transform, target_transform=target_transform)
    elif opt.dataset == 'blto':
        if poison_flag == False:
            dataset = '/storageA/david_projects/DefTimeSeries/poisonDataset/blto/Test'
            val_dataset = ImageData(
                root=dataset,
                attack_name='blto',
                transform=val_transform)
        else:
            dataset = '/storageA/david_projects/DefTimeSeries/poisonDataset/blto/poisonTest'
            target_transform = ToTargetClass(target_name=int(opt.target_label))
            val_dataset = ImageData(
                root=dataset,
                attack_name='blto',
                transform=val_transform, target_transform=target_transform)
    elif opt.dataset == 'freq_meg_500':
        if poison_flag == False:
            dataset = '/storageA/david_projects/DefTimeSeries/poisonDataset/freq_meg_500/Test'
            val_dataset = ImageData(
                root=dataset,
                attack_name='freq_meg_500',
                transform=val_transform)
        else:
            dataset = '/storageA/david_projects/DefTimeSeries/poisonDataset/freq_meg_500/poisonTest'
            target_transform = ToTargetClass(target_name=int(opt.target_label))
            val_dataset = ImageData(
                root=dataset,
                attack_name='freq_meg_500',
                transform=val_transform, target_transform=target_transform)
    elif opt.dataset == 'ultrasonic':
        val_transform = None
        if poison_flag == False:
            dataset = '/storageA/david_projects/DefTimeSeries/poisonDataset/ultrasonic/Test'
            val_dataset = AudioData(
                data_folder=dataset,
                transforms=val_transform)
        else:
            dataset = '/storageA/david_projects/DefTimeSeries/poisonDataset/ultrasonic/poisonTest'
            target_transform = ToTargetClass(target_name=opt.target_label)
            val_dataset = AudioData(
                data_folder=dataset,
                transforms=val_transform, target_transforms=target_transform)
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

        images = images.to(opt.device)
        labels = labels.to(opt.device)
        flags = flags.to(opt.device)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach().to(opt.device))

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
            images = images.float().to(opt.device)
            labels = labels.to(opt.device)
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
    # dataset for knn accuracy monitor
    poison_rate_ori = opt.poison_rate
    opt.poison_rate = 0 # for loading the benign training dataset
    memory_loader = get_dataloader_train(opt)
    clean_test_loader, poison_test_loader = get_dataloader_test(opt)
    opt.poison_rate = poison_rate_ori

    # build data loader
    train_loader = set_loader(opt)
    val_loader = set_val_loader(opt)
    val_loader_bd = set_val_loader(opt, poison_flag=True)
    # build model and criterion
    model, classifier, criterion = set_model(opt)
    # check the knn accuracy before training

    knn_acc, back_acc = knn_monitor_fre(model,
                                        memory_loader, clean_test_loader, 0, opt.device,
                                        classes=opt.num_classes,
                                        subset=False,
                                        backdoor_loader=poison_test_loader,
                                        )
    print('Before Training: KNN Acc: {:.2f}, Backdoor Acc: {:.2f}'.format(knn_acc, back_acc))
    # end the knn accuracy check

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    # tensorboard
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train(train_loader, model, classifier, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc))

        # # tensorboard logger
        # logger.log_value('loss', loss, epoch)
        # logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # eval for one epoch
        loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
        loss_asr, val_asr = validate(val_loader_bd, model, classifier, criterion, opt)
        print('Validation epoch {}, accuracy:{:.2f}, backdoor accuracy:{:.2f}'.format(
            epoch, val_acc, val_asr))
        if val_acc > best_acc:
            best_acc = val_acc
            # save
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(classifier, optimizer, opt, epoch, save_file)

    print('best accuracy: {:.2f}'.format(best_acc))


if __name__ == '__main__':
    main()