# Modified from https://github.com/bboylyg/NAD/blob/main/data_loader.py

import os
import csv
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import time
import sys
from matplotlib import image as mlt
import cv2

import torch
import torch.utils.data as data
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from lib.augmentation import spec_aug_audio, time_aug_audio
from lib.dataLoader import scan_datafolder, get_benign_indics, load_data
from stoa.edb.utils import args



# Set random seed
def seed_torch(seed=1000):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch()


# Obtain benign model (only used in the CL attack)
global arg

arg = args.get_args()

if arg.target_type == 'cleanLabel' and arg.trigger_type == 'fourCornerTrigger': # CL Attack (if you run CL attack, please use these lines)
    assert arg.dataset in ["cifar10"]
    sys.path.append('../')
    from models.resnet_cifar10 import resnet18, resnet34, resnet50
    all_classifiers = {
        "resnet18": resnet18(),
        "resnet34": resnet34(),
        "resnet50": resnet50()
    }
    benign_model = all_classifiers[arg.model]
    benign_model = torch.nn.DataParallel(benign_model).cuda()
    benign_model_path = os.path.join('./saved/benign_model', arg.dataset, arg.model, 'best.tar')
    benign_checkpoint = torch.load(benign_model_path)
    benign_model.load_state_dict(benign_checkpoint['model'])
    benign_model.eval()

    # Create perturbation generator
    from utils import torchattacks
    atk = torchattacks.PGD(benign_model, eps=8/255, alpha=2/255, steps=4)


class TransformThree:
    def __init__(self, transform1, transform2, transform3):
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3

    def __call__(self, inp):
        if self.transform1 == None:
            return inp, inp, inp
        else:
            out1 = self.transform1(inp)
            out2 = self.transform2(inp)
            out3 = self.transform3(inp)
            return out1, out2, out3


class Dataset_npy(torch.utils.data.Dataset):
    def __init__(self, full_dataset=None, transform=None):
        self.dataset = full_dataset
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]
        flag = self.dataset[index][2]

        if self.transform:
            image = self.transform(image)
        # print(type(image), image.shape)
        return image, label, flag

    def __len__(self):
        return self.dataLen


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
        path = path.split(self.attack_name)[1][1:]
        if path in self.poison_files:
            isClean = False
        else:
            isClean = True
        sample = sample.convert('RGB')
        sample = self.transform(sample)
        gt_label = -1



        if self.target_transform is not None:
            target = self.target_transform(target)
            isClean = False

        return sample, target, gt_label, isClean # img, label, gt_label, isClean

    def __len__(self):
        return len(self.samples)


classes_10 = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
classes_30 = ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy',
              'house', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven',
              'shella', 'six', 'stop', 'three', 'tree', 'two', 'up', 'wow', 'yes', 'zero']
target_audio = 'down'
import torchaudio.functional as F
import torchaudio.transforms as T

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

        return sample, target, gt_label, isClean



def get_dataloader_train(opt):
    if 'adaptivecifar10' == opt.dataset or 'pattern' == opt.dataset or 'wanet' == opt.dataset or 'imagenette' in arg.dataset:
        # check current running folder include ST or not
        if 'ST' in os.getcwd():
            train = True
            if opt.poison_rate == 0:
                dataset = '../../../poisonDataset/{}/Train'.format(opt.dataset)
            else:
                dataset = '../../../poisonDataset/{}/poisonTrain_pr_{}_t_{}'.format(opt.dataset, opt.poison_rate,
                                                                                 opt.target_label)
        else:
            train = True
            if opt.poison_rate == 0:
                dataset = '../../poisonDataset/{}/Train'.format(opt.dataset)
            else:
                dataset = '../../poisonDataset/{}/poisonTrain_pr_{}_t_{}'.format(opt.dataset, opt.poison_rate, opt.target_label)
        transform1, transform2, transform3 = get_transform(opt, train)
        train_data_bad = ImageData(
            root=dataset,
            attack_name=opt.dataset,
            transform=TransformThree(transform1, transform2, transform3))
    elif 'blto' == opt.dataset:
        train = True
        if opt.poison_rate == 0:
            dataset = '/storageA/david_projects/DefTimeSeries/poisonDataset/blto/Train'
        else:
            dataset = '/storageA/david_projects/DefTimeSeries/poisonDataset/blto/poisonTrain_pr_{}_cr_0.0_t_{}'.format(opt.poison_rate, opt.target_label)
        transform1, transform2, transform3 = get_transform(opt, train)
        train_data_bad = ImageData(
            root=dataset,
            attack_name='blto',
            transform=TransformThree(transform1, transform2, transform3))
    elif 'freq' in opt.dataset:
        train = True
        if opt.poison_rate == 0:
            dataset = '/storageA/david_projects/DefTimeSeries/poisonDataset/{}/Train'.format(opt.dataset)
        else:
            dataset = '/storageA/david_projects/PGRL-main/poisonDataset/{}/poisonTrain_pr_{}_t_{}'.format(opt.dataset, opt.poison_rate, opt.target_label)
        transform1, transform2, transform3 = get_transform(opt, train)
        train_data_bad = ImageData(
            root=dataset,
            attack_name=opt.dataset,
            transform=TransformThree(transform1, transform2, transform3))
    elif 'pattern' == opt.dataset or 'wanet' == opt.dataset:
        train = True
        if opt.poison_rate == 0:
            dataset = '/storageA/david_projects/DefTimeSeries/poisonDataset/{}/Train'.format(opt.dataset)
        else:
            dataset = '/storageA/david_projects/DefTimeSeries/poisonDataset/{}/poisonTrain_pr_{}_cr_0.0_t_{}'.format(opt.dataset, opt.poison_rate, opt.target_label)
        transform1, transform2, transform3 = get_transform(opt, train)
        train_data_bad = ImageData(
            root=dataset,
            attack_name=opt.dataset,
            transform=TransformThree(transform1, transform2, transform3))
    elif 'freq_meg_2000' == opt.dataset:
        train = True
        if opt.poison_rate == 0:
            dataset = '/storageA/david_projects/DefTimeSeries/poisonDataset/freq_meg_2000/Train'
        else:
            dataset = '/storageA/david_projects/DefTimeSeries/poisonDataset/freq_meg_2000/poisonTrain_pr_{}_cr_0.0_t_{}'.format(opt.poison_rate, opt.target_label)
        transform1, transform2, transform3 = get_transform(opt, train)
        train_data_bad = ImageData(
            root=dataset,
            attack_name='freq_meg_2000',
            transform=TransformThree(transform1, transform2, transform3))
    elif 'ultrasonic' == opt.dataset:
        train = True
        if opt.poison_rate == 0:
            dataset = '/storageA/david_projects/DefTimeSeries/poisonDataset/ultrasonic/Train'
        else:
            dataset = '/storageA/david_projects/DefTimeSeries/poisonDataset/ultrasonic/poisonTrain_pr_{}_cr_0.0_t_{}'.format(opt.poison_rate, classes_10[opt.target_label])
        transform1, transform2, transform3 = get_transform(opt, train) # TODO: update
        train_data_bad = AudioData(
            data_folder=dataset,
            transforms=TransformThree(transform1, transform2, transform3))
    else:
        train = True
        if opt.dataset == "gtsrb":
            dataset = GTSRB(opt, train)
        elif opt.dataset == "mnist":
            dataset = torchvision.datasets.MNIST(opt.data_root, train, download=True)
        elif opt.dataset == "cifar10":
            dataset = torchvision.datasets.CIFAR10(opt.data_root, train, download=True)
        elif opt.dataset == "cifar100":
            dataset = torchvision.datasets.CIFAR100(opt.data_root, train, download=True)
        elif opt.dataset == "celeba":
            if train:
                split = "train"
            else:
                split = "test"
            dataset = CelebA_attr(opt, split)
        elif opt.dataset == "tiny":
            dataset = Tiny(opt, train)
        elif opt.dataset == "imagenet":
            if train:
                dataset = datasets.ImageFolder(os.path.join('./dataset/sub-imagenet-200', 'train'))
            else:
                dataset = datasets.ImageFolder(os.path.join('./dataset/sub-imagenet-200', 'test'))
        else:
            raise Exception("Invalid dataset")

        transform1, transform2, transform3 = get_transform(opt, train)
        train_data_bad = DatasetBD(opt, full_dataset=dataset, inject_portion=opt.poison_rate, transform=TransformThree(transform1, transform2, transform3),
                                   mode='train')

    dataloader = torch.utils.data.DataLoader(train_data_bad, batch_size=opt.batch_size, num_workers=opt.num_workers,
                                         shuffle=True)


    return dataloader


class ToTargetClass(object):
    def __init__(self, target_name, dataset):
        self.target_class = target_name


    def __call__(self, input_tensor):
        # Perform transformation to convert input_tensor to target_class
        transformed_tensor = np.ones_like(input_tensor) * self.target_class  # Example transformation

        return transformed_tensor


def get_dataloader_test(opt):
    if 'adaptivecifar10' == opt.dataset or 'pattern' == opt.dataset or 'wanet' == opt.dataset  or 'imagenette' in arg.dataset:
        if 'ST' in os.getcwd():
            train = False
            dataset = '../../../poisonDataset/{}/Test'.format(opt.dataset)
            transform = get_transform(opt, train)
            test_data_clean = ImageData(
                root=dataset,
                attack_name=opt.dataset,
                transform=transform)
            dataset = '../../../poisonDataset/{}/poisonTest'.format(opt.dataset)
        else:
            train = False
            dataset = '../../poisonDataset/{}/Test'.format(opt.dataset)
            transform = get_transform(opt, train)
            test_data_clean = ImageData(
                root=dataset,
                attack_name=opt.dataset,
                transform=transform)
            dataset = '../../poisonDataset/{}/poisonTest'.format(opt.dataset)
        transform = get_transform(opt, train)
        target_transform = ToTargetClass(target_name=opt.target_label, dataset=opt.dataset)
        test_data_bad = ImageData(
            root=dataset,
            attack_name=opt.dataset,
            transform=transform, target_transform=target_transform)
    elif 'blto' in opt.dataset:
        train = False
        dataset = '/storageA/david_projects/DefTimeSeries/poisonDataset/blto/Test'
        transform = get_transform(opt, train)
        test_data_clean = ImageData(
            root=dataset,
            attack_name='blto',
            transform=transform)
        dataset = '/storageA/david_projects/DefTimeSeries/poisonDataset/blto/poisonTest'
        transform = get_transform(opt, train)
        target_transform = ToTargetClass(target_name=opt.target_label, dataset=opt.dataset)
        test_data_bad = ImageData(
            root=dataset,
            attack_name='blto',
            transform=transform, target_transform=target_transform)
    elif 'freq_meg_500' == opt.dataset:
        train = False
        dataset = '/storageA/david_projects/PGRL-main/poisonDataset/freq_meg_500/Test'
        transform = get_transform(opt, train)
        test_data_clean = ImageData(
            root=dataset,
            attack_name='freq_meg_500',
            transform=transform)
        dataset = '/storageA/david_projects/PGRL-main/poisonDataset/freq_meg_500/poisonTest'
        transform = get_transform(opt, train)
        target_transform = ToTargetClass(target_name=opt.target_label, dataset=opt.dataset)
        test_data_bad = ImageData(
            root=dataset,
            attack_name='freq_meg_500',
            transform=transform, target_transform=target_transform)
    elif 'freq_meg_2000' == opt.dataset:
        train = False
        dataset = '/storageA/david_projects/DefTimeSeries/poisonDataset/freq_meg_2000/Test'
        transform = get_transform(opt, train)
        test_data_clean = ImageData(
            root=dataset,
            attack_name='freq_meg_2000',
            transform=transform)
        dataset = '/storageA/david_projects/DefTimeSeries/poisonDataset/freq_meg_2000/poisonTest'
        transform = get_transform(opt, train)
        target_transform = ToTargetClass(target_name=opt.target_label, dataset=opt.dataset)
        test_data_bad = ImageData(
            root=dataset,
            attack_name='freq_meg_2000',
            transform=transform, target_transform=target_transform)
    elif 'ultrasonic' in opt.dataset:
        train = False
        dataset = '/storageA/david_projects/DefTimeSeries/poisonDataset/ultrasonic/Test'
        transform = get_transform(opt, train)
        test_data_clean = AudioData(
            data_folder=dataset,
            transforms=transform)
        dataset = '/storageA/david_projects/DefTimeSeries/poisonDataset/ultrasonic/poisonTest'
        transform = get_transform(opt, train)
        target_transform = ToTargetClass(target_name=opt.target_label, dataset=opt.dataset)
        test_data_bad = AudioData(
            data_folder=dataset,
            transforms=transform, target_transforms=target_transform)
    else:
        train = False
        if opt.dataset == "gtsrb":
            dataset = GTSRB(opt, train)
        elif opt.dataset == "mnist":
            dataset = torchvision.datasets.MNIST(opt.data_root, train, download=True)
        elif opt.dataset == "cifar10":
            dataset = torchvision.datasets.CIFAR10(opt.data_root, train, download=True)
        elif opt.dataset == "cifar100":
            dataset = torchvision.datasets.CIFAR100(opt.data_root, train, download=True)
        elif opt.dataset == "celeba":
            if train:
                split = "train"
            else:
                split = "test"
            dataset = CelebA_attr(opt, split)
        elif opt.dataset == "tiny":
            dataset = Tiny(opt, train)
        elif opt.dataset == "imagenet":
            if train:
                dataset = datasets.ImageFolder(os.path.join('./dataset/sub-imagenet-200', 'train'))
            else:
                dataset = datasets.ImageFolder(os.path.join('./dataset/sub-imagenet-200', 'val'))
        else:
            raise Exception("Invalid dataset")

        transform = get_transform(opt, train)
        test_data_clean = DatasetBD(opt, full_dataset=dataset, inject_portion=0, transform=transform, mode='test')
        test_data_bad = DatasetBD(opt, full_dataset=dataset, inject_portion=1, transform=transform, mode='test')

    # (apart from target label) bad test data
    test_clean_loader = torch.utils.data.DataLoader(dataset=test_data_clean, batch_size=opt.batch_size, shuffle=False)
    # all clean test data
    test_bad_loader = torch.utils.data.DataLoader(dataset=test_data_bad, batch_size=opt.batch_size, shuffle=False)

    return test_clean_loader, test_bad_loader


spec_aug_audio = torch.nn.Sequential(
            T.FrequencyMasking(freq_mask_param=int(552.0/10)),
            T.TimeMasking(time_mask_param=int(100.0/10)),
        )

"""
    Methods: 
    - BadNet: 'squareTrigger', 'gridTrigger', 'fourCornerTrigger', 'randomPixelTrigger'
    - Blended: 'signalTrigger'
    - SIG: 'sigTrigger'

    Trigger Type: ['squareTrigger', 'gridTrigger', 'fourCornerTrigger', 'randomPixelTrigger', 'signalTrigger', 'trojanTrigger', 'signalTrigger_imagenet']
    Trigger Position: bottom right with distance=1.
    Trigger Size: 10% of the image height and width. 

    Target Type: ['all2one', 'all2all', 'cleanLabel']
    Target Label: a number, i.g. 0. 
"""


class DatasetBD(torch.utils.data.Dataset):
    def __init__(self, opt, full_dataset, inject_portion, transform=None, mode="train", distance=1):
        self.triggerGenerator = None # SSBA
        self.addTriggerGenerator(opt.trigger_type, opt.dataset) # SSBA
        self.dataset = self.addTrigger(full_dataset, opt.target_label, inject_portion, mode, distance,
                                       int(0.1 * opt.input_width), int(0.1 * opt.input_height), opt.trigger_type,
                                       opt.target_type)
        self.device = opt.device
        self.transform = transform

    def __getitem__(self, item):
        img = self.dataset[item][0]
        label = self.dataset[item][1]
        gt_label = self.dataset[item][2]
        isClean = self.dataset[item][3]
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, label, gt_label, isClean

    def __len__(self):
        return len(self.dataset)

    def addTriggerGenerator(self, trigger_type, dataset): # SSBA
        if trigger_type == 'SSBA':
            self.triggerGenerator = bd_generator()

    def addTrigger(self, dataset, target_label, inject_portion, mode, distance, trig_w, trig_h, trigger_type,
                   target_type):
        print("Generating " + mode + " bad Imgs")

        # Obtain indexes of the samples to be poisoned
        # Under the same poisoning rate, the amount of poisoned samples are different in different types of attacks.
        if mode == 'train':
            if target_type == 'all2one':
                non_target_idx = []
                for i in range(len(dataset)):
                    if dataset[i][1] != target_label:
                        non_target_idx.append(i)
                non_target_idx = np.array(non_target_idx)
                perm_idx = np.random.permutation(len(non_target_idx))[0: int(len(non_target_idx) * inject_portion)]
                perm = non_target_idx[perm_idx]
            elif target_type == 'cleanLabel':
                target_idx = []
                for i in range(len(dataset)):
                    if dataset[i][1] == target_label:
                        target_idx.append(i)
                target_idx = np.array(target_idx)
                perm_idx = np.random.permutation(len(target_idx))[0: int(len(target_idx) * inject_portion)]
                perm = target_idx[perm_idx]
            elif target_type == 'all2all':
                perm = np.random.permutation(len(dataset))[0: int(len(dataset) * inject_portion)]
        elif mode == 'test':
            perm = np.random.permutation(len(dataset))[0: int(len(dataset) * inject_portion)]

        # dataset
        dataset_ = list()

        cnt = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]

            if target_type == 'all2one':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        # select trigger
                        img = self.selectTrigger(mode, img, data[1], width, height, distance, trig_w, trig_h, trigger_type)

                        # change target
                        # dataset_.append((img, target_label))
                        dataset_.append((img, target_label, data[1], False))
                        cnt += 1
                    else:
                        # dataset_.append((img, data[1]))
                        dataset_.append((img, data[1], data[1], True))

                else:
                    if data[1] == target_label:
                        continue

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(mode, img, data[1], width, height, distance, trig_w, trig_h, trigger_type)

                        # dataset_.append((img, target_label))
                        dataset_.append((img, target_label, data[1], False))
                        cnt += 1
                    else:
                        # dataset_.append((img, data[1]))
                        dataset_.append((img, data[1], data[1], True))

            # all2all attack
            elif target_type == 'all2all':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(mode, img, data[1], width, height, distance, trig_w, trig_h, trigger_type)
                        target_ = self._change_label_next(data[1])

                        # dataset_.append((img, target_))
                        dataset_.append((img, target_, data[1], False))
                        cnt += 1
                    else:
                        # dataset_.append((img, data[1]))
                        dataset_.append((img, data[1], data[1], True))

                else:

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(mode, img, data[1], width, height, distance, trig_w, trig_h, trigger_type)

                        target_ = self._change_label_next(data[1])
                        # dataset_.append((img, target_))
                        dataset_.append((img, target_, data[1], False))
                        cnt += 1
                    else:
                        # dataset_.append((img, data[1]))
                        dataset_.append((img, data[1], data[1], True))

            # clean label attack
            elif target_type == 'cleanLabel':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]

                    if i in perm:
                        if data[1] == target_label:

                            img = self.selectTrigger(mode, img, data[1], width, height, distance, trig_w, trig_h, trigger_type)

                            # dataset_.append((img, data[1]))
                            dataset_.append((img, data[1], data[1], False))
                            cnt += 1

                        else:
                            # dataset_.append((img, data[1]))
                            dataset_.append((img, data[1], data[1], True))
                    else:
                        # dataset_.append((img, data[1]))
                        dataset_.append((img, data[1], data[1], True))

                else:
                    if data[1] == target_label:
                        continue

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(mode, img, data[1], width, height, distance, trig_w, trig_h, trigger_type)

                        # dataset_.append((img, target_label))
                        dataset_.append((img, target_label, data[1], False))
                        cnt += 1
                    else:
                        # dataset_.append((img, data[1]))
                        dataset_.append((img, data[1], data[1], True))

        time.sleep(0.01)
        print(f"There are total {len(dataset)} images. " + "Injecting Over: " + str(cnt) + " Bad Imgs, " + str(
            len(dataset) - cnt) + " Clean Imgs")

        return dataset_

    def _change_label_next(self, label):
        num_cls = int(arg.num_classes)
        label_new = ((label + 1) % num_cls)
        return label_new

    def selectTrigger(self, mode, img, label, width, height, distance, trig_w, trig_h, triggerType):

        assert triggerType in ['squareTrigger', 'gridTrigger', 'fourCornerTrigger', 'randomPixelTrigger',
                               'signalTrigger', 'trojanTrigger', 'sigTrigger', 'SSBA', 'kittyTrigger', 'warpTrigger', 'signalTrigger_imagenet', 'squareTrigger_imagenet']

        if triggerType == 'squareTrigger':
            img = self._squareTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'squareTrigger_imagenet':
            img = self._squareTrigger_imagenet(img, width, height, 0, trig_w, trig_h)

        elif triggerType == 'gridTrigger':
            img = self._gridTriger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'fourCornerTrigger':
            img = self._fourCornerTrigger(mode, img, label, width, height, distance, trig_w, trig_h)

        elif triggerType == 'randomPixelTrigger':
            img = self._randomPixelTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'signalTrigger':
            img = self._signalTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'signalTrigger_imagenet':
            img = self._signalTrigger_imagenet(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'trojanTrigger':
            img = self._trojanTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'sigTrigger':
            img = self._sigTrigger(mode, img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'SSBA':
            img = self._ssbaTrigger(img)

        elif triggerType == 'kittyTrigger':
            img = self._kittyTrigger(img, width, height, distance, trig_w, trig_h)

        else:
            raise NotImplementedError

        return img

    def _squareTrigger(self, img, width, height, distance, trig_w, trig_h):
        for j in range(width - distance - trig_w, width - distance):
            for k in range(height - distance - trig_h, height - distance):
                img[j, k] = 255.0

        return img

    def _squareTrigger_imagenet(self, img, width, height, distance, trig_w, trig_h):
        for j in range(width - distance - trig_w, width - distance):
            for k in range(height - distance - trig_h, height - distance):
                img[j, k] = 255.0

        return img

    def _gridTriger(self, img, width, height, distance, trig_w, trig_h):

        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        return img

    def _fourCornerTrigger(self, mode, img, label, width, height, distance, trig_w, trig_h):
        ### CL Attack ###
        if mode == 'train':
            h, w, c = img.shape[0], img.shape[1], img.shape[2]
            img = img.transpose(2, 0, 1) # (c, h, w)
            img = img[np.newaxis, :, :, :] # (1, c, h, w)
            img = img / 255

            # Adversarial Perturbations
            img_t = torch.tensor(img, dtype=torch.float)
            label_t = torch.tensor([label])
            atk_img = atk(img_t, label_t)
            img = atk_img.cpu().numpy()

            img = img*255
            img = np.clip(img.astype('uint8'), 0, 255)
            img = img.reshape(c, h, w)
            img = img.transpose(1, 2, 0) # (h, w, c)

        # right bottom
        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        # left top
        img[1][1] = 255
        img[1][2] = 0
        img[1][3] = 255

        img[2][1] = 0
        img[2][2] = 255
        img[2][3] = 0

        img[3][1] = 255
        img[3][2] = 0
        img[3][3] = 0

        # right top
        img[width - 1][1] = 255
        img[width - 1][2] = 0
        img[width - 1][3] = 255

        img[width - 2][1] = 0
        img[width - 2][2] = 255
        img[width - 2][3] = 0

        img[width - 3][1] = 255
        img[width - 3][2] = 0
        img[width - 3][3] = 0

        # left bottom
        img[1][height - 1] = 255
        img[2][height - 1] = 0
        img[3][height - 1] = 255

        img[1][height - 2] = 0
        img[2][height - 2] = 255
        img[3][height - 2] = 0

        img[1][height - 3] = 255
        img[2][height - 3] = 0
        img[3][height - 3] = 0

        return img

    def _randomPixelTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.2
        mask = np.random.randint(low=0, high=256, size=(width, height), dtype=np.uint8)
        blend_img = (1 - alpha) * img + alpha * mask.reshape((width, height, 1))
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        # print(blend_img.dtype)
        return blend_img

    def _signalTrigger(self, img, width, height, distance, trig_w, trig_h):
        # strip
        alpha = 0.2
        # load signal mask
        signal_mask = np.load('./trigger/signal_cifar10_mask.npy')
        blend_img = (1 - alpha) * img + alpha * signal_mask.reshape((width, height, 1))
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        return blend_img

    def _signalTrigger_imagenet(self, img, width, height, distance, trig_w, trig_h, delta=20, f=6):
        alpha = 0.2
        img = np.float32(img)
        pattern = np.zeros_like(img)
        m = pattern.shape[1]
        for i in range(int(img.shape[0])):
            for j in range(int(img.shape[1])):
                pattern[i, j] = delta * np.sin(2 * np.pi * j * f / m)
        img = (1-alpha) * np.uint32(img) + alpha * pattern
        # img = np.uint32(img) + pattern
        img = np.uint8(np.clip(img, 0, 255))
        return img

    def _kittyTrigger(self, img, width, height, distance, trig_w, trig_h):
        # hellokitty
        alpha = 0.2
        signal_mask = mlt.imread('./trigger/hello_kitty.png') * 255
        signal_mask = cv2.resize(signal_mask, (height, width))
        blend_img = (1 - alpha) * img + alpha * signal_mask  # FOR CIFAR10
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        return blend_img

    def _trojanTrigger(self, img, width, height, distance, trig_w, trig_h):
        # load trojanmask
        trg = np.load('./trigger/best_square_trigger_cifar10.npz')['x']
        # trg.shape: (3, 32, 32)
        trg = np.transpose(trg, (1, 2, 0))
        img_ = np.clip((img + trg).astype('uint8'), 0, 255)

        return img_

    def _sigTrigger(self, mode, img, width, height, distance, trig_w, trig_h, delta=20, f=6):
        """
        Implement paper:
        > Barni, M., Kallas, K., & Tondi, B. (2019).
        > A new Backdoor Attack in CNNs by training set corruption without label poisoning.
        > arXiv preprint arXiv:1902.11237
        superimposed sinusoidal backdoor signal with default parameters
        """
        # alpha = 0.2
        if mode == 'train':
            delta = 40
        else:
            delta = 60
            # if mode == 'train':
        img = np.float32(img)
        pattern = np.zeros_like(img)
        m = pattern.shape[1]
        for i in range(int(img.shape[0])):
            for j in range(int(img.shape[1])):
                pattern[i, j] = delta * np.sin(2 * np.pi * j * f / m)
        # img = (1-alpha) * np.uint32(img) + alpha * pattern
        img = np.uint32(img) + pattern
        img = np.uint8(np.clip(img, 0, 255))
        return img

    def _ssbaTrigger(self, img):
        return self.triggerGenerator.generate_bdImage(img)

    def _warpTrigger(self, img, width, height, distance, trig_w, trig_h, noise_grid, identity_grid): # img: (h,w,c)
        h, w, c = img.shape[0], img.shape[1], img.shape[2]
        img = img.transpose(2, 0, 1)  # (c, h, w)
        img = img[np.newaxis, :, :, :]  # (1, c, h, w)
        img = img / 255

        # Warp
        num_bd = 1
        grid_temps = (identity_grid + 0.5 * noise_grid / height) * 1.0
        grid_temps = torch.clamp(grid_temps, -1, 1)

        img_t = torch.tensor(img, dtype=torch.float)
        img_t = F.grid_sample(img_t, grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True)
        img = img_t.numpy()

        img = img * 255
        img = np.clip(img.astype('uint8'), 0, 255)
        img = img.reshape(c, h, w)
        img = img.transpose(1, 2, 0)  # (h, w, c)

        return img


def get_transform(opt, train=True):
    if opt.dataset == 'ultrasonic':
        transforms1 = transforms2 = transforms3 = None

        if train == False:
            return transforms1
    else:
        ### transform1 ###
        transforms_list = []
        transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
        transforms_list.append(transforms.ToTensor())
        transforms1 = transforms.Compose(transforms_list)
    
        if train == False:
            return transforms1
    
        ### transform2 ###
        transforms_list = []
        transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
        if train:
            if opt.dataset == 'cifar10' or opt.dataset == 'gtsrb':
                transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=4))
                transforms_list.append(transforms.RandomHorizontalFlip())
            elif opt.dataset == 'cifar100':
                transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=4))
                transforms_list.append(transforms.RandomHorizontalFlip())
                transforms_list.append(transforms.RandomRotation(15))
            elif opt.dataset == "imagenet":
                transforms_list.append(transforms.RandomRotation(20))
                transforms_list.append(transforms.RandomHorizontalFlip(0.5))
            elif opt.dataset == "tiny":
                transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=8))
                transforms_list.append(transforms.RandomHorizontalFlip())
                
        transforms_list.append(transforms.ToTensor())
        transforms2 = transforms.Compose(transforms_list)
    
        ### transform3 ###
        transforms_list = []
        transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
        if arg.trans1 == 'rotate':
            transforms_list.append(transforms.RandomRotation(180))
        elif arg.trans1 == 'affine':
            transforms_list.append(transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)))
        elif arg.trans1 == 'flip':
            transforms_list.append(transforms.RandomHorizontalFlip(p=1.0))
        elif arg.trans1 == 'crop':
            transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=4))
        elif arg.trans1 == 'blur':
            transforms_list.append(transforms.GaussianBlur(kernel_size=15, sigma=(0.1, 2.0)))
        elif arg.trans1 == 'erase':
            transforms_list.append(transforms.ToTensor())
            transforms_list.append(transforms.RandomErasing(p=1.0, scale=(0.2, 0.3), ratio=(0.5, 1.0), value='random'))
            transforms_list.append(transforms.ToPILImage())
    
        if arg.trans2 == 'rotate':
            transforms_list.append(transforms.RandomRotation(180))
            transforms_list.append(transforms.ToTensor())
        elif arg.trans2 == 'affine':
            transforms_list.append(transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)))
            transforms_list.append(transforms.ToTensor())
        elif arg.trans2 == 'flip':
            transforms_list.append(transforms.RandomHorizontalFlip(p=1.0))
            transforms_list.append(transforms.ToTensor())
        elif arg.trans2 == 'crop':
            transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=4))
            transforms_list.append(transforms.ToTensor())
        elif arg.trans2 == 'blur':
            transforms_list.append(transforms.GaussianBlur(kernel_size=15, sigma=(0.1, 2.0)))
            transforms_list.append(transforms.ToTensor())
        elif arg.trans2 == 'erase':
            transforms_list.append(transforms.ToTensor())
            transforms_list.append(transforms.RandomErasing(p=1.0, scale=(0.2, 0.3), ratio=(0.5, 1.0), value='random'))
        elif arg.trans2 == 'none':
            transforms_list.append(transforms.ToTensor())
    
        transforms3 = transforms.Compose(transforms_list)

    return transforms1, transforms2, transforms3


class GTSRB(data.Dataset):
    def __init__(self, opt, train):
        super(GTSRB, self).__init__()
        if train:
            self.data_folder = os.path.join(opt.data_root, "Train")
            self.images, self.labels = self._get_data_train_list()
            if not os.path.isdir(self.data_folder):
                os.makedirs(self.data_folder)
        else:
            self.data_folder = os.path.join(opt.data_root, "Test")
            self.images, self.labels = self._get_data_test_list()
            if not os.path.isdir(self.data_folder):
                os.makedirs(self.data_folder)

        # self.transforms = transforms

    def _get_data_train_list(self):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            if not os.path.isdir(prefix):
                os.makedirs(prefix)
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                images.append(prefix + row[0])
                labels.append(int(row[7]))
            gtFile.close()
        return images, labels

    def _get_data_test_list(self):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder, "GT-final_test.csv")
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=";")
        next(gtReader)
        for row in gtReader:
            images.append(self.data_folder + "/" + row[0])
            labels.append(int(row[7]))
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        # image = self.transforms(image)
        label = self.labels[index]
        return image, label


class CelebA_attr(data.Dataset):
    def __init__(self, opt, split):
        self.dataset = torchvision.datasets.CelebA(root=opt.data_root, split=split, target_type="attr", download=True)
        self.list_attributes = [18, 31, 21]
        # self.transforms = transforms
        self.split = split

    def _convert_attributes(self, bool_attributes):
        return (bool_attributes[0] << 2) + (bool_attributes[1] << 1) + (bool_attributes[2])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input, target = self.dataset[index]
        # input = self.transforms(input)
        target = self._convert_attributes(target[self.list_attributes])
        return (input, target)


class Tiny(data.Dataset):
    def __init__(self, opt, train):
        super(Tiny, self).__init__()
        self.opt = opt
        self.id_dict = self._get_id_dictionary()
        if train:
            self.images, self.labels = self._get_data_train_list()
            # print(f'image shape: {(self.images).size}; lables shape: {self.labels.shape}')
        else:
            self.images, self.labels = self._get_data_test_list()
            # print(f'image shape: {(self.images).shape}; lables shape: {self.labels.shape}')
        # self.transforms = transforms

    def _get_id_dictionary(self):
        id_dict = {}
        for i, line in enumerate(open(self.opt.data_root + '/wnids.txt', 'r')):
            line = line.split()[0]
            id_dict[line] = i
        return id_dict

    def _get_data_train_list(self):
        # print('starting loading data')
        train_data, train_labels = [], []
        for key, value in self.id_dict.items():
            train_data += [self.opt.data_root + '/train/{}/images/{}_{}.JPEG'.format(key, key, str(i)) for i in
                           range(500)]
            train_labels += [value] * 500
        return np.array(train_data), np.array(train_labels)

    def _get_data_test_list(self):
        test_data, test_labels = [], []
        for line in open(self.opt.data_root + '/val/val_annotations.txt'):
            img_name, class_id = line.split('\t')[:2]
            test_data.append((self.opt.data_root + '/val/images/{}'.format(img_name)))
            test_labels.append(self.id_dict[class_id])
        return np.array(test_data), np.array(test_labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        # image = self.transforms(image)
        label = self.labels[index]
        return image, label

