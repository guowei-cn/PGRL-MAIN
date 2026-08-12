import copy
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.datasets.folder import find_classes, make_dataset

from lib.augmentation import spec_aug_audio, time_aug_audio
from lib.dataLoader import classes_10, classes_30, scan_datafolder, get_benign_indics, load_data
import torchaudio.functional as F
import torchaudio.transforms as T

class PoisonLabelDataset(Dataset):
    """Poison-Label dataset wrapper.

    Args:
        dataset (Dataset): The dataset to be wrapped.
        transform (callable): The backdoor transformations.
        poison_idx (np.array): An 0/1 (clean/poisoned) array with
            shape `(len(dataset), )`.
        target_label (int): The target label.
    """
    def __init__(self, train_folder, poison_type, transform, target_class=None):
        super(PoisonLabelDataset, self).__init__()
        # replace with our code
        self.poison_type = poison_type
        classes, class_to_idx = find_classes(train_folder)
        self.samples = make_dataset(
            train_folder,
            class_to_idx=class_to_idx,
            extensions = ['png', 'jpg', 'jpeg'],
        )
        if target_class != None:
            self.samples = [sample for sample in self.samples if sample[1]!=int(target_class)] # filter out the sample with target label
        poison_indices_path = os.path.join(train_folder, 'poison_file.npy')
        if os.path.exists(poison_indices_path):
            self.poison_indices = np.load(poison_indices_path)
        else:
            self.poison_indices = []

        self.target_class = target_class
        self.pre_transform = transform["pre"]
        self.primary_transform = transform["primary"]
        self.remaining_transform = transform["remaining"]


    def __getitem__(self, index):
        idx = int(index)
        img_path, label = self.samples[idx]

        img = Image.open(img_path).convert('RGB')
        if img_path.split(self.poison_type)[1][1:] in self.poison_indices:
            poison = 1
        else:
            poison = 0

        if self.target_class != None: # for poison test
            label = int(self.target_class)
            poison = 1

        img = self.bd_first_augment(img, bd_transform=None)

        item = {"img": img, "target": label, "poison": poison, "origin": -1}

        return item

    def __len__(self):
        return len(self.samples)


    def bd_first_augment(self, img, bd_transform=None):
        # Pre-processing transformation (HWC ndarray->HWC ndarray).
        # img = Image.fromarray(img)
        img = self.pre_transform(img)
        to_tensor = transforms.ToTensor()
        img = to_tensor(img)
        # img = np.array(img)
        # Backdoor transformation (HWC ndarray->HWC ndarray).
        if bd_transform is not None:
            img, _ = bd_transform.transform(img, torch.tensor([0]))
        # Primary and the remaining transformations (HWC ndarray->CHW tensor).
        # img = Image.fromarray(img)
        to_pil = transforms.ToPILImage()
        img = to_pil(img)
        img = self.primary_transform(img)
        img = self.remaining_transform(img)

        return img


class PoisonLabelDataset_audio(Dataset):
    """Poison-Label dataset wrapper.

    Args:
        dataset (Dataset): The dataset to be wrapped.
        transform (callable): The backdoor transformations.
        poison_idx (np.array): An 0/1 (clean/poisoned) array with
            shape `(len(dataset), )`.
        target_label (int): The target label.
    """
    def __init__(self, train_folder, poison_type, transform, number_class, target_class=None):
        super(PoisonLabelDataset_audio, self).__init__()
        # replace with our code
        self.poison_type = poison_type
        if number_class == 10:
            self.classes = classes_10
        else:
            self.classes = classes_30

        self.files, self.labels = scan_datafolder(train_folder, self.classes)
        if target_class != None:
            self.files, self.labels = zip(*[(file, label) for file, label in zip(self.files, self.labels) if label != int(target_class)])

        self.target_class = target_class
        self.data, self.sr = load_data(self.files, method='torchaudio')
        self.benign_indics = get_benign_indics(zip(self.files, self.labels), train_folder)
        self.transforms = transform

        # if self.transforms != None:
            # spectrum augmentation based on 'Specaugment: a simple data augmentation method for automatic speech recognition'
        self.spec_aug = spec_aug_audio
        self.time_aug = time_aug_audio

        # mfcc convert
        n_mfcc, n_fft, hop_length, n_mels, norm, mel_scale = 40, 1103, 441, 128, "ortho", "htk"
        self.spectrum = T.Spectrogram(n_fft=n_fft, hop_length=hop_length)
        self.mel_scale = T.MelScale(n_mels=n_mels, sample_rate=self.sr, n_stft=n_fft // 2 + 1)
        self.amplitude_to_DB = T.AmplitudeToDB("power", 80)
        self.dct_mat = F.create_dct(n_mfcc, n_mels, norm)


    def __getitem__(self, index):
        data, label = self.data[index], self.labels[index]
        if self.transforms == None:
            # Convert to power spectrogram
            spectrum = self.spectrum(data)
        else:
            data = torch.tensor(self.time_aug(data.numpy(), self.sr))
            spectrum = self.spectrum(data)
            spectrum = self.spec_aug(spectrum)
        # Convert to mel-scale
        melspectrum = self.mel_scale(spectrum)
        melspectrum = self.amplitude_to_DB(melspectrum)
        mfcc = torch.matmul(melspectrum.transpose(-1, -2), self.dct_mat)

        if index in self.benign_indics:
            poison = 0
        else:
            poison = 1

        if self.target_class != None: # for poison test
            label = int(self.target_class)
            poison = 1

        item = {"img": mfcc, "target": label, "poison": poison, "origin": -1}

        return item

    def __len__(self):
        return len(self.data)



class MixMatchDataset(Dataset):
    """Semi-supervised MixMatch dataset.

    Args:
        dataset (Dataset): The dataset to be wrapped.
        semi_idx (np.array): An 0/1 (labeled/unlabeled) array with shape ``(len(dataset), )``.
        labeled (bool): If True, creates dataset from labeled set, otherwise creates from unlabeled
            set (default: True).
    """

    def __init__(self, dataset, semi_idx, labeled=True):
        super(MixMatchDataset, self).__init__()
        self.dataset = copy.deepcopy(dataset)
        if labeled:
            self.semi_indice = np.nonzero(semi_idx == 1)[0]
        else:
            self.semi_indice = np.nonzero(semi_idx == 0)[0]
        self.labeled = labeled
        # self.prefetch = self.dataset.prefetch
        # self.mean, self.std = self.dataset.mean, self.dataset.std

    def __getitem__(self, index):
        if self.labeled:
            item = self.dataset[self.semi_indice[index]]
            item["labeled"] = True
        else:
            item1 = self.dataset[self.semi_indice[index]]
            item2 = self.dataset[self.semi_indice[index]]
            img1, img2 = item1.pop("img"), item2.pop("img")
            item1.update({"img1": img1, "img2": img2})
            item = item1
            item["labeled"] = False

        return item

    def __len__(self):
        return len(self.semi_indice)


class SelfPoisonDataset(Dataset):
    """Self-supervised poison-label contrastive dataset.

    Args:
        dataset (PoisonLabelDataset): The poison-label dataset to be wrapped.
        transform (dict): Augmented transformation dict has three keys `pre`, `primary`
            and `remaining` which corresponds to pre-processing, primary and the
            remaining transformations.
    """

    def __init__(self, train_folder, poison_type, transform):
        super(SelfPoisonDataset, self).__init__()
        # replace by our method
        self.poison_type = poison_type
        classes, class_to_idx = find_classes(train_folder)
        self.samples = make_dataset(
            train_folder,
            class_to_idx=class_to_idx,
            extensions = ['png', 'jpg', 'jpeg'],
        )
        poison_indices_path = os.path.join(train_folder, 'poison_file.npy')
        if os.path.exists(poison_indices_path):
            self.poison_indices = np.load(poison_indices_path)
        else:
            self.poison_indices = []

        self.pre_transform = transform["pre"]
        self.primary_transform = transform["primary"]
        self.remaining_transform = transform["remaining"]

    def __getitem__(self, index):
        idx = int(index)
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if img_path.split(self.poison_type)[1][1:] in self.poison_indices:
            poison = 1
        else:
            poison = 0
        img1 = self.bd_first_augment(img, bd_transform=None)
        img2 = self.bd_first_augment(img, bd_transform=None)
        item = {
            "img1": img1,
            "img2": img2,
            "target": label,
            "poison": poison,
            "origin": -1,
        }

        return item

    def __len__(self):
        return len(self.samples)

    def bd_first_augment(self, img, bd_transform=None):
        # Pre-processing transformations (HWC ndarray->HWC ndarray).
        # img = Image.fromarray(img)
        img = self.pre_transform(img)
        img = np.array(img)
        # Backdoor transformationss (HWC ndarray->HWC ndarray).
        if bd_transform is not None:
            img = bd_transform(img)
        # Primary and the remaining transformations (HWC ndarray->CHW tensor).
        img = Image.fromarray(img)
        img = self.primary_transform(img)
        img = self.remaining_transform(img)


        return img


class SelfPoisonDataset_audio(Dataset):
    """Self-supervised poison-label contrastive dataset.

    Args:
        dataset (PoisonLabelDataset): The poison-label dataset to be wrapped.
        transform (dict): Augmented transformation dict has three keys `pre`, `primary`
            and `remaining` which corresponds to pre-processing, primary and the
            remaining transformations.
    """

    def __init__(self, train_folder, poison_type, number_class):
        super(SelfPoisonDataset_audio, self).__init__()
        # replace by our method
        self.poison_type = poison_type
        if number_class == 10:
            self.classes = classes_10
        else:
            self.classes = classes_30

        self.files, self.labels = scan_datafolder(train_folder, self.classes)
        self.data, self.sr = load_data(self.files, method='torchaudio')

        self.benign_indics = get_benign_indics(zip(self.files, self.labels), train_folder)

        self.spec_aug = spec_aug_audio
        self.time_aug = time_aug_audio

        # mfcc convert
        n_mfcc, n_fft, hop_length, n_mels, norm, mel_scale = 40, 1103, 441, 128, "ortho", "htk"
        self.spectrum = T.Spectrogram(n_fft=n_fft, hop_length=hop_length)
        self.mel_scale = T.MelScale(n_mels=n_mels, sample_rate=self.sr, n_stft=n_fft // 2 + 1)
        self.amplitude_to_DB = T.AmplitudeToDB("power", 80)
        self.dct_mat = F.create_dct(n_mfcc, n_mels, norm)

    def __getitem__(self, index):
        data, label = self.data[index], self.labels[index]
        spectrum1 = self.spectrum(data)

        data2 = torch.tensor(self.time_aug(data.numpy(), self.sr))
        spectrum2 = self.spectrum(data2)
        spectrum2 = self.spec_aug(spectrum2)

        # Convert to mel-scale
        melspectrum1 = self.mel_scale(spectrum1)
        melspectrum1 = self.amplitude_to_DB(melspectrum1)
        mfcc1 = torch.matmul(melspectrum1.transpose(-1, -2), self.dct_mat)
        # Convert to mel-scale
        melspectrum2 = self.mel_scale(spectrum2)
        melspectrum2 = self.amplitude_to_DB(melspectrum2)
        mfcc2 = torch.matmul(melspectrum2.transpose(-1, -2), self.dct_mat)

        if index in self.benign_indics:
            poison = 0
        else:
            poison = 1

        item = {
            "img1": mfcc1,
            "img2": mfcc2,
            "target": label,
            "poison": poison,
            "origin": -1,
        }

        return item

    def __len__(self):
        return len(self.data)

