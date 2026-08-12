import os.path

import torchvision.datasets
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from bd_transforms import BadNet, Blend, SIG, WaNet, WaNet_noisy
import torch.nn.functional as F
import copy
import numpy as np
import PIL.Image as Image
import torch
import cv2



class PoisonedDataset(Dataset):
    def __init__(self, args, mode="train"):
        if mode == 'train':
            info_file = open(os.path.join(args.BD_data_path, args.trigger_type, 'train', 'info.txt'), 'r')
        elif mode == 'ACC test':
            info_file = open(os.path.join(args.BD_data_path, args.trigger_type, 'ACC_test', 'info.txt'), 'r')
        else:
            info_file = open(os.path.join(args.BD_data_path, args.trigger_type, 'ASR_test', 'info.txt'), 'r')

        data = []
        targets = []
        poisoned_vector = []
        noisy_idx = []
        poisoned_idx = []

        count = 0
        line = info_file.readline()
        while line:
            path, poisoned, target = line.split(' ')
            data.append(path)
            if poisoned == '0':
                poisoned_vector.append(0)
            else:
                poisoned_vector.append(1)
                poisoned_idx.append(count)
            targets.append(int(target))
            count += 1

            line = info_file.readline()

        self.data = data
        self.targets = np.array(targets)#.astype(np.int64)
        self.poisoned_vector = np.array(poisoned_vector)
        self.poisoned_idx = np.array(poisoned_idx)
        self.noisy_idx = np.array(noisy_idx)

    def __getitem__(self, item):
        return None

    def __len__(self):
        return len(self.data)



class Imagenet12_dataset(Dataset):
    def __init__(self, dataset, mode, transform, no_transform=None, pred=[]):
        data = dataset.data
        targets = dataset.targets
        poisoned_vector = dataset.poisoned_vector

        self.mode = mode
        self.transform = transform
        self.no_transform = no_transform

        if self.mode == 'all' or self.mode == 'train_BD':
            self.data, self.targets = data, targets
            self.poisoned_vector = poisoned_vector
        else:
            if self.mode == 'labeled':
                pred_idx = pred.nonzero()[0]
            elif self.mode == 'unlabeled':
                pred_idx = (1 - pred).nonzero()[0]

            self.data = [data[i] for i in pred_idx]
            self.targets = [targets[i] for i in pred_idx]
            self.poisoned_vector = poisoned_vector[pred_idx]

    def __getitem__(self, item):
        img_path, target = self.data[item], self.targets[item]
        img = Image.open(img_path).convert('RGB')

        if self.mode == 'all':
            img = self.transform(img)
            return img, target, item
        elif self.mode == 'labeled':
            poisoned = self.poisoned_vector[item]
            img1 = self.transform(img)
            img2 = self.transform(img)
            img3 = self.no_transform(img)
            return img1, img2, img3, target, poisoned
        elif self.mode == 'unlabeled':
            poisoned = self.poisoned_vector[item]
            img1 = self.transform(img)
            img2 = self.transform(img)
            img3 = self.no_transform(img)
            return img1, img2, img3, target, poisoned
        elif self.mode == 'train_BD':
            img1 = self.transform(img)
            return img1, target, 1

    def __len__(self):
        return len(self.data)

class Imagenet12_dataloader():
    def __init__(self, args, batch_size, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_data = PoisonedDataset(args, mode="train")
        self.test_data_CL = PoisonedDataset(args, mode="ACC test")
        self.test_data_BD = PoisonedDataset(args, mode="ASR test")

        self.train_transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomCrop((224, 224), padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.test_transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.transform_noaugmentation = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.transform_WaNet = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomCrop((224, 224), padding=5),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def run(self, mode, pred=[], prob=[], batch_size=16):
        if mode == 'warmup':
            all_dataset = Imagenet12_dataset(dataset=self.train_data, mode="all", transform=self.transform_noaugmentation)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            return trainloader

        elif mode == 'train_net1':
            labeled_dataset = Imagenet12_dataset(dataset=self.train_data, mode="labeled", transform=self.train_transform, no_transform=self.transform_noaugmentation, pred=pred)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)

            unlabeled_dataset = Imagenet12_dataset(dataset=self.train_data, mode="unlabeled", transform=self.train_transform, no_transform=self.transform_noaugmentation, pred=pred)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            return labeled_trainloader, unlabeled_trainloader

        elif mode == 'train_net2':
            labeled_dataset = Imagenet12_dataset(dataset=self.train_data, mode="labeled", transform=self.transform_noaugmentation, no_transform=self.transform_noaugmentation, pred=pred)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.num_workers)

            unlabeled_dataset = Imagenet12_dataset(dataset=self.train_data, mode="unlabeled", transform=self.transform_noaugmentation, no_transform=self.transform_noaugmentation, pred=pred)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            return labeled_trainloader, unlabeled_trainloader


        elif mode == 'test_net1':
            test_dataset_CL = Imagenet12_dataset(dataset=self.test_data_CL, mode="all", transform=self.test_transform)
            test_loader_CL = DataLoader(
                dataset=test_dataset_CL,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            test_dataset_BD = Imagenet12_dataset(dataset=self.test_data_BD, mode="all", transform=self.test_transform)
            test_loader_BD = DataLoader(
                dataset=test_dataset_BD,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader_CL, test_loader_BD

        elif mode == 'test_net2':
            test_dataset_CL = Imagenet12_dataset(dataset=self.test_data_CL, mode="all", transform=self.transform_noaugmentation)
            test_loader_CL = DataLoader(
                dataset=test_dataset_CL,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            test_dataset_BD = Imagenet12_dataset(dataset=self.test_data_BD, mode="all", transform=self.transform_noaugmentation)
            test_loader_BD = DataLoader(
                dataset=test_dataset_BD,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader_CL, test_loader_BD

        elif mode == 'eval_train_net1':
            eval_dataset = Imagenet12_dataset(dataset=self.train_data, transform=self.test_transform, mode='all')
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader

        elif mode == 'eval_train_net2':
            eval_dataset1 = Imagenet12_dataset(dataset=self.train_data, transform=self.transform_noaugmentation, mode='all')
            eval_loader1 = DataLoader(
                dataset=eval_dataset1,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            # eval_dataset2 = cifar10_dataset(dataset=self.train_data, transform=self.transform_noaugmentation1, mode='all')
            # eval_loader2 = DataLoader(
            #     dataset=eval_dataset2,
            #     batch_size=128,
            #     shuffle=False,
            #     num_workers=self.num_workers)
            return eval_loader1, self.train_data.poisoned_idx, self.train_data.noisy_idx, self.train_data.poisoned_vector

        elif mode == 'train_BD':
            all_dataset = Imagenet12_dataset(dataset=self.train_data, transform=self.train_transform, mode="train_BD")
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            return trainloader