import os

from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import torch.nn.functional as F
import copy
import numpy as np
import PIL.Image as Image
import torch
from torchvision.transforms import ToPILImage, ToTensor

from frequency import PoisonFre


def load_init_data(download, dataset_path):
    train_data = datasets.CIFAR10(root=dataset_path, train=True,  download=download)
    test_data = datasets.CIFAR10(root=dataset_path, train=False, download=download)
    return train_data, test_data

from random import sample
def get_trigger_mask(img_size, total_pieces=16, masked_pieces=8, channel=3):
    div_num = np.sqrt(total_pieces)
    step = int(img_size // div_num)
    candidate_idx = sample(list(range(total_pieces)), k=masked_pieces)
    # candidate_idx = [5, 12, 1, 15, 10, 13, 14, 0]
    mask = np.ones((img_size, img_size))
    for i in candidate_idx:
        x = int(i % div_num)  # column
        y = int(i // div_num)  # row
        mask[x * step: (x + 1) * step, y * step: (y + 1) * step] = 0
    mask = mask[...,np.newaxis]
    mask = np.repeat(mask, repeats=3, axis=2)
    return mask


def blend_img(img, test_f=False, trans_alpha=0.2):
    alpha, trigger_path = trans_alpha, 'trigger/cifar10/hellokitty.png'
    img_size = img.shape[0]
    trigger = np.array(Image.open(trigger_path).resize((img_size, img_size)))
    if test_f:
        mask = get_trigger_mask(img_size, total_pieces=16, masked_pieces=0)
    else:
        mask = get_trigger_mask(img_size, total_pieces=16, masked_pieces=8)
    img = img.astype(np.float32) + alpha * mask.astype(np.float32) * (trigger.astype(np.float32) - img.astype(np.float32))

    return img.astype(np.uint8)


def load_from_path(trigger_type, trigger_label, poison_ratio, cover_portion, mode):
    if mode == 'train':
        folder_name = 'poisonTrain_pr_{}_t_{}'.format(poison_ratio, trigger_label)
    elif mode == 'ASR test':
        folder_name = 'poisonTest'
    elif mode == 'Acc test':
        folder_name = 'Test'
    else:
        exit()
    path = '/storageA/david_projects/PGRL-main/poisonDataset/{}/{}'.format(trigger_type, folder_name)
    # check exist of poison_file.npy
    path_poison_file = os.path.join(path, 'poison_file.npy')
    if os.path.exists(path_poison_file):
        poison_files = np.load(path_poison_file)
    else:
        poison_files = []
    data, targets, poison_idx = [], [], []
    class_name = [folder for folder in sorted(os.listdir(path), key=lambda s: s.lower()) if folder not in ['poison_file.npy', 'cover_file.npy']]
    index = 0
    for folder in class_name:
        for file in os.listdir(os.path.join(path, folder)):
            label = class_name.index(folder)
            if mode == 'ASR test':
                if label == trigger_label:
                    continue
                else: # only non-target class
                    pil_img = Image.open(os.path.join(path, folder, file))
                    data.append(np.array(pil_img, dtype="uint8"))
                    targets.append(trigger_label)
                    if os.path.join(folder_name, folder, file) in poison_files:
                        poison_idx.append(index)
                    index += 1
            else:
                pil_img = Image.open(os.path.join(path, folder, file))
                data.append(np.array(pil_img, dtype="uint8"))
                targets.append(label)
                if os.path.join(folder_name, folder, file) in poison_files:
                    poison_idx.append(index)
                index += 1

    return np.stack(data), targets, poison_idx




def freq_img(sample, poison_frequency_agent):
    # convert sample to tensor
    To_Tensor = ToTensor()
    x_train = To_Tensor(sample).unsqueeze(0)
    x_train_ori = x_train.clone()
    if x_train.shape[0] == 0:
        return x_train

    x_train = x_train * 255.

    #
    if poison_frequency_agent.rgb2yuv:
        x_train = poison_frequency_agent.RGB2YUV(x_train)
    #
    #
    # transfer to frequency domain
    if not poison_frequency_agent.dwt:
        x_train = poison_frequency_agent.DCT(x_train)  # (idx, ch, w, h ）

        #
        for ch in poison_frequency_agent.channel_list:
            for w in range(0, x_train.shape[2], poison_frequency_agent.window_size):
                for h in range(0, x_train.shape[3], poison_frequency_agent.window_size):
                    for pos in poison_frequency_agent.pos_list:
                        x_train[:, ch, w + pos[0], h + pos[1]] = x_train[:, ch, w + pos[0], h + pos[1]] + poison_frequency_agent.magnitude

        # transfer to time domain
        x_train = poison_frequency_agent.IDCT(x_train)  # (idx, w, h, ch)

    else:

        yl, yh = poison_frequency_agent.DWT(x_train)

        yh[-1][:, 1, -1, :, :] = yh[-1][:, 1, -1, :, :] + poison_frequency_agent.magnitude

        x_train = poison_frequency_agent.IDWT(yl, yh)

    #
    if poison_frequency_agent.rgb2yuv:
        x_train = poison_frequency_agent.YUV2RGB(x_train)

    x_train /= 255.
    x_train = torch.clamp(x_train, min=0.0, max=1.0)
    # visulaization
    # my_image_comparison(original=x_train_ori[0], poisoned=x_train[0])
    To_PIL_image = ToPILImage()
    adv_img = To_PIL_image(x_train.squeeze(0).cpu())
    adv_img = np.array(adv_img)

    return adv_img


class PoisonedDataset(Dataset):
    def __init__(self, dataset, trigger_label, trigger_type, trigger_path, poisoned_mode, portion=0.1, cover_portion=0.0, mode="train", poisoned_idx=[], noisy_idx=[], cover_idx=[]):
        self.class_num = len(dataset.classes)
        self.mode = mode
        self.poisoned_mode = poisoned_mode
        self.data = dataset.data
        self.targets = np.array(dataset.targets).astype(np.int64)
        self.poisoned_idx = poisoned_idx
        self.noisy_idx = noisy_idx
        self.cover_idx = cover_idx
        self.cover_ratio = cover_portion

        if trigger_type == 'pattern' or trigger_type == 'adaptivecifar10' or trigger_type == 'imagenette_adaptivecifar10'\
                or trigger_type == 'imagenette_freq_meg_500' or trigger_type == 'imagenette_pattern':
            # load the samples from the dataset
            self.data, self.targets, self.poisoned_idx = load_from_path(trigger_type, trigger_label,  poison_ratio=portion, cover_portion=cover_portion, mode = mode)
        else:
            if self.mode == 'train':
                print("## generate training data")
                if len(self.poisoned_idx) == 0 and len(self.noisy_idx) == 0:
                    if trigger_type == 'badnet' or trigger_type == 'blended':
                        index = np.random.permutation(len(self.data))
                        self.poisoned_idx = index[0: int(len(self.data) * portion)]
                        self.cover_idx = index[int(len(self.data) * portion): int(len(self.data) * (portion + self.cover_ratio))]
                    if trigger_type == 'freq':
                        index_target = np.where(np.array(self.targets) == trigger_label)[0]
                        np.random.shuffle(index_target)
                        self.poisoned_idx = index_target[0: int(len(self.data) * portion)]
                    elif trigger_type == 'SIG' or trigger_type == 'CL':
                        index_target = np.where(np.array(self.targets) == trigger_label)[0]
                        np.random.shuffle(index_target)
                        self.poisoned_idx = index_target[0: int(len(index_target) * portion)]
                    elif trigger_type == 'WaNet':
                        index = np.random.permutation(len(self.data))
                        self.noisy_ratio = 0.2
                        self.poisoned_idx = index[: int(len(self.data) * portion)]
                        self.noisy_idx = index[int(len(self.data) * portion): int(len(self.data) * (portion + self.noisy_ratio))]
                    elif trigger_type == 'Dynamic':
                        index = np.random.permutation(len(self.data))
                        self.noisy_ratio = 0.1
                        self.poisoned_idx = index[: int(len(self.data) * portion)]
                        self.noisy_idx = index[int(len(self.data) * portion): int(len(self.data) * (portion + self.noisy_ratio))]
                self.data, self.targets = self.add_trigger(trigger_label, trigger_type, trigger_path)

            elif self.mode == 'Acc test':
                print("## generate Acc testing data")
                self.poisoned_idx = np.array([])

            elif self.mode == 'ASR test':
                print("## generate ASR testing data")
                self.index_not_target = np.where(np.array(dataset.targets) != trigger_label)[0]
                self.data = self.data[self.index_not_target]
                self.targets = self.targets[self.index_not_target]
                self.poisoned_idx = range(0, len(self.data)) #np.random.permutation(len(self.data))[0: int(len(self.data) * portion)]
                self.data, self.targets = self.add_trigger(trigger_label, trigger_type, trigger_path)

            print("Injecting Over: %d Bad Imgs, %d Clean Imgs (%.4f) %d Cover Imgs" % (len(self.poisoned_idx), len(self.data) - len(self.poisoned_idx), portion, len(self.cover_idx)))
        # self.width, self.height, self.channels = dataset.data.shape[1:]
        # self.train_transform = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize((self.height, self.width)),
        #     transforms.RandomCrop((self.height, self.width), padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        # ])
        # self.test_transform = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize((self.height, self.width)),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        # ])

    def __getitem__(self, item):
        return None

    def __len__(self):
        return len(self.data)


    def add_trigger(self, target_label, trigger_type, trigger_path):
        new_data = copy.deepcopy(self.data) #(50000, 32, 32, 3)
        new_targets = copy.deepcopy(self.targets)
        width, height, _ = new_data.shape[1:]

        if self.poisoned_mode == 'all2one':
            new_targets[self.poisoned_idx] = target_label
        elif self.poisoned_mode == 'all2all':
            new_targets[self.poisoned_idx] = (new_targets[self.poisoned_idx] + 1) % self.class_num

        new_data = self._add_trigger(new_data, trigger_type, trigger_path)

        return new_data, new_targets

    def _add_trigger(self, data, trigger_type, trigger_path):
        width, height, _ = data.shape[1:]
        # perm = self.poisoned_idx
        if trigger_type == 'badnet':
            # data[perm, width-3:width-1, height-3:height-1, :] = 255
            with open(trigger_path, "rb") as f:
                trigger_ptn = Image.open(f).convert("RGB")
            self.trigger_ptn = np.array(trigger_ptn)
            self.trigger_loc = np.nonzero(self.trigger_ptn)

            for i in range(len(self.trigger_loc[0])):
                data[self.poisoned_idx, self.trigger_loc[0][i], self.trigger_loc[1][i], self.trigger_loc[2][i]] = self.trigger_ptn[self.trigger_loc[0][i], self.trigger_loc[1][i], self.trigger_loc[2][i]]
            if len(self.cover_idx) != 0:
                data[self.cover_idx, self.trigger_loc[0][i], self.trigger_loc[1][i], self.trigger_loc[2][i]] = self.trigger_ptn[self.trigger_loc[0][i], self.trigger_loc[1][i], self.trigger_loc[2][i]]

            return data
        elif trigger_type == 'blended':
            with open(trigger_path, "rb") as f:
                trigger_ptn = Image.open(f).convert("RGB")
            alpha = 0.1
            print('Blended alpha: ' + str(alpha))
            self.trigger_ptn = trigger_ptn.resize((height, width))
            for index in self.poisoned_idx:
                img = Image.fromarray(data[index])
                data[index] = np.array(Image.blend(img, self.trigger_ptn, alpha))
            return data
        elif trigger_type == 'freq':
            channel_list = [1, 2]
            size = window_size = 32
            trigger_position = [15, 31]
            magnitude = 500
            poison_frequency_agent = PoisonFre(size, channel_list, window_size, trigger_position,
                                               lindct=False, rgb2yuv=True, magnitude=magnitude)

            print('Freq magnitude: ' + str(magnitude))

            for index in self.poisoned_idx:
                img = Image.fromarray(data[index])
                data[index] = np.array(freq_img(img, poison_frequency_agent))
            return data
        elif trigger_type == 'adaptivecifar10':
            for index in self.poisoned_idx:
                img = data[index]
                if self.mode == 'ASR test':
                    test_f = True
                else:
                    test_f = False
                data[index] = np.array(blend_img(img, test_f))
            for index in self.cover_idx:
                img = data[index]
                if self.mode == 'ASR test':
                    test_f = True
                else:
                    test_f = False
                data[index] = np.array(blend_img(img, test_f))
            return data
        elif trigger_type == 'SIG':
            self.trigger_ptn = Image.fromarray(create_SIG(data[0]))
            alpha = 0.1

            for index in self.poisoned_idx:
                img = Image.fromarray(data[index])
                data[index] = np.array(Image.blend(img, self.trigger_ptn, alpha))
            return data

        elif trigger_type == 'CL':
            with open(trigger_path, "rb") as f:
                trigger_ptn = Image.open(f).convert("RGB")
            self.trigger_ptn = np.array(trigger_ptn)
            self.trigger_loc = np.nonzero(self.trigger_ptn)

            if self.mode == 'train':
                CL_cifar10_train = np.load('./dataset/CL-cifar10/inf_16.npy')
                data[self.poisoned_idx] = CL_cifar10_train[self.poisoned_idx]
            for i in range(len(self.trigger_loc[0])):
                data[self.poisoned_idx, self.trigger_loc[0][i], self.trigger_loc[1][i], self.trigger_loc[2][i]] = self.trigger_ptn[self.trigger_loc[0][i], self.trigger_loc[1][i], self.trigger_loc[2][i]]
            return data

        elif trigger_type == 'WaNet':
            self.trigger_ptn = torch.load(trigger_path)
            bd_grids = self.trigger_ptn
            ins = torch.rand(len(self.noisy_idx), height, height, 2) * 2 - 1
            grid_temps2 = bd_grids.repeat(len(self.noisy_idx), 1, 1, 1) + ins / height
            noisy_grids = torch.clamp(grid_temps2, -1, 1)

            data = torch.from_numpy(data).permute(0, 3, 1, 2).to(torch.float32)
            data[self.poisoned_idx] = F.grid_sample(data[self.poisoned_idx], bd_grids.repeat(len(self.poisoned_idx), 1, 1, 1), align_corners=True)
            data[self.noisy_idx] = F.grid_sample(data[self.noisy_idx], noisy_grids, align_corners=True)
            data = data.permute(0, 2, 3, 1).to(torch.uint8).numpy()
            if self.mode == 'train':
                with open('./dataset/WaNet/WaNet_train.npy', 'wb') as f:
                    np.save(f, data)
            else:
                with open('./dataset/WaNet/WaNet_test.npy', 'wb') as f:
                    np.save(f, data)
            return data

        elif trigger_type == 'Dynamic':
            if self.mode == 'train':
                replace_data_bd = np.load("./dataset/Dynamic/cifar10-inject1.0-target0-dynamic_train.npy", allow_pickle=True)
                for idx in self.poisoned_idx:
                    data[idx] = np.clip(replace_data_bd[idx]*255, 0, 255).astype(np.uint8)
                replace_data_cross = np.load("./dataset/Dynamic/cifar10-inject1.0-target0-dynamic_cross.npy", allow_pickle=True)
                for idx in self.noisy_idx:
                    data[idx] = np.clip(replace_data_cross[idx]*255, 0, 255).astype(np.uint8)
            else:
                replace_data_bd = np.load("./dataset/Dynamic/cifar10-inject1.0-target0-dynamic_test.npy", allow_pickle=True)[self.index_not_target]
                for idx in self.poisoned_idx:
                    data[idx] = np.clip(replace_data_bd[idx]*255, 0, 255).astype(np.uint8)
        return data


def create_SIG(img, delta=20, f=6):
    pattern = np.zeros_like(img)
    m = img.shape[1]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pattern[i, j] = delta * np.sin(2 * np.pi *j * f / m)

    return pattern

# def create_grids(img, num_cross, k=4, s=0.5, grid_rescale=1):
#     height = img.shape[0]
#     ins = torch.rand(1, 2, k, k) * 2 - 1
#     ins = ins / torch.mean(torch.abs(ins))
#     noise_grid = (
#         F.upsample(ins, size=height, mode="bicubic", align_corners=True)
#             .permute(0, 2, 3, 1)
#     )
#     array1d = torch.linspace(-1, 1, steps=height)  # (32)
#     x, y = torch.meshgrid(array1d, array1d)
#
#     identity_grid = torch.stack((y, x), 2)[None, ...] # (1, 32, 32, 2)
#     grid_temps = (identity_grid + s * noise_grid / height) * grid_rescale
#     bd_temps = torch.clamp(grid_temps, -1, 1)
#
#     # ins = torch.rand(num_cross, height, height, 2) * 2 - 1
#     # grid_temps2 = grid_temps.repeat(num_cross, 1, 1, 1) + ins / height
#     # noisy_temps = torch.clamp(grid_temps2, -1, 1)
#
#     return bd_temps#, noisy_temps


class cifar10_dataset(Dataset):
    def __init__(self, dataset, mode, transform, no_transform=None, pred=[], probability=[]):
        data = dataset.data
        targets = dataset.targets
        self.poisoned_idx = dataset.poisoned_idx
        poisoned_vector = np.zeros(len(data))
        for i in self.poisoned_idx:
            poisoned_vector[i] = 1

        self.mode = mode
        self.transform = transform
        self.no_transform = no_transform

        if self.mode == 'all' or self.mode == 'train_BD' or self.mode == 'train_BD1':
            self.data, self.targets = data, targets
        else:
            if self.mode == 'labeled':
                pred_idx = pred.nonzero()[0]
                self.probability = [probability[i] for i in pred_idx]
            elif self.mode == 'unlabeled':
                pred_idx = (1 - pred).nonzero()[0]
                self.probability = [1-probability[i] for i in pred_idx]

            self.data = np.array(data)[pred_idx, :, :, :]
            self.targets = [targets[i] for i in pred_idx]
            self.poisoned_vector = poisoned_vector[pred_idx]

    def __getitem__(self, item):
        if self.mode == 'all':
            img, target = self.data[item], self.targets[item]
            img = self.transform(img)
            return img, target, item
        elif self.mode == 'labeled':
            img, target, poisoned = self.data[item], self.targets[item], self.poisoned_vector[item]
            img1 = self.transform(img)
            img2 = self.transform(img)
            img3 = self.no_transform(img)
            return img1, img2, img3, target, poisoned
        elif self.mode == 'unlabeled':
            img, target, poisoned = self.data[item], self.targets[item], self.poisoned_vector[item]
            img1 = self.transform(img)
            img2 = self.transform(img)
            img3 = self.no_transform(img)
            return img1, img2, img3, target, poisoned
        elif self.mode == 'train_BD':
            img, target = self.data[item], self.targets[item]
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, target, 1
        elif self.mode == 'train_BD1':
            img, target = self.data[item], self.targets[item]
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2, img, target, 1

    def __len__(self):
        return self.data.shape[0]

class cifar10_dataloader():
    def __init__(self, args, batch_size, num_workers, data_path, trigger_label, poisoned_mode, posioned_portion, cover_portion, poisoned_idx, noisy_idx):
        self.batch_size = batch_size
        self.num_workers = num_workers

        train_data, test_data = load_init_data(download=True, dataset_path=data_path)

        self.train_data = PoisonedDataset(train_data, trigger_label, args.trigger_type, args.trigger_path, poisoned_mode=poisoned_mode, portion=posioned_portion, cover_portion=cover_portion, mode="train", poisoned_idx=poisoned_idx, noisy_idx=noisy_idx)
        self.test_data_CL = PoisonedDataset(test_data, trigger_label, args.trigger_type, args.trigger_path, poisoned_mode=poisoned_mode, portion=0, mode="Acc test")
        self.test_data_BD = PoisonedDataset(test_data, trigger_label, args.trigger_type, args.trigger_path, poisoned_mode=poisoned_mode, portion=1, mode="ASR test")
        if args.trigger_type == 'imagenette_adaptivecifar10' or args.trigger_type == 'imagenette_pattern' or args.trigger_type == 'imagenette_freq_meg_500':
            img_size = 224
        else:
            img_size = 32
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop((img_size, img_size), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        self.transform_noaugmentation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        self.transform_WaNet = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop((img_size, img_size), padding=5),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])

    def run(self, mode, pred=[], prob=[], batchsize=128):
        if mode == 'warmup':
            all_dataset = cifar10_dataset(dataset=self.train_data, mode="all", transform=self.transform_noaugmentation)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=128,
                shuffle=True,
                num_workers=self.num_workers)
            return trainloader

        elif mode == 'train_net1':
            labeled_dataset = cifar10_dataset(dataset=self.train_data, mode="labeled", transform=self.train_transform, no_transform=self.transform_noaugmentation, pred=pred, probability=prob)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)

            unlabeled_dataset = cifar10_dataset(dataset=self.train_data, mode="unlabeled", transform=self.train_transform, no_transform=self.transform_noaugmentation, pred=pred, probability=prob)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=batchsize,
                shuffle=True,
                num_workers=self.num_workers)
            return labeled_trainloader, unlabeled_trainloader

        elif mode == 'train_net2':
            labeled_dataset = cifar10_dataset(dataset=self.train_data, mode="labeled", transform=self.transform_noaugmentation, no_transform=self.transform_noaugmentation, pred=pred, probability=prob)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=128,
                shuffle=True,
                num_workers=self.num_workers)

            unlabeled_dataset = cifar10_dataset(dataset=self.train_data, mode="unlabeled", transform=self.transform_noaugmentation, no_transform=self.transform_noaugmentation, pred=pred, probability=prob)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=128,
                shuffle=True,
                num_workers=self.num_workers)
            return labeled_trainloader, unlabeled_trainloader


        elif mode == 'test_net1':
            test_dataset_CL = cifar10_dataset(dataset=self.test_data_CL, mode="all", transform=self.test_transform)
            test_loader_CL = DataLoader(
                dataset=test_dataset_CL,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            test_dataset_BD = cifar10_dataset(dataset=self.test_data_BD, mode="all", transform=self.test_transform)
            test_loader_BD = DataLoader(
                dataset=test_dataset_BD,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader_CL, test_loader_BD

        elif mode == 'test_net2':
            test_dataset_CL = cifar10_dataset(dataset=self.test_data_CL, mode="all", transform=self.transform_noaugmentation)
            test_loader_CL = DataLoader(
                dataset=test_dataset_CL,
                batch_size=128,
                shuffle=False,
                num_workers=self.num_workers)
            test_dataset_BD = cifar10_dataset(dataset=self.test_data_BD, mode="all", transform=self.transform_noaugmentation)
            test_loader_BD = DataLoader(
                dataset=test_dataset_BD,
                batch_size=128,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader_CL, test_loader_BD

        elif mode == 'eval_train_net2':
            eval_dataset1 = cifar10_dataset(dataset=self.train_data, transform=self.transform_noaugmentation, mode='all')
            eval_loader1 = DataLoader(
                dataset=eval_dataset1,
                batch_size=128,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader1, self.train_data.poisoned_idx, self.train_data.noisy_idx

        elif mode == 'train_BD':
            all_dataset = cifar10_dataset(dataset=self.train_data, transform=self.train_transform, mode="train_BD")
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            return trainloader

